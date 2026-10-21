"""
Multimodal (Image + Document) Subsystem for Parentheses 0.9.

Operates strictly under the Mamba-2 / RWKV / Selective Linear Attention (SSM)
paradigm:
  - Recurrent SSM Visual Patch Projector (`SSMVisualProjector`, `prefill_visual_state`):
    tokenizes 2D images into non-overlapping raster patches, projects them into
    `n_embd` using a deterministic Xavier/orthogonal projection derived from
    `model.tok_emb.weight` (zero external checkpoint required), and steps each
    patch sequentially through the Parentheses SSM blocks via `block.step(x_p, state)`.
    Maintains an exact constant O(1) recurrent state size `(B, n_head, head_dim, head_dim)`
    per layer across any number of patches or images -- zero O(N^2) ViT self-attention
    matrix and zero growing KV-cache.
  - Structured Visual Descriptor (`ImageProcessor`, `describe_image`):
    computes deterministic visual attributes (resolution, aspect ratio, color mode,
    dominant RGB/HSL palette names, Rec.601 luminance mean/std, RMS & Michelson
    contrast, Sobel gradient edge/texture density, 2x2 spatial quadrant layout,
    and embedded PNG/EXIF metadata) and formats a grounded natural-language summary
    (supporting English and Dravidian tags) for RAG indexing and prompt grounding.
  - Universal Document Processor (`DocumentProcessor`, `extract_document_text`):
    pure-Python extraction of clean UTF-8 text (preserving English and Dravidian
    scripts `Knda`, `Taml`, `Telu`, `Mlym`, `Orya`) from PDF (`.pdf`, uncompressed
    and `/FlateDecode` zlib streams, `Tj`/`TJ`/UTF-16BE `<FEFF...>` hex strings),
    Word (`.docx`, paragraphs + tables), HTML/XML (`.html`, `.htm`, `.xml` with
    `<script>`/`<style>` removal), CSV/TSV (`.csv`, `.tsv`), JSON/JSONL (`.json`,
    `.jsonl`), plain text/code (`.txt`, `.md`, `.rst`, `.py`), and images
    (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`).

    python -m features.multimodal --self-test
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import math
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
import zlib
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import ExifTags, Image

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif", ".tiff"})
DOCUMENT_EXTENSIONS = frozenset({
    ".pdf", ".docx", ".html", ".htm", ".xml",
    ".csv", ".tsv", ".json", ".jsonl",
    ".txt", ".md", ".rst", ".py",
})
MULTIMODAL_EXTENSIONS = IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS

# Localized visual summary headers for BPE Dravidian & English prompts
_LANG_VISUAL_HEADER = {
    "eng": "Visual Image Summary",
    "kan": "ಚಿತ್ರದ ದೃಶ್ಯ ವಿವರಣೆ (Visual Summary)",
    "tam": "படத்தின் காட்சி சுருக்கம் (Visual Summary)",
    "tel": "చిత్ర దృశ్య సారాంశం (Visual Summary)",
    "mal": "ചിത്രത്തിന്റെ ദൃശ്യ സംഗ്രഹം (Visual Summary)",
    "tcy": "ಚಿತ್ರೊದ ವಿವರಣೆ (Visual Summary)",
    "kff": "ಚಿತ್ರ ವಿವರಣೆ (Visual Summary)",
    "kxv": "ಚಿತ್ರ ವಿವರಣೆ (Visual Summary)",
    "nit": "చిత్ర వివరణ (Visual Summary)",
}


class VisualDescription(dict):
    """Dictionary of structured image attributes whose `str()` is the natural-language summary."""

    @property
    def summary(self) -> str:
        return str(self.get("summary", ""))

    @property
    def text(self) -> str:
        return str(self.get("summary", ""))

    def to_text(self) -> str:
        return str(self.get("summary", ""))

    def __str__(self) -> str:
        return str(self.get("summary", ""))


# ---------------------------------------------------------------------------
# Part A: Image Loading, Visual Feature Analysis & Recurrent SSM State Prefill
# ---------------------------------------------------------------------------

def load_image(source: str | Path | bytes | bytearray | Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    """Load an image from a file path, raw bytes, base64/data URI, numpy array, torch.Tensor, or PIL.Image."""
    if isinstance(source, Image.Image):
        return source.copy()

    if isinstance(source, torch.Tensor):
        t = source.detach().cpu()
        if t.dim() == 4:
            if t.size(0) != 1:
                raise ValueError(f"load_image expects a single image tensor, got batch shape {tuple(t.shape)}")
            t = t[0]
        if t.dim() != 3:
            raise ValueError(f"Expected 3D image tensor (C, H, W) or (H, W, C), got shape {tuple(t.shape)}")
        if t.shape[0] in (1, 3, 4) and t.shape[2] not in (1, 3, 4):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= -1.0:
                if arr.min() < 0.0:
                    arr = (arr + 1.0) * 0.5
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return Image.fromarray(arr)

    if isinstance(source, np.ndarray):
        arr = source
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 and arr.min() >= 0.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if isinstance(source, (bytes, bytearray)):
        if not source:
            raise ValueError("Cannot load image from empty bytes")
        try:
            img = Image.open(io.BytesIO(source))
            img.load()
            return img
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image bytes: {exc}") from exc

    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Image file not found: {source}")
        try:
            img = Image.open(source)
            img.load()
            return img
        except Exception as exc:
            raise ValueError(f"Invalid or corrupted image file {source}: {exc}") from exc

    if isinstance(source, str):
        s = source.strip()
        if s.startswith("data:image/") and ";base64," in s:
            b64_part = s.split(";base64,", 1)[1]
            try:
                raw = base64.b64decode(b64_part)
                img = Image.open(io.BytesIO(raw))
                img.load()
                return img
            except Exception as exc:
                raise ValueError(f"Invalid base64 data URI image: {exc}") from exc

        p = Path(s)
        if p.exists() and p.is_file():
            try:
                img = Image.open(p)
                img.load()
                return img
            except Exception as exc:
                raise ValueError(f"Invalid or corrupted image file {p}: {exc}") from exc

        # Try raw base64 string if it looks like base64
        if len(s) > 32 and not any(c in s for c in ("\n", "\r", "\\")) and p.suffix.lower() not in IMAGE_EXTENSIONS:
            try:
                raw = base64.b64decode(s, validate=True)
                img = Image.open(io.BytesIO(raw))
                img.load()
                return img
            except Exception:
                pass

        raise FileNotFoundError(f"Image path does not exist or is not a valid image: {source!r}")

    raise TypeError(f"Unsupported image source type: {type(source).__name__}")


def _rgb_to_hsl(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert RGB in [0, 255] to (H in [0, 360), S in [0, 1], L in [0, 1])."""
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    cmax = max(rf, gf, bf)
    cmin = min(rf, gf, bf)
    delta = cmax - cmin
    l = (cmax + cmin) * 0.5
    if delta < 1e-6:
        return 0.0, 0.0, l
    s = delta / (1.0 - abs(2.0 * l - 1.0) + 1e-9)
    if cmax == rf:
        h = 60.0 * (((gf - bf) / delta) % 6.0)
    elif cmax == gf:
        h = 60.0 * (((bf - rf) / delta) + 2.0)
    else:
        h = 60.0 * (((rf - gf) / delta) + 4.0)
    return h % 360.0, min(max(s, 0.0), 1.0), min(max(l, 0.0), 1.0)


def _classify_color_name(r: float, g: float, b: float) -> str:
    """Deterministic human-readable color name from (R, G, B) in [0, 255]."""
    h, s, l = _rgb_to_hsl(r, g, b)
    if l < 0.12:
        return "black"
    if l > 0.88 and s < 0.20:
        return "white"
    if s < 0.15:
        return "gray"
    if h < 18.0 or h >= 345.0:
        return "red"
    if h < 42.0:
        return "orange"
    if h < 70.0:
        return "yellow"
    if h < 165.0:
        return "green"
    if h < 200.0:
        return "cyan"
    if h < 260.0:
        return "blue"
    if h < 295.0:
        return "purple"
    return "magenta"


def _extract_image_metadata(img: Image.Image) -> dict[str, str]:
    """Extract clean text metadata from PNG info chunks and EXIF tags."""
    meta: dict[str, str] = {}
    if getattr(img, "info", None):
        for k, v in img.info.items():
            if isinstance(k, str) and isinstance(v, (str, int, float)):
                ks = k.strip()
                vs = str(v).strip()
                if ks and vs and ks.lower() not in ("dpi", "gamma", "chromaticity", "icc_profile"):
                    meta[ks] = vs[:256]
    try:
        exif = img.getexif()
        if exif:
            for tag_id, val in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(val, (str, int, float)):
                    vs = str(val).strip()
                    if vs:
                        meta[str(tag_name)] = vs[:256]
    except Exception:
        pass
    return meta


def describe_image(
    source: str | Path | bytes | bytearray | Image.Image | np.ndarray | torch.Tensor,
    lang: str = "eng",
) -> VisualDescription:
    """Compute deterministic visual features and a natural-language summary for an image.

    Returns a `VisualDescription(dict)` containing structured keys (`width`, `height`,
    `resolution`, `aspect_ratio`, `mode`, `brightness_mean`, `brightness_std`,
    `brightness_level`, `contrast_rms`, `contrast_michelson`, `contrast_level`,
    `edge_density`, `texture_level`, `dominant_colors`, `dominant_rgb`, `dominant_hsl`,
    `quadrants`, `metadata`, `summary`, `text`) whose `str()` is the grounded summary.
    """
    source_name = Path(source).name if isinstance(source, (str, Path)) and len(str(source)) < 260 else None
    img = load_image(source)
    orig_mode = img.mode
    width, height = img.size
    aspect_ratio = round(width / max(height, 1), 3)
    meta = _extract_image_metadata(img)

    rgb_img = img.convert("RGB")
    # Downsample very large images to at most 256x256 for fast, deterministic stats
    stat_img = rgb_img
    if width > 256 or height > 256:
        stat_img = rgb_img.resize((min(width, 256), min(height, 256)), Image.Resampling.BILINEAR)

    arr = np.asarray(stat_img, dtype=np.float32)  # (H, W, 3) in [0, 255]
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Rec. 601 luminance Y in [0, 255]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    lum_mean = float(np.mean(lum))
    lum_std = float(np.std(lum))
    lum_min = float(np.min(lum))
    lum_max = float(np.max(lum))

    if lum_mean >= 170.0:
        brightness_level = "bright"
    elif lum_mean <= 85.0:
        brightness_level = "dark"
    else:
        brightness_level = "medium"

    # RMS contrast (in [0, 255] and normalized [0, 1]) & Michelson contrast
    contrast_rms = lum_std
    contrast_rms_norm = lum_std / 255.0
    denom = lum_max + lum_min
    contrast_michelson = float((lum_max - lum_min) / denom) if denom > 1e-5 else 0.0

    if contrast_rms < 5.0:
        contrast_level = "uniform"
    elif contrast_rms >= 35.0:
        contrast_level = "high-contrast"
    else:
        contrast_level = "moderate-contrast"

    # Sobel edge/texture density via pure numpy (3x3 Sobel + 1px finite differences for Nyquist textures)
    lum_norm = lum / 255.0
    if lum_norm.shape[0] >= 3 and lum_norm.shape[1] >= 3:
        gx = (
            (lum_norm[:-2, 2:] + 2.0 * lum_norm[1:-1, 2:] + lum_norm[2:, 2:])
            - (lum_norm[:-2, :-2] + 2.0 * lum_norm[1:-1, :-2] + lum_norm[2:, :-2])
        )
        gy = (
            (lum_norm[2:, :-2] + 2.0 * lum_norm[2:, 1:-1] + lum_norm[2:, 2:])
            - (lum_norm[:-2, :-2] + 2.0 * lum_norm[:-2, 1:-1] + lum_norm[:-2, 2:])
        )
        sobel_mag = float(np.mean(np.hypot(gx, gy) / 4.0))
        dx1 = float(np.mean(np.abs(lum_norm[:, 1:] - lum_norm[:, :-1])))
        dy1 = float(np.mean(np.abs(lum_norm[1:, :] - lum_norm[:-1, :])))
        edge_density = max(sobel_mag, 0.5 * (dx1 + dy1))
    else:
        edge_density = 0.0

    if edge_density >= 0.12:
        texture_level = "detailed/textured"
    elif edge_density <= 0.02:
        texture_level = "smooth/uniform"
    else:
        texture_level = "moderate-edge"

    # Spatial 2x2 quadrant luminance layout
    h_mid = max(lum.shape[0] // 2, 1)
    w_mid = max(lum.shape[1] // 2, 1)
    q_tl = float(np.mean(lum[:h_mid, :w_mid]))
    q_tr = float(np.mean(lum[:h_mid, w_mid:])) if lum.shape[1] > 1 else q_tl
    q_bl = float(np.mean(lum[h_mid:, :w_mid])) if lum.shape[0] > 1 else q_tl
    q_br = float(np.mean(lum[h_mid:, w_mid:])) if (lum.shape[0] > 1 and lum.shape[1] > 1) else q_tl
    quadrants = {
        "top_left": round(q_tl, 2),
        "top_right": round(q_tr, 2),
        "bottom_left": round(q_bl, 2),
        "bottom_right": round(q_br, 2),
    }

    # Dominant RGB / HSL & color palette names across a 4x4 grid + global mean
    mean_r, mean_g, mean_b = float(np.mean(r)), float(np.mean(g)), float(np.mean(b))
    dom_h, dom_s, dom_l = _rgb_to_hsl(mean_r, mean_g, mean_b)

    color_counts: dict[str, int] = {}
    grid_img = rgb_img.resize((16, 16), Image.Resampling.BILINEAR)
    grid_arr = np.asarray(grid_img, dtype=np.float32).reshape(-1, 3)
    for px in grid_arr:
        cname = _classify_color_name(float(px[0]), float(px[1]), float(px[2]))
        color_counts[cname] = color_counts.get(cname, 0) + 1
    dominant_colors = [c for c, _ in sorted(color_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]]
    overall_color = _classify_color_name(mean_r, mean_g, mean_b)
    if overall_color not in dominant_colors:
        dominant_colors.insert(0, overall_color)

    header = _LANG_VISUAL_HEADER.get(lang, _LANG_VISUAL_HEADER["eng"])
    name_prefix = f" ({source_name})" if source_name else ""
    color_str = ", ".join(dominant_colors)
    meta_str = ""
    if meta:
        meta_pairs = [f"{k}: {v}" for k, v in meta.items()]
        meta_str = " Metadata: " + "; ".join(meta_pairs) + "."

    summary = (
        f"{header}{name_prefix}: {width}x{height} {orig_mode} image "
        f"(aspect ratio {aspect_ratio:.2f}), {brightness_level} luminance "
        f"(mean {lum_mean:.1f}/255, std {lum_std:.1f}), {contrast_level} "
        f"(RMS {contrast_rms:.1f}, Michelson {contrast_michelson:.2f}), "
        f"{texture_level} edges (Sobel density {edge_density:.3f}). "
        f"Dominant colors: {color_str} (mean RGB {int(round(mean_r))},{int(round(mean_g))},{int(round(mean_b))}). "
        f"Quadrant luminance (TL={q_tl:.1f}, TR={q_tr:.1f}, BL={q_bl:.1f}, BR={q_br:.1f})."
        f"{meta_str}"
    )

    return VisualDescription({
        "filename": source_name or "image",
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "aspect_ratio": aspect_ratio,
        "mode": orig_mode,
        "brightness_mean": round(lum_mean, 3),
        "brightness_norm": round(lum_mean / 255.0, 4),
        "brightness_std": round(lum_std, 3),
        "brightness_level": brightness_level,
        "contrast_rms": round(contrast_rms, 3),
        "contrast_rms_norm": round(contrast_rms_norm, 4),
        "contrast_michelson": round(contrast_michelson, 4),
        "contrast_level": contrast_level,
        "edge_density": round(edge_density, 4),
        "texture_level": texture_level,
        "dominant_colors": dominant_colors,
        "dominant_rgb": (int(round(mean_r)), int(round(mean_g)), int(round(mean_b))),
        "dominant_hsl": (round(dom_h, 1), round(dom_s, 3), round(dom_l, 3)),
        "quadrants": quadrants,
        "metadata": meta,
        "lang": lang,
        "summary": summary,
        "text": summary,
    })


def state_memory_bytes(states: list) -> int:
    """Total memory footprint in bytes of an SSM recurrent state list."""
    total = 0
    for st in states:
        if isinstance(st, tuple):
            for t in st:
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        elif isinstance(st, torch.Tensor):
            total += st.numel() * st.element_size()
    return total


def state_shapes(states: list) -> list[tuple[int, ...]]:
    """Return the primary recurrent matrix shape `(B, n_head, head_dim, head_dim)` per layer."""
    shapes = []
    for st in states:
        mat = st[0] if isinstance(st, tuple) else st
        shapes.append(tuple(mat.shape))
    return shapes


class SSMVisualProjector(nn.Module):
    """Recurrent SSM Visual Patch Projector for Parentheses Selective Linear Attention.

    Converts 2D images into raster-ordered non-overlapping patch sequences and steps
    them sequentially through `block.step(x_patch, state)`, preserving a strictly
    constant O(1) recurrent state size `(B, n_head, head_dim, head_dim)` per layer.
    """

    def __init__(self, patch_size: int = 16, grid_size: int = 64):
        super().__init__()
        if patch_size <= 0 or grid_size <= 0 or grid_size % patch_size != 0:
            raise ValueError(
                f"grid_size ({grid_size}) must be a positive multiple of patch_size ({patch_size})"
            )
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.grid_side = grid_size // patch_size
        self.num_patches = (grid_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size

    def extract_patches(
        self,
        image: str | Path | bytes | bytearray | Image.Image | np.ndarray | torch.Tensor,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Extract non-overlapping 2D patches `(B, N_patches, 3 * P * P)` in raster scan order."""
        P = self.patch_size
        G = self.grid_size
        if isinstance(image, torch.Tensor) and image.dim() == 4 and image.size(1) == 3:
            # Batched tensor (B, 3, H, W)
            t = image.to(device=device, dtype=torch.float32)
            if t.max() > 1.0:
                t = t / 255.0
            if t.shape[2] != G or t.shape[3] != G:
                t = torch.nn.functional.interpolate(t, size=(G, G), mode="bilinear", align_corners=False)
            t = (t - 0.5) * 2.0
        else:
            img = load_image(image).convert("RGB")
            if img.size != (G, G):
                img = img.resize((G, G), Image.Resampling.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # (G, G, 3)
            arr = (arr - 0.5) * 2.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, G, G)
            if device is not None:
                t = t.to(device=device)

        B = t.size(0)
        gh = G // P
        gw = G // P
        # Raster order: (B, 3, gh, P, gw, P) -> (B, gh, gw, 3, P, P) -> (B, N_patches, 3*P*P)
        patches = (
            t.view(B, 3, gh, P, gw, P)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(B, gh * gw, 3 * P * P)
        )
        return patches.to(dtype=dtype)

    @staticmethod
    def _build_projection_matrix(model, patch_dim: int, n_embd: int, device, dtype) -> torch.Tensor:
        """Deterministically derive a `(patch_dim, n_embd)` Xavier-scaled projection from `model.tok_emb.weight`."""
        w = model.tok_emb.weight.detach().float()  # (vocab_size, n_embd)
        flat = w.reshape(-1)
        needed = patch_dim * n_embd
        if flat.numel() < needed:
            repeats = (needed + flat.numel() - 1) // flat.numel()
            flat = flat.repeat(repeats)
        mat = flat[:needed].view(patch_dim, n_embd).clone()
        # Zero-mean and normalize columns to Xavier scale 1/sqrt(patch_dim)
        mat = mat - mat.mean(dim=0, keepdim=True)
        col_norm = mat.norm(dim=0, keepdim=True).clamp_min(1e-6)
        mat = (mat / col_norm) * (math.sqrt(n_embd / float(patch_dim)) * 0.1)
        return mat.to(device=device, dtype=dtype)

    @staticmethod
    def _spatial_2d_pos_embed(grid_h: int, grid_w: int, n_embd: int, device, dtype) -> torch.Tensor:
        """Deterministic 2D sinusoidal raster positional embedding `(1, grid_h * grid_w, n_embd)`."""
        pos = torch.zeros(grid_h * grid_w, n_embd, device=device, dtype=torch.float32)
        half = max(n_embd // 2, 1)
        div = torch.exp(torch.arange(0, half, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
        idx = 0
        for r in range(grid_h):
            for c in range(grid_w):
                pos[idx, 0:half:2] = torch.sin(r * div)
                if half > 1:
                    pos[idx, 1:half:2] = torch.cos(r * div[: pos[idx, 1:half:2].numel()])
                rem = n_embd - half
                if rem > 0:
                    div_c = torch.exp(torch.arange(0, rem, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(rem, 1)))
                    pos[idx, half::2] = torch.sin(c * div_c)
                    if rem > 1:
                        pos[idx, half + 1 :: 2] = torch.cos(c * div_c[: pos[idx, half + 1 :: 2].numel()])
                idx += 1
        return (pos.unsqueeze(0) * 0.02).to(dtype=dtype)

    def project_patches(self, model, patches: torch.Tensor) -> torch.Tensor:
        """Project `(B, N_patches, patch_dim)` patches into `(B, N_patches, n_embd)`."""
        B, N, D = patches.shape
        n_embd = model.cfg.n_embd
        proj = self._build_projection_matrix(model, D, n_embd, patches.device, patches.dtype)
        gh = int(math.isqrt(N))
        gw = N // max(gh, 1)
        pos = self._spatial_2d_pos_embed(gh, gw, n_embd, patches.device, patches.dtype)
        return (patches @ proj) + pos[:, :N, :]

    @torch.no_grad()
    def prefill(self, model, image, states: list | None = None) -> list:
        """Step all 2D raster patches sequentially through `model.blocks` in O(1) recurrent state space."""
        model._require_selective("SSMVisualProjector.prefill()")
        device = next(model.parameters()).device
        dtype = model.tok_emb.weight.dtype
        patches = self.extract_patches(image, device=device, dtype=dtype)
        B, N, _ = patches.shape
        embeds = self.project_patches(model, patches)  # (B, N, n_embd)

        if states is None:
            states = model.empty_states(B, device=device, dtype=dtype)

        initial_shapes = state_shapes(states)
        initial_bytes = state_memory_bytes(states)

        for p_idx in range(N):
            x_p = embeds[:, p_idx, :]  # (B, n_embd)
            new_states = []
            for block, st in zip(model.blocks, states):
                x_p, st = block.step(x_p, st)
                new_states.append(st)
            states = new_states

        # Enforce the Parentheses SSM O(1) state memory invariant
        final_shapes = state_shapes(states)
        final_bytes = state_memory_bytes(states)
        assert final_shapes == initial_shapes, (
            f"SSM state shape mutated during visual prefill: {initial_shapes} -> {final_shapes}"
        )
        assert final_bytes == initial_bytes, (
            f"SSM state memory grew during visual prefill: {initial_bytes} -> {final_bytes} bytes"
        )
        return states


@torch.no_grad()
def prefill_visual_state(
    model,
    image: str | Path | bytes | bytearray | Image.Image | np.ndarray | torch.Tensor,
    states: list | None = None,
    patch_size: int = 16,
    grid_size: int = 64,
) -> list:
    """Inject an image into the Parentheses SSM recurrent state `S_t` via O(1) sequential patch stepping.

    Returns the updated `states` list (`n_layer` tensors of shape `(B, n_head, head_dim, head_dim)`).
    """
    projector = SSMVisualProjector(patch_size=patch_size, grid_size=grid_size)
    return projector.prefill(model, image, states=states)


class ImageProcessor:
    """High-level facade for image loading, structured visual description, and SSM state prefill."""

    def __init__(self, patch_size: int = 16, grid_size: int = 64, lang: str = "eng"):
        self.projector = SSMVisualProjector(patch_size=patch_size, grid_size=grid_size)
        self.lang = lang

    def load(self, source) -> Image.Image:
        return load_image(source)

    def describe(self, source, lang: str | None = None) -> VisualDescription:
        return describe_image(source, lang=lang or self.lang)

    def extract_patches(self, source, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.projector.extract_patches(source, device=device, dtype=dtype)

    def prefill_state(self, model, source, states: list | None = None) -> list:
        return self.projector.prefill(model, source, states=states)


# ---------------------------------------------------------------------------
# Part B: Universal Multi-Format Document Processor (PDF, DOCX, HTML, CSV, JSON)
# ---------------------------------------------------------------------------

def _decode_pdf_literal_string(raw: bytes) -> str:
    """Decode a PDF literal string `(...)`, unescaping `\\n`, `\\r`, `\\t`, `\\(`, `\\)`, and octal `\\ddd`."""
    out = bytearray()
    i = 0
    n = len(raw)
    while i < n:
        b = raw[i]
        if b == 0x5C:  # backslash '\'
            i += 1
            if i >= n:
                break
            esc = raw[i]
            if esc == ord("n"):
                out.append(0x0A)
            elif esc == ord("r"):
                out.append(0x0D)
            elif esc == ord("t"):
                out.append(0x09)
            elif esc == ord("b"):
                out.append(0x08)
            elif esc == ord("f"):
                out.append(0x0C)
            elif esc in (ord("("), ord(")"), ord("\\")):
                out.append(esc)
            elif esc in (0x0A, 0x0D):
                # Line continuation backslash + newline
                if esc == 0x0D and i + 1 < n and raw[i + 1] == 0x0A:
                    i += 1
            elif ord("0") <= esc <= ord("7"):
                oct_digits = bytes([esc])
                for _ in range(2):
                    if i + 1 < n and ord("0") <= raw[i + 1] <= ord("7"):
                        i += 1
                        oct_digits += bytes([raw[i]])
                    else:
                        break
                out.append(int(oct_digits, 8) & 0xFF)
            else:
                out.append(esc)
        else:
            out.append(b)
        i += 1

    # Check UTF-16BE / UTF-16LE BOM first
    b_out = bytes(out)
    if b_out.startswith(b"\xfe\xff"):
        return b_out[2:].decode("utf-16-be", errors="replace")
    if b_out.startswith(b"\xff\xfe"):
        return b_out[2:].decode("utf-16-le", errors="replace")
    try:
        return b_out.decode("utf-8")
    except UnicodeDecodeError:
        return b_out.decode("latin-1", errors="replace")


def _decode_pdf_hex_string(hex_bytes: bytes) -> str:
    """Decode a PDF `<hex>` string, supporting UTF-16BE (`FEFF`), UTF-8, and Latin-1."""
    cleaned = re.sub(rb"\s+", b"", hex_bytes)
    if len(cleaned) % 2 == 1:
        cleaned += b"0"
    try:
        raw = bytes.fromhex(cleaned.decode("ascii"))
    except Exception:
        return ""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace")
    if raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def _extract_tokens_from_pdf_stream(content: bytes) -> list[str]:
    """Parse PDF content stream (`BT ... ET`, `Tj`, `TJ`, `'`, `\"`, `Td`, `TD`, `T*`) into paragraphs."""
    blocks = re.findall(rb"BT\b(.*?)\bET", content, flags=re.DOTALL)
    if not blocks:
        blocks = [content]

    paragraphs: list[str] = []
    token_re = re.compile(
        rb"""
        (\[(?:[^\[\]]|\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]*>)*\]\s*TJ) |
        (\((?:\\.|[^\\()])*\)\s*(?:Tj|'|")) |
        (<[0-9A-Fa-f\s]+>\s*Tj) |
        (-?\d+(?:\.\d+)?\s+-?\d+(?:\.\d+)?\s+T[dD]\b) |
        (\bT\*)
        """,
        re.VERBOSE | re.DOTALL,
    )
    inner_re = re.compile(
        rb"\((?:\\.|[^\\()])*\)|<[0-9A-Fa-f\s]+>|-?\d+(?:\.\d+)?"
    )

    for blk in blocks:
        current_lines: list[str] = []
        current_line: list[str] = []
        for match in token_re.finditer(blk):
            tj_arr, tj_lit, tj_hex, td_op, tstar_op = match.groups()
            if tj_arr is not None:
                arr_body = tj_arr[: tj_arr.rfind(b"]")]
                if arr_body.startswith(b"["):
                    arr_body = arr_body[1:]
                pieces: list[str] = []
                for item in inner_re.findall(arr_body):
                    if item.startswith(b"(") and item.endswith(b")"):
                        pieces.append(_decode_pdf_literal_string(item[1:-1]))
                    elif item.startswith(b"<") and item.endswith(b">"):
                        pieces.append(_decode_pdf_hex_string(item[1:-1]))
                    else:
                        try:
                            kern = float(item)
                            if kern < -100.0 and pieces and not pieces[-1].endswith(" "):
                                pieces.append(" ")
                        except ValueError:
                            pass
                current_line.append("".join(pieces))
            elif tj_lit is not None:
                rparen = tj_lit.rfind(b")")
                lit_body = tj_lit[1:rparen]
                text_piece = _decode_pdf_literal_string(lit_body)
                if tj_lit.rstrip().endswith((b"'", b'"')) and current_line:
                    current_lines.append("".join(current_line).strip())
                    current_line = [text_piece]
                else:
                    current_line.append(text_piece)
            elif tj_hex is not None:
                rangle = tj_hex.rfind(b">")
                hex_body = tj_hex[1:rangle]
                current_line.append(_decode_pdf_hex_string(hex_body))
            elif td_op is not None:
                parts = td_op.split()
                if len(parts) >= 2:
                    try:
                        ty = float(parts[1])
                        if abs(ty) > 0.1 and current_line:
                            current_lines.append("".join(current_line).strip())
                            current_line = []
                    except ValueError:
                        pass
            elif tstar_op is not None:
                if current_line:
                    current_lines.append("".join(current_line).strip())
                    current_line = []

        if current_line:
            current_lines.append("".join(current_line).strip())
        para = " ".join(ln for ln in current_lines if ln).strip()
        if para:
            paragraphs.append(para)

    return paragraphs


def extract_pdf_text(source: str | Path | bytes | bytearray) -> str:
    """Pure-Python PDF text extractor supporting uncompressed and `/FlateDecode` streams."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raise TypeError(f"Unsupported PDF source type: {type(source).__name__}")

    if not raw or b"%PDF-" not in raw[:1024]:
        raise ValueError("Invalid or corrupted PDF: missing %PDF- header")

    stream_pattern = re.compile(
        rb"(<<[^>]*?>>)?\s*stream(?:\r\n|\n|\r)(.*?)(?:\r\n|\n|\r)?endstream",
        re.DOTALL,
    )
    all_paragraphs: list[str] = []
    for match in stream_pattern.finditer(raw):
        dict_bytes = match.group(1) or b""
        stream_data = match.group(2)
        is_flate = b"/FlateDecode" in dict_bytes or b"/Fl" in dict_bytes
        decoded_stream = None
        if is_flate:
            for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
                try:
                    decoded_stream = zlib.decompress(stream_data, wbits)
                    break
                except Exception:
                    try:
                        decoded_stream = zlib.decompress(stream_data.strip(b"\r\n"), wbits)
                        break
                    except Exception:
                        pass
            if decoded_stream is None:
                continue
        else:
            # Also opportunistically try zlib if it isn't plain ASCII/text
            if b"BT" in stream_data or b"Tj" in stream_data or b"TJ" in stream_data:
                decoded_stream = stream_data
            else:
                try:
                    decoded_stream = zlib.decompress(stream_data)
                except Exception:
                    continue

        if decoded_stream:
            paras = _extract_tokens_from_pdf_stream(decoded_stream)
            all_paragraphs.extend(paras)

    return "\n\n".join(p for p in all_paragraphs if p.strip())


def create_pdf_bytes(paragraphs: list[str], compress: bool = True) -> bytes:
    """Create a valid minimal PDF file in pure Python (for testing & self-contained documents).

    ASCII paragraphs are encoded with `(literal) Tj` inside `BT ... ET` blocks;
    non-ASCII Unicode paragraphs (e.g., Kannada/Tamil/Telugu/Malayalam) are encoded
    as `<FEFF...>` UTF-16BE hex strings so PDF readers and `extract_pdf_text` preserve
    100% of Unicode codepoints.
    """
    stream_chunks: list[bytes] = []
    y_pos = 750
    for para in paragraphs:
        clean = para.strip()
        if not clean:
            continue
        try:
            clean.encode("ascii")
            escaped = (
                clean.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
                .encode("ascii")
            )
            op = b"(" + escaped + b") Tj"
        except UnicodeEncodeError:
            utf16_hex = (b"\xfe\xff" + clean.encode("utf-16-be")).hex().upper().encode("ascii")
            op = b"<" + utf16_hex + b"> Tj"
        blk = f"BT\n/F1 12 Tf\n72 {y_pos} Td\n".encode("ascii") + op + b"\nET\n"
        stream_chunks.append(blk)
        y_pos -= 24

    raw_stream = b"".join(stream_chunks)
    if compress:
        payload = zlib.compress(raw_stream)
        filter_spec = b"/Filter /FlateDecode "
    else:
        payload = raw_stream
        filter_spec = b""

    objs: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
        b"4 0 obj\n<< " + filter_spec + f"/Length {len(payload)} >>\nstream\n".encode("ascii") + payload + b"\nendstream\nendobj\n",
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _extract_docx_paragraph_text(p_elem: ET.Element) -> str:
    """Extract text runs `<w:t>`, tabs `<w:tab/>`, and breaks `<w:br/>` from a `<w:p>` element."""
    parts: list[str] = []
    for elem in p_elem.iter():
        tag = elem.tag
        if tag == f"{_WORD_NS}t" or tag.endswith("}t") or tag == "t":
            if elem.text:
                parts.append(elem.text)
        elif tag == f"{_WORD_NS}tab" or tag.endswith("}tab"):
            parts.append("\t")
        elif tag == f"{_WORD_NS}br" or tag.endswith("}br"):
            parts.append("\n")
    return "".join(parts).strip()


def extract_docx_text(source: str | Path | bytes | bytearray) -> str:
    """Pure-Python `.docx` text extractor preserving paragraphs and table rows."""
    try:
        if isinstance(source, (str, Path)):
            zf = zipfile.ZipFile(Path(source), "r")
        elif isinstance(source, (bytes, bytearray)):
            zf = zipfile.ZipFile(io.BytesIO(source), "r")
        else:
            raise TypeError(f"Unsupported DOCX source type: {type(source).__name__}")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"Invalid or corrupted DOCX archive: {exc}") from exc

    with zf:
        if "word/document.xml" not in zf.namelist():
            raise ValueError("Invalid DOCX archive: missing word/document.xml")
        xml_bytes = zf.read("word/document.xml")

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"Corrupted word/document.xml in DOCX: {exc}") from exc

    body = root.find(f"{_WORD_NS}body")
    if body is None:
        body = root

    blocks: list[str] = []
    for child in body:
        tag = child.tag
        if tag == f"{_WORD_NS}p" or tag.endswith("}p") or tag == "p":
            p_text = _extract_docx_paragraph_text(child)
            if p_text:
                blocks.append(p_text)
        elif tag == f"{_WORD_NS}tbl" or tag.endswith("}tbl") or tag == "tbl":
            rows_out: list[str] = []
            for tr in child.iter():
                if tr.tag == f"{_WORD_NS}tr" or tr.tag.endswith("}tr") or tr.tag == "tr":
                    cells_out: list[str] = []
                    for tc in tr:
                        if tc.tag == f"{_WORD_NS}tc" or tc.tag.endswith("}tc") or tc.tag == "tc":
                            cell_paras = [
                                _extract_docx_paragraph_text(p)
                                for p in tc
                                if p.tag == f"{_WORD_NS}p" or p.tag.endswith("}p") or p.tag == "p"
                            ]
                            cell_str = " ".join(cp for cp in cell_paras if cp).strip()
                            if cell_str:
                                cells_out.append(cell_str)
                    if cells_out:
                        rows_out.append(" | ".join(cells_out))
            if rows_out:
                blocks.append("\n\n".join(rows_out))

    return "\n\n".join(blocks)


def create_docx_bytes(
    paragraphs: list[str],
    tables: list[list[list[str]]] | None = None,
) -> bytes:
    """Create a valid OpenXML `.docx` archive in pure Python with UTF-8 paragraphs and tables."""
    w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body_parts: list[str] = []
    for para in paragraphs:
        esc = html.escape(para, quote=False)
        body_parts.append(f"<w:p><w:r><w:t>{esc}</w:t></w:r></w:p>")

    if tables:
        for tbl in tables:
            tr_parts: list[str] = []
            for row in tbl:
                tc_parts: list[str] = []
                for cell in row:
                    esc = html.escape(str(cell), quote=False)
                    tc_parts.append(f"<w:tc><w:p><w:r><w:t>{esc}</w:t></w:r></w:p></w:tc>")
                tr_parts.append(f"<w:tr>{''.join(tc_parts)}</w:tr>")
            body_parts.append(f"<w:tbl>{''.join(tr_parts)}</w:tbl>")

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{w_ns}"><w:body>'
        + "".join(body_parts)
        + "</w:body></w:document>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml.encode("utf-8"))
        zf.writestr("_rels/.rels", rels_xml.encode("utf-8"))
        zf.writestr("word/document.xml", doc_xml.encode("utf-8"))
    return buf.getvalue()


class _CleanHTMLParser(HTMLParser):
    """Pure-Python HTML parser that strips `<script>`/`<style>` and preserves block structure."""

    _IGNORE_TAGS = frozenset({"script", "style", "noscript", "svg", "template"})
    _BLOCK_TAGS = frozenset({
        "p", "div", "section", "article", "header", "footer", "main", "aside",
        "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "pre",
        "title", "dt", "dd", "figcaption",
    })

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._in_cell = False
        self.blocks: list[str] = []
        self._current: list[str] = []

    def _flush_block(self):
        if self._current:
            line = re.sub(r"[ \t]+", " ", "".join(self._current)).strip()
            if line:
                self.blocks.append(line)
            self._current = []

    def handle_starttag(self, tag: str, attrs):
        t = tag.lower()
        if t in self._IGNORE_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth > 0:
            return
        if t in self._BLOCK_TAGS:
            self._flush_block()
        elif t in ("br", "hr"):
            self._flush_block()
        elif t in ("td", "th"):
            if self._current and not "".join(self._current).rstrip().endswith("|"):
                self._current.append(" | ")
            self._in_cell = True

    def handle_endtag(self, tag: str):
        t = tag.lower()
        if t in self._IGNORE_TAGS:
            self._ignore_depth = max(0, self._ignore_depth - 1)
            return
        if self._ignore_depth > 0:
            return
        if t in self._BLOCK_TAGS:
            self._flush_block()
        elif t in ("td", "th"):
            self._in_cell = False

    def handle_data(self, data: str):
        if self._ignore_depth > 0:
            return
        cleaned = data.replace("\r", " ").replace("\n", " ")
        if cleaned.strip():
            self._current.append(cleaned)
        elif self._current and not self._current[-1].endswith(" "):
            self._current.append(" ")

    def get_text(self) -> str:
        self._flush_block()
        return "\n\n".join(self.blocks)


def extract_html_text(source: str | Path | bytes | bytearray) -> str:
    """Extract clean readable text from HTML or XML, stripping `<script>` and `<style>` blocks."""
    if isinstance(source, Path):
        raw_text = source.read_text(encoding="utf-8", errors="replace")
    elif isinstance(source, (bytes, bytearray)):
        raw_text = bytes(source).decode("utf-8", errors="replace")
    elif isinstance(source, str):
        p = Path(source)
        if len(source) < 260 and "\n" not in source and "<" not in source and p.exists() and p.is_file():
            raw_text = p.read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = source
    else:
        raise TypeError(f"Unsupported HTML source type: {type(source).__name__}")

    parser = _CleanHTMLParser()
    parser.feed(raw_text)
    parser.close()
    return parser.get_text()


def extract_csv_text(source: str | Path | bytes | bytearray, delimiter: str = ",") -> str:
    """Convert CSV or TSV rows into self-contained readable paragraph blocks."""
    if isinstance(source, Path):
        raw_text = source.read_text(encoding="utf-8", errors="replace")
    elif isinstance(source, (bytes, bytearray)):
        raw_text = bytes(source).decode("utf-8", errors="replace")
    elif isinstance(source, str):
        p = Path(source)
        if len(source) < 260 and "\n" not in source and p.exists() and p.is_file():
            raw_text = p.read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = source
    else:
        raise TypeError(f"Unsupported CSV source type: {type(source).__name__}")

    reader = csv.reader(io.StringIO(raw_text), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return ""
    if len(rows) == 1:
        return " | ".join(c for c in rows[0] if c)

    headers = rows[0]
    blocks: list[str] = []
    for row in rows[1:]:
        pairs = []
        for idx, val in enumerate(row):
            if not val:
                continue
            col_name = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx + 1}"
            pairs.append(f"{col_name}: {val}")
        if pairs:
            blocks.append(" | ".join(pairs))
    return "\n\n".join(blocks)


def _format_json_record(obj: Any) -> list[str]:
    """Flatten a JSON object or list of objects into readable paragraph strings."""
    if isinstance(obj, dict):
        parts = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v)
            parts.append(f"{k}: {v_str}")
        return [" | ".join(parts)] if parts else []
    if isinstance(obj, list):
        out: list[str] = []
        for item in obj:
            out.extend(_format_json_record(item))
        return out
    return [str(obj)] if obj is not None else []


def extract_json_text(source: str | Path | bytes | bytearray, is_jsonl: bool = False) -> str:
    """Convert JSON or JSONL records into clean UTF-8 paragraph blocks."""
    if isinstance(source, Path):
        raw_text = source.read_text(encoding="utf-8", errors="replace")
    elif isinstance(source, (bytes, bytearray)):
        raw_text = bytes(source).decode("utf-8", errors="replace")
    elif isinstance(source, str):
        p = Path(source)
        if len(source) < 260 and "\n" not in source and not source.lstrip().startswith(("{", "[")) and p.exists():
            raw_text = p.read_text(encoding="utf-8", errors="replace")
        else:
            raw_text = source
    else:
        raise TypeError(f"Unsupported JSON source type: {type(source).__name__}")

    blocks: list[str] = []
    if is_jsonl:
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            blocks.extend(_format_json_record(obj))
    else:
        obj = json.loads(raw_text)
        blocks.extend(_format_json_record(obj))
    return "\n\n".join(b for b in blocks if b.strip())


def extract_document_text(
    source: str | Path | bytes | bytearray | Image.Image,
    filename: str | None = None,
    lang: str = "eng",
) -> str:
    """Universal multimodal document & image text extractor.

    Dispatches by file extension or content magic bytes to extract clean UTF-8
    paragraphs from `.pdf`, `.docx`, `.html`/`.xml`, `.csv`/`.tsv`, `.json`/`.jsonl`,
    `.txt`/`.md`/`.py`, or images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`).
    """
    if isinstance(source, Image.Image):
        return describe_image(source, lang=lang).summary

    ext = ""
    if filename:
        ext = Path(filename).suffix.lower()
    elif isinstance(source, (str, Path)):
        p = Path(source)
        ext = p.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_text(source)
    if ext == ".docx":
        return extract_docx_text(source)
    if ext in (".html", ".htm", ".xml"):
        return extract_html_text(source)
    if ext == ".csv":
        return extract_csv_text(source, delimiter=",")
    if ext == ".tsv":
        return extract_csv_text(source, delimiter="\t")
    if ext == ".json":
        return extract_json_text(source, is_jsonl=False)
    if ext == ".jsonl":
        return extract_json_text(source, is_jsonl=True)
    if ext in IMAGE_EXTENSIONS:
        return describe_image(source, lang=lang).summary

    # Magic-byte detection when `source` is raw bytes without an explicit filename
    if isinstance(source, (bytes, bytearray)):
        b_src = bytes(source)
        if b_src.startswith(b"%PDF-"):
            return extract_pdf_text(b_src)
        if b_src.startswith(b"PK\x03\x04"):
            return extract_docx_text(b_src)
        if b_src.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"BM", b"RIFF")):
            return describe_image(b_src, lang=lang).summary
        return b_src.decode("utf-8", errors="replace")

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="replace")
        return str(source)

    raise TypeError(f"Unsupported document source: {type(source).__name__}")


class DocumentProcessor:
    """High-level universal document processor for Parentheses Modular Free Think / RAG."""

    def __init__(self, lang: str = "eng"):
        self.lang = lang

    def extract(self, source: str | Path | bytes | bytearray | Image.Image, filename: str | None = None) -> str:
        return extract_document_text(source, filename=filename, lang=self.lang)


# ---------------------------------------------------------------------------
# Self-Test & CLI
# ---------------------------------------------------------------------------

def _self_test():
    import tempfile
    from PIL import PngImagePlugin
    from model import ModelConfig, Parentheses, PRESETS

    torch.manual_seed(0)

    # 1. Image loading + visual feature analysis + PNG metadata
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Bright red PNG with embedded metadata
        red_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        red_arr[:, :, 0] = 240
        red_arr[:, :, 1] = 20
        red_arr[:, :, 2] = 20
        red_img = Image.fromarray(red_arr)
        png_info = PngImagePlugin.PngInfo()
        png_info.add_text("Title", "Solar Flare Observation")
        png_info.add_text("Location", "Kodaikanal Solar Observatory")
        red_path = tmp_path / "solar_red.png"
        red_img.save(red_path, pnginfo=png_info)

        desc = describe_image(red_path, lang="eng")
        assert desc["width"] == 64 and desc["height"] == 64
        assert "red" in desc["dominant_colors"], desc["dominant_colors"]
        assert desc["contrast_level"] == "uniform", desc["contrast_level"]
        assert "Solar Flare Observation" in desc["summary"], desc["summary"]
        assert "Kodaikanal" in str(desc)

        # Kannada localized visual summary
        desc_kan = describe_image(red_path, lang="kan")
        assert "ಚಿತ್ರದ ದೃಶ್ಯ ವಿವರಣೆ" in desc_kan.summary

        # 2. SSM O(1) Recurrent Visual State Prefill across 16 and 64 patches
        model = Parentheses(PRESETS["parentheses-0.9-300k-selective"]).eval()
        states_16 = prefill_visual_state(model, red_path, patch_size=16, grid_size=64)
        states_64 = prefill_visual_state(model, red_path, patch_size=8, grid_size=64)
        assert state_shapes(states_16) == state_shapes(states_64)
        assert state_memory_bytes(states_16) == state_memory_bytes(states_64)
        assert all(s.abs().sum().item() > 0 for s in states_16)

        # 3. PDF generation + extraction (FlateDecode compressed + Kannada Unicode)
        pdf_path = tmp_path / "test_doc.pdf"
        pdf_bytes = create_pdf_bytes([
            "Parentheses uses linear-time selective state space recurrence.",
            "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಕೇಂದ್ರ.",
        ], compress=True)
        pdf_path.write_bytes(pdf_bytes)
        pdf_text = extract_document_text(pdf_path)
        assert "selective state space recurrence" in pdf_text, pdf_text
        assert "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ" in pdf_text, pdf_text

        # 4. DOCX generation + extraction (with Dravidian paragraphs & table)
        docx_path = tmp_path / "test_doc.docx"
        docx_bytes = create_docx_bytes(
            paragraphs=[
                "Western Ghats biodiversity report.",
                "ಕಾವೇರಿ ನದಿ ಕೊಡಗು ಜಿಲ್ಲೆಯ ತಲಕಾವೇರಿಯಲ್ಲಿ ಹುಟ್ಟುತ್ತದೆ.",
                "தமிழ்நாடு மற்றும் கேரளா மலைத்தொடர்கள்.",
            ],
            tables=[[["River", "Origin"], ["Kaveri", "Talakaveri"]]],
        )
        docx_path.write_bytes(docx_bytes)
        docx_text = extract_document_text(docx_path)
        assert "Western Ghats biodiversity" in docx_text
        assert "ತಲಕಾವೇರಿಯಲ್ಲಿ" in docx_text
        assert "தமிழ்நாடு" in docx_text
        assert "Kaveri | Talakaveri" in docx_text

        # 5. HTML extraction (stripping script/style)
        html_path = tmp_path / "page.html"
        html_path.write_text(
            "<html><head><title>SSM Architecture</title>"
            "<script>alert('xss');</script><style>body{color:red;}</style></head>"
            "<body><h1>Recurrent State Space</h1><p>Constant O(1) inference memory.</p></body></html>",
            encoding="utf-8",
        )
        html_text = extract_document_text(html_path)
        assert "alert" not in html_text and "color:red" not in html_text
        assert "Recurrent State Space" in html_text and "Constant O(1) inference memory." in html_text

        # 6. CSV & JSONL extraction
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("language,bpb\nkan,0.654\ntam,0.678\n", encoding="utf-8")
        csv_text = extract_document_text(csv_path)
        assert "language: kan | bpb: 0.654" in csv_text

        jsonl_path = tmp_path / "records.jsonl"
        jsonl_path.write_text(
            json.dumps({"city": "Mysuru", "script": "Kannada"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        jsonl_text = extract_document_text(jsonl_path)
        assert "city: Mysuru | script: Kannada" in jsonl_text

    print("[self-test] multimodal ok (SSM O(1) visual prefill + PDF/DOCX/HTML/CSV/JSONL/Image verified)")


def main():
    p = argparse.ArgumentParser(description="Parentheses Multimodal (Image + Document) Subsystem.")
    p.add_argument("--self-test", action="store_true", help="run CPU self-tests and exit")
    p.add_argument("--describe-image", default=None, help="path to an image to describe")
    p.add_argument("--extract-doc", default=None, help="path to a document (.pdf, .docx, .html, .csv, .json) to extract")
    p.add_argument("--lang", default="eng", help="language tag (eng, kan, tam, tel, mal, ...)")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.describe_image:
        print(describe_image(args.describe_image, lang=args.lang).summary)
    if args.extract_doc:
        print(extract_document_text(args.extract_doc, lang=args.lang))


if __name__ == "__main__":
    main()
