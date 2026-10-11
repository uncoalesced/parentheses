"""
Byte-Level NPU Text Filter -- Inline DataLoader Co-Processor.
Runs a lightweight 1D-CNN directly on raw UTF-8 byte streams before tokenization
to detect mojibake, corrupted Unicode, noise, and verify language script.

Engineered for Parentheses 0.9 / 120M on AMD Ryzen AI NPU (XDNA 1).
Supports VitisAIExecutionProvider with seamless CPUExecutionProvider fallback.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnxruntime as ort
except ImportError:
    ort = None

CLASSES = [
    "kan",     # 0: Kannada
    "tam",     # 1: Tamil
    "tel",     # 2: Telugu
    "mal",     # 3: Malayalam
    "lowres",  # 4: Low-resource Dravidian (Tulu, Kodava, Gondi, Kuvi, Kolami)
    "eng",     # 5: English
    "markup",  # 6: HTML, XML, Markdown, Code boilerplate
    "noise",   # 7: Mojibake, corrupt UTF-8, high entropy junk, repetitive noise
]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
FIXED_SEQ_LEN = 512


class ByteFilterCNN(nn.Module):
    """Compact 1D-CNN for byte-level sequence classification and quality scoring.
    
    Operates on uint8/int64 byte indices in range [0, 255].
    Architecture uses depthwise-separable 1D convolutions for minimal compute
    and 100% native compatibility with AMD AIE tile compilation.
    """

    def __init__(self, emb_dim: int = 64, hidden_dim: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(256, emb_dim)

        # Stage 1: Initial conv (downsample x2)
        self.conv1 = nn.Conv1d(emb_dim, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Stage 2: Depthwise separable block 1 (downsample x2)
        self.dw2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, groups=hidden_dim)
        self.pw2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # Stage 3: Depthwise separable block 2 (downsample x2)
        self.dw3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, groups=hidden_dim)
        self.pw3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Stage 4: Depthwise separable block 3 (downsample x2)
        self.dw4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, groups=hidden_dim)
        self.pw4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(hidden_dim)

        # Heads
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len) int64 in [0, 255]
        Returns:
            logits: (batch, num_classes)
            quality: (batch, 1) in [0, 1]
        """
        # (B, L) -> (B, L, D) -> (B, D, L)
        h = self.embedding(x).transpose(1, 2)

        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.pw2(self.dw2(h))))
        h = F.relu(self.bn3(self.pw3(self.dw3(h))))
        h = F.relu(self.bn4(self.pw4(self.dw4(h))))

        pooled = self.pool(h).squeeze(-1)  # (B, hidden_dim)

        logits = self.classifier(pooled)
        quality = torch.sigmoid(self.quality_head(pooled))

        return logits, quality


DEFAULT_ONNX_MODEL = Path(__file__).resolve().parent.parent / "checkpoints" / "npu_byte_filter" / "byte_filter_fixed512.onnx"


class NpuByteFilter:
    """Inference runner for the Byte-Level Text Filter.
    Dispatches to AMD XDNA 1 NPU via VitisAIExecutionProvider if available,
    falling back seamlessly to CPUExecutionProvider.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        provider: str = "auto",
        seq_len: int = FIXED_SEQ_LEN,
        quality_threshold: float = 0.50,
    ):
        if model_path is None and DEFAULT_ONNX_MODEL.exists():
            model_path = str(DEFAULT_ONNX_MODEL)

        self.model_path = model_path
        self.seq_len = seq_len
        self.quality_threshold = quality_threshold
        self.provider = provider
        self.session = None
        self.active_provider = "uninitialized"

        if model_path and Path(model_path).exists():
            self._init_session(model_path)

    def _init_session(self, path: str):
        if ort is None:
            raise RuntimeError("onnxruntime is required for NpuByteFilter ONNX mode.")

        available = ort.get_available_providers()
        requested = []

        if self.provider in ("auto", "npu"):
            if "VitisAIExecutionProvider" in available:
                requested.append("VitisAIExecutionProvider")
            elif self.provider == "npu":
                print("[NpuByteFilter] VitisAIExecutionProvider not found; using CPU.")

        requested.append("CPUExecutionProvider")

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        try:
            self.session = ort.InferenceSession(str(path), sess_options=so, providers=requested)
            self.active_provider = self.session.get_providers()[0]
        except Exception as e:
            print(f"[NpuByteFilter] Provider init error ({e}), falling back to CPU.")
            self.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
            self.active_provider = "CPUExecutionProvider"

    def filter_bytes(self, raw_bytes: bytes) -> Dict[str, Union[str, float, bool]]:
        """Evaluate raw bytes, returning predicted language, quality, and keep recommendation."""
        # Truncate or pad to self.seq_len
        b_arr = np.frombuffer(raw_bytes[: self.seq_len], dtype=np.uint8)
        padded = np.zeros(self.seq_len, dtype=np.int64)
        if len(b_arr) > 0:
            padded[: len(b_arr)] = b_arr

        if self.session is not None:
            inputs = {"bytes": padded[None, :]}
            logits, quality = self.session.run(["logits", "quality"], inputs)
            probs = self._softmax(logits[0])
            pred_class_idx = int(np.argmax(probs))
            q_score = float(quality[0][0])
        else:
            # Fallback heuristic if ONNX model is not loaded
            return self._heuristic_check(raw_bytes)

        pred_lang = CLASSES[pred_class_idx]
        is_clean = (pred_lang not in ("noise", "markup")) and (q_score >= self.quality_threshold)

        return {
            "lang": pred_lang,
            "confidence": float(probs[pred_class_idx]),
            "quality": q_score,
            "keep": is_clean,
            "provider": self.active_provider,
        }

    def filter_text(self, text: str) -> Dict[str, Union[str, float, bool]]:
        """Convenience method accepting unicode string."""
        return self.filter_bytes(text.encode("utf-8", errors="replace"))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    @staticmethod
    def _heuristic_check(raw_bytes: bytes) -> Dict[str, Union[str, float, bool]]:
        """Fast fallback rule check when ONNX model is uninitialized."""
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return {"lang": "noise", "confidence": 1.0, "quality": 0.0, "keep": False, "provider": "heuristic"}

        # Check for mojibake markers
        if "\ufffd" in text or text.count("&amp;") > 3:
            return {"lang": "noise", "confidence": 0.9, "quality": 0.1, "keep": False, "provider": "heuristic"}

        # Basic Indic unicode block ranges
        kannada_chars = sum(1 for c in text if "\u0C80" <= c <= "\u0CFF")
        tamil_chars = sum(1 for c in text if "\u0B80" <= c <= "\u0BFF")
        telugu_chars = sum(1 for c in text if "\u0C00" <= c <= "\u0C7F")
        malayalam_chars = sum(1 for c in text if "\u0D00" <= c <= "\u0D7F")
        total = max(len(text), 1)

        counts = [
            ("kan", kannada_chars),
            ("tam", tamil_tamil := tamil_chars),
            ("tel", telugu_chars),
            ("mal", malayalam_chars),
        ]
        best_lang, best_count = max(counts, key=lambda x: x[1])

        if best_count / total > 0.3:
            return {"lang": best_lang, "confidence": best_count / total, "quality": 0.95, "keep": True, "provider": "heuristic"}

        return {"lang": "eng", "confidence": 0.8, "quality": 0.9, "keep": True, "provider": "heuristic"}


def self_test():
    """Self-test verifying torch model shapes, forward pass, and heuristic fallback."""
    print("[self-test] Running ByteFilterCNN self-test...")
    torch.manual_seed(42)

    # 1. Model instantiation & forward pass
    model = ByteFilterCNN()
    x = torch.randint(0, 256, (2, FIXED_SEQ_LEN), dtype=torch.long)
    logits, quality = model(x)

    assert logits.shape == (2, NUM_CLASSES), f"Unexpected logits shape: {logits.shape}"
    assert quality.shape == (2, 1), f"Unexpected quality shape: {quality.shape}"
    assert (quality >= 0.0).all() and (quality <= 1.0).all(), "Quality scores out of [0, 1] range"

    # 2. Heuristic fallback test
    filter_obj = NpuByteFilter(model_path="")
    res_kn = filter_obj.filter_text("ನಮಸ್ಕಾರ, ಕನ್ನಡ ಭಾಷೆ ತುಂಬಾ ಸುಂದರವಾಗಿದೆ.")
    assert res_kn["lang"] == "kan", f"Expected 'kan', got {res_kn['lang']}"
    assert res_kn["keep"] is True, "Expected valid Kannada text to be kept"

    res_noise = filter_obj.filter_bytes(b"\xff\xfe\x00\x00\x80\x81\x82 corrupted mojibake bytes")
    assert res_noise["keep"] is False, "Expected corrupted bytes to be dropped"

    # 3. ONNX model test if exported
    if DEFAULT_ONNX_MODEL.exists():
        onnx_filter = NpuByteFilter(model_path=str(DEFAULT_ONNX_MODEL))
        onnx_res = onnx_filter.filter_text("ನಮಸ್ಕಾರ, ಕನ್ನಡ ಭಾಷೆ ತುಂಬಾ ಸುಂದರವಾಗಿದೆ.")
        assert onnx_res["keep"] is True, f"Expected clean text to pass ONNX filter: {onnx_res}"
        print(f"[self-test] ONNX filter active provider: {onnx_filter.active_provider}")

    print(f"[self-test] ByteFilterCNN: PASSED (Param count: {sum(p.numel() for p in model.parameters()):,})")


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        idx = sys.argv.index("--check")
        test_str = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "ಕನ್ನಡ"
        filt = NpuByteFilter()
        print(filt.filter_text(test_str))
    else:
        self_test()
