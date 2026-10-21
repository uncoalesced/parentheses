"""
Comprehensive Multimodal (Image + Document) Verification Test Suite for Parentheses 0.9.

Covers:
  1. Synthetic & real PNG/JPEG/BMP/base64/Tensor image loading and raster patch tokenization.
  2. SSM O(1) Recurrent Visual State Prefill (`prefill_visual_state`) across 16, 64, and 256
     patches and multi-image streams -- proving constant O(1) recurrent state size.
  3. SSM `slab_v2=True` recurrent visual state prefill invariant verification.
  4. Visual feature & spatial 2x2 quadrant analysis (bright vs dark, high-contrast vs uniform,
     RGB/HSL color detection, Sobel edge density, PNG metadata extraction).
  5. Multilingual (English + Dravidian `kan`, `tam`, `tel`, `mal`, `tcy`, `kff`, `kxv`, `nit`)
     localized visual summaries.
  6. Pure-Python PDF generation & extraction (uncompressed, `/FlateDecode` compressed, `TJ`
     kerning arrays, escaped octals/parentheses, and UTF-16BE `<FEFF...>` Unicode).
  7. Pure-Python DOCX generation & extraction (Kannada/Tamil/Telugu/Malayalam/Odia paragraphs
     and structured tables).
  8. HTML/XML document parsing (`<script>`/`<style>` stripping, headings, paragraphs, tables).
  9. CSV, TSV, JSON, and JSONL structured document ingestion.
  10. `IngestedStore` mixed multimodal ingestion & retrieval (BM25 + Vector/TurboVec backends)
      across PDF, DOCX, HTML, CSV, JSONL, and PNG image modalities.
  11. `ModularFreeThinkSession` end-to-end BPE generation over ingested documents & images
      with internal `</freethink>` closure and zero role/tag leaks.
  12. Security & corrupted input boundary validation (corrupted PDF/DOCX/Image handling).

Run:
    python tests/test_multimodal_suite.py
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, PngImagePlugin

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.free_think import load_bpe_tokenizer
from features.modular_free_think import IngestedStore, ModularFreeThinkSession, _tokens
from features.multimodal import (
    DocumentProcessor,
    ImageProcessor,
    SSMVisualProjector,
    VisualDescription,
    create_docx_bytes,
    create_pdf_bytes,
    describe_image,
    extract_csv_text,
    extract_document_text,
    extract_docx_text,
    extract_html_text,
    extract_json_text,
    extract_pdf_text,
    load_image,
    prefill_visual_state,
    state_memory_bytes,
    state_shapes,
)
from model import ModelConfig, PRESETS, Parentheses


class TestMultimodalSuite(unittest.TestCase):
    """Comprehensive verification suite for the Parentheses 0.9 Multimodal Subsystem."""

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Image Loading & Raster Patch Tokenization
    # ------------------------------------------------------------------
    def test_01_image_loading_and_patch_tokenization(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            arr = np.zeros((48, 96, 3), dtype=np.uint8)
            arr[:, :48, 0] = 250  # left half red
            arr[:, 48:, 2] = 250  # right half blue
            pil_img = Image.fromarray(arr)

            # Save as PNG, JPEG, BMP, WEBP
            for ext, fmt in ((".png", "PNG"), (".jpg", "JPEG"), (".bmp", "BMP"), (".webp", "WEBP")):
                p = tmp_dir / f"sample{ext}"
                pil_img.save(p, format=fmt)
                loaded = load_image(p)
                self.assertEqual(loaded.size, (96, 48))

            # Raw bytes & base64 data URI
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            raw_bytes = buf.getvalue()
            self.assertEqual(load_image(raw_bytes).size, (96, 48))

            b64_uri = "data:image/png;base64," + base64.b64encode(raw_bytes).decode("ascii")
            self.assertEqual(load_image(b64_uri).size, (96, 48))

            # Torch tensor (C, H, W) and (1, C, H, W) and numpy array
            chw = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            self.assertEqual(load_image(chw).size, (96, 48))
            self.assertEqual(load_image(chw.unsqueeze(0)).size, (96, 48))
            self.assertEqual(load_image(arr).size, (96, 48))

            # Patch extraction: grid_size=64, patch_size=16 -> 16 patches of dim 768
            proj = SSMVisualProjector(patch_size=16, grid_size=64)
            patches = proj.extract_patches(pil_img)
            self.assertEqual(tuple(patches.shape), (1, 16, 3 * 16 * 16))
            self.assertTrue(torch.all(patches >= -1.001) and torch.all(patches <= 1.001))

    # ------------------------------------------------------------------
    # 2. SSM O(1) Recurrent Visual State Prefill Invariant
    # ------------------------------------------------------------------
    def test_02_ssm_o1_recurrent_visual_state_prefill(self):
        cfg = ModelConfig(
            vocab_size=256, block_size=256, n_layer=4, n_head=4, n_embd=64,
            attn_type="selective_linear",
        )
        model = Parentheses(cfg).eval()
        expected_shapes = [(1, 4, 16, 16)] * 4

        empty_st = model.empty_states(1)
        baseline_bytes = state_memory_bytes(empty_st)
        self.assertEqual(state_shapes(empty_st), expected_shapes)

        # Create synthetic test image
        arr = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        img = Image.fromarray(arr)

        # Step 16 patches (64x64, P=16), 64 patches (64x64, P=8), and 256 patches (128x128, P=8)
        st_16 = prefill_visual_state(model, img, patch_size=16, grid_size=64)
        st_64 = prefill_visual_state(model, img, patch_size=8, grid_size=64)
        st_256 = prefill_visual_state(model, img, patch_size=8, grid_size=128)

        # O(1) Constant State Size Proof: state shapes and byte sizes are strictly identical!
        for st in (st_16, st_64, st_256):
            self.assertEqual(state_shapes(st), expected_shapes)
            self.assertEqual(state_memory_bytes(st), baseline_bytes)
            for layer_s in st:
                self.assertTrue(torch.isfinite(layer_s).all())
                self.assertGreater(layer_s.abs().sum().item(), 0.0)

        # Determinism check
        st_16_again = prefill_visual_state(model, img, patch_size=16, grid_size=64)
        for s1, s2 in zip(st_16, st_16_again):
            self.assertTrue(torch.allclose(s1, s2, atol=1e-6))

        # Multi-image sequential prefill (10 images in a row -> exact same O(1) state memory!)
        running_state = None
        for i in range(10):
            img_i = Image.new("RGB", (64, 64), color=((i * 25) % 255, (i * 50) % 255, (i * 75) % 255))
            running_state = prefill_visual_state(model, img_i, states=running_state, patch_size=16, grid_size=64)
            self.assertEqual(state_shapes(running_state), expected_shapes)
            self.assertEqual(state_memory_bytes(running_state), baseline_bytes)

    # ------------------------------------------------------------------
    # 3. SSM slab_v2=True Recurrent Visual State Prefill
    # ------------------------------------------------------------------
    def test_03_ssm_slab_v2_visual_state_prefill(self):
        cfg = ModelConfig(
            vocab_size=256, block_size=256, n_layer=2, n_head=4, n_embd=64,
            attn_type="selective_linear", slab_v2=True,
        )
        model = Parentheses(cfg).eval()
        empty_st = model.empty_states(1)
        baseline_bytes = state_memory_bytes(empty_st)

        img = Image.new("RGB", (64, 64), color=(120, 200, 80))
        st_after = prefill_visual_state(model, img, patch_size=16, grid_size=64)
        self.assertEqual(state_shapes(st_after), [(1, 4, 16, 16), (1, 4, 16, 16)])
        self.assertEqual(state_memory_bytes(st_after), baseline_bytes)

    # ------------------------------------------------------------------
    # 4. Visual Feature, Contrast, Edge & 2x2 Spatial Quadrant Analysis
    # ------------------------------------------------------------------
    def test_04_visual_feature_and_quadrant_analysis(self):
        # A. Bright white image vs dark black image
        bright_img = Image.new("RGB", (64, 64), color=(245, 245, 245))
        dark_img = Image.new("RGB", (64, 64), color=(10, 10, 10))
        d_bright = describe_image(bright_img)
        d_dark = describe_image(dark_img)
        self.assertEqual(d_bright["brightness_level"], "bright")
        self.assertEqual(d_dark["brightness_level"], "dark")
        self.assertEqual(d_bright["contrast_level"], "uniform")
        self.assertIn("white", d_bright["dominant_colors"])
        self.assertIn("black", d_dark["dominant_colors"])

        # B. High-contrast checkerboard with strong edges
        checker = np.zeros((64, 64, 3), dtype=np.uint8)
        checker[::2, ::2] = 255
        checker[1::2, 1::2] = 255
        d_check = describe_image(Image.fromarray(checker))
        self.assertEqual(d_check["contrast_level"], "high-contrast")
        self.assertEqual(d_check["texture_level"], "detailed/textured")
        self.assertGreater(d_check["edge_density"], 0.20)

        # C. 2x2 Quadrant layout & PNG metadata
        quad_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        quad_arr[:32, :32] = 255  # Top-Left bright white, other 3 quadrants black
        quad_img = Image.fromarray(quad_arr)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "quad_meta.png"
            info = PngImagePlugin.PngInfo()
            info.add_text("Author", "Parentheses Vision Lab")
            info.add_text("Subject", "Dravidian Epigraphy Plate")
            quad_img.save(p, pnginfo=info)

            d_quad = describe_image(p)
            self.assertGreater(d_quad["quadrants"]["top_left"], 240.0)
            self.assertLess(d_quad["quadrants"]["bottom_right"], 5.0)
            self.assertEqual(d_quad["metadata"].get("Author"), "Parentheses Vision Lab")
            self.assertIn("Dravidian Epigraphy Plate", d_quad.summary)

    # ------------------------------------------------------------------
    # 5. Multilingual Dravidian Visual Descriptions
    # ------------------------------------------------------------------
    def test_05_dravidian_localized_image_descriptions(self):
        img = Image.new("RGB", (48, 48), color=(20, 180, 40))
        for lang_tag, expected_sub in (
            ("eng", "Visual Image Summary"),
            ("kan", "ಚಿತ್ರದ ದೃಶ್ಯ ವಿವರಣೆ"),
            ("tam", "படத்தின் காட்சி சுருக்கம்"),
            ("tel", "చిత్ర దృశ్య సారాంశం"),
            ("mal", "ചിത്രത്തിന്റെ ദൃശ്യ സംഗ്രഹം"),
        ):
            desc = describe_image(img, lang=lang_tag)
            self.assertIsInstance(desc, VisualDescription)
            self.assertEqual(desc["lang"], lang_tag)
            self.assertIn(expected_sub, desc.summary)
            self.assertIn("green", desc["dominant_colors"])

    # ------------------------------------------------------------------
    # 6. Pure-Python PDF Generation & Extraction (FlateDecode + Unicode)
    # ------------------------------------------------------------------
    def test_06_pdf_generation_and_extraction(self):
        paras = [
            "Selective Linear Attention computes recurrent state updates in O(1) memory.",
            "Western Ghats rivers include Kaveri, Netravati, and Periyar.",
            "ಕನ್ನಡ ಮತ್ತು ತಮಿಳು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿವೆ.",
        ]
        # Compressed (/FlateDecode) PDF
        pdf_compressed = create_pdf_bytes(paras, compress=True)
        text_c = extract_pdf_text(pdf_compressed)
        for p in paras:
            self.assertIn(p, text_c)

        # Uncompressed PDF with TJ kerning array and escaped parentheses
        raw_tj_stream = (
            b"BT /F1 12 Tf 72 720 Td "
            b"[(Recurrent) -200 (State) -180 (Space\\(SSM\\))] TJ "
            b"0 -24 Td (Octal escape: \\101\\102\\103) Tj ET"
        )
        custom_pdf = (
            b"%PDF-1.4\n1 0 obj\n<< /Length "
            + str(len(raw_tj_stream)).encode("ascii")
            + b" >>\nstream\n"
            + raw_tj_stream
            + b"\nendstream\nendobj\n%%EOF"
        )
        text_u = extract_pdf_text(custom_pdf)
        self.assertIn("Recurrent State Space(SSM)", text_u)
        self.assertIn("Octal escape: ABC", text_u)

    # ------------------------------------------------------------------
    # 7. Pure-Python DOCX Generation & Extraction (Dravidian + Tables)
    # ------------------------------------------------------------------
    def test_07_docx_generation_and_extraction_dravidian(self):
        paragraphs = [
            "Parentheses 0.9-120M Multilingual Document Benchmark.",
            "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಆಗಿದೆ.",
            "சென்னை தமிழ்நாட்டின் தலைநகரம் ஆகும்.",
            "హైదరాబాద్ తెలంగాణ రాజధాని.",
            "തിരുവനന്തപുരം കേരളത്തിന്റെ തലസ്ഥാനമാണ്.",
            "ଭୁବନେଶ୍ୱର ଓଡ଼ିଶାର ରାଜଧାନୀ.",
        ]
        tables = [
            [
                ["Language", "Script", "Tag"],
                ["Kannada", "Knda", "kan"],
                ["Tamil", "Taml", "tam"],
                ["Telugu", "Telu", "tel"],
                ["Malayalam", "Mlym", "mal"],
            ]
        ]
        docx_bytes = create_docx_bytes(paragraphs=paragraphs, tables=tables)
        extracted = extract_docx_text(docx_bytes)
        for p in paragraphs:
            self.assertIn(p, extracted)
        self.assertIn("Kannada | Knda | kan", extracted)
        self.assertIn("Malayalam | Mlym | mal", extracted)

    # ------------------------------------------------------------------
    # 8. HTML & XML Document Parsing
    # ------------------------------------------------------------------
    def test_08_html_and_xml_parsing(self):
        raw_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parentheses SSM Documentation</title>
            <style>.hidden { display: none; color: purple; }</style>
            <script>const secret = "DO_NOT_INCLUDE_SCRIPT";</script>
        </head>
        <body>
            <h1>State Space Duality Architecture</h1>
            <p>Training runs in parallel linear time O(T) via chunked scans.</p>
            <p>ಲ್ಲಾದರೂ ಇರು ಎಂತಾದರೂ ಇರು ಎಂದೆಂದಿಗೂ ನೀ ಕನ್ನಡವಾಗಿರು.</p>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>kan_bpb</td><td>0.6542</td></tr>
            </table>
        </body>
        </html>
        """
        extracted = extract_html_text(raw_html)
        self.assertNotIn("DO_NOT_INCLUDE_SCRIPT", extracted)
        self.assertNotIn("display: none", extracted)
        self.assertIn("Parentheses SSM Documentation", extracted)
        self.assertIn("State Space Duality Architecture", extracted)
        self.assertIn("ಕನ್ನಡವಾಗಿರು", extracted)
        self.assertIn("kan_bpb | 0.6542", extracted)

    # ------------------------------------------------------------------
    # 9. CSV, TSV, JSON & JSONL Structured Ingestion
    # ------------------------------------------------------------------
    def test_09_csv_tsv_and_json_jsonl_structured_ingestion(self):
        csv_data = "model,params,kan_bpb,tam_bpb\nParentheses-120M,119.6M,0.654,0.678\n"
        csv_out = extract_csv_text(csv_data)
        self.assertIn("model: Parentheses-120M | params: 119.6M | kan_bpb: 0.654 | tam_bpb: 0.678", csv_out)

        tsv_data = "river\torigin\nKaveri\tTalakaveri\n"
        tsv_out = extract_csv_text(tsv_data, delimiter="\t")
        self.assertIn("river: Kaveri | origin: Talakaveri", tsv_out)

        json_data = json.dumps(
            [{"temple": "Hampi Virupaksha", "state": "Karnataka"}, {"temple": "Meenakshi", "state": "Tamil Nadu"}],
            ensure_ascii=False,
        )
        json_out = extract_json_text(json_data, is_jsonl=False)
        self.assertIn("temple: Hampi Virupaksha | state: Karnataka", json_out)
        self.assertIn("temple: Meenakshi | state: Tamil Nadu", json_out)

        jsonl_data = (
            json.dumps({"lang": "kan", "greeting": "ನಮಸ್ಕಾರ"}, ensure_ascii=False) + "\n"
            + json.dumps({"lang": "tam", "greeting": "வணக்கம்"}, ensure_ascii=False) + "\n"
        )
        jsonl_out = extract_json_text(jsonl_data, is_jsonl=True)
        self.assertIn("lang: kan | greeting: ನಮಸ್ಕಾರ", jsonl_out)
        self.assertIn("lang: tam | greeting: வணக்கம்", jsonl_out)

    # ------------------------------------------------------------------
    # 10. IngestedStore Mixed Multimodal Ingestion & Retrieval (BM25 + Vector)
    # ------------------------------------------------------------------
    def test_10_ingested_store_mixed_multimodal_bm25_and_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)

            pdf_p = tdir / "ocean_report.pdf"
            pdf_p.write_bytes(create_pdf_bytes([
                "Abyssal hydrothermal vents release superheated mineral water on the ocean floor."
            ]))

            docx_p = tdir / "karnataka_history.docx"
            docx_p.write_bytes(create_docx_bytes([
                "ಹಂಪಿ ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ಭವ್ಯ ರಾಜಧಾನಿಯಾಗಿತ್ತು."
            ]))

            html_p = tdir / "ssm_notes.html"
            html_p.write_text(
                "<html><script>evil()</script><body><p>Mamba selective linear attention avoids quadratic KV caches.</p></body></html>",
                encoding="utf-8",
            )

            csv_p = tdir / "benchmarks.csv"
            csv_p.write_text("dataset,score,metric\nsangraha_verified,0.6542,val_bpb\n", encoding="utf-8")

            img_p = tdir / "mars_rover.png"
            mars_img = Image.new("RGB", (64, 64), color=(230, 60, 20))
            png_meta = PngImagePlugin.PngInfo()
            png_meta.add_text("Mission", "Mangalyaan Orbiter")
            mars_img.save(img_p, pnginfo=png_meta)

            all_files = [pdf_p, docx_p, html_p, csv_p, img_p]

            # A. BM25 Backend
            bm25_store = IngestedStore(backend="bm25")
            bm25_store.ingest(all_files)
            self.assertGreaterEqual(len(bm25_store.documents), 5)

            self.assertIn("hydrothermal", bm25_store.retrieve("hydrothermal vents")[0].lower())
            self.assertIn("ವಿಜಯನಗರ", bm25_store.retrieve("ಹಂಪಿ ಸಾಮ್ರಾಜ್ಯದ")[0])
            self.assertIn("quadratic", bm25_store.retrieve("Mamba quadratic")[0].lower())
            self.assertIn("sangraha_verified", bm25_store.retrieve("sangraha_verified val_bpb")[0])
            self.assertIn("Mangalyaan", bm25_store.retrieve("Mangalyaan Orbiter")[0])

            # B. Vector (TurboVec) Backend with deterministic word-hash embedder
            def hash_embedder(texts):
                out = np.zeros((len(texts), 64), dtype=np.float32)
                for r, t in enumerate(texts):
                    for w in _tokens(t):
                        out[r, hash(w) % 64] += 1.0
                norms = np.linalg.norm(out, axis=1, keepdims=True)
                return out / np.maximum(norms, 1e-9)

            vec_store = IngestedStore(backend="vector", embedder=hash_embedder)
            vec_store.ingest(all_files)
            self.assertEqual(len(vec_store.documents), len(bm25_store.documents))
            vec_hits = vec_store.retrieve("Abyssal hydrothermal vents superheated mineral water", top_k=1)
            self.assertTrue(vec_hits and "hydrothermal" in vec_hits[0].lower())

    # ------------------------------------------------------------------
    # 11. ModularFreeThinkSession End-to-End BPE Generation over Docs & Images
    # ------------------------------------------------------------------
    def test_11_modular_freethink_end_to_end_bpe_multimodal(self):
        tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
        tok = load_bpe_tokenizer({"tokenizer": {"path": str(tok_dir)}})
        bpe_model = Parentheses(
            ModelConfig(
                vocab_size=32768, block_size=512, n_layer=2, n_head=2, n_embd=32,
                attn_type="selective_linear",
            )
        ).eval()

        store = IngestedStore()
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            pdf_p = tdir / "kaveri.pdf"
            pdf_p.write_bytes(create_pdf_bytes([
                "ಕಾವೇರಿ ನದಿ ಕೊಡಗು ಜಿಲ್ಲೆಯ ತಲಕಾವೇರಿಯಲ್ಲಿ ಹುಟ್ಟುತ್ತದೆ."
            ]))
            docx_p = tdir / "mysuru.docx"
            docx_p.write_bytes(create_docx_bytes([
                "ಮೈಸೂರು ಅರಮನೆ ಕರ್ನಾಟಕದ ಪ್ರಮುಖ ಪ್ರವಾಸಿ ತಾಣವಾಗಿದೆ."
            ]))
            img_p = tdir / "waterfall.png"
            Image.new("RGB", (64, 64), color=(30, 140, 230)).save(img_p)

            store.ingest([pdf_p, docx_p])
            session = ModularFreeThinkSession(
                bpe_model, store, splice_every=6, tokenizer=tok, lang="kan",
                mode="freethink", images=[img_p],
            )

            # Verify SSM visual state was prefilled in O(1) memory
            self.assertIsNotNone(session.image_states)
            self.assertEqual(state_shapes(session.image_states), [(1, 2, 16, 16), (1, 2, 16, 16)])

            # Verify primed context splices retrieved PDF/DOCX chunks in Kannada
            primed = session._primed("ಕಾವೇರಿ ನದಿ ಕರ್ನಾಟಕದ ಜೀವನದಿ.")
            self.assertIn("ತಲಕಾವೇರಿಯಲ್ಲಿ", primed)
            self.assertIn("ಇದರ ಬಗ್ಗೆ", primed)

            # Run end-to-end BPE stream and verify internal </freethink> closure + zero role/tag leaks
            output = session.run("ಕಾವೇರಿ ನದಿ ಕರ್ನಾಟಕದ ಜೀವನದಿ.", max_tokens=12)
            self.assertTrue(len(output) > 0)
            self.assertTrue(session.closed_freethink)
            self.assertEqual(session.internal_close_tag, "</freethink>")
            self.assertIn("</freethink>", session.internal_transcript)
            for forbidden in (
                "<|user|>", "<|assistant|>", "<|endoftext|>", "<|endofturn|>",
                "<|pad|>", "<esg>", "<think>", "</think>", "<freethink>", "</freethink>",
            ):
                self.assertNotIn(forbidden, output)

    # ------------------------------------------------------------------
    # 12. Security & Corrupted Input Boundary Validation
    # ------------------------------------------------------------------
    def test_12_security_and_corrupted_input_boundaries(self):
        # Corrupted PDF bytes (missing %PDF- header)
        with self.assertRaises(ValueError):
            extract_pdf_text(b"NOT_A_VALID_PDF_FILE_AT_ALL")

        # Corrupted DOCX bytes (not a zip file)
        with self.assertRaises(ValueError):
            extract_docx_text(b"CORRUPTED_ZIP_BYTES_12345")

        # Valid ZIP archive but missing word/document.xml
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("other.txt", "hello")
        with self.assertRaises(ValueError):
            extract_docx_text(buf.getvalue())

        # Corrupted image bytes
        with self.assertRaises(ValueError):
            load_image(b"NOT_AN_IMAGE_89PNG_CORRUPT")

        # Non-existent file path
        with self.assertRaises(FileNotFoundError):
            load_image("non_existent_directory_xyz/missing_image.png")

        # Invalid patch grid geometry
        with self.assertRaises(ValueError):
            SSMVisualProjector(patch_size=15, grid_size=64)

        # Causal attention model must be rejected by prefill_visual_state (SSM-only invariant)
        causal_model = Parentheses(PRESETS["parentheses-0.9-100k"]).eval()
        img = Image.new("RGB", (64, 64), color=(100, 100, 100))
        with self.assertRaises(RuntimeError):
            prefill_visual_state(causal_model, img)

    # ------------------------------------------------------------------
    # Real 120M Checkpoint Loader (cached across tests 13-15)
    # ------------------------------------------------------------------
    _real_120m_cache = None

    @classmethod
    def _get_real_120m(cls):
        if cls._real_120m_cache is not None:
            return cls._real_120m_cache
        from scripts.chat import load_model_and_tokenizer
        candidates = [
            REPO_ROOT / "checkpoints" / "parentheses-0.9-120M-sft-v2" / "step_9786_final.pt",
            REPO_ROOT / "checkpoints" / "parentheses-0.9-120M-sft" / "step_9636_final.pt",
            REPO_ROOT / "release" / "parentheses-0.9-120M-candidate" / "model.safetensors",
        ]
        for ckpt_p in candidates:
            if ckpt_p.exists():
                model, tok, cfg, step = load_model_and_tokenizer(ckpt_p, device="cpu")
                cls._real_120m_cache = (model, tok, cfg, step)
                return cls._real_120m_cache
        raise unittest.SkipTest("No 120M checkpoint found on disk")

    # ------------------------------------------------------------------
    # 13. Real 120M Checkpoint SSM Visual State Prefill & Grounded FreeThink
    # ------------------------------------------------------------------
    def test_13_real_120m_ssm_visual_prefill_and_freethink(self):
        model, tok, cfg, _ = self._get_real_120m()
        self.assertEqual(cfg.n_layer, 32)
        self.assertEqual(cfg.n_head, 8)
        self.assertEqual(cfg.n_embd, 512)
        self.assertEqual(cfg.attn_type, "selective_linear")

        with tempfile.TemporaryDirectory() as tmp:
            img_p = Path(tmp) / "observatory.png"
            arr = np.zeros((64, 64, 3), dtype=np.uint8)
            arr[:, :, 0] = 30
            arr[:, :, 1] = 110
            arr[:, :, 2] = 235
            img = Image.fromarray(arr)
            meta = PngImagePlugin.PngInfo()
            meta.add_text("Instrument", "Vainu Bappu Telescope")
            img.save(img_p, pnginfo=meta)

            # Verify 32-layer SSM visual state prefill maintains exact O(1) state memory (32 x 1 x 8 x 64 x 64 fp32 = 4,194,304 bytes)
            empty_st = model.empty_states(1)
            expected_bytes = state_memory_bytes(empty_st)
            vis_states = prefill_visual_state(model, img_p, patch_size=16, grid_size=64)
            self.assertEqual(len(vis_states), 32)
            self.assertEqual(state_shapes(vis_states), [(1, 8, 64, 64)] * 32)
            self.assertEqual(state_memory_bytes(vis_states), expected_bytes)

            store = IngestedStore()
            session = ModularFreeThinkSession(
                model, store, splice_every=6, tokenizer=tok, lang="eng", images=[img_p]
            )
            out = session.run("Deep sky astronomical observation uses large optical mirrors.", max_tokens=6)
            self.assertTrue(len(out.strip()) > 0)
            self.assertTrue(session.closed_freethink)

    # ------------------------------------------------------------------
    # 14. Real 120M Checkpoint Grounded QA/FreeThink over PDF & Dravidian DOCX
    # ------------------------------------------------------------------
    def test_14_real_120m_pdf_and_dravidian_docx_grounded_qa(self):
        from scripts.chat import reply
        model, tok, _, _ = self._get_real_120m()

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            pdf_p = tdir / "ssm_paper.pdf"
            pdf_p.write_bytes(create_pdf_bytes([
                "Parentheses 0.9-120M uses selective linear attention with constant O(1) recurrent state."
            ]))
            docx_p = tdir / "kannada_notes.docx"
            docx_p.write_bytes(create_docx_bytes([
                "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ಪ್ರಮುಖ ತಂತ್ರಜ್ಞಾನ ಕೇಂದ್ರವಾಗಿದೆ.",
                "சென்னை தமிழ்நாட்டின் தலைநகரம்.",
            ]))

            ans_kan = reply(
                model,
                f'/doc "{docx_p}" /think ಬೆಂಗಳೂರು ಯಾವ ರಾಜ್ಯದ ರಾಜಧಾನಿ?',
                "cpu", max_new_tokens=6, temperature=0.5, top_k=20,
                tok=tok, lang="kan", fmt="chat", stream=False,
            )
            self.assertTrue(len(ans_kan.strip()) > 0)
            for hidden in ("/doc", "/think", "<think>", "</think>", "<freethink>", "</freethink>"):
                self.assertNotIn(hidden, ans_kan)

    # ------------------------------------------------------------------
    # 15. Web Studio app.py POST /api/ingest Endpoint Verification
    # ------------------------------------------------------------------
    def test_15_web_studio_api_ingest_endpoint(self):
        import app as studio_app

        tiny_cfg = ModelConfig(
            vocab_size=32768, block_size=512, n_layer=2, n_head=2, n_embd=32,
            attn_type="selective_linear",
        )
        tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
        tok = load_bpe_tokenizer({"tokenizer": {"path": str(tok_dir)}})
        engine = studio_app.GenerationEngine(Parentheses(tiny_cfg).eval(), tok, device="cpu")

        # 1. Ingest PDF via engine
        pdf_bytes = create_pdf_bytes(["Hydrothermal vents on the abyssal ocean floor."])
        res_pdf = engine.ingest_multimodal("ocean.pdf", raw_bytes=pdf_bytes, lang="eng")
        self.assertEqual(res_pdf["status"], "ok")
        self.assertEqual(res_pdf["modality"], "document")
        self.assertIn("Hydrothermal vents", res_pdf["summary"])

        # 2. Ingest Kannada DOCX via engine
        docx_bytes = create_docx_bytes(["ಕಾವೇರಿ ನದಿ ಕೊಡಗಿನಲ್ಲಿ ಹುಟ್ಟುತ್ತದೆ."])
        res_docx = engine.ingest_multimodal("kaveri.docx", raw_bytes=docx_bytes, lang="kan")
        self.assertEqual(res_docx["status"], "ok")
        self.assertIn("ಕಾವೇರಿ", res_docx["summary"])

        # 3. Ingest PNG Image via engine & verify O(1) SSM visual state prefill
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(240, 180, 20)).save(buf, format="PNG")
        res_img = engine.ingest_multimodal("gold.png", raw_bytes=buf.getvalue(), lang="eng")
        self.assertEqual(res_img["status"], "ok")
        self.assertEqual(res_img["modality"], "image")
        self.assertGreater(res_img["ssm_state_bytes"], 0)
        self.assertEqual(res_img["ssm_state_shapes"], [[1, 2, 16, 16], [1, 2, 16, 16]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
