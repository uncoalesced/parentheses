#!/usr/bin/env python3
# Engineered by uncoalesced
"""
tests/test_sft_abstention_v4.py
Unit and property tests for multilingual epistemic abstention generator (scripts/generate_sft_abstention_v4.py).
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_sft_abstention_v4 import (
    build_english_catalog,
    build_kannada_catalog,
    build_tamil_catalog,
    build_telugu_catalog,
    build_malayalam_catalog,
)


def test_catalogs_length_and_structure():
    """Verify that every language catalog returns exactly 800 non-empty, well-formed tuples."""
    catalogs = {
        "eng": build_english_catalog(),
        "kan": build_kannada_catalog(),
        "tam": build_tamil_catalog(),
        "tel": build_telugu_catalog(),
        "mal": build_malayalam_catalog(),
    }

    for lang, items in catalogs.items():
        assert len(items) == 800, f"Expected 800 items for {lang}, got {len(items)}"
        for prompt, think, resp in items:
            assert prompt.strip(), f"Empty prompt in {lang}"
            assert think.strip(), f"Empty think trace in {lang}"
            assert resp.strip(), f"Empty response in {lang}"
            # Ensure no unclosed tags inside raw think string
            assert think.count("<think>") == think.count("</think>")


def test_generate_sft_abstention_cli_execution():
    """Run CLI generation end-to-end and test dataset distribution and invariants."""
    script_path = REPO_ROOT / "scripts" / "generate_sft_abstention_v4.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_jsonl = Path(tmp_dir) / "test_abstention.jsonl"
        cmd = [
            sys.executable,
            str(script_path),
            "--out",
            str(out_jsonl),
            "--seed",
            "123",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "[OK] Generated 4000 multilingual epistemic abstention SFT traces" in res.stdout
        assert out_jsonl.exists()

        records = []
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        assert len(records) == 4000

        counts_by_lang = {}
        think_count = 0
        direct_count = 0

        for r in records:
            assert r["kind"] == "epistemic_abstention"
            lang = r["lang"]
            counts_by_lang[lang] = counts_by_lang.get(lang, 0) + 1

            response = r["response"]
            if "<think>" in response:
                assert response.count("<think>") == response.count("</think>") == 1
                assert response.startswith("<think>")
                think_count += 1
            else:
                assert "</think>" not in response
                direct_count += 1

            # Format verification
            formatted = r["formatted"]
            assert formatted.startswith(f"<|user|><{lang}>")
            assert "<|assistant|>" in formatted
            assert formatted.endswith("<|endoftext|>")

        # Invariant: exactly 800 per language
        for expected_lang in ["eng", "kan", "tam", "tel", "mal"]:
            assert counts_by_lang[expected_lang] == 800

        # Invariant: ~85% with <think> (17/20 = 85.0% -> 3400 items), 15% direct (600 items)
        assert think_count == 3400
        assert direct_count == 600
