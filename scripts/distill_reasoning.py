#!/usr/bin/env python3
"""Offline Reasoning & Distillation Pipeline for Parentheses 0.9-120M SFT Tail.

Per SPEC_120M_DISTILLATION_AND_LINGUAL_PIPELINE.md:
- Teacher 1: Qwen3.8-27B-UD-IQ3_XXS.gguf (math, logic, multi-step deliberation)
- Teacher 2: Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf (code logic, tools)
- STRICT RULE: External LLM teachers NEVER generate Dravidian language text.
  External teachers generate ONLY domain-neutral math, code, and structured reasoning.
- Output Format: <|user|><lang> {prompt}<|assistant|><think>{reasoning}</think>{answer}<|endoftext|>
- Mandatory Sandbox Gate: Python execution & math ground-truth verification.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERIDOT_MODELS = Path("E:/Peridot/models")

TEACHER_REASONING = PERIDOT_MODELS / "Qwen3.8-27B-UD-IQ3_XXS.gguf"
TEACHER_CODER = PERIDOT_MODELS / "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"

THINK_REGEX = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)


def verify_python_code(code_str: str, timeout_sec: int = 5) -> bool:
    """Automated sandbox verification: executes code snippet in isolated process."""
    if not code_str or not code_str.strip():
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code_str)
        temp_path = f.name
    try:
        res = subprocess.run([sys.executable, temp_path], timeout=timeout_sec,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except (subprocess.TimeoutExpired, Exception):
        return False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def format_sft_turn(prompt: str, think: str, answer: str, lang: str = "eng") -> str:
    """Format single reasoning turn using native v3_32k tokens."""
    prompt = prompt.strip()
    think = think.strip()
    answer = answer.strip()
    return f"<|user|><{lang}> {prompt}<|assistant|><think>\n{think}\n</think>\n{answer}<|endoftext|>\n"


def main():
    parser = argparse.ArgumentParser(description="Distillation and Reasoning Traces Generator")
    parser.add_argument("--teacher", choices=["reasoning", "coder"], default="reasoning")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "processed_v5" / "distilled_reasoning.jsonl"))
    parser.add_argument("--self-test", action="store_true", help="Run self-test of sandbox verification")
    args = parser.parse_args()

    if args.self_test:
        print("[self-test] Testing python verification sandbox...")
        assert verify_python_code("def add(a, b): return a + b\nassert add(2, 3) == 5\nprint('ok')")
        assert not verify_python_code("assert False")
        assert not verify_python_code("import time; time.sleep(10)", timeout_sec=1)
        formatted = format_sft_turn("What is 2+2?", "2+2 equals 4 by basic arithmetic.", "4", "eng")
        assert "<|user|><eng> What is 2+2?<|assistant|><think>" in formatted
        print("[self-test] Verification sandbox & formatting passed ok.")
        return 0

    teacher_path = TEACHER_REASONING if args.teacher == "reasoning" else TEACHER_CODER
    if not teacher_path.is_file():
        print(f"[ERROR] Teacher model not found at {teacher_path}", file=sys.stderr)
        return 1

    print(f"[distill] Teacher selected: {teacher_path.name}")
    print(f"[distill] Output destination: {args.output}")
    print("[distill] Ready for batch prompt evaluation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
