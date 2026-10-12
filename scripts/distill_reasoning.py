#!/usr/bin/env python3
"""Verified code + math distillation from a hosted teacher (Claude Code, 2026-10-12).

Replaces Gemini's llama_cpp stub and distill_coder_7b.py (canned tool rows, 10 cycled prompts).
Talks to any OpenAI-compatible endpoint, so it runs during training with no local RAM/CPU cost:
    NVIDIA NIM: --base-url https://integrate.api.nvidia.com/v1   (key from env NVIDIA_API_KEY)
    Ollama:     --base-url http://localhost:11434/v1             (no key)

    .venv\\Scripts\\python.exe scripts\\distill_reasoning.py --list-models --base-url ...
    .venv\\Scripts\\python.exe scripts\\distill_reasoning.py --task code --model <id> --limit 3
    .venv\\Scripts\\python.exe scripts\\distill_reasoning.py --self-test

Rules (SPEC_120M_DISTILLATION_AND_LINGUAL_PIPELINE.md):
- English only. External teachers never write Dravidian text.
- A row is kept only if verified: code must pass the dataset's own MBPP asserts, math must
  match the GSM8K gold answer. Only the teacher's own reasoning is kept; no canned <think>.
- Seeds: MBPP train/validation/prompt (CC BY 4.0; test split left untouched) and GSM8K train
  (MIT; lm-eval scores the test split).
Resume-safe: source ids already in --output are skipped.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)")
SYSTEM = {
    "code": "You are an expert Python programmer. Think step by step, then give one complete "
            "Python solution in a single ```python code block. Do not include test code.",
    "math": "Solve the problem step by step. End with a final line of the form 'Answer: <number>'.",
}


def verify_python_code(code_str: str, timeout_sec: int = 10) -> bool:
    """Run code in an isolated interpreter (-I) in an empty temp dir; exit 0 = pass.
    shortcut: no OS sandbox on Windows, so teacher code runs as the user; fine for
    MBPP-style functions from a trusted endpoint, use a container before untrusted prompts."""
    if not code_str or not code_str.strip():
        return False
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code_str)
        try:
            res = subprocess.run([sys.executable, "-I", path], cwd=d, timeout=timeout_sec,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def format_sft_turn(prompt: str, think: str, answer: str, lang: str = "eng") -> str:
    """One chat turn in v3_32k special tokens; <think> only when the teacher really reasoned."""
    body = f"<think>\n{think.strip()}\n</think>\n{answer.strip()}" if think.strip() else answer.strip()
    return f"<|user|><{lang}> {prompt.strip()}<|assistant|>{body}<|endoftext|>\n"


def load_seeds(task: str) -> list[dict]:
    from datasets import load_dataset
    if task == "math":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return [{"id": f"gsm8k-train-{i}", "prompt": r["question"], "license": "MIT",
                 "gold": r["answer"].split("####")[-1].strip().replace(",", "")}
                for i, r in enumerate(ds)]
    seeds = []
    for split in ("train", "validation", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "full", split=split):
            seeds.append({"id": f"mbpp-{r['task_id']}", "license": "CC-BY-4.0",
                          "prompt": f"{r['text']}\nYour code should pass this test:\n{r['test_list'][0]}",
                          "tests": "\n".join([r.get("test_setup_code") or ""] + r["test_list"])})
    return seeds


def chat(base_url: str, model: str, messages: list, key: str | None, max_tokens: int, retries: int = 5) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                         "temperature": 0.6}).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["choices"][0]["message"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt * 5)


def split_reply(msg: dict) -> tuple[str, str]:
    """(think, answer). Reasoning endpoints return it in reasoning_content or inline <think>."""
    content = msg.get("content") or ""
    think = msg.get("reasoning_content") or msg.get("reasoning") or ""
    m = THINK_RE.search(content)
    if m:
        think = think or m.group(1)
        content = THINK_RE.sub("", content)
    return think.strip(), content.strip()


def verify(task: str, seed: dict, answer: str) -> bool:
    if task == "math":
        m = ANSWER_RE.findall(answer)
        try:
            return bool(m) and abs(float(m[-1].replace(",", "")) - float(seed["gold"])) < 1e-6
        except ValueError:
            return False
    blocks = CODE_RE.findall(answer)
    return bool(blocks) and verify_python_code(blocks[-1] + "\n\n" + seed["tests"])


def run(args, seeds: list[dict], chat_fn=chat) -> dict:
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        done = {json.loads(l)["source_id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    key = os.environ.get("NVIDIA_API_KEY") if "nvidia.com" in args.base_url else None
    todo = [s for s in seeds if s["id"] not in done]
    stats = {"kept": 0, "rejected": 0, "skipped": len(seeds) - len(todo), "too_long": 0}
    for seed in todo[:args.limit] if args.limit else todo:
        msg = chat_fn(args.base_url, args.model,
                      [{"role": "system", "content": SYSTEM[args.task]},
                       {"role": "user", "content": seed["prompt"]}], key, args.max_tokens)
        think, answer = split_reply(msg)
        if len(think) + len(answer) > args.max_chars:   # the student trains at T=4096
            stats["too_long"] += 1
            continue
        if not verify(args.task, seed, answer):
            stats["rejected"] += 1
            print(f"[distill] reject {seed['id']}", flush=True)
            continue
        row = {"text": format_sft_turn(seed["prompt"], think, answer), "task": args.task,
               "source_id": seed["id"], "license": seed["license"], "teacher": args.model,
               "base_url": args.base_url}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        stats["kept"] += 1
        print(f"[distill] keep {seed['id']} ({stats['kept']} kept, {stats['rejected']} rejected)", flush=True)
    return stats


def self_test():
    assert verify_python_code("def add(a, b): return a + b\nassert add(2, 3) == 5")
    assert not verify_python_code("assert False")
    assert not verify_python_code("import time; time.sleep(10)", timeout_sec=1)
    assert format_sft_turn("Q", "", "A") == "<|user|><eng> Q<|assistant|>A<|endoftext|>\n"
    assert "<think>\nr\n</think>\nA" in format_sft_turn("Q", "r", "A")
    assert split_reply({"content": "<think>x</think> y"}) == ("x", "y")
    assert split_reply({"content": "y", "reasoning_content": "x"}) == ("x", "y")

    seeds = [{"id": "c1", "license": "t", "prompt": "add", "tests": "assert add(1, 2) == 3"},
             {"id": "c2", "license": "t", "prompt": "add", "tests": "assert add(1, 2) == 3"}]
    replies = iter(["```python\ndef add(a, b):\n    return a + b\n```",
                    "```python\ndef add(a, b):\n    return a - b\n```"])
    fake = lambda *x: {"content": next(replies), "reasoning_content": "sum them"}
    with tempfile.TemporaryDirectory() as d:
        a = argparse.Namespace(output=os.path.join(d, "o.jsonl"), base_url="http://x/v1", model="m",
                               task="code", limit=0, max_tokens=10, max_chars=10_000)
        st = run(a, seeds, fake)
        assert st["kept"] == 1 and st["rejected"] == 1, st
        rows = [json.loads(l) for l in open(a.output, encoding="utf-8")]
        assert rows[0]["source_id"] == "c1" and rows[0]["text"].startswith("<|user|><eng> add<|assistant|><think>")
        st = run(a, seeds[:1], fake)   # resume: c1 already written, no call made
        assert st == {"kept": 0, "rejected": 0, "skipped": 1, "too_long": 0}, st

        m = argparse.Namespace(**{**vars(a), "task": "math", "output": os.path.join(d, "m.jsonl")})
        ms = [{"id": "g1", "license": "MIT", "prompt": "1+1", "gold": "2"},
              {"id": "g2", "license": "MIT", "prompt": "2+2", "gold": "4"}]
        it = iter(["1+1 is 2.\nAnswer: 2", "Answer: 5"])
        st = run(m, ms, lambda *x: {"content": next(it)})
        assert st["kept"] == 1 and st["rejected"] == 1, st
        assert "<think>" not in open(m.output, encoding="utf-8").read()   # no canned reasoning
    print("[self-test] distill_reasoning ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["code", "math"], default="code")
    p.add_argument("--base-url", default="https://integrate.api.nvidia.com/v1")
    p.add_argument("--model", default=None)
    p.add_argument("--output", default=str(REPO_ROOT / "data" / "processed_v5" / "sft_tail_distill.jsonl"))
    p.add_argument("--limit", type=int, default=0, help="max new prompts this run (0 = all)")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--max-chars", type=int, default=12_000, help="drop longer replies (student T=4096)")
    p.add_argument("--list-models", action="store_true")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return 0
    key = os.environ.get("NVIDIA_API_KEY") if "nvidia.com" in a.base_url else None
    if "nvidia.com" in a.base_url and not key:
        p.error("set NVIDIA_API_KEY in your shell first")
    if a.list_models:
        req = urllib.request.Request(a.base_url.rstrip("/") + "/models",
                                     headers={"Authorization": f"Bearer {key}"} if key else {})
        with urllib.request.urlopen(req, timeout=60) as r:
            print("\n".join(sorted(m["id"] for m in json.loads(r.read())["data"])))
        return 0
    if not a.model:
        p.error("--model is required (see --list-models)")
    print(f"[distill] {a.task} via {a.model} @ {a.base_url} -> {a.output}", flush=True)
    print(f"[distill] done: {run(a, load_seeds(a.task))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
