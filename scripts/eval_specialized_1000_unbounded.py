#!/usr/bin/env python3
"""Run the 1,000-case Parentheses specialized evaluation suite.

Generation is intentionally uncapped: model end/control tokens stop a response;
--watchdog-seconds is an optional time guard (0 disables it). Inputs are never
silently truncated and must fit the checkpoint's configured context.

Examples:
  .venv/Scripts/python.exe -B scripts/eval_specialized_1000_unbounded.py --self-test
  .venv/Scripts/python.exe -B scripts/eval_specialized_1000_unbounded.py \
      --checkpoint checkpoints/parentheses-1.0-253M-s9/final.pt --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model import Parentheses  # noqa: E402
from models.model_slab_v4 import SlabV4Model  # noqa: E402

SUITE_DIR = REPO_ROOT / "data" / "eval_suites"
PREFIX = "parentheses_253m_specialized_1000_v1"
DEFAULT_PROMPTS = SUITE_DIR / f"{PREFIX}.prompts.jsonl"
DEFAULT_KEYS = SUITE_DIR / f"{PREFIX}.keys.jsonl"
DEFAULT_MANIFEST = SUITE_DIR / f"{PREFIX}.manifest.json"
DEFAULT_TOKENIZER = REPO_ROOT / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
DEFAULT_OUTPUT = REPO_ROOT / "out" / "specialized_1000_v1.jsonl"
STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_suite(prompts_path: Path, keys_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompts = load_jsonl(prompts_path)
    keys = load_jsonl(keys_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(prompts) != 1000 or len(keys) != 1000:
        raise ValueError(f"suite must have 1,000 prompts and keys; got {len(prompts)} / {len(keys)}")
    if len({row.get("id") for row in prompts}) != 1000:
        raise ValueError("prompt IDs are not unique")
    if [row.get("id") for row in prompts] != [row.get("id") for row in keys]:
        raise ValueError("prompt/key row order or IDs differ")
    expected = Counter({f"D{i:02d}": 100 for i in range(1, 11)})
    if Counter(row.get("domain_id") for row in prompts) != expected:
        raise ValueError("domain distribution is not exactly 10 × 100")
    if Counter(row.get("language") for row in prompts) != Counter({"en": 800, "kn": 50, "ta": 50, "te": 50, "ml": 50}):
        raise ValueError("language distribution is not 800 English and 50 each kn/ta/te/ml")
    if sha256_file(prompts_path) != manifest.get("prompts_sha256"):
        raise ValueError("prompt file SHA-256 does not match manifest")
    if sha256_file(keys_path) != manifest.get("keys_sha256"):
        raise ValueError("answer-key file SHA-256 does not match manifest")
    return prompts, manifest


def make_prompt_ids(tokenizer: Tokenizer, row: dict[str, Any], device: str) -> torch.Tensor:
    user_id = tokenizer.token_to_id("<|user|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    language_id = tokenizer.token_to_id(row.get("language_tag", ""))
    if None in (user_id, assistant_id, language_id):
        raise ValueError(f"missing special token for {row['id']} ({row.get('language_tag')})")
    body = tokenizer.encode(row["prompt"], add_special_tokens=False).ids
    ids = [int(user_id), int(language_id), *body, int(assistant_id)]
    return torch.tensor([ids], dtype=torch.long, device=device)


def context_limit(cfg: Any) -> int:
    for name in ("block_size", "max_seq_len", "context_length"):
        value = getattr(cfg, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 4096


def sample_next(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    top_k = min(50, logits.size(-1))
    values, _ = torch.topk(logits, top_k, dim=-1)
    logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def unbounded_token_stream(
    model: Any,
    input_ids: torch.Tensor,
    temperature: float,
    clock: dict[str, float] | None = None,
) -> Iterable[int]:
    """Yield tokens until the caller stops; there is deliberately no token cap.

    If ``clock`` is given, ``clock["prefill_end"]`` is set once the prompt has been
    consumed, so the watchdog can time decoding only. Token-by-token prefill on the
    recurrent path runs at decode speed (~45 tok/s); timing it against the watchdog
    penalised long prompts (D06) before a single output token was produced.
    """
    if isinstance(model, SlabV4Model):
        states = model.empty_states(input_ids.size(0), device=input_ids.device)
        with torch.inference_mode():
            for pos in range(input_ids.size(1) - 1):
                _, states = model.step(input_ids[:, pos:pos + 1], states)
            if input_ids.is_cuda:
                torch.cuda.synchronize()
            if clock is not None:
                clock["prefill_end"] = time.monotonic()
            current = input_ids[:, -1:]
            while True:
                logits, states = model.step(current, states)
                next_id = sample_next(logits, temperature)
                yield int(next_id[0, 0].item())
                current = next_id
    else:
        with torch.inference_mode():
            first = True
            for next_id in model.stream(
                input_ids,
                max_new_tokens=None,
                temperature=temperature,
                top_k=1 if temperature <= 0 else 50,
                repetition_penalty=1.0,
            ):
                if first and clock is not None:
                    # stream() hides its prefill; the first token marks its end.
                    clock["prefill_end"] = time.monotonic()
                first = False
                yield int(next_id[0, 0].item())


def collect_until_stop(
    token_stream: Iterable[int],
    stop_ids: set[int],
    *,
    started: float,
    watchdog_seconds: float = 0.0,
    collected: list[int] | None = None,
    clock: dict[str, float] | None = None,
) -> tuple[list[int], str, bool]:
    """Collect tokens until a stop token or the watchdog fires.

    The watchdog measures from ``clock["prefill_end"]`` when available (decode time
    only), otherwise from ``started``.
    """
    output = collected if collected is not None else []
    for token_id in token_stream:
        if token_id in stop_ids:
            return output, "stop_token", True
        output.append(token_id)
        t_ref = clock.get("prefill_end", started) if clock is not None else started
        if watchdog_seconds > 0 and time.monotonic() - t_ref >= watchdog_seconds:
            return output, "watchdog", False
    return output, "stream_ended", True


def load_model(checkpoint: Path, device: str) -> tuple[Any, Any, int, int]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = bundle.get("cfg")
    state = bundle.get("model")
    if cfg is None or not isinstance(state, dict):
        raise ValueError("checkpoint must contain 'cfg' and a 'model' state dictionary")
    model = SlabV4Model(cfg) if getattr(cfg, "slab_v4", False) else Parentheses(cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    step = int(bundle.get("step", -1))
    params = sum(p.numel() for p in model.parameters())
    return model, cfg, step, params


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


def run(args: argparse.Namespace) -> int:
    prompts_path = Path(args.prompts)
    keys_path = Path(args.keys)
    manifest_path = Path(args.manifest)
    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output)
    for path in (prompts_path, keys_path, manifest_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    prompts, manifest = validate_suite(prompts_path, keys_path, manifest_path)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Self-test does only CPU/file/tokenizer checks; it never loads a checkpoint or uses a GPU.
    if args.self_test:
        tags = {tokenizer.token_to_id(tag) for tag in ("<eng>", "<kan>", "<tam>", "<tel>", "<mal>")}
        if None in tags:
            raise ValueError("tokenizer is missing one or more required language tags")
        special_ids = {tokenizer.token_to_id(tag) for tag in STOP_TOKENS + ("<|user|>", "<|assistant|>")}
        if None in special_ids:
            raise ValueError("tokenizer is missing one or more chat/stop tokens")
        lengths = [len(make_prompt_ids(tokenizer, row, "cpu")[0]) for row in prompts]
        limit = 4096
        too_long = [(prompts[i]["id"], length) for i, length in enumerate(lengths) if length > limit]
        if too_long:
            raise ValueError(f"{len(too_long)} prompts exceed the declared 4,096-token input context: {too_long[:5]}")

        class FakeModel:
            def stream(self, _idx: torch.Tensor, max_new_tokens: int | None, **_kwargs):
                assert max_new_tokens is None, "stream must be open-ended"
                yield torch.tensor([[11]])
                yield torch.tensor([[12]])
                yield torch.tensor([[99]])

        fake = FakeModel()
        got, why, complete = collect_until_stop(
            unbounded_token_stream(fake, torch.tensor([[1, 2]]), 0.0), {99}, started=time.monotonic()
        )
        if (got, why, complete) != ([11, 12], "stop_token", True):
            raise AssertionError((got, why, complete))

        # Watchdog must time decoding only: a slow prefill must not consume the budget.
        class SlowPrefillModel:
            def stream(self, _idx: torch.Tensor, max_new_tokens: int | None, **_kwargs):
                time.sleep(0.3)  # simulated prefill longer than the watchdog
                for tok in (21, 22, 23, 99):
                    yield torch.tensor([[tok]])

        clock: dict[str, float] = {}
        got, why, complete = collect_until_stop(
            unbounded_token_stream(SlowPrefillModel(), torch.tensor([[1, 2]]), 0.0, clock=clock),
            {99},
            started=time.monotonic(),
            watchdog_seconds=0.2,
            clock=clock,
        )
        if (got, why, complete) != ([21, 22, 23], "stop_token", True) or "prefill_end" not in clock:
            raise AssertionError(("prefill counted against watchdog", got, why, complete, clock))
        print("PASS: watchdog times decode only (a 0.3 s prefill did not trip a 0.2 s watchdog).")
        print(f"PASS: {len(prompts)} prompts/keys; exact domain/language quotas; manifest hashes verified.")
        print(f"PASS: all prompt token IDs fit 4,096 input tokens; required tokenizer tags and stop tokens exist.")
        print("PASS: generation calls the model stream with max_new_tokens=None and stops only on control token (unless watchdog is enabled).")
        return 0

    if not args.checkpoint:
        raise ValueError("--checkpoint is required for a model run")
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"output exists; choose another --output or pass --resume: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model, cfg, step, params = load_model(Path(args.checkpoint), device)
    max_input = context_limit(cfg)
    raw_stop_ids = {tokenizer.token_to_id(token) for token in STOP_TOKENS}
    if None in raw_stop_ids:
        raise ValueError("tokenizer is missing a required stop token")
    stop_ids = {int(x) for x in raw_stop_ids}
    run_id = uuid.uuid4().hex

    completed: set[str] = set()
    if args.resume and output_path.exists():
        for old in load_jsonl(output_path):
            if old.get("is_complete") is True:
                completed.add(str(old.get("id")))

    chosen = prompts
    if args.prompt_ids:
        wanted = {x.strip() for x in args.prompt_ids.split(",") if x.strip()}
        unknown = wanted - {p["id"] for p in prompts}
        if unknown:
            raise ValueError(f"unknown prompt IDs: {sorted(unknown)}")
        chosen = [p for p in prompts if p["id"] in wanted]
    if args.limit is not None:
        chosen = chosen[:args.limit]
    if args.shuffle:
        random.Random(args.seed).shuffle(chosen)

    total = len(chosen)
    done_now = 0
    print(f"Running {total} prompts on {device}; model={type(model).__name__} params={params:,} checkpoint_step={step}.")
    print("Generation token cap: NONE. Stop: model control token; optional watchdog is time-based only.")
    for index, row in enumerate(chosen, 1):
        if row["id"] in completed:
            continue
        prompt_ids = make_prompt_ids(tokenizer, row, device)
        if prompt_ids.size(1) > max_input:
            raise ValueError(f"{row['id']} has {prompt_ids.size(1)} input tokens, exceeding checkpoint context {max_input}; no truncation performed")
        t0 = time.monotonic()
        clock: dict[str, float] = {}
        token_ids: list[int] = []
        finish_reason = "interrupted"
        is_complete = False
        try:
            token_ids, finish_reason, is_complete = collect_until_stop(
                unbounded_token_stream(model, prompt_ids, args.temperature, clock=clock),
                stop_ids,
                started=t0,
                watchdog_seconds=args.watchdog_seconds,
                collected=token_ids,
                clock=clock,
            )
        except KeyboardInterrupt:
            finish_reason = "interrupted"
            is_complete = False
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        t_end = time.monotonic()
        prefill_end = clock.get("prefill_end", t0)
        decode_seconds = t_end - prefill_end
        record = {
            "run_id": run_id,
            "id": row["id"],
            "domain_id": row["domain_id"],
            "language": row["language"],
            "difficulty": row["difficulty"],
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_step": step,
            "model_class": type(model).__name__,
            "device": device,
            "temperature": args.temperature,
            "prompt_tokens": int(prompt_ids.size(1)),
            "completion_tokens": len(token_ids),
            "generated_text": text,
            "finish_reason": finish_reason,
            "watchdog_seconds": args.watchdog_seconds,
            "is_complete": is_complete,
            "generation_seconds": round(t_end - t0, 3),
            "prefill_seconds": round(prefill_end - t0, 3),
            "decode_seconds": round(decode_seconds, 3),
            "tok_per_s": round(len(token_ids) / decode_seconds, 2) if decode_seconds > 0 else None,
            "watchdog_clock": "decode_only",
        }
        append_record(output_path, record)
        done_now += 1
        print(f"[{index}/{total}] {row['id']} tokens={len(token_ids)} stop={finish_reason} elapsed={record['generation_seconds']:.1f}s")
        if finish_reason == "interrupted":
            print("Interrupted; partial response was saved. Resume with --resume.")
            return 130
    print(f"Finished this batch: {done_now} new records appended to {output_path}.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="model checkpoint (required except with --self-test)")
    parser.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    parser.add_argument("--keys", default=str(DEFAULT_KEYS), help="used only for integrity checks, never fed into the model")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy deterministic evaluation; no output token cap")
    parser.add_argument("--watchdog-seconds", type=float, default=0.0, help="optional per-response time stop; 0 disables it; never a token cap")
    parser.add_argument("--limit", type=int, help="run only the first N prompts (smoke/testing only)")
    parser.add_argument("--prompt-ids", help="comma-separated exact IDs to run")
    parser.add_argument("--shuffle", action="store_true", help="shuffle selected prompt order with --seed")
    parser.add_argument("--seed", type=int, default=2531000)
    parser.add_argument("--resume", action="store_true", help="append to output and skip IDs already completed")
    parser.add_argument("--self-test", action="store_true", help="verify suite, tokenizer, context lengths, and uncapped stop behavior; no checkpoint/GPU")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    if args.watchdog_seconds < 0:
        parser.error("--watchdog-seconds cannot be negative")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted before a complete response was available.", file=sys.stderr)
        raise SystemExit(130)
