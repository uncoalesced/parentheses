"""
Free Think Mode.

Engineered by uncoalesced

From the plan: given a non-question / "safe statement" prompt, the model keeps
generating a live, viewable/exportable stream of "thinking" tokens about the
input, using whatever's in its trained knowledge -- rather than stopping after
a normal response.

Purely a decoding-time layer, no architecture changes: Parentheses.stream()
(KV-cached, so per-token cost stays at normal decoding cost -- the plan's
efficiency requirement) plus a priming prefix, an incremental byte->text
decoder, and an export.

    python -m features.free_think --checkpoint checkpoints/step_161000.pt \
        --prompt "The ocean is very deep."
    python -m features.free_think --checkpoint checkpoints/step_161000.pt \
        --prompt "The ocean is very deep." --max-tokens 20000 \
        --timing t.csv    # per-token latency, flat-vs-growing summary
    python -m features.free_think --self-test    # no checkpoint needed

A callback API isn't provided on purpose: start() is a generator, so a UI wants
`for chunk in session.start(prompt): show(chunk)`, which is the callback.
"""

import argparse
import codecs
import json
import statistics
import time
from pathlib import Path

import torch

from model import Parentheses

# Where export() is allowed to write. Not just a default -- a boundary: export()
# refuses any path that resolves outside this directory, so a future caller
# (e.g. an auto-save trigger, not a human typing --export) can't be tricked or
# mistaken into writing somewhere else on disk. See handoff-conversation-memory.md.
EXPORT_DIR = Path(__file__).resolve().parent.parent / "exports"

# Tunable without touching code below. The model is byte-level and trained on
# plain Wikipedia/Gutenberg prose, so this is a text-continuation cue, not a
# system prompt it was instruction-tuned to obey.
PRIMING = "{prompt}\n\nThinking about this: "

# Cheap heuristic rather than the "small classifier head" the design sketch
# offered as an alternative -- Free Think only has to catch the obvious
# question shapes, and a classifier head would need its own labelled data and
# training.
_QUESTION_STARTS = (
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "might", "has", "have", "had",
)


def is_question(prompt: str) -> bool:
    """True if `prompt` looks like a question -- Free Think Mode is for statements."""
    text = prompt.strip()
    if not text:
        return False
    if text.endswith("?"):
        return True
    return text.split()[0].strip("\"'(").lower() in _QUESTION_STARTS


# The two helpers below are the parts features/modular_free_think.py reuses: it
# builds a different primed string (retrieved chunks spliced in) but must reject
# questions and decode bytes identically. Kept private -- FreeThinkSession and
# is_question stay the public interface.

def _reject_questions(prompt: str, force: bool):
    if not force and is_question(prompt):
        raise ValueError(
            f"Free Think Mode is for statements, not questions: {prompt!r} "
            "(pass force=True to free-think anyway)"
        )


def _stream_text(model: Parentheses, primed: str, max_tokens: int | None,
                 temperature: float, top_k: int, timings: list[float] | None = None):
    """Yield decoded text as `model` continues `primed`.

    `timings`, if given, gets one wall-clock seconds-per-token entry appended
    per sampled byte -- the measurement behind the selective-linear backbone's
    O(1)-per-token claim (does latency stay flat as position grows, or creep
    up because something is still secretly O(n)?). Opt-in rather than always
    on: an open-ended session (max_tokens=None) runs until the consumer stops,
    so an unconditional list would grow without bound.
    """
    device = next(model.parameters()).device
    idx = torch.tensor([list(primed.encode("utf-8"))], dtype=torch.long, device=device)
    # Pin the primed prompt as an attention sink: on a run that goes on
    # long enough to slide past block_size (the whole point of Free
    # Think), the model would otherwise have zero memory it ever started
    # here -- see Parentheses.stream's sink_tokens docstring. stream()
    # itself clamps this further (at most block_size//4), this cap just
    # keeps a short prompt from sinking its own entire length.
    sink_tokens = min(idx.size(1), model.cfg.block_size // 4)
    # a multi-byte character can span several sampled tokens, so decode
    # incrementally instead of one bytes() call per token
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    # driven by next() rather than `for`, so the timed region is exactly the
    # model's work for one token -- a `for` body would also fold in however
    # long the consumer (printing, a UI) spends between yields.
    stream = model.stream(idx, max_tokens, temperature, top_k, sink_tokens)
    while True:
        started = time.perf_counter()
        try:
            next_id = next(stream)
        except StopIteration:
            break
        # int() forces the device->host copy, so on CUDA it waits for the step
        # that produced this token. Must stay inside the timed region or the
        # GPU work lands in the *next* token's slice and every number is skewed.
        byte = int(next_id)
        if timings is not None:
            timings.append(time.perf_counter() - started)
        chunk = decoder.decode(bytes([byte]))
        if chunk:
            yield chunk


def resolve_inside(base_dir: Path | str, path: str) -> Path:
    """Resolve `path` under `base_dir`, refusing anything that escapes it.

    The containment check both this file's export() and
    features/conversation_memory.py's notes file rely on: a caller that isn't
    a human typing a filename (an auto-save trigger, a conversation log the
    model itself names) can't be tricked into touching the rest of the disk.
    Creates `base_dir` -- callers only ever want it to exist.
    """
    base_dir = Path(base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    target = (base_dir / path).resolve()
    if not target.is_relative_to(base_dir):
        raise ValueError(f"path escapes {base_dir}: {path!r}")
    return target


def write_timings(timings: list[float], path: str, base_dir: Path | str = EXPORT_DIR) -> Path:
    """Write per-token latencies as `token_index,ms` CSV; return the path.

    Sandboxed under `base_dir` exactly like export(). Row 0 also carries the
    prefill of the primed prompt (a generator body doesn't run until the first
    next()), so it is always the slow one -- read the trend from row 1 on.
    """
    target = resolve_inside(base_dir, path)
    with open(target, "w", encoding="utf-8") as f:
        f.write("token_index,ms\n")
        for i, seconds in enumerate(timings):
            f.write(f"{i},{seconds * 1000:.4f}\n")
    return target


def timing_summary(timings: list[float], window: int = 100) -> str:
    """One line: does per-token latency stay flat, or grow with position?"""
    body = timings[1:]  # row 0 is prefill, not a steady-state token
    if not body:
        return f"{len(timings)} tokens (too few to compare)"
    window = min(window, len(body))
    first = statistics.median(body[:window]) * 1000
    last = statistics.median(body[-window:]) * 1000
    return (f"{len(timings)} tokens, median first {window} = {first:.3f} ms, "
            f"median last {window} = {last:.3f} ms, ratio {last / first:.3f}x")


class FreeThinkSession:
    """A live, exportable stream of open-ended "thinking" about a statement."""

    def __init__(self, model: Parentheses, temperature: float = 0.9, top_k: int = 40):
        if model.cfg.vocab_size > 256:
            raise ValueError(
                "FreeThinkSession decodes model output as raw UTF-8 bytes; the "
                f"vocab_size={model.cfg.vocab_size} presets need their trained BPE "
                "tokenizer wired in here first (see data/tokenizer/)."
            )
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.prompt: str | None = None
        self.history: list[str] = []

    def start(self, prompt: str, max_tokens: int | None = None, force: bool = False,
              timings: list[float] | None = None):
        """Yield decoded text as the model thinks, appending it to self.history.

        max_tokens=None keeps thinking until the consumer stops iterating.
        `timings` is passed straight through to _stream_text() -- see there.
        """
        _reject_questions(prompt, force)
        self.prompt = prompt
        self.history = []
        primed = PRIMING.format(prompt=prompt.strip())
        for chunk in _stream_text(self.model, primed, max_tokens, self.temperature,
                                  self.top_k, timings):
            self.history.append(chunk)
            yield chunk

    def run(self, prompt: str, max_tokens: int = 400, **kwargs) -> str:
        """Non-streaming convenience: consume start() and return the whole text."""
        return "".join(self.start(prompt, max_tokens, **kwargs))

    def export(self, path: str, base_dir: Path | str = EXPORT_DIR):
        """Write the session to `path` -- JSON if it ends in .json, else plain text
        (.md included -- it's just prose, no special-casing needed).

        `path` is resolved relative to `base_dir` and must stay inside it --
        rejects `../` traversal or an absolute path pointing elsewhere, so this
        can't be used to write outside the sandboxed export directory.
        """
        target = resolve_inside(base_dir, path)
        text = "".join(self.history)
        with open(target, "w", encoding="utf-8") as f:
            if target.suffix == ".json":
                json.dump({"prompt": self.prompt, "thinking": text}, f, indent=2)
            else:
                f.write(f"{self.prompt}\n\n{text}\n")
        return target


def main():
    p = argparse.ArgumentParser(description="Free Think Mode: open-ended thinking about a statement.")
    p.add_argument("--checkpoint", default="checkpoints/step_161000.pt")
    p.add_argument("--prompt", default="The ocean is very deep.")
    p.add_argument("--max-tokens", type=int, default=400, help="0 = think until Ctrl-C")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--export", default=None,
                   help=f"filename to write inside {EXPORT_DIR} (.json, .txt, .md -- "
                        "can't escape that directory)")
    p.add_argument("--timing", default=None,
                   help=f"filename to write per-token latencies to inside {EXPORT_DIR} "
                        "(token_index,ms CSV); also prints a first-vs-last summary")
    p.add_argument("--force", action="store_true", help="free-think even if the prompt looks like a question")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true",
                   help="check the session logic on an untrained tiny model and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    # weights_only=False: train.py pickles the ModelConfig dataclass into the ckpt
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    session = FreeThinkSession(model, args.temperature, args.top_k)
    print(f"[free-think] {args.checkpoint} (step {ckpt['step']}) on {args.device}")
    print(f"[free-think] prompt: {args.prompt}\n")
    timings = [] if args.timing else None
    try:
        for chunk in session.start(args.prompt, args.max_tokens or None,
                                   force=args.force, timings=timings):
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    if args.export:
        target = session.export(args.export)
        print(f"[free-think] exported to {target}")
    if timings:
        target = write_timings(timings, args.timing)
        print(f"[free-think] {timing_summary(timings)}")
        print(f"[free-think] per-token latencies written to {target}")


def _self_test():
    import tempfile

    from model import PRESETS

    torch.manual_seed(0)
    assert is_question("Why is the sky blue?")
    assert is_question("is this a question")
    assert is_question('"Would you look at that')
    assert not is_question("The sky is blue.")
    assert not is_question("")

    # smallest *byte-level* preset: tiny-smoke's vocab_size=64 can't even
    # embed an ASCII prompt, and Free Think feeds raw UTF-8 bytes in
    model = Parentheses(PRESETS["parentheses-0.9-100k"]).eval()
    session = FreeThinkSession(model)

    try:
        list(session.start("Why?"))
        raise AssertionError("questions should be rejected")
    except ValueError:
        pass
    assert list(session.start("Why?", max_tokens=3, force=True))  # ...unless forced

    text = session.run("A statement.", max_tokens=20)
    assert text == "".join(session.history) and text, repr(text)

    # open-ended stream: no max_tokens, the consumer decides when to stop
    seen = 0
    for _ in session.start("Another statement."):
        seen += 1
        if seen == 5:
            break
    assert seen == 5

    # timing is opt-in and counts sampled *bytes*, not decoded chunks -- a
    # multi-byte character is several tokens that yield one chunk between them
    timings = []
    session.run("A fourth statement.", max_tokens=12, timings=timings)
    assert len(timings) == 12, len(timings)
    assert all(t >= 0 for t in timings), timings
    assert "12 tokens" in timing_summary(timings), timing_summary(timings)
    assert "too few" in timing_summary([0.001])  # nothing left after dropping prefill

    with tempfile.TemporaryDirectory() as tmp:
        rows = open(write_timings(timings, "t.csv", base_dir=tmp), encoding="utf-8").read().splitlines()
        assert rows[0] == "token_index,ms" and len(rows) == 13, rows[:2]

    with tempfile.TemporaryDirectory() as tmp:
        session.run("A third statement.", max_tokens=10)
        txt = session.export("s.txt", base_dir=tmp)
        md = session.export("s.md", base_dir=tmp)  # .md is plain text, same branch as .txt
        js = session.export("s.json", base_dir=tmp)
        assert "A third statement." in open(txt, encoding="utf-8").read()
        assert "A third statement." in open(md, encoding="utf-8").read()
        assert json.load(open(js, encoding="utf-8"))["prompt"] == "A third statement."

        # sandbox: can't escape base_dir via traversal or an absolute path
        for escape in ("../escaped.txt", str(Path(tmp).parent / "escaped.txt")):
            try:
                session.export(escape, base_dir=tmp)
                raise AssertionError(f"export should have rejected {escape!r}")
            except ValueError:
                pass
    print("[self-test] free_think ok")


if __name__ == "__main__":
    main()
