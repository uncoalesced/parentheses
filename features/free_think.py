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
import re
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

# Internal slash commands and XML reasoning tags. These are strictly internal
# engine directives and must NEVER be fed as literal text or leaked in output.
INTERNAL_COMMANDS = (
    "/think",
    "/freethink",
    "/free-think",
    "/nothink",
    "/no-think",
    "/chat",
    "/base",
)
INTERNAL_TAGS = (
    "<think>",
    "</think>",
    "<freethink>",
    "</freethink>",
)

_CMD_RE = re.compile(
    r"(?<!\w)/(?:freethink|free-think|nothink|no-think|think|chat|base)(?!\w)",
    re.IGNORECASE,
)
_TAG_RE = re.compile(
    r"</?(?:freethink|free-think|think)>|<\|(?:user|assistant|system|endoftext|pad)\|>|<esg>|</?tool_call>",
    re.IGNORECASE,
)


def parse_internal_command(prompt: str, default_mode: str | None = None) -> tuple[str, str | None]:
    """Extract and hide internal slash commands (/think, /freethink, /nothink, /chat, /base)
    and raw <think>/<freethink> tags from `prompt`, returning `(cleaned_prompt, resolved_mode)`.
    """
    if not prompt:
        return "", default_mode

    detected_cmd: str | None = None
    lower = prompt.lower()
    if any(tok in lower for tok in ("/freethink", "/free-think", "<freethink>", "</freethink>")):
        detected_cmd = "freethink"
    elif any(tok in lower for tok in ("/nothink", "/no-think")):
        detected_cmd = "nothink"
    elif any(tok in lower for tok in ("/think", "<think>", "</think>")):
        detected_cmd = "think"
    elif "/chat" in lower:
        detected_cmd = "chat"
    elif "/base" in lower:
        detected_cmd = "base"

    cleaned = _TAG_RE.sub(" ", prompt)
    cleaned = _CMD_RE.sub(" ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    if detected_cmd is None:
        return cleaned, default_mode

    if default_mode in ("think", "chat", "freethink", "base"):
        if detected_cmd == "freethink":
            return cleaned, "freethink"
        if detected_cmd == "think":
            return cleaned, "think"
        if detected_cmd in ("nothink", "chat"):
            return cleaned, "chat"
        if detected_cmd == "base":
            return cleaned, "base"

    # FreeThinkSession mode space ('continuation' | 'structured')
    if detected_cmd == "think":
        return cleaned, "structured"
    if detected_cmd == "freethink":
        return cleaned, ("structured" if default_mode in (None, "structured") else "continuation")
    if detected_cmd in ("nothink", "chat", "base"):
        return cleaned, "continuation"
    return cleaned, default_mode


LEGACY_ARTIFACT_PHRASES: tuple[str, ...] = (
    "Examining the key principles behind this statement,",
    "Examining the key principles behind this statement.",
    "Examining the key principles behind this statement:",
    "Examining the key principles behind this statement",
    "ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ,",
    "ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ.",
    "ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ",
    "இதன் முக்கியக் கூறுகளை ஆராய்ந்தால்,",
    "இதன் முக்கியக் கூறுகளை ஆராய்ந்தால்.",
    "இதன் முக்கியக் கூறுகளை ஆராய்ந்தால்",
    "దీనిలోని ముఖ్య అంశాలను విశ్లేషిస్తే,",
    "దీనిలోని ముఖ్య అంశాలను విశ్లేషిస్తే.",
    "దీనిలోని ముఖ్య అంశాలను విశ్లేషిస్తే",
    "ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ,",
    "ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ.",
    "ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ",
    "ಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ,",
    "ಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ.",
    "ಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ",
    "ଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ,",
    "ଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ.",
    "ଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ",
    "ଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ,",
    "ଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ.",
    "ଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ",
    "ఇదా ఎన్నె అని మరి,",
    "ఇదా ఎన్నె అని మరి.",
    "ఇదా ఎన్నె అని మరి",
)


def strip_internal_tags(text: str) -> str:
    """Remove any internal <think>, </think>, <freethink>, </freethink>, or /think commands from visible text,
    and strip legacy synthetic artifact phrases if they appear at the start."""
    if not text:
        return ""
    out = _TAG_RE.sub("", text)
    out = _CMD_RE.sub("", out)
    stripped = out.lstrip(" \t\r\n")
    for phrase in LEGACY_ARTIFACT_PHRASES:
        if stripped.startswith(phrase):
            return stripped[len(phrase):].lstrip(" \t\r\n,.:;")
    return out


# Tunable without touching code below. The model is byte-level and trained on
# plain Wikipedia/Gutenberg prose, so this is a text-continuation cue, not a
# system prompt it was instruction-tuned to obey.
PRIMING = "{prompt}\n\nThinking about this: "

# Per-language native-script continuation cues for BPE checkpoints so non-English
# Free Think prompts stay 100% in their target script rather than switching to Latin.
# IMPORTANT: Do NOT end BPE priming strings with a trailing space ' ' -- in ByteLevel
# BPE (v3_32k), a trailing space encodes as an isolated 'Ġ' token, whereas real words
# carry their own leading 'Ġ' (so an isolated 'Ġ' forces the next token to be a digit).
LANG_PRIMING = {
    "eng": "{prompt}\n\nThinking about this further,",
    "kan": "{prompt}\n\nಇದರ ಬಗ್ಗೆ ಮತ್ತಷ್ಟು ವಿವರಿಸುವುದಾದರೆ,",
    "tam": "{prompt}\n\nஇதைப் பற்றி மேலும் விரிவாகக் கூறினால்,",
    "tel": "{prompt}\n\nదీని గురించి మరింత వివరంగా చెప్పాలంటే,",
    "mal": "{prompt}\n\nഇതിനെക്കുറിച്ച് കൂടുതൽ വിശദമായി പറഞ്ഞാൽ,",
    "tcy": "{prompt}\n\nಇಂದೆತ ಬಗ್ಗೆ ದುಂಬು ಪನ್ಪಿನಂಚಿನವು ಪಂಡ",
    "kff": "{prompt}\n\nଇଦିନ୍ ପେର୍କେ ଆରି କେତୁତାନ୍ ମେଣ୍ଡେ",
    "kxv": "{prompt}\n\nଏ଼ଦାଆଁତାକି ଅ଼ଡ଼େ ମେହ୍‌ଦୁ",
    "nit": "{prompt}\n\nఇదా ఎన్నె అని మరి",
}

# Canned analytical anchors are PURGED from runtime generation:
# Parentheses 0.9-120M must never have synthetic prefix anchors prefilled into its prompt.
STRUCTURED_THINK_ANCHOR: dict[str, str] = {}

# Cheap heuristic rather than the "small classifier head" the design sketch
# offered as an alternative -- Free Think only has to catch the obvious
# question shapes, and a classifier head would need its own labelled data and
# training.
_QUESTION_STARTS = (
    "who", "what", "when", "where", "why", "how", "which", "whose", "whom",
    "is", "are", "was", "were", "am", "do", "does", "did", "can", "could",
    "will", "would", "should", "shall", "may", "might", "has", "have", "had",
    # Dravidian interrogative starters (kan, tam, tel, mal)
    "ಏಕೆ", "ಹೇಗೆ", "ಏನು", "ಯಾರು", "ಎಲ್ಲಿ", "ಯಾವ", "ಎಷ್ಟು",
    "ஏன்", "எப்படி", "என்ன", "யார்", "எங்கே", "எந்த", "எவ்வளவு",
    "ఎందుకు", "ఎలా", "ఏమిటి", "ఎవరు", "ఎక్కడ", "ఏది", "ఎంత",
    "എന്തുകൊണ്ട്", "എങ്ങനെ", "എന്ത്", "ആര്", "എവിടെ", "ഏത്", "എത്ര",
    # Low-resource Dravidian interrogatives (tcy, kff, kxv, nit)
    "ದಾಯೆ", "ಎಂಚ", "ದಾದ", "ಏರ್", "ಓಡೆ", "ಏತ್",
    "ବାତା", "ବେନୋ", "ଇଞ୍ଚୋ", "ନାତା",
    "ଇନାକା", "ଇମ୍ବାଣା", "ଇନେସି", "ଇମ୍ବାୟା",
    "తాని", "ఎంద్క్", "బోన్", "ఎంద్కు",
)


def is_question(prompt: str) -> bool:
    """True if `prompt` looks like a question -- Free Think Mode is for statements."""
    clean_text, _ = parse_internal_command(prompt)
    text = clean_text.strip()
    if not text:
        return False
    clean_tail = text.rstrip("\"'”’) \t\r\n")
    if clean_tail.endswith("?") or clean_tail.endswith("\uff1f"):
        return True
    first_word = text.split()[0].strip("\"'“‘(").lower()
    return first_word in _QUESTION_STARTS


def format_primed(prompt: str, lang: str = "eng", mode: str = "continuation") -> str:
    """Build the primed prompt string for `lang` and `mode` ('continuation' or 'structured')."""
    clean, _ = parse_internal_command(prompt, default_mode=mode)
    clean = clean.strip()
    if mode == "structured":
        return clean
    template = LANG_PRIMING.get(lang, PRIMING)
    return template.format(prompt=clean)


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


def load_bpe_tokenizer(ckpt: dict):
    """The checkpoint's HF tokenizer (BPE runs save {path, sha256}), or None for byte models."""
    meta = ckpt.get("tokenizer")
    if not meta:
        return None
    from tokenizers import Tokenizer
    tok_path = Path(meta["path"]) / "tokenizer.json"
    if not tok_path.exists():
        tok_path = Path(__file__).resolve().parent.parent / meta["path"] / "tokenizer.json"
    tok = Tokenizer.from_file(str(tok_path))
    tok.encode_special_tokens = True  # tags typed in the prompt stay text; specials go in by id
    return tok


def _stream_bpe(model: Parentheses, tok, primed: str, lang: str, max_tokens: int | None,
                temperature: float, top_k: int, timings: list[float] | None = None,
                mode: str = "continuation", min_tokens: int = 0):
    """BPE twin of _stream_text: <lang> + primed text in (or <|user|><lang> prompt<|assistant|><think>
    + native analytical cue when mode='structured'), decoded text out. All internal <think>, </think>,
    <freethink>, and </freethink> tags are kept strictly internal and never emitted into visible text."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from eval_queries import STOP_TOKENS, looping

    device = next(model.parameters()).device
    tag = tok.token_to_id(f"<{lang}>")
    if tag is None:
        raise ValueError(f"tokenizer has no <{lang}> tag")

    clean_primed = strip_internal_tags(primed).strip()
    if mode == "structured":
        user_id = tok.token_to_id("<|user|>")
        asst_id = tok.token_to_id("<|assistant|>")
        think_id = tok.token_to_id("<think>")
        body_ids = tok.encode(clean_primed, add_special_tokens=False).ids
        nl_ids = tok.encode("\n", add_special_tokens=False).ids
        ids = [user_id, tag] + body_ids + [asst_id, think_id] + nl_ids
    else:
        ids = [tag] + tok.encode(clean_primed.rstrip(" "), add_special_tokens=False).ids

    # Never allow role/control tags to leak into visible FreeThink prose.
    # Note: <think> and </think> are handled internally below (not as stop tokens!)
    # so structured mode streams both the <think> reasoning trace and the post-</think>
    # structured answer without cutting off at </think>.
    extra_stops = ("<|assistant|>", "<|pad|>", "<esg>", "<tool_call>", "</tool_call>")
    stops = {tok.token_to_id(s) for s in STOP_TOKENS + extra_stops} - {None}
    think_open_id = tok.token_to_id("<think>")
    think_close_id = tok.token_to_id("</think>")
    para_ids = tok.encode("\n\n", add_special_tokens=False).ids

    out, pending = [], []
    restarts = 0
    max_restarts = 3
    first_chunk_emitted = False
    last_ended_nl = False
    just_closed_think = False
    filter_artifact = True
    strip_leading_punct = False
    artifact_buf = ""

    def _prepare_chunk(text: str) -> str:
        nonlocal first_chunk_emitted, last_ended_nl, just_closed_think, filter_artifact, strip_leading_punct, artifact_buf
        clean = strip_internal_tags(text)
        if not clean:
            return ""
        if filter_artifact:
            artifact_buf += clean
            s = artifact_buf.lstrip(" \t\r\n")
            if any(p.startswith(s) and len(p) > len(s) for p in LEGACY_ARTIFACT_PHRASES):
                return ""
            matched_phrase = False
            for p in LEGACY_ARTIFACT_PHRASES:
                if s.startswith(p):
                    artifact_buf = s[len(p):].lstrip(" \t\r\n,.:;")
                    matched_phrase = True
                    break
            if matched_phrase:
                filter_artifact = False
                strip_leading_punct = True
                clean = artifact_buf
                artifact_buf = ""
            elif any(p.startswith(s) for p in LEGACY_ARTIFACT_PHRASES):
                return ""
            else:
                filter_artifact = False
                clean = artifact_buf
                artifact_buf = ""

        if strip_leading_punct and clean:
            clean = clean.lstrip(" \t\r\n,.:;")
            if clean:
                strip_leading_punct = False

        if not clean:
            return ""
        if not first_chunk_emitted:
            if not clean.strip():
                return ""
            clean = clean.lstrip()
            if clean and "a" <= clean[0] <= "z":
                clean = clean[0].upper() + clean[1:]
            first_chunk_emitted = True
        elif just_closed_think:
            if not clean.strip("\n\r "):
                return ""
            clean = clean.lstrip("\n\r ")
            if clean and "a" <= clean[0] <= "z":
                clean = clean[0].upper() + clean[1:]
            just_closed_think = False
        last_ended_nl = clean.endswith("\n")
        return clean

    while max_tokens is None or len(out) < max_tokens:
        rem = None if max_tokens is None else (max_tokens - len(out))
        cur_ids = ids + out
        idx = torch.tensor([cur_ids[-model.cfg.block_size:]], dtype=torch.long, device=device)
        sink_tokens = min(idx.size(1), model.cfg.block_size // 4)
        stream = model.stream(
            idx,
            rem,
            temperature,
            top_k,
            min_p=0.06,
            sink_tokens=sink_tokens,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
        )
        hit_early_stop = False
        while True:
            started = time.perf_counter()
            try:
                t = int(next(stream))
            except StopIteration:
                break
            if timings is not None:
                timings.append(time.perf_counter() - started)
            if t == think_open_id:
                out.append(t)
                continue
            if t == think_close_id:
                out.append(t)
                if pending:
                    flushed = _prepare_chunk(tok.decode(pending, skip_special_tokens=False))
                    pending = []
                    if flushed:
                        yield flushed
                if first_chunk_emitted and not last_ended_nl:
                    yield "\n\n"
                    last_ended_nl = True
                just_closed_think = True
                continue
            if t in stops:
                if len(out) < min_tokens and restarts < max_restarts and para_ids:
                    restarts += 1
                    hit_early_stop = True
                    out.extend(para_ids)
                    pending.extend(para_ids)
                break
            out.append(t)
            pending.append(t)
            # byte-level BPE can split a character across tokens: hold them until it decodes whole
            chunk = tok.decode(pending, skip_special_tokens=False)
            if chunk and not chunk.endswith("\ufffd"):
                pending = []
                clean_chunk = _prepare_chunk(chunk)
                if clean_chunk:
                    yield clean_chunk
            if looping(out):
                hit_early_stop = False
                break
        if not hit_early_stop:
            break
    if filter_artifact and artifact_buf:
        s = artifact_buf.lstrip(" \t\r\n")
        for p in LEGACY_ARTIFACT_PHRASES:
            if s.startswith(p):
                artifact_buf = s[len(p):].lstrip(" \t\r\n,.:;")
                break
        if artifact_buf:
            clean_chunk = _prepare_chunk(artifact_buf)
            if clean_chunk:
                yield clean_chunk
        artifact_buf = ""
    if pending:  # stream ended mid-character: emit what's left rather than drop it
        clean_chunk = _prepare_chunk(tok.decode(pending, skip_special_tokens=False))
        if clean_chunk:
            yield clean_chunk
        pending = []


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
    stream = model.stream(idx, max_tokens, temperature, top_k, sink_tokens=sink_tokens)
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
            clean_chunk = strip_internal_tags(chunk)
            if clean_chunk:
                yield clean_chunk


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
    """A live, exportable stream of open-ended "thinking" about a statement.

    All `/think`, `/freethink`, `<think>`, `</think>`, `<freethink>`, and `</freethink>`
    commands and tags are handled strictly internally (`self.internal_open_tag = "<freethink>"`,
    `self.internal_close_tag = "</freethink>"`, `self.closed_freethink = True`) and never leaked
    into the user-visible output stream.
    """

    def __init__(self, model: Parentheses, temperature: float = 0.75, top_k: int = 35,
                 tokenizer=None, lang: str = "eng", mode: str = "continuation",
                 min_tokens: int = 0):
        if model.cfg.vocab_size > 256 and tokenizer is None:
            raise ValueError(
                f"vocab_size={model.cfg.vocab_size} is a BPE model: pass its tokenizer "
                "(load_bpe_tokenizer(ckpt)); without one output is decoded as raw UTF-8 bytes."
            )
        if mode not in ("continuation", "structured"):
            raise ValueError(f"mode must be 'continuation' or 'structured', got {mode!r}")
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.tokenizer = tokenizer
        self.lang = lang
        self.mode = mode
        self.min_tokens = min_tokens
        self.prompt: str | None = None
        self.history: list[str] = []
        self.internal_open_tag: str = "<freethink>"
        self.internal_close_tag: str = "</freethink>"
        self.closed_freethink: bool = False
        self.internal_transcript: str = ""

    @property
    def closed_think(self) -> bool:
        """Backward-compatible alias for `closed_freethink`."""
        return self.closed_freethink

    def start(self, prompt: str, max_tokens: int | None = None, force: bool = False,
              timings: list[float] | None = None, mode: str | None = None,
              min_tokens: int | None = None):
        """Yield decoded text as the model thinks, appending it to self.history.

        Automatically hides and strips any leading/inline `/think` or `/freethink`
        commands and closes the internal state with `</freethink>` without emitting
        raw tag strings to the caller.
        """
        clean_prompt, parsed_mode = parse_internal_command(prompt, default_mode=mode or self.mode)
        if not clean_prompt and prompt.strip():
            clean_prompt = prompt.strip()
        _reject_questions(clean_prompt, force)
        self.prompt = clean_prompt
        self.history = []
        self.closed_freethink = False
        self.internal_transcript = ""
        active_mode = parsed_mode or mode or self.mode
        active_min = self.min_tokens if min_tokens is None else min_tokens
        if self.tokenizer is not None:
            primed = format_primed(clean_prompt, self.lang, active_mode)
            chunks = _stream_bpe(self.model, self.tokenizer, primed, self.lang, max_tokens,
                                 self.temperature, self.top_k, timings,
                                 mode=active_mode, min_tokens=active_min)
        else:
            primed = PRIMING.format(prompt=clean_prompt.strip())
            chunks = _stream_text(self.model, primed, max_tokens, self.temperature, self.top_k, timings)
        try:
            for chunk in chunks:
                clean_chunk = strip_internal_tags(chunk)
                if clean_chunk:
                    self.history.append(clean_chunk)
                    yield clean_chunk
        finally:
            self.closed_freethink = True
            body = "".join(self.history).strip()
            self.internal_transcript = f"{self.internal_open_tag}\n{body}\n{self.internal_close_tag}"

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
        text = strip_internal_tags("".join(self.history))
        with open(target, "w", encoding="utf-8") as f:
            if target.suffix == ".json":
                json.dump(
                    {
                        "prompt": self.prompt,
                        "thinking": text,
                        "internal_tag": self.internal_close_tag,
                        "closed_freethink": self.closed_freethink,
                    },
                    f,
                    indent=2,
                )
            else:
                f.write(f"{self.prompt}\n\n{text}\n")
        return target


def main():
    p = argparse.ArgumentParser(description="Free Think Mode: open-ended thinking about a statement.")
    p.add_argument("--checkpoint", default="checkpoints/parentheses-0.9-120M/step_9536_final.pt")
    p.add_argument("--prompt", default="The ocean is very deep.")
    p.add_argument("--max-tokens", type=int, default=400, help="0 = think until Ctrl-C")
    p.add_argument("--min-tokens", type=int, default=64, help="minimum thinking tokens before allowing stop")
    p.add_argument("--mode", choices=["continuation", "structured"], default="continuation",
                   help="'continuation' (<lang> + native priming) or 'structured' (<|user|><lang>...<|assistant|><think>)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--export", default=None,
                   help=f"filename to write inside {EXPORT_DIR} (.json, .txt, .md -- "
                        "can't escape that directory)")
    p.add_argument("--timing", default=None,
                   help=f"filename to write per-token latencies to inside {EXPORT_DIR} "
                        "(token_index,ms CSV); also prints a first-vs-last summary")
    p.add_argument("--force", action="store_true", help="free-think even if the prompt looks like a question")
    p.add_argument("--lang", default="eng", help="language tag for BPE checkpoints (kan, tam, tel, mal, ...)")
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

    session = FreeThinkSession(model, args.temperature, args.top_k,
                               tokenizer=load_bpe_tokenizer(ckpt), lang=args.lang,
                               mode=args.mode, min_tokens=args.min_tokens)
    print(f"[free-think] {args.checkpoint} (step {ckpt['step']}, mode={args.mode}, lang={args.lang}) on {args.device}")
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
    assert is_question("ಏಕೆ ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ")
    assert not is_question("The sky is blue.")
    assert not is_question("ಬೆಂಗಳೂರು ಒಂದು ನಗರ.")
    assert not is_question("")

    # Native per-language priming checks
    assert "Thinking about this" in format_primed("The sky is blue.", "eng", "continuation")
    assert "ಇದರ ಬಗ್ಗೆ" in format_primed("ಬೆಂಗಳೂರು ಒಂದು ನಗರ.", "kan", "continuation")
    assert format_primed("ಬೆಂಗಳೂರು ಒಂದು ನಗರ.", "kan", "structured") == "ಬೆಂಗಳೂರು ಒಂದು ನಗರ."

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

    # BPE path: untrained tiny model with the real v3_32k vocab (tokenizer file only, no data)
    from model.config import ModelConfig
    tok_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer" / "v3_32k"
    tok = load_bpe_tokenizer({"tokenizer": {"path": str(tok_dir)}})
    bpe = Parentheses(ModelConfig(vocab_size=32768, block_size=256, n_layer=2, n_head=2, n_embd=32,
                                  attn_type="selective_linear")).eval()
    try:
        FreeThinkSession(bpe)
        raise AssertionError("a BPE model without a tokenizer should be rejected")
    except ValueError:
        pass
    s = FreeThinkSession(bpe, tokenizer=tok, lang="kan")
    timings = []
    text = s.run("ಬೆಂಗಳೂರು ಒಂದು ನಗರ.", max_tokens=30, timings=timings)
    assert text == "".join(s.history) and "\ufffd" not in text[-1:], repr(text[-5:])
    assert 1 <= len(timings) <= 30, len(timings)

    # Structured FreeThink mode on BPE (internal </freethink> closure + hidden commands)
    s_struct = FreeThinkSession(bpe, tokenizer=tok, lang="kan", mode="structured", min_tokens=5)
    t_struct = s_struct.run("/think ಬೆಂಗಳೂರು ಒಂದು ನಗರ.", max_tokens=15)
    assert t_struct == "".join(s_struct.history) and len(t_struct) > 0
    assert s_struct.prompt == "ಬೆಂಗಳೂರು ಒಂದು ನಗರ.", s_struct.prompt
    assert s_struct.closed_freethink and s_struct.internal_close_tag == "</freethink>"
    assert s_struct.internal_transcript.endswith("</freethink>")
    assert all(tag not in t_struct for tag in INTERNAL_TAGS + INTERNAL_COMMANDS), repr(t_struct)
    print("[self-test] free_think ok")


if __name__ == "__main__":
    main()
