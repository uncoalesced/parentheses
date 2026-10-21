"""Lexical retrieval over the distilled knowledge corpus.

Builds a BM25 index over the parsed Nemotron-v2 facts (knowledge QA) and prose
(self-contained article sentences). For a factual question, returns the single
best-matching fact as a clean context line for grounded answering.

The index is built lazily on first use and cached for the process lifetime.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARSED = ROOT / "out" / "distill_v2" / "parsed"

STOP = set(
    "what is are the a an of in on for to and or which who whom whose when where why how "
    "does do did was were be been being with by as at from that this it its you your "
    "please answer only one name word briefly question tell me about "
    "has have had many current latest recent reliable source sources confirm confirms "
    "confirmed verify verified information documentation explain explanation means mean "
    "provide provides give using use government website link links "
    "and or but if then than so such very can could you find check".split()
)

_rows: list[dict] | None = None
_docs: list[list[str]] | None = None
_bm25 = None
_idf: dict[str, float] | None = None


def _toks(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def _load() -> None:
    global _rows, _docs, _bm25, _idf
    if _rows is not None:
        return
    from rank_bm25 import BM25Okapi

    rows: list[dict] = []
    for name in ("facts", "prose"):
        p = PARSED / f"{name}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                q = (r.get("q") or "").strip()
                a = (r.get("a") or "").strip()
                if q and a and len(a) <= 400:
                    rows.append({"q": q, "a": a, "src": name})
    docs = [_toks(r["q"] + " " + r["a"]) for r in rows]
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    _idf = {w: math.log((n - df[w] + 0.5) / (df[w] + 0.5) + 1) for w in df}
    _bm25 = BM25Okapi(docs)
    _rows, _docs = rows, docs


def _context(row: dict) -> str:
    q, a = row["q"].strip(), row["a"].strip()
    m = re.match(r"tell me (?:a fact )?about (.+?)\.?$", q, re.I)
    if m:
        topic = m.group(1).strip().rstrip(".")
        return f"{topic}: {a}"
    if len(a.split()) >= 4:
        return a
    return f"{q} {a}"


def _first_phrase_bonus(prompt: str, row: dict, idf: dict[str, float]) -> float:
    raw = re.findall(r"[a-z0-9]+", prompt.lower())
    words = set(_toks(prompt))
    pairs = []
    for x, y in zip(raw, raw[1:]):
        if x in words and y in words and x != y and (x, y) not in pairs:
            pairs.append((x, y))
    if not pairs:
        return 0.0
    a, b = pairs[0]
    text = (row["q"] + " " + row["a"]).lower()
    pat = r"(?<!second-)\b" + re.escape(a) + r"\s+" + re.escape(b) + r"\b"
    if re.search(pat, text):
        return 2.5 * (idf.get(a, 0.0) + idf.get(b, 0.0))
    return 0.0


_NOUN_STOP = {
    "what", "who", "whom", "whose", "which", "when", "where", "why", "how",
    "is", "are", "was", "were", "can", "could", "do", "does", "did",
    "explain", "write", "return", "answer", "list", "give", "reply", "find",
    "verify", "use", "tell", "the", "a", "an", "and", "or", "of", "in", "on",
    "for", "to", "with", "by", "at", "from", "this", "that", "it", "its",
}


def _proper_nouns(prompt: str) -> list[str]:
    found = []
    groups = re.findall(r"\b([A-Z][A-Za-z0-9-]*)\b", prompt)
    for i, g in enumerate(groups):
        w = g.lower()
        if i == 0 or w in _NOUN_STOP:
            continue
        if w not in found:
            found.append(w)
    return found


def _years(prompt: str) -> list[str]:
    return re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", prompt)


def _clean_query(prompt: str) -> str:
    q = re.sub(
        r"\b(?:answer|reply|respond)(?:\s+(?:this|the|next))?(?:\s+question)?"
        r"(?:\s+(?:using|with|in))?(?:\s+exactly)?(?:\s+[a-z0-9]+)?\s+words?\b",
        " ", prompt, flags=re.I)
    q = re.sub(r"\bin\s+lower\s*case(?:\s+only)?\b", " ", q, flags=re.I)
    return q.strip()


def _entity(prompt: str) -> str:
    raw = re.findall(r"[a-z0-9]+", prompt.lower())
    if "of" not in raw:
        return ""
    words = set(_toks(prompt))
    for w in raw[raw.index("of") + 1:]:
        if w in words:
            return w
    return ""


def _head_term_bonus(prompt: str, row: dict, idf: dict[str, float]) -> float:
    raw = re.findall(r"[a-z0-9]+", prompt.lower())
    words = set(_toks(prompt))
    if "of" not in raw:
        return 0.0
    idx = raw.index("of")
    head = ""
    for w in reversed(raw[:idx]):
        if w in words:
            head = w
            break
    if not head:
        return 0.0
    text = (row["q"] + " " + row["a"]).lower()
    if re.search(r"\b" + re.escape(head) + r"\b", text):
        return 6.0 * idf.get(head, 0.0)
    return 0.0


def context_for(prompt: str, min_cov: float = 0.5, min_score: float = 2.5) -> str | None:
    """Return the best knowledge context for a factual prompt, or None if unsure."""
    _load()
    assert _rows is not None and _docs is not None and _bm25 is not None and _idf is not None
    qp = _clean_query(prompt)
    qs = set(_toks(qp))
    if len(qs) < 2:
        return None
    scores = _bm25.get_scores(list(qs))
    cand = scores.argsort()[-30:][::-1]
    rarest = sorted(qs, key=lambda w: -_idf.get(w, 0.0))[:2]
    entity = _entity(qp)
    proper = _proper_nouns(prompt)
    years = _years(prompt)
    best_s, best_i = -1e9, -1
    for i in cand:
        d = set(_docs[int(i)])
        if entity and entity not in d:
            continue
        if proper and any(w not in d for w in proper):
            continue
        if years:
            text = (_rows[int(i)]["q"] + " " + _rows[int(i)]["a"]).lower()
            if any(y not in text for y in years):
                continue
        ov = sum(_idf.get(w, 0.0) for w in qs & d)
        cov = len(qs & d) / len(qs)
        s = float(scores[int(i)]) + 4 * ov + 8 * cov
        if not all(w in d for w in rarest):
            s -= 15
        s += _first_phrase_bonus(qp, _rows[int(i)], _idf)
        s += _head_term_bonus(qp, _rows[int(i)], _idf)
        if s > best_s:
            best_s, best_i = s, int(i)
    if best_i < 0:
        return None
    cov = len(qs & set(_docs[best_i])) / len(qs)
    if cov < min_cov or float(scores[best_i]) < min_score:
        return None
    return _context(_rows[best_i])


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        assert context_for("What is the capital of Australia?") is not None
        ctx = context_for("What is the capital of Australia?")
        assert ctx is not None and "canberra" in ctx.lower(), ctx
        ctx = context_for("Which planet is closest to the Sun?")
        assert ctx is not None and "mercury" in ctx.lower(), ctx
        assert context_for("zzzz qqqq") is None
        print("[self-test] knowledge_rag ok")
