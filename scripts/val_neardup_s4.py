"""
S4 STEM val near-copy audit (Claude Code, 2026-11-03): how much of data/processed_s4_stem_240m/val.bin
has a near-copy in its train.bin. Reuses the v4/v5 audit's machinery unchanged (out/audit_v5_data.py:
32-token span hashing, non-overlapping val spans, every train window) and leak_probe.py's thresholds
(fraction of a val doc's spans found in train; tiers only count docs with >= 4 spans). leak_probe.py
itself is hardcoded to processed_v4/val, hence this driver.

Source per doc is not stored in the .bin files; it is recovered by replaying build_s4_stem_240m.py's
assemble() shuffle (default_rng(4001) over the per-source kept counts in manifest.json), then checked
against document content (<think>, "The answer is:", \\boxed) before being reported.

CPU only, single pass over train.bin, no torch.
    .venv\\Scripts\\python.exe scripts\\val_neardup_s4.py
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "out"))
from audit_v5_data import W, docs, span_hashes, val_span_index  # noqa: E402

TIERS = (0.1, 0.25, 0.5, 0.9)
HEADLINE = 0.5  # the tier the v4/v5 audit quoted (135 / 118 val docs)


def replay_sources(manifest: dict, seed: int, val_frac: float):
    """-> (names, val_src, train_src): source index of each doc in val.bin / train.bin order."""
    names = list(manifest["sources"])
    kept = [manifest["sources"][n]["kept"] for n in names]
    rng = np.random.default_rng(seed)
    d = np.concatenate([np.stack([np.full(k, i), np.arange(k)], 1) for i, k in enumerate(kept)])
    rng.shuffle(d)
    n_val = max(1, int(len(d) * val_frac))
    return names, d[:n_val, 0], d[n_val:, 0]


def content_label(text: str) -> str:
    """Source guess from the body shapes build_s4_stem_240m.body_for() writes."""
    if "</think>" not in text:
        return "orca_math_or_unboxed_numina"
    tail = text.rsplit("</think>", 1)[1].replace("<|endoftext|>", "").strip()
    if tail.startswith("The answer is"):
        return "metamathqa"
    if tail.startswith("\\boxed{") and tail.endswith("}") and "\n" not in tail:
        return "numinamath_cot"
    return "bespoke_stratos"


def scan(train: np.ndarray, vh: np.ndarray, chunk: int):
    """near_dup_hits() from audit_v5_data, plus the first train position of each matched span."""
    first_pos = {}
    for s in range(0, len(train), chunk):
        h = span_hashes(train[s:s + chunk + W - 1])
        i = np.searchsorted(vh, h)
        i[i == len(vh)] = 0
        where = np.nonzero(vh[i] == h)[0]
        if len(where) == 0:
            continue
        u, idx = np.unique(h[where], return_index=True)
        for v, p in zip(u.tolist(), (where[idx] + s).tolist()):
            first_pos.setdefault(v, p)
    return first_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(REPO / "data/processed_s4_stem_240m"))
    ap.add_argument("--tokenizer", default=str(REPO / "data/tokenizer/v3_32k/tokenizer.json"))
    ap.add_argument("--seed", type=int, default=4001)
    ap.add_argument("--val-frac", type=float, default=0.005)
    ap.add_argument("--chunk", type=int, default=20_000_000)
    ap.add_argument("--json", default=str(REPO / "out/eval_s4/val_neardup_s4.json"))
    a = ap.parse_args()
    d = Path(a.dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)

    varr = np.fromfile(d / "val.bin", dtype=np.uint16)
    vbnd = np.load(d / "val_boundaries.npy")
    val_docs = [doc for _, doc in docs(varr, vbnd)]
    train = np.fromfile(d / "train.bin", dtype=np.uint16)
    tbnd = np.load(d / "train_boundaries.npy").astype(np.int64)
    print(f"val {len(val_docs):,} docs / {len(varr):,} tok; train {len(tbnd):,} docs / {len(train):,} tok", flush=True)

    names, val_src, train_src = replay_sources(manifest, a.seed, a.val_frac)
    assert len(val_src) == len(val_docs) and len(train_src) == len(tbnd), "replayed split sizes differ"
    texts = [tok.decode(doc.tolist(), skip_special_tokens=False) for doc in val_docs]
    labels = [content_label(t) for t in texts]
    agree = Counter()
    for k in range(len(val_docs)):
        rep = names[val_src[k]]
        ok = labels[k] == rep or (labels[k] == "orca_math_or_unboxed_numina" and rep in ("orca_math", "numinamath_cot"))
        agree[(rep, labels[k], ok)] += 1
    n_ok = sum(c for (_, _, ok), c in agree.items() if ok)
    print(f"source replay vs content: {n_ok}/{len(val_docs)} agree", flush=True)

    vh, vo = val_span_index(val_docs)
    n_spans = np.bincount(vo, minlength=len(val_docs))
    first_pos = scan(train, vh, a.chunk)
    hits = Counter()
    owner_pos = defaultdict(list)
    for v, p in first_pos.items():  # each distinct val span counts once, as in near_dup_hits()
        lo, hi = np.searchsorted(vh, np.uint64(v)), np.searchsorted(vh, np.uint64(v), side="right")
        for o in set(vo[lo:hi].tolist()):
            hits[o] += 1
            owner_pos[o].append(p)
    frac = np.array([hits.get(k, 0) / max(1, n_spans[k]) for k in range(len(val_docs))])
    doc_tok = np.array([len(x) for x in val_docs])

    out = dict(val_docs=len(val_docs), val_tokens=int(len(varr)), train_docs=int(len(tbnd)),
               train_tokens=int(len(train)), span_len=W, total_val_spans=int(n_spans.sum()),
               matched_val_spans=int(sum(hits.values())), source_replay_agree=n_ok,
               source_replay_detail=[dict(replayed=r, content=c, ok=ok, n=n) for (r, c, ok), n in sorted(agree.items())])
    elig = n_spans >= 4
    out["docs_with_ge4_spans"] = int(elig.sum())
    out["docs_lt4_spans_by_source"] = dict(Counter(names[val_src[k]] for k in range(len(val_docs)) if not elig[k]))
    out["docs_with_any_shared_span"] = len(hits)
    out["g3_rule_docs"] = sum(1 for k, c in hits.items() if c >= min(4, n_spans[k]))
    out["tiers"] = {}
    for lo in TIERS:
        sel = elig & (frac >= lo)
        out["tiers"][f"{lo:.2f}"] = dict(
            docs=int(sel.sum()), pct_of_all_val_docs=100 * sel.sum() / len(val_docs),
            pct_of_ge4_docs=100 * sel.sum() / max(1, elig.sum()),
            val_tokens_in_these_docs=int(doc_tok[sel].sum()), pct_val_tokens=100 * doc_tok[sel].sum() / len(varr),
            by_source=dict(Counter(names[val_src[k]] for k in np.nonzero(sel)[0])))
    out["by_source"] = {}
    for i, n in enumerate(names):
        m = val_src == i
        sel = m & elig & (frac >= HEADLINE)
        out["by_source"][n] = dict(val_docs=int(m.sum()), ge4_span_docs=int((m & elig).sum()),
                                   ge50_docs=int(sel.sum()), pct_ge50_of_ge4=100 * sel.sum() / max(1, (m & elig).sum()),
                                   val_tokens=int(doc_tok[m].sum()), spans=int(n_spans[m].sum()),
                                   matched_spans=int(sum(hits.get(k, 0) for k in np.nonzero(m)[0])))
    # train-side source of each >= 50% doc: majority source of the train docs holding its matched spans
    cross = Counter()
    order = sorted((k for k in range(len(val_docs)) if elig[k]), key=lambda k: (-frac[k], -n_spans[k]))
    for k in order:
        if frac[k] < HEADLINE:
            break
        tdocs = np.searchsorted(tbnd, np.array(owner_pos[k]), side="right") - 1
        tsrc = Counter(names[train_src[t]] for t in tdocs)
        cross[(names[val_src[k]], tsrc.most_common(1)[0][0])] += 1
    worst = []
    for k in order[:5]:
        rec = dict(val_id=f"val#{k}", source=names[val_src[k]], spans=int(n_spans[k]), found=int(hits.get(k, 0)),
                   frac=float(frac[k]), tokens=int(doc_tok[k]), val_snippet=texts[k][:220].replace("\n", " "))
        if owner_pos[k]:
            tdocs = np.searchsorted(tbnd, np.array(owner_pos[k]), side="right") - 1
            t0 = int(Counter(tdocs.tolist()).most_common(1)[0][0])
            rec.update(train_id=f"train#{t0}", train_source=names[train_src[t0]],
                       train_snippet=tok.decode(train[tbnd[t0]:tbnd[t0] + 80].tolist(),
                                                skip_special_tokens=False)[:220].replace("\n", " "))
        worst.append(rec)
    out["ge50_val_source_to_train_source"] = [dict(val=v, train=t, n=c) for (v, t), c in sorted(cross.items())]
    out["worst5"] = worst
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
