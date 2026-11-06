"""
S5 textbook corpus near-copy audit (Claude Code, 2026-11-03): how many docs of each eval set have a
near-copy in data/processed_s5_textbooks_240m/train.bin. Eval sets: the S5 textbook val split, the
processed_v5 val split (val_v5 gate), the S4 STEM val split, and the 60 probe prompts.

Same machinery and thresholds as scripts/val_neardup_s4.py (out/audit_v5_data.py: 32-token span hashes,
non-overlapping eval spans, every train window; tiers count docs with >= 4 spans). One pass over train.
CPU only, no torch.
    .venv\\Scripts\\python.exe scripts\\neardup_s5_textbooks.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "out"))
sys.path.insert(0, str(REPO))
from audit_v5_data import docs, near_dup_hits, val_span_index  # noqa: E402

TIERS = (0.1, 0.25, 0.5, 0.9)
VAL_SETS = {
    "s5_textbook_val": "data/processed_s5_textbooks_240m",
    "v5_val": "data/processed_v5",
    "s4_stem_val": "data/processed_s4_stem_240m",
}


def probe_docs(tok, eng_id: int) -> list[np.ndarray]:
    """The 60 probe prompts, tokenized with a leading tag slot (val_span_index skips token 0)."""
    from scripts.eval_60p_adversarial_suite import PROBES
    return [np.asarray([eng_id, *tok.encode(p["prompt"], add_special_tokens=False).ids], dtype=np.uint16)
            for p in PROBES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", default=str(REPO / "data/processed_s5_textbooks_240m"))
    ap.add_argument("--tokenizer", default=str(REPO / "data/tokenizer/v3_32k/tokenizer.json"))
    ap.add_argument("--json", default=str(REPO / "out/neardup_s5_textbooks.json"))
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)

    sets: dict[str, list[np.ndarray]] = {}
    for name, d in VAL_SETS.items():
        arr = np.fromfile(REPO / d / "val.bin", dtype=np.uint16)
        bnd = np.load(REPO / d / "val_boundaries.npy").astype(np.int64)
        sets[name] = [doc for _, doc in docs(arr, bnd)]
    sets["probes_60"] = probe_docs(tok, tok.token_to_id("<eng>"))

    all_docs, owner_set = [], []
    for name, ds in sets.items():
        all_docs += ds
        owner_set += [name] * len(ds)
    vh, vo = val_span_index(all_docs)
    n_spans = np.bincount(vo, minlength=len(all_docs))
    train = np.fromfile(Path(a.train_dir) / "train.bin", dtype=np.uint16)
    print(f"{len(all_docs):,} eval docs, {len(vh):,} spans; train {len(train):,} tok", flush=True)
    hits = near_dup_hits(train, vh, vo, chunk=20_000_000)

    out = dict(train_dir=a.train_dir, train_tokens=int(len(train)), span_len=32, sets={})
    owner_set = np.asarray(owner_set)
    frac = np.array([hits.get(k, 0) / max(1, n_spans[k]) for k in range(len(all_docs))])
    for name in sets:
        m = owner_set == name
        elig = m & (n_spans >= 4)
        out["sets"][name] = dict(
            docs=int(m.sum()), ge4_span_docs=int(elig.sum()),
            any_shared_span=int(sum(1 for k in np.nonzero(m)[0] if hits.get(k, 0) > 0)),
            tiers={f"{t:.2f}": int((elig & (frac >= t)).sum()) for t in TIERS})
        if name == "probes_60":  # short prompts: also report any probe with a shared span, by index
            out["sets"][name]["probe_idx_with_shared_span"] = [int(k - np.nonzero(m)[0][0])
                                                               for k in np.nonzero(m)[0] if hits.get(k, 0) > 0]
    Path(a.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["sets"], indent=2))


if __name__ == "__main__":
    main()
