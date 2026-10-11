#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates training_data_ledger_2026-10-07.jsonl for F:\Lingual_Data\Trained-On\LEDGER\
covering all files touched by 2026-10-07 runs:
- data/processed_dravidian_v1/* (v1/v2/v3 byte runs)
- data/processed_dravidian_lowres_v1/* (v3; esg flagged research-only)
- data/processed_all_sources_v1/* (replay)
- F:\Lingual_Data\data\processed_v3\text\*.jsonl (stage-1 text)
- ...\pools\v3_32k|v3_64k|v3_48k(partial)
- data/processed_v3/* and its F: original
- the Sangraha parquet shards actually read (kan/tam/tel/mal first N shards)
"""

import os
import json
import hashlib
from pathlib import Path

REPO = Path("E:/Parentheses/parentheses-0.9")
FLAKE = Path("F:/Lingual_Data")
OUT_JSONL = FLAKE / "Trained-On/LEDGER/training_data_ledger_2026-10-07.jsonl"

# Pre-known SHA-256 hashes from existing manifests
DRAV_V1_HASHES = {
    "train.bin": "4111c8c217f484efbf696561813dbad7077fd27dc053a6af86225975818c7304",
    "val.bin": "7ddcd8f275473e5fa30298bd23f1419fb6a77ff9b7dc6005df2900ec9d23b274",
    "train_boundaries.npy": "9d1f13dac959cdae3744b6759c0b04e0fd5205fee5ebd44066c800fe1a9a26ed",
    "val_boundaries.npy": "81a8d599fb13dc962b3d1966fb9d4c528b2a32dedee13621299018f57283cd00",
}

V3_HASHES = {
    "train.bin": "4e9322d7ddaf05cd0da01f7d8fef75ab2b61adb26c5ed2d5e0d4237bd403b9cd",
    "train_boundaries.npy": "b5b3e9404f3c8c6e86d5d818425e1148ee9fa89ae33c5c73594b77ea14b8a0cb",
    "val/eng.bin": "c0cc58242bbe191b958252a1de68f345871535c9e5e948ab4588019e65b7b722",
    "val/eng_boundaries.npy": "7e67675b37c21b30e1d5391bfeb628954c5e3b667e4f4f68b46e11f6dbc9fb41",
    "val/kan.bin": "52187219ac71b91881a3c2a489fba79b2a192657730ed04bed3008f59d7a1143",
    "val/kan_boundaries.npy": "fec423c4f8126be08d2122cf240d40fa21b93c6d9382aa1dc4380538bff0345a",
    "val/kff.bin": "42c78da19cbe79278c65df011787838225267af5785e6230935d98e8469b82ff",
    "val/kff_boundaries.npy": "1fad2697c01f159784f610e55499fa995e089ce5db4e44e8ca899f53b6ee42f4",
    "val/kxv.bin": "255deff65e5938557009b8dec62e2e0bf75c03c57b551e7a290a5ac22eeec7be",
    "val/kxv_boundaries.npy": "519b069133271ef63b6829978d46b428b6ea3b27ba45c33bfe7bcad7794f09d5",
    "val/mal.bin": "bd3c66f8adb52b71a255fe83e3e3e2c8603714bd8452f22302a92e17ce752c51",
    "val/mal_boundaries.npy": "215db6f6de91e5313b2bb5512b43bd90b4e2cae4b1e72337c68cf06ffc16a09a",
    "val/nit.bin": "1ffdd96baa6c251fb0b9a5ae85e094cbecad1ea44c24047a2fab9cd7aaae6b29",
    "val/nit_boundaries.npy": "d8c6d70aba0790de3faac7c9470c3c0d031c1ba8a89c3c84bfb803889e5a4ef9",
    "val/tam.bin": "47635a4e1332677d9f6df03bd136e565b1321d4015aaed989bf5be41b51a2f2c",
    "val/tam_boundaries.npy": "808ed9621bcd58df73d4a8c56bff5e67c1a3d23a59eb0e9e08994b2bd2a035ce",
    "val/tcy.bin": "ddd488def6e57e9369cba363afbef278fea3ddd7e3cc4526c4b2a8c04db59217",
    "val/tcy_boundaries.npy": "58861e9f7c114c157830028f6d7da1324569dbc7bc3577fd1b7e8b334b06bc01",
    "val/tel.bin": "935b0d8df159ca1ea927acac456dbd1a75110c993405fc35519533fa573dd690",
    "val/tel_boundaries.npy": "6546c4518a80db7335ec03dce8f3a2e76448320c3911fe06362ce2fa6138bb7c",
}

# Lowres manifest hashes
with open(REPO / "data/processed_dravidian_lowres_v1/manifest.json", "r", encoding="utf-8") as f:
    lowres_data = json.load(f)

LOWRES_HASHES = {}
for lang, linfo in lowres_data.get("languages", {}).items():
    for ftype, hval in linfo.get("checksums_sha256", {}).items():
        LOWRES_HASHES[f"{lang}/{ftype}"] = hval


def hash_file_or_defer(path: Path, max_hash_bytes=50 * 1024 * 1024) -> tuple[str, int, int]:
    """Return (sha256_or_deferred, bytes, mtime)."""
    sz = path.stat().st_size
    mtime = int(path.stat().st_mtime)
    # Check pre-known hashes
    # If on F: and size > max_hash_bytes, defer
    if sz > max_hash_bytes and str(path).startswith("F:"):
        return "deferred", sz, mtime
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while chunk := fp.read(1024 * 1024 * 8):
            h.update(chunk)
    return h.hexdigest(), sz, mtime


def make_record(path: Path, kind: str, source: str, license_str: str,
                used_by: list, rederive: str, known_hash: str = None, date_str="2026-10-07") -> dict:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    st = path.stat()
    sz = st.st_size
    mtime = int(st.st_mtime)

    if known_hash:
        sha256_val = known_hash
    else:
        sha256_val, sz, mtime = hash_file_or_defer(path)

    norm_path = str(path).replace("\\", "/")
    rec = {
        "path": norm_path,
        "bytes": sz,
        "sha256": sha256_val,
        "kind": kind,
        "source": source,
        "license": license_str,
        "used_by": used_by,
        "rederive": rederive,
        "date": date_str
    }
    if sha256_val == "deferred":
        rec["mtime"] = mtime
    return rec


def build_all_records():
    records = []

    # 1. data/processed_dravidian_v1 (E: and F:)
    for base in [REPO / "data/processed_dravidian_v1", FLAKE / "data/processed_dravidian_v1"]:
        if not base.exists():
            continue
        used_by_train = [
            {"run": "dravidian-v1-selective", "tokens": 1638400000, "epochs": 1.0436},
            {"run": "dravidian-v2-replay", "tokens": 1484800000, "epochs": 0.9457},
            {"run": "dravidian-v3-lowres", "tokens": 783360000, "epochs": 0.4989}
        ]
        used_by_val = [
            {"run": "dravidian-v1-selective", "tokens": 82151244, "epochs": 1.0},
            {"run": "dravidian-v2-replay", "tokens": 82151244, "epochs": 1.0},
            {"run": "dravidian-v3-lowres", "tokens": 82151244, "epochs": 1.0}
        ]
        for f in base.iterdir():
            if f.is_file():
                if f.name == "train.bin":
                    records.append(make_record(f, "train_bin", "AI4Bharat Sangraha + IndicCorp v2 (kan, tam, tel, mal)",
                                               "CC BY 4.0 / CC BY-SA 4.0", used_by_train,
                                               "python scripts/tokenize_corpus.py --preset parentheses-0.9-300k-selective",
                                               DRAV_V1_HASHES["train.bin"]))
                elif f.name == "train_boundaries.npy":
                    records.append(make_record(f, "boundaries", "AI4Bharat Sangraha + IndicCorp v2 doc boundaries",
                                               "CC BY 4.0 / CC BY-SA 4.0", used_by_train,
                                               "python scripts/tokenize_corpus.py --preset parentheses-0.9-300k-selective",
                                               DRAV_V1_HASHES["train_boundaries.npy"]))
                elif f.name == "val.bin":
                    records.append(make_record(f, "val", "AI4Bharat Sangraha + IndicCorp v2 val split (5%)",
                                               "CC BY 4.0 / CC BY-SA 4.0", used_by_val,
                                               "python scripts/tokenize_corpus.py --preset parentheses-0.9-300k-selective",
                                               DRAV_V1_HASHES["val.bin"]))
                elif f.name == "val_boundaries.npy":
                    records.append(make_record(f, "boundaries", "AI4Bharat Sangraha + IndicCorp v2 val doc boundaries",
                                               "CC BY 4.0 / CC BY-SA 4.0", used_by_val,
                                               "python scripts/tokenize_corpus.py --preset parentheses-0.9-300k-selective",
                                               DRAV_V1_HASHES["val_boundaries.npy"]))
                elif f.suffix == ".json":
                    records.append(make_record(f, "manifest", "Parentheses 0.9 Dravidian Tokenization metadata",
                                               "Apache-2.0", [], "python scripts/tokenize_corpus.py"))

    # 2. data/processed_dravidian_lowres_v1 (E: and F:)
    for base in [REPO / "data/processed_dravidian_lowres_v1", FLAKE / "data/processed_dravidian_lowres_v1"]:
        if not base.exists():
            continue
        # manifest.json
        mf = base / "manifest.json"
        if mf.exists():
            records.append(make_record(mf, "manifest", "Parentheses 0.9 low-resource Dravidian metadata",
                                       "Apache-2.0", [], "python scripts/tokenize_lowres.py"))
        # language dirs
        lowres_configs = {
            "tcy": ("Wikimedia: tcy.wikipedia, tcy.wikisource", "CC BY-SA 4.0", [{"run": "dravidian-v3-lowres", "tokens": 46080000, "epochs": 1.672}]),
            "kff": ("eBible (The Word for the World)", "CC BY-SA 4.0", [{"run": "dravidian-v3-lowres", "tokens": 15360000, "epochs": 4.879}]),
            "kxv": ("eBible (The Word for the World)", "CC BY-SA 4.0", [{"run": "dravidian-v3-lowres", "tokens": 15360000, "epochs": 5.553}]),
            "nit": ("eBible (The Word for the World)", "CC BY-SA 4.0", [{"run": "dravidian-v3-lowres", "tokens": 15360000, "epochs": 6.198}]),
            "esg": ("Wycliffe Bible Translators (RESEARCH-ONLY, all rights reserved)", "research-only", [{"run": "dravidian-v3-lowres", "tokens": 15360000, "epochs": 5.459}])
        }
        for lang, (src, lic, used) in lowres_configs.items():
            ldir = base / lang
            if ldir.exists():
                for f in ldir.iterdir():
                    if f.is_file():
                        key = f"{lang}/{f.name}"
                        hval = LOWRES_HASHES.get(key)
                        kind = "train_bin" if f.name == "train.bin" else ("val" if f.name == "val.bin" else "boundaries")
                        rederive_cmd = f"python scripts/tokenize_lowres.py --lang {lang}"
                        records.append(make_record(f, kind, src, lic, used if "train" in f.name else [], rederive_cmd, hval))

    # 3. data/processed_all_sources_v1 (E:)
    allsrc_base = REPO / "data/processed_all_sources_v1"
    if allsrc_base.exists():
        records.append(make_record(
            allsrc_base / "train.bin", "train_bin",
            "Parentheses all-sources mix (Gutenberg, proof-pile-2, hermes, parallel, books)",
            "Mixed Open (see split_manifest.json)",
            [
                {"run": "dravidian-v2-replay", "tokens": 153600000, "epochs": 0.050},
                {"run": "dravidian-v3-lowres", "tokens": 92160000, "epochs": 0.030},
                {"run": "corpus_v3_extract_eng", "tokens": 1073741824, "epochs": 0.350}
            ],
            "python scripts/build_all_sources_dataset.py",
            "921c3cb62906170500349941648cfee949d41cc2f9be302a67e9e8699068087e"
        ))
        records.append(make_record(
            allsrc_base / "val.bin", "val",
            "Parentheses all-sources validation split",
            "Mixed Open (see split_manifest.json)",
            [], "python scripts/build_all_sources_dataset.py",
            "f624e70f2b47304e599cd596415cb3055c1614d4cf69bb2738bb3c161c680da3"
        ))
        records.append(make_record(
            allsrc_base / "split_manifest.json", "manifest",
            "Parentheses all-sources split manifest",
            "Apache-2.0",
            [], "python scripts/build_all_sources_dataset.py",
            "fee71af1d4d2ac6b712e758af17e348d282340b54c5c464f6280af3ec0c6d5f0"
        ))

    # 4. data/processed_v3 (E: and F:)
    v3_used = [{"run": "parentheses-0.9-41M", "tokens": 1059454976, "epochs": 1.394}]
    for v3_dir in [REPO / "data/processed_v3", FLAKE / "data/processed_v3"]:
        if not v3_dir.exists():
            continue
        # train.bin
        records.append(make_record(
            v3_dir / "train.bin", "train_bin",
            "Parentheses Corpus v3 (Sangraha verified Dravidian + English replay + low-res Dravidian)",
            "CC BY 4.0 / CC BY-SA 4.0 / Mixed Open (no esg)",
            v3_used,
            "python F:/Lingual_Data/scripts/build_corpus_v3.py mix --budget 760e6",
            V3_HASHES["train.bin"]
        ))
        records.append(make_record(
            v3_dir / "train_boundaries.npy", "boundaries",
            "Parentheses Corpus v3 doc boundaries (900,161 docs)",
            "CC BY 4.0 / CC BY-SA 4.0 / Mixed Open",
            v3_used,
            "python F:/Lingual_Data/scripts/build_corpus_v3.py mix --budget 760e6",
            V3_HASHES["train_boundaries.npy"]
        ))
        records.append(make_record(
            v3_dir / "manifest.json", "manifest",
            "Parentheses Corpus v3 manifest with checksums and licenses",
            "Apache-2.0",
            [],
            "python F:/Lingual_Data/scripts/build_corpus_v3.py mix --budget 760e6",
            "94d00308a0b8ba71bcc38581a2b59012aabd1893f8d9f6803ec00ac18a65aff4"
        ))
        val_dir = v3_dir / "val"
        if val_dir.exists():
            for vf in val_dir.iterdir():
                if vf.is_file():
                    key = f"val/{vf.name}"
                    hval = V3_HASHES.get(key)
                    kind = "val" if vf.suffix == ".bin" else "boundaries"
                    records.append(make_record(
                        vf, kind,
                        f"Parentheses Corpus v3 held-out validation split ({vf.stem.split('_')[0]})",
                        "CC BY 4.0 / CC BY-SA 4.0 / Mixed Open",
                        [],
                        "python F:/Lingual_Data/scripts/build_corpus_v3.py pool --tokenizer data/tokenizer/v3_32k",
                        hval
                    ))

    # 5. F:\Lingual_Data\data\processed_v3\text\*.jsonl (and stats)
    text_dir = FLAKE / "data/processed_v3/text"
    if text_dir.exists():
        text_licenses = {
            "kan": ("Sangraha verified (ai4bharat, CC BY 4.0) + v1 sources", "CC BY 4.0"),
            "tam": ("Sangraha verified (ai4bharat, CC BY 4.0) + v1 sources", "CC BY 4.0"),
            "tel": ("Sangraha verified (ai4bharat, CC BY 4.0) + v1 sources", "CC BY 4.0"),
            "mal": ("Sangraha verified (ai4bharat, CC BY 4.0) + v1 sources", "CC BY 4.0"),
            "eng": ("processed_all_sources_v1 mix (Gutenberg PD, proof-pile-2, etc.)", "Mixed Open"),
            "tcy": ("Wikimedia: tcy.wikipedia, tcy.wikisource", "CC BY-SA 4.0"),
            "kff": ("eBible (The Word for the World)", "CC BY-SA 4.0"),
            "kxv": ("eBible (The Word for the World)", "CC BY-SA 4.0"),
            "nit": ("eBible (The Word for the World)", "CC BY-SA 4.0"),
        }
        for tf in text_dir.iterdir():
            if tf.is_file():
                lang = tf.stem.split(".")[0]
                src, lic = text_licenses.get(lang, ("Parentheses Corpus v3 source", "Open"))
                if tf.suffix == ".jsonl":
                    rederive_cmd = f"python F:/Lingual_Data/scripts/build_corpus_v3.py extract --lang {lang}"
                    records.append(make_record(tf, "text", src, lic, v3_used, rederive_cmd))
                elif tf.name.endswith(".stats.json"):
                    rederive_cmd = f"python F:/Lingual_Data/scripts/build_corpus_v3.py extract --lang {lang}"
                    records.append(make_record(tf, "manifest", f"QC extract stats for {lang}", lic, [], rederive_cmd))

    # 6. F:\Lingual_Data\data\processed_v3\pools
    pools_dir = FLAKE / "data/processed_v3/pools"
    if pools_dir.exists():
        for pdir in pools_dir.iterdir():
            if pdir.is_dir():
                pool_name = pdir.name
                # pool files
                for pf in pdir.iterdir():
                    if pf.is_file():
                        lang = pf.stem.split("_")[0]
                        src, lic = text_licenses.get(lang, ("Parentheses Corpus v3 pool", "Open"))
                        rederive_cmd = f"python F:/Lingual_Data/scripts/build_corpus_v3.py pool --tokenizer data/tokenizer/{pool_name}"
                        if pf.suffix == ".bin":
                            records.append(make_record(pf, "pool", f"Encoded token pool for {lang} ({pool_name})", lic, v3_used if pool_name == "v3_32k" else [], rederive_cmd))
                        elif pf.suffix == ".npy":
                            records.append(make_record(pf, "boundaries", f"Doc starts for {lang} ({pool_name})", lic, v3_used if pool_name == "v3_32k" else [], rederive_cmd))
                        elif pf.suffix == ".json":
                            records.append(make_record(pf, "manifest", f"Pool metadata ({pool_name})", "Apache-2.0", [], rederive_cmd))
                    elif pf.is_dir() and pf.name == "val":
                        for vf in pf.iterdir():
                            if vf.is_file():
                                lang = vf.stem.split("_")[0]
                                src, lic = text_licenses.get(lang, ("Parentheses Corpus v3 val", "Open"))
                                rederive_cmd = f"python F:/Lingual_Data/scripts/build_corpus_v3.py pool --tokenizer data/tokenizer/{pool_name}"
                                kind = "val" if vf.suffix == ".bin" else "boundaries"
                                records.append(make_record(vf, kind, f"Encoded val for {lang} ({pool_name})", lic, [], rederive_cmd))

    # 7. Sangraha parquet shards actually read
    sangraha_base = FLAKE / "raw/hf/ai4bharat_Sangraha_verified/verified"
    shards_read = {
        "kan": ["data-0.parquet", "data-1.parquet"],
        "tam": ["data-0.parquet", "data-1.parquet"],
        "tel": ["data-0.parquet", "data-1.parquet"],
        "mal": ["data-0.parquet", "data-1.parquet", "data-2.parquet"]
    }
    for lang, shard_list in shards_read.items():
        ldir = sangraha_base / lang
        for sname in shard_list:
            spf = ldir / sname
            if spf.exists():
                records.append(make_record(
                    spf, "raw",
                    f"AI4Bharat Sangraha verified {lang} ({sname})",
                    "CC BY 4.0",
                    v3_used,
                    "downloaded from huggingface.co/datasets/ai4bharat/Sangraha"
                ))

    return records


def main():
    print("Building records...")
    records = build_all_records()
    print(f"Total candidate records generated: {len(records)}")

    # Ensure parent dir exists
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {OUT_JSONL}")


if __name__ == "__main__":
    main()
