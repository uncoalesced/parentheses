"""
Download + clean English<->X parallel sentence pairs into plain-text shards.

Engineered by uncoalesced

Third data source alongside prepare_wikipedia.py (monolingual English) and
prepare_books.py (monolingual English): this one pulls *parallel* text for
the translation layer (see handoff-translation-layer.md), from OPUS
(https://opus.nlpl.eu) -- the standard aggregator that already redistributes
Tatoeba, CCMatrix, NLLB, Samanantar and friends in one uniform format.

Not named translate.py on purpose -- see data/tokenize_corpus.py's name for
the same lesson (`tokenize` is stdlib). `prepare_parallel` shadows nothing.

Sourcing bar: same "legal, no piracy" standard the rest of data/ uses. Only
corpora with an explicit open license on OPUS are pulled -- see LICENSES
below. Everything else is skipped with a printed reason rather than silently
included; `--allow CORPUS` overrides one at a time if you've checked its
terms yourself. Notably this excludes OpenSubtitles (no stated license, and
it's a scrape of subtitle files whose copyright sits with the studios) even
though it is the single largest source for several Tier A languages.

Output goes through data/_shard_io.py's write_shards, same as the other two
prep scripts, so data/tokenize_corpus.py picks it up with no changes:

    python3 data/prepare_parallel.py --lang kn
    python3 data/prepare_parallel.py --lang kn --list      # survey only, no download
    python3 data/prepare_parallel.py --self-test           # no network

Each run also writes a manifest.json next to the shards recording which
corpus supplied what, under which license, at what pair and byte-token
count -- that's the per-language sourcing record the handoff asks for.
"""

import argparse
import glob
import io
import json
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile

from _shard_io import write_shards, self_test as _shard_io_self_test

OPUS_API = "https://opus.nlpl.eu/opusapi/"


def _ssl_context():
    """TLS context that can actually verify OPUS's download host.

    OPUS serves its zips from object.pouta.csc.fi, whose chain this venv's
    Python has no root for -- plain urlopen dies with CERTIFICATE_VERIFY_FAILED
    even though opus.nlpl.eu itself works. certifi ships the missing root and
    is already installed (a `datasets` dependency), so use its bundle and fall
    back to the system store if it isn't there. Verification stays on either
    way; never disable it to make a download go through.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

# Corpora with an explicit open license stated on their OPUS page, checked
# 2026-08-30. Anything not in here is skipped by default -- absence means
# "no license stated on OPUS", not "known bad", so --allow is the escape
# hatch once you've read the corpus's own terms.
#
# NLLB is the workhorse: it covers every Tier A/B language, and its pair
# counts are identical to CCMatrix's for 15 of the 22 and strictly larger
# for the other 7 (CCMatrix has *no* English-Kannada pair at all), so
# skipping CCMatrix -- which states no license on OPUS -- costs nothing.
LICENSES = {
    "NLLB": "ODC-By 1.0",
    "Samanantar": "CC0 1.0",
    "Anuvaad": "CC-BY-4.0",
    "Tatoeba": "CC BY 2.0 FR",
    "bible-uedin": "CC0 1.0",
    "wikimedia": "CC-BY-SA-4.0",
    "pmindia": "CC-BY-SA-4.0",
    "WikiTitles": "CC-BY-SA-4.0",
    "translatewiki": "CC BY 3.0",
    "tico-19": "CC0 1.0",
    "EuroPat": "CC0 1.0",
    "ALT": "CC BY 4.0",
    "ELITR-ECA": "CC-BY-4.0",
    "tldr-pages": "CC-BY-4.0",
    "MDN_Web_Docs": "CC-BY-SA-2.5",
    "GoURMET": "CC0 1.0",
    "SUMMA": "CC0 1.0",
    "Joshua-IPC": "CC-BY-3.0",
    "Salome": "CC-BY-SA-3.0",
}

# Deliberately not enumerated: the ~300 ELRC-*/ELRA-* corpora (EU public-sector
# translation memories). They do state licenses on OPUS -- mostly CC-BY-4.0,
# publicDomain or openUnder-PSI, but a minority are CC-BY-NC-4.0, which is a
# restriction worth a human decision rather than a blanket rule. Each is also
# only a few hundred to a few thousand pairs, so listing all of them buys
# almost no volume. Pull one with --allow ELRC-whatever after reading its terms.

# Target-side script check: mined corpora are full of rows where the
# "translation" is just the English copied over. For a non-Latin-script
# language that's caught by asking the target side to actually contain
# characters from its own script. (start, end) is one representative
# Unicode block per language; languages written in Latin script are None
# and fall back to the src != tgt check alone.
SCRIPTS = {
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "hi": (0x0900, 0x097F),   # Devanagari
    "bn": (0x0980, 0x09FF),   # Bengali
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "ml": (0x0D00, 0x0D7F),   # Malayalam
    "zh": (0x4E00, 0x9FFF),   # CJK unified ideographs
    "ja": (0x3040, 0x30FF),   # kana (kanji-only lines are rare in sentences)
    "ko": (0xAC00, 0xD7AF),   # hangul syllables
    "ar": (0x0600, 0x06FF),   # Arabic
    "fa": (0x0600, 0x06FF),   # Arabic script (Persian)
    "ru": (0x0400, 0x04FF),   # Cyrillic
    "uk": (0x0400, 0x04FF),
    "el": (0x0370, 0x03FF),
    "he": (0x0590, 0x05FF),
}

MIN_BYTES = 8          # shorter than this is a UI string, not a sentence
MAX_BYTES = 1000       # longer than this is a runaway paragraph
# Length-ratio sanity check between the two sides, on *characters* not bytes:
# UTF-8 costs 1 byte/char for English and 3 for Kannada/Devanagari/CJK, so a
# byte ratio would call a perfectly good en-kn pair lopsided purely because of
# the encoding. ponytail: 4.0 is a flat cap across every language pair; CJK
# packs more meaning per character than English, so a per-language ratio
# would keep more good zh/ja pairs if this ever filters too hard.
MAX_RATIO = 4.0
# Floor for "is this declared download size believable". Two real sentences
# plus a separator never compress to 20 bytes, so a zip claiming less than
# this per pair is reporting a placeholder, not a size.
MIN_BYTES_PER_PAIR = 20


def opus_index(src: str, tgt: str) -> list[dict]:
    """Every latest-version OPUS corpus carrying the src-tgt pair."""
    url = OPUS_API + "?" + urllib.parse.urlencode(
        {"source": src, "target": tgt, "preprocessing": "moses", "version": "latest"})
    with urllib.request.urlopen(url, timeout=120, context=_ssl_context()) as r:
        corpora = json.load(r)["corpora"]
    for c in corpora:  # the API returns these as strings for some corpora
        for k in ("alignment_pairs", "size", "source_tokens", "target_tokens"):
            try:
                c[k] = int(c[k])
            except (TypeError, ValueError):
                c[k] = 0
    corpora = [c for c in corpora if c["alignment_pairs"] > 0]
    corpora.sort(key=lambda c: -c["alignment_pairs"])
    return corpora


def selectable(corpora: list[dict], allow: list[str], max_mb: float, max_corpora: int = 0,
               force: list[str] = ()):
    """Split the index into (usable, [(corpus, why-skipped)]).

    `max_corpora` bounds how many get downloaded: a high-resource language
    like French has 125 OPUS corpora, ~40 of them openly licensed, and
    fetching every one to fill a 200K-pair budget is all download and no
    benefit. Keeps the largest ones; 0 means no cap.

    `force` names corpora that bypass every guard here -- license allowlist,
    size plausibility and size cap alike. That's how you reach NLLB for the
    languages where OPUS misreports its size; it also means accepting a
    download that can run to tens of GB.
    """
    keep, skipped = [], []
    for c in corpora:
        name = c["corpus"]
        if name in force:
            keep.append(c)
        elif name not in LICENSES and name not in allow:
            skipped.append((c, "no open license stated on OPUS"))
        elif c["size"] * 1024 < c["alignment_pairs"] * MIN_BYTES_PER_PAIR:
            # The declared size has to be big enough to actually hold the
            # declared pairs. OPUS reports 0 for some entries and a flat 1KB
            # for NLLB en-es and en-fr, which are tens of GB -- both sail
            # through a plain size cap and start an unbounded download.
            skipped.append((c, "OPUS's declared size is too small to be real -- "
                               "treating as unknown (could be tens of GB)"))
        elif c["size"] / 1024 > max_mb:
            skipped.append((c, f"{c['size'] / 1024:.0f}MB download > --max-download-mb {max_mb:g}"))
        else:
            keep.append(c)
    if max_corpora and len(keep) > max_corpora:
        keep.sort(key=lambda c: -c["alignment_pairs"])
        keep, extra = keep[:max_corpora], keep[max_corpora:]
        skipped += [(c, f"beyond --max-corpora {max_corpora}") for c in extra]
    return keep, skipped


def cache_name(url: str) -> str:
    """Cache filename for `url` -- the whole path, not just its tail.

    OPUS urls are .../OPUS-<corpus>/<version>/moses/<pair>.txt.zip, and both
    the corpus and the version have to be in the key: most corpora call their
    version "v1", so keying on version+pair alone makes NLLB, Anuvaad,
    pmindia and bible-uedin all collide on one cached file and silently read
    each other's data.
    """
    path = urllib.parse.urlsplit(url).path.strip("/")
    return re.sub(r"[^A-Za-z0-9._-]", "_", path)


def _fetch(url: str, cache_dir: str) -> str:
    """Download `url` into cache_dir once; return the local path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, cache_name(url))
    if not os.path.exists(path):
        tmp = path + ".part"          # so an interrupted download isn't cached as complete
        with urllib.request.urlopen(url, timeout=300, context=_ssl_context()) as r, \
                open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(tmp, path)
    return path


def read_moses_zip(path: str, src: str, tgt: str):
    """Yield (source_line, target_line) from an OPUS moses .txt.zip.

    A moses zip holds the two sides as separate line-aligned files ending
    in .<src> and .<tgt>, plus README/LICENSE members to ignore.
    """
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        s = next((n for n in names if n.endswith("." + src)), None)
        t = next((n for n in names if n.endswith("." + tgt)), None)
        if not s or not t:
            return
        with z.open(s) as fs, z.open(t) as ft:
            fs = io.TextIOWrapper(fs, encoding="utf-8", errors="replace")
            ft = io.TextIOWrapper(ft, encoding="utf-8", errors="replace")
            for a, b in zip(fs, ft):
                yield a, b


def in_script(text: str, code: str) -> bool:
    """True if `text` contains a character from `code`'s script (or we don't check it)."""
    rng = SCRIPTS.get(code)
    if rng is None:
        return True
    lo, hi = rng
    return any(lo <= ord(ch) <= hi for ch in text)


def clean_pair(a: str, b: str, src: str, tgt: str) -> tuple[str, str] | None:
    """Normalise one pair, or None if it isn't usable training text."""
    a = re.sub(r"\s+", " ", a).strip()
    b = re.sub(r"\s+", " ", b).strip()
    if not a or not b or a == b:
        return None
    na, nb = len(a.encode("utf-8")), len(b.encode("utf-8"))
    if not (MIN_BYTES <= na <= MAX_BYTES) or not (MIN_BYTES <= nb <= MAX_BYTES):
        return None
    if max(len(a), len(b)) / min(len(a), len(b)) > MAX_RATIO:
        return None
    if not in_script(b, tgt) or not in_script(a, src):
        return None
    return a, b


def format_pair(a: str, b: str, src: str, tgt: str) -> str:
    """One training block.

    Provisional format, straight from handoff-translation-layer.md's
    "<lang tag> source text -> <lang tag> target text". The final prompt
    convention is explicitly deferred to a later doc, so this stays a
    one-line function to change rather than a format baked through the
    pipeline.
    """
    return f"<{src}> {a} -> <{tgt}> {b}"


def iter_blocks(corpora, src, tgt, cache_dir, max_pairs, stats, fetch=None):
    """Yield formatted, deduped training blocks across `corpora`, largest
    first, stopping at max_pairs. Records per-corpus counts in `stats`.

    `fetch` is the download step, injected so the self-test can hand over a
    local zip instead of monkey-patching a module global."""
    fetch = fetch or _fetch
    # Max-min fair share, smallest corpus first: each corpus may take up to an
    # equal cut of what's left, so the small ones contribute everything they
    # have and the big ones absorb the remainder. Largest-first instead would
    # let one corpus eat the entire budget -- Samanantar alone fills 200K pairs
    # of en-kn, and a mix of news, scripture, Wikipedia and UI strings is
    # better training data than 200K rows of the same register.
    corpora = sorted(corpora, key=lambda c: c["alignment_pairs"])
    seen = set()
    total = 0
    for i, c in enumerate(corpora):
        left = max_pairs - total
        if left <= 0:
            stats.append({"corpus": c["corpus"], "license": LICENSES.get(c["corpus"], "allowed via --allow"),
                          "available_pairs": c["alignment_pairs"], "kept_pairs": 0,
                          "byte_tokens": 0, "note": "budget already full"})
            continue
        share = left if i == len(corpora) - 1 else max(1, left // (len(corpora) - i))
        kept = tokens = 0
        try:
            path = fetch(c["url"], cache_dir)
            pairs = read_moses_zip(path, src, tgt)
        except Exception as e:                      # one dead mirror shouldn't kill the run
            stats.append({"corpus": c["corpus"], "license": LICENSES.get(c["corpus"], "allowed via --allow"),
                          "available_pairs": c["alignment_pairs"], "kept_pairs": 0,
                          "byte_tokens": 0, "note": f"download/read failed: {e}"})
            continue
        for a, b in pairs:
            if kept >= share or total >= max_pairs:
                break
            cleaned = clean_pair(a, b, src, tgt)
            if cleaned is None:
                continue
            # ponytail: hashes, not the pairs themselves, to keep the dedupe set
            # small on multi-million-pair corpora. Collisions are ~1e-9 at these
            # counts; store the tuples if that ever needs to be exact.
            key = hash(cleaned)
            if key in seen:
                continue
            seen.add(key)
            block = format_pair(*cleaned, src, tgt)
            kept += 1
            total += 1
            tokens += len(block.encode("utf-8"))
            yield block
        stats.append({"corpus": c["corpus"], "license": LICENSES.get(c["corpus"], "allowed via --allow"),
                      "available_pairs": c["alignment_pairs"], "kept_pairs": kept,
                      "byte_tokens": tokens, "note": ""})


def print_report(root: str):
    """Summarise every manifest.json under `root` -- the after-the-run table."""
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "en-*", "manifest.json"))):
        d = json.load(open(path, encoding="utf-8"))
        used = [s for s in d["sources"] if s["kept_pairs"]]
        rows.append((d["target"], d["pairs"], d["byte_tokens"], used))
    if not rows:
        print(f"[report] no manifests under {root} yet")
        return
    rows.sort(key=lambda r: -r[1])
    print(f"{'pair':<10}{'pairs':>12}{'byte tokens':>15}{'B/pair':>8}  sources (license)")
    for tgt, pairs, tok, used in rows:
        srcs = ", ".join(f"{s['corpus']} [{s['license']}]"
                         for s in sorted(used, key=lambda s: -s["kept_pairs"]))
        print(f"{'en-' + tgt:<10}{pairs:>12,}{tok:>15,}{tok / max(pairs, 1):>8.0f}  {srcs}")
    print(f"\n{len(rows)} languages, {sum(r[1] for r in rows):,} pairs, "
          f"{sum(r[2] for r in rows):,} byte tokens")


def print_survey(corpora, keep, skipped, lang):
    print(f"[survey] en-{lang}: {len(corpora)} OPUS corpora, "
          f"{sum(c['alignment_pairs'] for c in corpora):,} raw pairs")
    for c in keep:
        print(f"  use  {c['corpus']:<20} {c['alignment_pairs']:>12,} pairs  "
              f"{c['size'] / 1024:>8.1f}MB  {LICENSES.get(c['corpus'], 'via --allow')}")
    for c, why in skipped:
        print(f"  skip {c['corpus']:<20} {c['alignment_pairs']:>12,} pairs  "
              f"{c['size'] / 1024:>8.1f}MB  -- {why}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--lang", default="kn", help="target language code (OPUS/ISO-639-1), English is the source")
    p.add_argument("--source-lang", default="en")
    p.add_argument("--out-dir", default=None, help="default: data/raw/parallel/en-<lang>")
    p.add_argument("--cache-dir", default="data/raw/parallel/_zips", help="downloaded OPUS zips")
    p.add_argument("--max-pairs", type=int, default=200_000,
                   help="cap on kept pairs per language; the whole English corpus is only ~265M byte tokens")
    p.add_argument("--max-download-mb", type=float, default=300.0,
                   help="skip any single OPUS corpus whose zip is larger than this")
    p.add_argument("--force-corpus", action="append", default=[], metavar="CORPUS",
                   help="use CORPUS whatever its license or size says (repeatable). This is how "
                        "you reach NLLB on languages where OPUS misreports its size -- and how you "
                        "start a tens-of-GB download, so pass it deliberately")
    p.add_argument("--max-corpora", type=int, default=12,
                   help="download at most this many corpora per language, largest first (0 = no cap)")
    p.add_argument("--allow", action="append", default=[],
                   help="use a corpus that states no license on OPUS (repeatable) -- check its terms first")
    p.add_argument("--shard-size", type=int, default=5000, help="pairs per shard file")
    p.add_argument("--list", action="store_true", help="survey what's available and exit, no download")
    p.add_argument("--report", action="store_true",
                   help="print the per-language table from every manifest already written, and exit")
    p.add_argument("--self-test", action="store_true",
                   help="check the clean/format/select logic on fake data and exit (no network)")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.report:
        print_report(os.path.join("data", "raw", "parallel"))
        return

    src, tgt = args.source_lang, args.lang
    out_dir = args.out_dir or os.path.join("data", "raw", "parallel", f"{src}-{tgt}")
    corpora = opus_index(src, tgt)
    keep, skipped = selectable(corpora, args.allow, args.max_download_mb, args.max_corpora,
                               args.force_corpus)
    print_survey(corpora, keep, skipped, tgt)
    if args.list:
        return
    if not keep:
        print(f"[warn] no openly-licensed corpus for {src}-{tgt} within the size cap", file=sys.stderr)
        return

    # Clear old shards first: write_shards numbers from 0, so a re-run with a
    # smaller budget would leave the previous run's tail behind and
    # tokenize_corpus.py would happily read both.
    for stale in glob.glob(os.path.join(out_dir, "shard_*.txt")):
        os.remove(stale)

    stats = []
    n = write_shards(iter_blocks(keep, src, tgt, args.cache_dir, args.max_pairs, stats),
                     out_dir, args.shard_size)
    pairs = sum(s["kept_pairs"] for s in stats)
    tokens = sum(s["byte_tokens"] for s in stats)
    manifest = {"source": src, "target": tgt, "pairs": pairs, "byte_tokens": tokens,
                "shards": n, "format": format_pair("SRC", "TGT", src, tgt), "sources": stats}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[done] {src}-{tgt}: {pairs:,} pairs, {tokens:,} byte tokens, "
          f"{n} shard(s) -> {out_dir}")


def _self_test():
    import tempfile

    # --- pair cleaning ------------------------------------------------------
    assert clean_pair("  The sky is blue. ", "ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ.\n", "en", "kn") == \
        ("The sky is blue.", "ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ.")
    assert clean_pair("The sky is blue.", "The sky is blue.", "en", "kn") is None   # copy-through
    assert clean_pair("The sky is blue.", "Il cielo e blu.", "en", "kn") is None    # wrong script
    assert clean_pair("", "ಆಕಾಶ", "en", "kn") is None                                # empty side
    assert clean_pair("Hi", "ಹಾಯ್", "en", "kn") is None                              # under MIN_BYTES
    assert clean_pair("x" * (MAX_BYTES + 1), "ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ.", "en", "kn") is None  # over MAX_BYTES
    assert clean_pair("The sky is blue and very wide indeed today.", "ಆಕಾಶ.", "en", "kn") is None  # ratio
    # a Latin-script target has no script gate, only the copy-through check
    assert clean_pair("The sky is blue.", "Le ciel est bleu.", "en", "fr") is not None
    assert in_script("anything", "fr") and not in_script("plain ascii", "kn")

    # --- format -------------------------------------------------------------
    assert format_pair("A dog.", "ಒಂದು ನಾಯಿ.", "en", "kn") == "<en> A dog. -> <kn> ಒಂದು ನಾಯಿ."

    # --- cache keys ---------------------------------------------------------
    # most OPUS corpora call their version "v1", so the corpus name has to be
    # part of the key or they overwrite each other's downloads
    base = "https://object.pouta.csc.fi/OPUS-{}/v1/moses/en-kn.txt.zip"
    assert cache_name(base.format("NLLB")) != cache_name(base.format("Anuvaad"))
    assert cache_name(base.format("NLLB")) == cache_name(base.format("NLLB"))
    assert "NLLB" in cache_name(base.format("NLLB")) and "v1" in cache_name(base.format("NLLB"))
    assert "/" not in cache_name(base.format("NLLB"))
    # same corpus, different pair or version, must not share a file either
    assert cache_name(base.format("NLLB")) != cache_name(
        "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/en-ta.txt.zip")
    assert cache_name(base.format("NLLB")) != cache_name(
        "https://object.pouta.csc.fi/OPUS-NLLB/v2/moses/en-kn.txt.zip")

    # --- license/size selection --------------------------------------------
    index = [
        {"corpus": "NLLB", "alignment_pairs": 34_347_151, "size": 2_228_705, "url": "u"},
        {"corpus": "Anuvaad", "alignment_pairs": 1_352_029, "size": 154_047, "url": "u"},
        {"corpus": "OpenSubtitles", "alignment_pairs": 42_521, "size": 1_100, "url": "u"},
        {"corpus": "CCMatrix", "alignment_pairs": 9_999, "size": 500, "url": "u"},
    ]
    keep, skipped = selectable(index, allow=[], max_mb=300.0)
    assert [c["corpus"] for c in keep] == ["Anuvaad"], keep
    reasons = {c["corpus"]: why for c, why in skipped}
    assert "no open license" in reasons["OpenSubtitles"], reasons
    assert "no open license" in reasons["CCMatrix"], reasons
    assert "max-download-mb" in reasons["NLLB"], reasons          # licensed, just too big here
    keep2, _ = selectable(index, allow=["CCMatrix"], max_mb=300.0)
    assert {c["corpus"] for c in keep2} == {"Anuvaad", "CCMatrix"}
    keep3, _ = selectable(index, allow=[], max_mb=99_999.0)
    assert "NLLB" in {c["corpus"] for c in keep3}                  # size cap was the only blocker

    # a blank or placeholder size must not read as "small enough". OPUS
    # reports 0 for some entries and a flat 1KB for NLLB en-es/en-fr, whose
    # real zips are tens of GB -- both would otherwise pass any size cap.
    for bogus in (0, 1):
        fake = [{"corpus": "NLLB", "alignment_pairs": 409_061_333, "size": bogus, "url": "u"}]
        keep4, skip4 = selectable(fake, allow=[], max_mb=99_999.0)
        assert keep4 == [] and "too small to be real" in skip4[0][1], (bogus, skip4)
    # ...but a genuinely small corpus with an honest small size still passes
    tiny = [{"corpus": "Tatoeba", "alignment_pairs": 232, "size": 14, "url": "u"}]
    assert [c["corpus"] for c in selectable(tiny, [], 300.0)[0]] == ["Tatoeba"]

    # --force-corpus overrides every guard at once: unlicensed, oversized and
    # bogus-size entries all come back
    forced = [{"corpus": "OpenSubtitles", "alignment_pairs": 42_521, "size": 1_100, "url": "u"},
              {"corpus": "NLLB", "alignment_pairs": 409_061_333, "size": 1, "url": "u"}]
    keepf, skipf = selectable(forced, [], 1.0, 0, force=["OpenSubtitles", "NLLB"])
    assert {c["corpus"] for c in keepf} == {"OpenSubtitles", "NLLB"} and skipf == []

    # --max-corpora keeps the largest and records the rest as skipped
    keep5, skip5 = selectable(index, allow=["CCMatrix", "OpenSubtitles"], max_mb=99_999.0, max_corpora=2)
    assert [c["corpus"] for c in keep5] == ["NLLB", "Anuvaad"], keep5
    assert all("max-corpora" in why for c, why in skip5), skip5

    # --- moses zip reading + end-to-end blocks (no network) -----------------
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "en-kn.txt.zip")
        with zipfile.ZipFile(zpath, "w") as z:
            z.writestr("README", "ignore me")
            z.writestr("Fake.en-kn.en", "The sky is blue.\nA dog barks loudly.\nThe sky is blue.\ncopy me\n")
            z.writestr("Fake.en-kn.kn", "ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ.\nನಾಯಿ ಜೋರಾಗಿ ಬೊಗಳುತ್ತದೆ.\nಆಕಾಶ ನೀಲಿಯಾಗಿದೆ.\ncopy me\n")
        assert len(list(read_moses_zip(zpath, "en", "kn"))) == 4
        assert list(read_moses_zip(zpath, "en", "xx")) == []       # missing side -> nothing, no raise

        stats = []
        corpus = {"corpus": "Fake", "alignment_pairs": 4, "size": 1, "url": zpath}
        local = lambda url, cache: url          # the only network step, stubbed out
        blocks = list(iter_blocks([corpus], "en", "kn", tmp, 100, stats, fetch=local))
        # 4 rows in -> dupe dropped, copy-through dropped, 2 kept
        assert len(blocks) == 2, blocks
        assert blocks[0] == "<en> The sky is blue. -> <kn> ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ."
        assert stats[0]["kept_pairs"] == 2 and stats[0]["license"] == "allowed via --allow"
        assert stats[0]["byte_tokens"] == sum(len(b.encode("utf-8")) for b in blocks)

        # max_pairs is a hard cap, and later corpora are recorded, not silently dropped
        stats2 = []
        capped = list(iter_blocks([corpus, dict(corpus, corpus="Fake2")], "en", "kn", tmp,
                                  1, stats2, fetch=local))
        assert len(capped) == 1, capped
        assert stats2[1]["note"] == "budget already full"

        # a dead mirror is recorded, not fatal -- the other corpora still run
        stats3 = []
        def boom(url, cache): raise OSError("mirror down")
        assert list(iter_blocks([corpus], "en", "kn", tmp, 100, stats3, fetch=boom)) == []
        assert "mirror down" in stats3[0]["note"]

        # fair share: a huge corpus must not swallow a budget two corpora share.
        # `big` claims millions of pairs but the zip behind it holds the same 2
        # usable rows, so the small one still gets its cut.
        stats4 = []
        big = dict(corpus, corpus="Big", alignment_pairs=4_000_000)
        list(iter_blocks([big, corpus], "en", "kn", tmp, 2, stats4, fetch=local))
        by_name = {s["corpus"]: s["kept_pairs"] for s in stats4}
        assert by_name == {"Fake": 1, "Big": 1}, stats4

        # shards land in the format tokenize_corpus.py already reads
        n = write_shards(iter(blocks), os.path.join(tmp, "out"), shard_size=5000)
        assert n == 1
        written = open(os.path.join(tmp, "out", "shard_00000.txt"), encoding="utf-8").read()
        assert written == "\n\n".join(blocks)

    _shard_io_self_test()
    print("[self-test] prepare_parallel ok")


if __name__ == "__main__":
    main()
