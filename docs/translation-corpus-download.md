# Translation corpus — download commands

Everything here is a copy-paste command for **you** to run; the code is
finished and self-tested. `data/prepare_parallel.py` does the work: it asks
the OPUS API what parallel corpora exist for `en-<lang>`, keeps only the ones
with an explicit open license, downloads them, cleans the pairs, and writes
plain-text shards plus a `manifest.json` recording exactly what came from
where.

PowerShell 5.1 has no `&&`, so every line below stands alone. A new tab does
not inherit the venv — run this first:

```powershell
.\venv\Scripts\Activate.ps1
```

## 0. Sanity check (no network, ~1 second)

```powershell
python data\prepare_parallel.py --self-test
```

Expected output:

```
[self-test] _shard_io ok
[self-test] prepare_parallel ok
```

## 1. Look before downloading (no download, one API call per language)

```powershell
python data\prepare_parallel.py --lang kn --list
```

Prints every OPUS corpus for the pair, which ones will be used, and a reason
for each one skipped. Change `--lang` for any other language.

## 2. The default pass — all 22 Tier A/B languages

About **3.3 GB** of downloads total, and up to 200K sentence pairs per
language. Zips are cached under `data/raw/parallel/_zips/`, so a re-run costs
nothing and an interrupted run resumes cleanly.

```powershell
foreach ($L in @('kn','hi','zh','fr','es','ar','ru','pt','uk','de','pl','ja','ko','vi','tr','fa','bn','nl','id','ms','ta','ml')) { python data\prepare_parallel.py --lang $L }
```

If you would rather go one at a time, or restart partway through, each
language is independent:

```powershell
python data\prepare_parallel.py --lang kn
python data\prepare_parallel.py --lang hi
python data\prepare_parallel.py --lang zh
python data\prepare_parallel.py --lang fr
python data\prepare_parallel.py --lang es
python data\prepare_parallel.py --lang ar
python data\prepare_parallel.py --lang ru
python data\prepare_parallel.py --lang pt
python data\prepare_parallel.py --lang uk
python data\prepare_parallel.py --lang de
python data\prepare_parallel.py --lang pl
python data\prepare_parallel.py --lang ja
python data\prepare_parallel.py --lang ko
python data\prepare_parallel.py --lang vi
python data\prepare_parallel.py --lang tr
python data\prepare_parallel.py --lang fa
python data\prepare_parallel.py --lang bn
python data\prepare_parallel.py --lang nl
python data\prepare_parallel.py --lang id
python data\prepare_parallel.py --lang ms
python data\prepare_parallel.py --lang ta
python data\prepare_parallel.py --lang ml
```

Output per language: `data/raw/parallel/en-<lang>/shard_*.txt` +
`manifest.json`.

## 3. See what you got

```powershell
python data\prepare_parallel.py --report
```

Reads every manifest written so far and prints pairs, byte tokens, bytes per
pair and the source corpora with their licenses, per language.

## 4. Feed it to the existing tokenizer

`tokenize_corpus.py` needs no changes — the shards are the same format
`prepare_wikipedia.py` and `prepare_books.py` write.

Translation data on its own:

```powershell
python data\tokenize_corpus.py --raw-dirs data\raw\parallel --out-dir data\processed_translation
```

Or mixed with the existing English corpus (what you'd actually train on, once
the param budget question is settled):

```powershell
python data\tokenize_corpus.py --raw-dirs data\raw\wikipedia data\raw\books data\raw\parallel --out-dir data\processed_multilingual
```

## Optional — the big NLLB pull

NLLB (ODC-By 1.0) is the largest openly-licensed source and covers every
language here, but its zips run 1–5 GB per pair, so the default run skips it.
Two separate reasons it gets skipped, and the fix differs:

- **Size over the cap** (kn, hi, ar, uk, ja, ko, vi, tr, fa, bn, ms, ta, ml —
  OPUS reports a real size, it's just large): raise the cap.

  ```powershell
  python data\prepare_parallel.py --lang kn --max-download-mb 5000
  ```

- **Size misreported** (zh, fr, es, ru, pt, de, pl, nl, id — OPUS declares
  0 KB or 1 KB for zips that are tens of GB): the size guard can't tell those
  from a genuinely tiny corpus, so it refuses them. Override per corpus, and
  know you are accepting a possibly very large download:

  ```powershell
  python data\prepare_parallel.py --lang fr --force-corpus NLLB
  ```

Either way `--max-pairs` still caps how much is *kept*, so the cost is
download bandwidth, not disk or training time. **You probably don't need this
at Plan One scale** — 200K pairs per language is already far past what a
few-hundred-K-parameter model can absorb.

## Optional — corpora the license bar excludes

The script only uses corpora with an explicit open license on OPUS. To pull
one it skipped, after reading that corpus's own terms yourself:

```powershell
python data\prepare_parallel.py --lang ar --allow OpenSubtitles
```

Worth knowing what that particular one is: OpenSubtitles states no license on
OPUS and is a scrape of subtitle files whose copyright sits with the studios.
It is the single largest source for Arabic, Korean, Turkish, Persian, Polish
and Indonesian, and it is excluded on purpose to keep the project's
"legal, no piracy" bar. Same reasoning excludes CCMatrix, ParaCrawl, CCAligned
and HPLT.

## Useful flags

| flag | default | what it does |
|---|---|---|
| `--lang` | `kn` | target language code; English is always the source |
| `--max-pairs` | `200000` | ceiling on kept pairs per language |
| `--max-download-mb` | `300` | skip any single corpus whose zip is bigger |
| `--max-corpora` | `12` | download at most this many corpora, largest first |
| `--allow CORPUS` | — | use a corpus that states no license on OPUS |
| `--force-corpus CORPUS` | — | bypass license *and* size guards for one corpus |
| `--list` | — | survey only, download nothing |
| `--report` | — | table of everything already downloaded |
| `--self-test` | — | assert-based logic check, no network |
