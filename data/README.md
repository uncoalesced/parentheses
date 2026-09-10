# Training data for Parentheses 0.9

Per the plan: general Wikipedia, legally-obtainable famous open-source
books, and public-domain religious/classic texts (the Bible, etc.).

Current target scale is ~100K-1M params (see docs/training-time-estimate.md)
-- that changes the corpus recommendation below from the original plan:
**use Simple English Wikipedia, not the full dump**, at least until the
model grows well past the 1M-param stretch tier. Full enwiki (~16GB, ~16B
byte-level tokens) is 100-1000x more data than this tier can usefully
absorb in one pass; Simple Wikipedia (~161MB, ~161M byte-level tokens,
226K articles) is sized to actually match the model's capacity and trains
in hours instead of days. See the doc for the full reasoning and numbers.

## Sources (all legal/no-piracy)

- **Wikipedia**: `pszemraj/simple_wikipedia` on Hugging Face (pre-cleaned
  Simple English Wikipedia) is the current default. Full `wikimedia/wikipedia`
  dumps (CC BY-SA 3.0 / GFDL, no scraping needed) remain the upgrade path
  once the model scales past the stretch tier.
- **Books**: Project Gutenberg (public domain, `https://www.gutenberg.org/`)
  covers most "very famous open-source books" -- classic literature, out of
  copyright. The Hugging Face `manu/project_gutenberg` or similar mirrors
  make bulk download easier than scraping gutenberg.org directly. At this
  scale, a handful of shorter books is plenty -- see the corpus-size
  reasoning above.
- **Religious/classic texts**: public-domain Bible translations (e.g. KJV,
  ASV -- both public domain) are available as plain text from sources like
  `https://www.gutenberg.org/ebooks/10` (KJV).
- **Parallel text (translation layer)**: OPUS (`https://opus.nlpl.eu`), the
  standard aggregator that redistributes Tatoeba, NLLB, Samanantar, Anuvaad,
  the Wikimedia translation memories and ~390 others in one uniform format,
  each with its own stated license. `prepare_parallel.py` pulls only corpora
  carrying an explicit open license (see its `LICENSES` table) and prints a
  reason for every corpus it skips, so the "legal, no piracy" bar is enforced
  in code rather than by memory. **OpenSubtitles is excluded** -- it states no
  license on OPUS and is a scrape of subtitle files whose copyright sits with
  the studios -- even though it is the single largest source for Arabic,
  Korean, Turkish, Persian, Polish and Indonesian. **CCMatrix is excluded**
  for the same no-stated-license reason, which costs nothing: NLLB (ODC-By)
  has identical pair counts for 15 of the 22 Tier A/B languages and strictly
  more for the rest, and CCMatrix has no English-Kannada pair at all.

## Tokenization

Default is **byte-level** (every UTF-8 byte is a token, vocab_size=256) for
the 100k/300k/600k presets -- no tokenizer training needed, just
`text.encode("utf-8")`. This is a deliberate choice at this param scale,
not a placeholder -- see model/config.py's comment above `PRESETS` for why
a normal subword vocab doesn't fit the budget here. The 1m stretch preset
uses a small trained BPE vocab (512) instead; `tokenizer/train_tokenizer.py`
covers that case.

## Pipeline (run for real, multiple times, on real hardware)

Not a scaffold description anymore -- this has produced real trained
checkpoints. English corpus: 265.5M train / 2.68M val byte tokens (Simple
Wikipedia + 50 Gutenberg books). Translation corpus: all 22 Tier A/B
languages from `Parentheses language training.md`, 200K pairs each, ~3.3GB
downloaded via OPUS. Two full `parentheses-0.9-300k` training runs have
consumed this pipeline's output end to end: one on English + 20 languages,
one (current) on English + all 22 after the validation-split fix below.

1. `prepare_wikipedia.py` -- download + clean Simple English Wikipedia (or
   full enwiki later) into plain text shards.
2. `prepare_books.py` -- download + clean Gutenberg-sourced books into plain
   text shards (`--limit`, default 50 -- "a handful of shorter books" per
   the corpus-size reasoning above).
3. `tokenizer/train_tokenizer.py` -- only needed for the 1m+ preset; skip
   for 100k/300k/600k (byte-level needs no training step).
3b. `prepare_parallel.py` -- **translation layer only**, not needed for the
   English model. Downloads English<->X sentence pairs from OPUS into the
   same shard format, one language per run
   (`--lang kn`), writing `raw/parallel/en-<lang>/` plus a `manifest.json`
   recording corpus, license, pair count and byte-token count per source.
   `--list` surveys availability without downloading anything. Pairs are
   emitted as `<en> source -> <kn> target` blocks -- a provisional format
   from handoff-translation-layer.md, deliberately a one-line function to
   change (`format_pair`) since the real prompt convention is still open.
4. `tokenize_corpus.py` -- reads the shards from steps 1-2 (or any
   `--raw-dirs` set, e.g. English + `raw/parallel/`), encodes them
   (byte-level by default, or `--tokenizer-dir` for a trained BPE
   tokenizer), and writes `<out-dir>/train.bin` / `val.bin` as a single
   `uint16` binary that `train.py`'s `np.memmap` loader expects (same
   layout nanoGPT uses). **Real bug found and fixed here 2026-08-31**: the
   validation split used to be a straight last-1%-of-the-concatenated-
   stream slice, and because shards are globbed per directory and Chinese
   (`en-zh`) sorted alphabetically last among the language directories,
   the entire validation set was silently 100% Chinese -- invalidating
   every val-loss number computed before that date. Now splits per source
   directory proportionally (verified: every source contributes exactly
   its own `val_fraction`, checkable via `--verify`), so `val.bin` is
   actually representative of the whole corpus.

## Stage 2: instruction/conversation data (SFT)

Steps 1-4 produce one undifferentiated byte stream, and next-token prediction
over it teaches the model what a transcript looks like, not how to answer when
asked. That is a separate stage, with its own corpus and its own loss:

5. `prepare_sft.py` -- converts conversation-shaped sources (ShareGPT today:
   `raw/manual/raw-sourced/hermes-function-calling-v1`) into turn-delimited
   shards under `raw/sft/<source>/`, each shard paired with a
   `shard_NNNNN.spans.json` recording the byte ranges of the assistant turns.
   Delimiters are control **bytes** (`\x01` start-of-turn, `\x02` role
   separator, `\x03` end-of-turn), not words -- the vocab is byte-level and a
   text marker collides with real content. `--pack` then encodes those shards
   to `processed_sft/{train,val}.bin` *plus* the parallel
   `{train,val}.mask.bin`, reusing `tokenize_corpus.write_bins` so the mask
   cannot drift out of step with the data it masks.
   `train.py --sft-spans processed_sft/train.mask.bin` sets every non-assistant
   target to the model's `ignore_index`, so the loss scores the response only.
   Real numbers on Hermes: 24 shards, 26,411 assistant turns, 51.4M byte
   tokens, 16.2% supervised. See `raw/sft/README.md`.

Two data-hygiene tools sit alongside the corpus rather than in the pipeline,
both report-first (they print what they would change; changing it is a second,
explicit run):

- `scrub_pii.py` -- regex redaction of emails, phone numbers and SSN-shaped
  strings, scoped by an allowlist to CommonCrawl-derived sources only
  (`open-web-math` today). It refuses to run against Gutenberg, Wikipedia,
  OpenStax or `personality-spec/` -- those are not scrapes of live web pages
  and carry no incidental-third-party-data risk. Data quality and third-party
  privacy, not content filtering: it removes an email address, not an opinion.
- `../scripts/dedup_raw_sourced.py` -- exact + near-duplicate scan across
  `raw-sourced/` (built by a separate session; report at `raw/DEDUP_REPORT.md`).

## Directories under `raw/` that no `--raw-dirs` default sweeps

Steps 1-4 above feed `raw/wikipedia/`, `raw/books/` and `raw/parallel/`, and
those three are what every `tokenize_corpus.py` invocation passes. The three
below are staged deliberately outside that set: they exist on disk, they are
not in any pretraining corpus, and moving them in is a decision someone has to
make on purpose rather than a side effect of running the pipeline.

5. `raw/monolingual/<lang>/` -- monolingual text in a language with no English
   pairing, so it does not fit `prepare_parallel.py`'s
   `<en> source -> <lang> target` shape. Currently holds Tulu (`tcy`): 2,202
   articles, 11.8MB, pulled with `prepare_wikipedia.py --source
   wikimedia/wikipedia --config 20231101.tcy` (CC BY-SA 3.0 / GFDL). Each
   language directory carries a `SOURCE.txt` with origin, license, pull date
   and size.
6. `raw/manual/` -- hand-supplied personality/character material, split into
   `raw-sourced/` (third-party open datasets, one `SOURCE.txt` per source) and
   `personality-spec/` (Joel's own writing). Files have started landing here
   -- see `raw/manual/README.md` for the format and licensing rules, and
   `personality-data-brief.md` in the repo root for the full brief.
   `../scripts/build_attribution.py` rolls every `SOURCE.txt` under
   `raw-sourced/` into one `ATTRIBUTION.md` -- the file that ships with a
   released checkpoint. Rerun it whenever a source lands.
7. `raw/sft/` -- the stage-2 conversation corpus from step 5. Separate from
   the pretraining shards on purpose: mixing them would put prompts back into
   the unmasked loss. Only `--pack` reads it, and only `--sft-spans` uses what
   it produces.

Each script has a `--self-test` flag that exercises its logic on fake data
with no network call -- 14/14 self-tests pass across the codebase as of
2026-08-31 (`model`, `free_think`, `modular_free_think`, `embedding_head`,
`prepare_parallel`, `tokenize_corpus`, `_shard_io`, `prepare_books`,
`nightly.ps1 -SelfTest`, and others). All of it has also been run for
real against real Hugging Face datasets and the real OPUS API, repeatedly,
on this laptop's RTX 5050 -- not just self-tested.
