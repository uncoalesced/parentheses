# Translation corpus sourcing — audit

The "real corpus-sourcing pass" `handoff-translation-layer.md` asks for as its
next concrete step. Kannada was done first and in full, as that doc suggests
(Joel's own fluency means output quality can be judged by ear later); the same
script and the same selection rules then apply unchanged to the rest of Tier A
and Tier B.

Commands to actually pull this data: `docs/translation-corpus-download.md`.
Implementation: `data/prepare_parallel.py`.

Everything below comes from the live OPUS API and OPUS's own per-corpus
license pages, queried 2026-08-30 — not from estimates.

## Where the data actually is

**OPUS (`https://opus.nlpl.eu`) is the answer to all three sources the handoff
names.** It already redistributes Tatoeba, CCMatrix, NLLB, Samanantar,
Anuvaad, the Wikimedia translation memories and ~390 others in one uniform
format, with a machine-readable index (`/opusapi/`) that reports pair counts,
token counts, download size and URL per corpus per language pair. There is no
reason to hit Tatoeba or CCMatrix separately.

Of the three the handoff names by name, only one turned out to matter:

- **NLLB** — ODC-By 1.0. The workhorse. Present for every Tier A/B language,
  20M–410M pairs each.
- **CCMatrix** — **states no license on OPUS**, and it is redundant anyway.
  Its pair counts are *identical* to NLLB's for 15 of the 22 languages and
  strictly smaller for the other 7. For **Kannada it does not exist at all**.
  Excluding it costs nothing.
- **Tatoeba** — CC BY 2.0 FR, and **far too small to matter for this project's
  languages**. Tatoeba's own totals confirm OPUS's mirror: 325 Kannada
  sentences in the entire database, 597 Tamil, 884 Malayalam. English-Kannada
  aligned: **232 pairs**. It's real, it's clean, it's free — and it is a
  rounding error next to Samanantar's 4 million.

## Kannada — the template case

| | |
|---|---|
| OPUS corpora carrying en-kn | 15 |
| Raw pairs across all 15 | 40,241,625 |
| Carrying an explicit open license | 8 |
| Selected under default caps | 7 |
| Openly-licensed pairs available | 39,987,478 |

Per-corpus, English–Kannada:

| corpus | license | pairs | zip |
|---|---|---|---|
| NLLB | ODC-By 1.0 | 34,347,151 | 2,176 MB |
| Samanantar | CC0 1.0 | 4,093,525 | 251 MB |
| Anuvaad | CC-BY-4.0 | 1,352,029 | 150 MB |
| wikimedia | CC-BY-SA-4.0 | 86,511 | 19 MB |
| bible-uedin | CC0 1.0 | 61,707 | 6 MB |
| pmindia | CC-BY-SA-4.0 | 35,232 | 2 MB |
| translatewiki | CC BY 3.0 | 11,091 | 0.2 MB |
| Tatoeba | CC BY 2.0 FR | 232 | 14 KB |
| — CCAligned | *none stated* | 163,922 | excluded |
| — OpenSubtitles | *none stated* | 42,521 | excluded |
| — XLEnt / KDE4 / TED2020 / QED / GNOME | *none stated* | 47,704 total | excluded |
| — CCMatrix | — | **absent for this pair** | — |

**Measured end-to-end**, restricted to the five corpora ≤25 MB (so the run
needed no large download):

```
121,733 pairs   64,421,465 byte tokens   529 bytes/pair   25 shards
  bible-uedin    61,021 pairs    29,607,573 byte tokens
  wikimedia      44,144 pairs    29,807,181 byte tokens
  pmindia         9,677 pairs     4,110,635 byte tokens
  translatewiki   6,661 pairs       865,076 byte tokens
  Tatoeba           230 pairs        31,000 byte tokens
```

Fed straight into the existing `data/tokenize_corpus.py` with no changes:
**81,112,855 train / 819,322 val byte tokens, `uint16`, max token id 240**.
That confirms the handoff's byte-level claim concretely — Kannada needs no
tokenizer work at all, every byte lands inside `vocab_size=256`.

At 529 bytes/pair, a full 200K-pair Kannada run lands around **100M byte
tokens**, against the existing English corpus's 265M. Latin-script languages
run 110–330 bytes/pair, so they cost proportionally less.

## Tier A / Tier B availability

`n` and `pairs` are what the default settings select (open license, ≤300 MB
per corpus, ≤12 corpora). `NLLB` is the extra available on request — it is
openly licensed but too big for the default cap.

| tier | language | code | n | pairs (default) | download | NLLB pairs |
|---|---|---|---|---|---|---|
| A | Kannada | kn | 7 | 5,640,327 | 429 MB | 34,347,151 |
| A | Turkish | tr | 6 | 3,106,586 | 349 MB | 47,045,956 |
| A | French | fr | 9 | 3,047,434 | 333 MB | 328,595,738 |
| A | Portuguese | pt | 7 | 2,083,070 | 214 MB | 173,743,166 |
| A | Ukrainian | uk | 4 | 1,585,747 | 192 MB | 20,240,171 |
| A | Bengali | bn | 9 | 1,520,227 | 183 MB | 62,006,746 |
| A | Russian | ru | 7 | 1,487,674 | 115 MB | 139,937,785 |
| A | Arabic | ar | 6 | 1,244,735 | 229 MB | 49,697,322 |
| A | German | de | 7 | 1,172,772 | 90 MB | 247,470,736 |
| A | Vietnamese | vi | 5 | 968,134 | 99 MB | 50,092,444 |
| A | Polish | pl | 7 | 955,827 | 101 MB | 74,070,714 |
| A | Japanese | ja | 6 | 910,594 | 100 MB | 40,883,733 |
| A | Spanish | es | 7 | 828,551 | 57 MB | 409,061,333 |
| A | Dutch | nl | 6 | 775,456 | 64 MB | 106,695,917 |
| A | Mandarin Chinese | zh | 6 | 634,452 | 98 MB | 71,383,325 |
| A | Persian | fa | 4 | 582,989 | 108 MB | 24,597,533 |
| A | Korean | ko | 6 | 450,589 | 43 MB | 19,358,582 |
| A | Hindi | hi | 9 | 336,052 | 41 MB | 33,193,629 |
| B | Tamil | ta | 7 | 1,839,054 | 222 MB | 42,588,178 |
| B | Malayalam | ml | 6 | 1,700,201 | 195 MB | 43,759,128 |
| B | Indonesian | id | 7 | 590,190 | 57 MB | 70,545,705 |
| B | Malay | ms | 4 | 413,227 | 44 MB | 56,832,366 |

Total for the default pass: **~3.3 GB, 22 languages**, capped at 200K kept
pairs each.

**Read the "pairs (default)" column as a download-budget artifact, not as a
statement about how much data exists.** Kannada tops it only because
Samanantar and Anuvaad happen to ship en-kn zips that fit under 300 MB while
their Hindi equivalents don't. Every language here has 19M–410M openly-licensed
pairs available once NLLB is included — vastly more than Plan One can absorb
either way.

## Flags — things to decide, not things I worked around

1. **Tier B is not the weak tier; Hindi and Persian are.** Tamil (1.84M) and
   Malayalam (1.70M) come out *ahead* of most of Tier A under the default
   caps, because Anuvaad and Joshua-IPC cover Dravidian languages well. Malay
   is the genuinely thin one (413K, 4 corpora — Samanantar covers ta/ml but
   not ms), and Persian is thin for a different reason: 61.5M of its
   parallel data is OpenSubtitles, which the license bar excludes. The
   handoff's A/B split doesn't match what's actually available.

2. **Hinglish has no parallel corpus.** Requirement 3 asks for Hindi *and*
   Hinglish (Hindi in Latin letters). OPUS has no `hi_rom` pair at all, and
   `hif` (Fiji Hindi) has 274 pairs and is a different language. Hinglish
   would have to be *generated* — romanising the Devanagari side of the Hindi
   corpus with a transliteration library — which is a build, not a pull, and
   is outside this pass's scope. Needs your call.

3. **"Top 5 Spanish dialects" and "Russian dialects of Eurasia" aren't
   sourceable this way.** OPUS tags `es_MX` and `es_AR` at 6,082 pairs each,
   and has no Russian dialect tags. Dialect coverage would have to come from
   dialect-labelled monolingual text, not parallel tags. Portuguese is the
   exception and works fine: `pt` (559M) plus `pt_BR` (115M across 5 corpora)
   covers "primary Portuguese + Brazilian" exactly as asked.

4. **Simplified vs Traditional Chinese isn't answered by the corpus.** OPUS
   tags `zh_CN` at 37.7M pairs and `zh_TW` at 39.5M — near-identical, so
   corpus size can't tell you which is "statistically more used". That's an
   outside-world fact (Simplified, by a wide margin — mainland China plus
   Singapore), not a data question. The generic `zh` pool (214M) mixes both.

5. **OpenSubtitles is excluded and it costs real volume.** It states no
   license on OPUS and is a scrape of subtitle files whose copyright sits with
   the studios, so it fails the project's "legal, no piracy" bar. It is the
   single largest source for Arabic (87.9M), Turkish (73.5M), Polish (75.3M),
   Indonesian (73.8M), Persian (61.5M) and Korean (31.1M). Excluded on
   purpose; `--allow OpenSubtitles` is there if you decide otherwise.

6. **OPUS's declared download sizes are not trustworthy.** It reports `1 KB`
   for NLLB en-es and en-fr, and `0` for seven more pairs, against real zips
   in the tens of GB. A plain size cap sails straight past that and starts an
   unbounded download. `prepare_parallel.py` cross-checks declared size
   against pair count and refuses anything claiming under 20 bytes per pair.
   Worth knowing before trusting that field anywhere else.

7. **The ~300 ELRC-*/ELRA-* EU corpora are excluded for now.** They do state
   licenses (mostly CC-BY-4.0, publicDomain or openUnder-PSI) but a minority
   are CC-BY-NC-4.0, and non-commercial is a restriction worth your decision
   rather than a blanket rule. Each is only a few hundred to a few thousand
   pairs, so the volume at stake is negligible.

## Still out of scope, per the handoff

Tier C (Latin, Ancient/Koine Greek, Hebrew), Tier D (Yiddish) and Tier E
(Sumerian, Akkadian) are untouched. No translation training was attempted —
that stays gated on Plan Two's parameter budget. The architecture question
(reuse `Parentheses` as a translation task vs. a separate seq2seq model) is
still yours to confirm; nothing here presumes either answer beyond the
provisional pair format.

## The pair format is provisional

Pairs are written as `<en> source text -> <kn> target text`, straight from the
handoff's proposal. It is deliberately isolated in one function
(`format_pair`) so settling the real prompt convention later is a one-line
change and a re-run, not a pipeline rewrite.
