# Judge rubric: Phase 3 query eval (v1)

Used by every LLM judge (Gemini 3.8 Flash, Hermes/NIM models) and the human reviewers. Same rubric for base, SFT and 120M runs, so scores compare.

## Input per item
One line from `out/queries_v1_<run>.jsonl`. It has the query fields (`id lang bucket prompt reference notes`) plus `output stop n_tokens`.

## Context the judge must keep in mind
- The model is tiny (41M params now, ~120M later). Grade what's on the page, not effort. A 1 is normal for hard items.
- The model is deliberately **not** refusal-trained. Don't reward refusals or penalize frankness. Penalize only wrong, incoherent, or off-language output.
- `lang` is the language the answer should be in. `hin` items are zero-shot probes: grade them, but they're expected to be low.

## Scores (integers 1–5; `null` only where the criterion doesn't apply)
| key | 1 | 3 | 5 |
|---|---|---|---|
| `fluency` | broken text, wrong script or mixed scripts | readable, some unnatural phrasing | reads like a native speaker |
| `language` | wrong language | right language with drift or code-switching not asked for | entirely the right language and script (Tulu ≠ Kannada) |
| `instruction` | ignores the prompt | partly follows it | does exactly what was asked (`lang_gen`: continues on topic) |
| `accuracy` | false or contradicts `reference` | partly right | correct. `null` if there's no factual content and no reference |
| `stopping` | rambles, loops or cuts off mid-thought | stops at an acceptable point | ends cleanly once the answer is done (`stop` = `eos` helps but isn't required) |
| `tool_choice` | (`tool` bucket only) no awareness a tool is needed | mentions the need vaguely | names or emits the right tool from `notes`. `null` for other buckets |

## Output: one JSON line per item, nothing else
```json
{"id": "kan-ins-003", "judge": "gemini-3.8-flash", "fluency": 3, "language": 5, "instruction": 2, "accuracy": null, "stopping": 2, "tool_choice": null, "comment": "<= 25 words, why"}
```
Save as `out/judged_<run>_<judge>.jsonl`. Judges never edit the outputs file.

## Disagreement
Claude merges the judges' files. An item is flagged for Joel when any key differs by 2 or more between judges. The human subset (~60 items: Tulu, Kannada, English) overrides the LLM judges where they overlap.
