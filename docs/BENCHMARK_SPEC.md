# Benchmark protocol

This is how every number published for parentheses 1.1 gets produced. If a number can't be produced this way, it doesn't get published.

## Why this file exists

The 1.0 release tables came from a script I wrote myself. Readers went through it and found real problems: 30 items per benchmark, keyword matching where there should have been scoring, MBPP code that was compiled but never run, the best of four runs reported as the headline, and comparison tables that set those numbers against other labs' full-benchmark scores. They were right. Those tables measured much less than they appeared to.

From 1.1 the model is scored by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), not by code written in this repository.

## Rules

1. Tasks come from lm-evaluation-harness, with its task definitions unmodified. No custom scorers.
2. Full test sets. A run with `--limit` is a smoke test and is never reported.
3. Greedy decoding, one run. No best-of-N, no sampling at temperature for a headline number.
4. Report the harness's standard error next to every score.
5. Multiple-choice tasks are reported next to their chance level. A score under chance is published as a score under chance.
6. Code tasks (MBPP, HumanEval) execute the tests, inside a sandbox with no network access.
7. MMLU means all 57 subjects. One subject is not MMLU.
8. Belebele is reported per language: English, Kannada, Tamil, Telugu, Malayalam.
9. The raw results JSON for every published number is committed under `eval/results/<checkpoint-sha256>/`, with the harness version and the exact command.
10. Comparison tables only include models that were run locally through the same harness, same task versions and same settings. Numbers are not copied from papers or model cards.
11. Any text that interprets a table is written by a person looking at that table.
12. The project's own held-out prompts are not a benchmark. They were written by the same person who trained the model, and checkpoints were selected against them. Where they appear they are labelled as an internal suite.
13. No metric enters a public file until the maintainer has read the raw results file and at least 20 scored outputs for each task.

## Task list for the 1.1 baseline

| Area | Harness tasks |
|---|---|
| General knowledge | `mmlu` |
| Reading, commonsense | `hellaswag`, `arc_easy`, `arc_challenge`, `lambada_openai` |
| Maths | `gsm8k` |
| Truthfulness | `truthfulqa_mc1`, `truthfulqa_mc2` |
| Multilingual reading | `belebele_eng_Latn`, `belebele_kan_Knda`, `belebele_tam_Taml`, `belebele_tel_Telu`, `belebele_mal_Mlym` |
| Code | `mbpp`, `humaneval` |

Task names are checked against the installed harness version before each run. A task the harness does not define is dropped, not reimplemented. Terminal-Bench is out until it can be run with real execution.

## What is not measured yet

Speech and image tasks have no agreed protocol here yet. They will be added to this file before any audio or vision number is published.
