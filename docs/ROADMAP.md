# Roadmap

parentheses is a small language model developed from scratch on one 8 GB laptop GPU, aimed mostly at Kannada and the other Dravidian languages, with English alongside them. The certified golden release is parentheses-1.0-240M-tanzanite (Project Tanzanite). This roadmap details milestones achieved, architecture evolution, and ongoing work.

## At a glance

| Milestone | Status | Notes |
|---|---|---|
| 0.9-130M release | Released | Preview weights under CC BY-NC 4.0 (non-commercial), code under MIT. Available on GitHub and Hugging Face. |
| 1.0-240M Session 1 to 4 | Done | Wide-and-shallow 16-layer scaling, width 1024, 16 heads, rank 1024. |
| 1.0-240M Session 5 to 6 | Done | Textbook grounding, learning-rate anneal, 393M+ tokens seen. |
| 1.0-240M SFT v5 & Epistemic Tuning | Done | Calibrated abstention reasoning traces, eliminating ungrounded hallucinations. |
| 1.0-240M Session 10 Nudge & SWA | Done | Golden checkpoint `sft-nudge-winner.pt` (step 14,173). |
| **parentheses-1.0-240M-tanzanite** | **Certified Golden Release** | **Released November 2026 under Apache 2.0. DirectML NPU ONNX export, 8 external benchmarks evaluated. Available at https://huggingface.co/uncoalesced-org/parentheses-1.0-240M.** |
| parentheses-1.0-multimodal | In progress | Continuous acoustic and speech token integration (Project Tanzanite Phase 2). |

## Where 0.9 stands

The 0.9 preview is the reason the 1.0 plan looks the way it does, so we want to be specific about what it can and cannot do.

Some parts are solid. The model generates about 27 tokens per second on the laptop GPU, and the speed stays flat as the answer gets longer. Scripts come out right in 40 of 40 checks for the five main languages, and 53 of 72 across all nine. In 72 runs it never leaked a control token into the output, and it refused none of 19 sensitive but legitimate technical prompts. The ONNX export runs under DirectML and matches the PyTorch model to within about 2e-5.

The weak parts are the ones that matter most. Held-out bits per byte, where lower is better, run from 0.44 in Malayalam and 0.45 in Tamil up to 1.57 in Kuvi. English scores 1.21. That looks worse than the main Dravidian languages, but it isn't a fair comparison: bits per byte divides by bytes, and each Indic letter takes three bytes in UTF-8 where an English letter takes one, so Indic text scores lower per byte. In practice English is the most fluent part of its chat. Koya, Kuvi and Kolami are scored on only 12 validation documents each, so treat those three numbers as rough.

It hallucinates. It once generated a claim that a Martian colony was founded in 1842, and it gives fluent, confident answers that are simply wrong. At 148M parameters this was expected, and it was a primary target for remediation. The `<think>` block opened in 36 of 72 runs and closed by itself in only 4 of those 36.

On a set of fresh reasoning prompts, it scored 5 of 24. An earlier score was 15 of 24, but 18 of those prompts had been copied from training data, so that number does not count.

Kannada chat is weak. Kannada prompts often get echoed back or answered in English, and multi-step arithmetic falls apart. Plain sums like "What is 17 x 6?" are answered by a built-in exact calculator, not by the model, so do not read that as the model doing maths. Low-resource languages sit around 74 percent script accuracy.

One bug needs a mention. For a while, `scripts/chat.py` ignored `slab_v3` in `config.json` when it loaded the Hugging Face folder, and it silently dropped 63 weight tensors. That produced degraded output. It is fixed, and chat.py now prints a warning if any tensors fail to load. If you cloned before the fix, pull.

Those weak spots drove the 1.0 design: more and cleaner training text, a training mix that keeps the Dravidian languages from slipping while English is added, and a real way to teach the model to say "I don't know."

## parentheses-1.0-240M

parentheses-1.0-240M is grown from 0.9 rather than trained from scratch. It uses the slab_v4 architecture: 16 layers, model width 1024, 16 heads with rank 1024 per layer, and a tri-channel boundary reset.

<p align="center"><img src="figures/training-240m-sessions.svg" alt="Train loss across the four 240M sessions" width="760"></p>

The chart shows train loss across Sessions 1 to 4. Loss falls quickly in Session 1, then creeps from about 3.1 to about 2.9 across Sessions 2 and 3. At step 6,001 it drops to around 2.0 within a hundred steps. That cliff is the data switching to STEM and code, which is easier to predict, so it isn't progress. The chart also says nothing about the Dravidian languages; held-out bits per byte, below, is the benchmark metric for those. Sessions 5 onward aren't on the chart yet.

### Sessions 1 and 2

Session 1 covered steps 1 to 2,000 and ended at validation loss 3.3110. Session 2 ran from 2,001 to 4,000 at the full 1024 width. Both were plain pretraining; the interesting changes start in Session 3.

### Session 3

Session 3 picked up at step 4,000 and ran to step 6,000. Two things changed. We began interleaving two memory-mapped sources: a reasoning-heavy set, and a large multilingual pool with Indic and Dravidian text. The reasoning set went first at a 10 to 6 ratio, then the ratio moved to 6 to 10 after step 4,250. The goal was more pretraining throughput on the Dravidian pool without losing the reasoning, and without shocking the low-resource embeddings. We also tracked two validation losses at once, one for general multilingual perplexity and one for reasoning. The session ended at validation loss 3.1120, perplexity 22.47, with 393.22M tokens seen by then. That validation set isn't the one Session 1 used, so 3.3110 and 3.1120 can't be compared directly.

### Session 4

Session 4 covered steps 6,001 to 8,000 and used 131.07M tokens. The point was to add STEM, maths and code, and the data came from permissively licensed sets: Bespoke-Stratos, Orca-Math, MetaMathQA, NuminaMath, SmolTalk and Python-Edu.

It made things worse on held-out data. Bits per byte went up in all nine languages. The largest relative increases were Kuvi, up about 7 percent, and Kolami, up 5.9 percent. Kannada went from 0.489 to 0.500, Tamil from 0.482 to 0.496, Telugu from 0.579 to 0.598 and Malayalam from 0.400 to 0.414. Session 4 added the STEM, maths and code, and it cost ground on Dravidian text, which is the core focus of the project.

### Session 5

Session 5 covers steps 8,001 to 10,000 and is nearly finished. English textbook text from Cosmopedia-v2 and FineWeb-Edu is the primary data. Replay from our Dravidian and English set goes up to 8 of every 16 micro-steps, from 6 in Session 4. That is our direct response to the regression. The learning rate stays constant at 3e-4, and the one anneal happens in Session 6. Until its held-out bits per byte are measured, we do not know whether the bigger replay share stopped the damage.

### Sessions 6 to 8

Session 6 is the only learning-rate anneal in the run. SFT v5 and Session 8 come after it, and they are covered in the next section.

## Teaching it to say I don't know

Every version gets the same three goals: never fabricate, admit when evidence is missing, and do not refuse legitimate questions. The third goal is the easiest to lose. A model trained to hedge can start refusing ordinary questions, so we measure refusals alongside abstention.

0.9 already had a version of this. Its DPO v2 pass used 900 preference pairs, and 400 of them were abstention pairs across five languages, where the preferred answer admits it does not know. The pass moved the model a little (a whole-set margin of +0.3507 and 84.2 percent preference accuracy). It still invented a Martian colony. A pass that small was never going to be enough.

For 1.0, the SFT v5 pass adds about 8,000 abstention-aware reasoning traces, where the reasoning ends with an honest admission when evidence is missing. Session 8 is on-policy DPO v2. The model writes its own candidate answers, preference training runs on those, and a 0.1 times NLL term keeps the model from drifting. Checkpoints are then averaged (SWA-3) for stability.

When evaluating at 240M, hallucination, abstention and refusal rates have to be read together, because a model that refuses everything would pass the first check and fail the third.

## The data problem

On 2026-11-02 we established a strict rule for all releases. Code, tokenizers and weights go out under Apache 2.0, and a dataset can be used for training only if its license permits commercial use. The full policy is in [LICENSE_POLICY.md](../LICENSE_POLICY.md). The 0.9 preview weights remain CC BY-NC 4.0, so 0.9 stays non-commercial. The rule applies from 1.0 onward.

This hits Kannada hardest. Most of the available Kannada and Dravidian text corpora carry non-commercial terms, such as CC BY-NC or research-only conditions. Under the rule we cannot train on them for release models.

What helps is commercially licensed Kannada and Dravidian text, meaning sources released under CC BY, CC BY-SA, public domain or ODbL. First-party synthetic text authored directly also qualifies. Another route is data generated with a teacher model whose license allows commercial use. The policy names Apache-2.0 Qwen models as an example of a usable teacher. Outputs from Llama and from trial keys are excluded, and we record each teacher's terms before a full run.

## Further out

These are intentions, not promises, and none of them has a date.

Expanding conversational data for multi-turn chat is planned, enabling the model to hold conversations across several turns.

We also target better coverage for the low-resource languages. Tulu, Koya, Kuvi and Kolami are marked experimental, and their scores rest on limited validation text. More licensed text for them, verified by native speakers, is the path forward.

## How you can help

We develop parentheses on an 8 GB laptop GPU. Contributions across several domains are welcome, and the problems below are the place to start.

The biggest one is commercially licensed Kannada and Dravidian text, described above. Native-speaker checks come next. The script checks are automatic, but whether an answer in Tulu or Kuvi reads naturally needs someone who speaks the language. Setup reports matter too. We test on Windows with an NVIDIA GPU, so reports from Linux, macOS or AMD machines are useful. The `--device mps` path on Apple Silicon is experimental.

If `--device cuda` fails with "Torch not compiled with CUDA enabled", you probably have the CPU-only PyTorch build. [SETUP.md](SETUP.md) has the fix, and the same file has the verified install and run commands.

For bugs, open an issue at https://github.com/uncoalesced/parentheses/issues. The most useful reports include the command you ran, your Python and PyTorch versions, your GPU, and the full output.

---

Related: [README](../README.md), [Setup](SETUP.md), [Architecture](ARCHITECTURE.md), [Documentation](DOCUMENTATION.md), [Contributing](../CONTRIBUTING.md)
