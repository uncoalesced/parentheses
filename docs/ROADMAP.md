# Roadmap

parentheses is a small recurrent language model for Kannada, Tamil, Telugu, Malayalam and English, trained on one 8 GB laptop GPU. The current release is parentheses-1.0-240M-tanzanite, with 253,544,176 parameters. parentheses-1.1 is in development in this repository. It keeps the 1.0 weights as a starting point and grows past 500M parameters. This roadmap covers what is done, what is being measured, what is still open, and what is only a target. Dates are targets, not commitments.

## At a glance

| Milestone | Status | Notes |
|---|---|---|
| parentheses-1.0-240M-tanzanite | Released, November 2026 | Weights Apache 2.0, code MIT. Available at https://huggingface.co/uncoalesced-org/parentheses-1.0-240M. |
| 1.1 runtime and tooling | Done | Clean Python rewrite, multi-turn chat state, chat CLI, self-test, lm-evaluation-harness adapter. |
| 1.1 measurements | In progress | How long 1.0 remembers a fact across a conversation, and tokenizer cost per language. |
| parentheses-1.1 (500M+, image, speech, TTS) | In development | Target mid-December 2026. This date may slip. |
| Parentheses family of three models | Planned | Target early 2027. |

## Where 1.0 stands

1.0 is the reason 1.1 looks the way it does, so this section is specific about what it cannot do.

It is a recurrent model with 16 layers, width 1024 and 16 heads. It is fluent in all five languages, and it is often confidently wrong. It knows less than its fluency suggests.

It was trained on one question and one answer at a time. Follow-up questions in the same chat usually get the previous answer again.

Its arithmetic and reasoning are weak. It gets grade-school maths wrong most of the time, and its thinking mode restates the question more than it reasons about it.

When a question is vague, it tends to say "I don't know" when it could have asked what the user meant.

It takes text only. It generates around 20 to 30 tokens per second, which is slow for its size.

Benchmark tables published with 1.0 are not a fair measure of the model. See [HISTORY.md](../HISTORY.md#the-benchmark-tables-november-2026) for what went wrong.

## Targets for 1.1

These are targets. None of them is met yet.

- **More knowledge.** Grow the 1.0 weights past 500M parameters, and train on more licensed text. See [The data](#the-data) below.
- **Multi-turn chat.** 1.0 saw only single-turn chats. 1.1 should be trained on conversations that run across several turns.
- **Better arithmetic and reasoning.** Less restating of the question, and more working through it.
- **Asking before giving up.** On a vague question, the model should ask what was meant before it says it does not know.
- **More modalities.** Image input, text-to-speech and speech input. Each one is planned for release with 1.1.
- **No attention.** 1.1 keeps the recurrent design. The state stays fixed in size however long the conversation runs, and no attention layer is added anywhere.

## Progress on 1.1

Done:

- A clean Python rewrite of the model. Its outputs match 1.0 exactly.
- Prefill-then-step generation.
- `ChatSession`, which carries memory between chat turns.
- A CUDA-graph decode step.
- A chat CLI and a self-test.
- An lm-evaluation-harness adapter.

In progress:

- Re-measuring 1.0 with lm-evaluation-harness. The run is done. The results are under review and are not published. Numbers will be published only after review and under [BENCHMARK_SPEC.md](BENCHMARK_SPEC.md).
- How long 1.0 keeps a fact across a conversation.
- Tokenizer cost per language.

Open decisions:

- Whether to keep the `v3_32k` tokenizer. The tokenizer cost measurement decides this.
- What change to the recurrent memory the measurements call for. No change has been chosen yet.

## Teaching it to say I don't know

Every version has the same three goals: never fabricate, admit when evidence is missing, and do not refuse legitimate questions. The third goal is the easiest to lose. A model trained to hedge can start refusing ordinary questions, so refusals are measured alongside abstention.

For 1.1 the new target is the vague-question case. The model should ask a clarifying question before it gives up. How 1.1 will be trained for this is not decided yet.

When evaluating, hallucination, abstention and refusal rates have to be read together. A model that refuses everything would pass the first check and fail the third.

## The data

1.1 uses a looser licence promise than 1.0. The inference code is MIT and the weights are Apache 2.0. The training data and the training pipeline are not published. Each source is logged privately with its licence, revision and hash. [LICENSE_TABLE.md](LICENSE_TABLE.md) publishes a summary by licence category.

Non-commercial data, and data whose authors ask that it not be used for training, stays out. Text with no stated licence, and text written by other models, may be used one source at a time, with the decision written down.

This matters most for Kannada. Most good Kannada text is non-commercial, so commercially licensed Kannada and Dravidian text remains the main constraint on the 1.1 data. If every input must be permissively licensed and listed, 1.0 is the model to use. 1.1 does not make that promise.

## Further out

These are intentions, not promises, and none of them has a firm date.

A three-model Parentheses family is planned for early 2027. Sizes and schedule are not final.

Tulu, Koya, Kuvi and Kolami are marked experimental, and their scores rest on limited validation text. The path forward is more licensed text for them, checked by native speakers.

## How you can help

Development happens on an 8 GB laptop GPU. Contributions in several areas are welcome, and the problems below are a good place to start.

The biggest one is commercially licensed Kannada and Dravidian text. Native-speaker checks come next. Automatic checks can confirm the script, but whether an answer in Tulu or Kuvi reads naturally needs someone who speaks the language. Setup reports also help. Reports from Linux, macOS or AMD machines are useful, since most testing runs on Windows with an NVIDIA GPU.

If `--device cuda` fails with "Torch not compiled with CUDA enabled", the CPU-only PyTorch build is probably installed. Reinstall PyTorch with CUDA support.

For bugs, open an issue at https://github.com/uncoalesced/parentheses/issues. The most useful reports include the command you ran, your Python and PyTorch versions, your GPU, and the full output.

---

Related: [README](../README.md), [Setup guide](SETUP.md), [How it works](ARCHITECTURE.md), [Technical reference](DOCUMENTATION.md), [Model card](../MODEL_CARD.md), [Release notes](../RELEASE_NOTES.md), [Benchmark protocol](BENCHMARK_SPEC.md), [Licence table](LICENSE_TABLE.md), [History](../HISTORY.md)
