<div align="center">
  <img src="assets/parentheses_lockup_transparent.svg" alt="parentheses" width="300">
</div>

<h3 align="center">History</h3>

<p align="center">
  How parentheses got from a transformer on a computer that later failed to a recurrent model on a laptop.
</p>

<p align="center">
  <a href="README.md">README</a> •
  <a href="docs/BENCHMARK_SPEC.md">Benchmark protocol</a> •
  <a href="docs/LICENSE_TABLE.md">Licence tiers</a>
</p>

---

This covers what each version was, why it was replaced, and the mistakes along the way, including the ones I'm not proud of.

<p align="center"><img src="docs/figures/history-growth.svg" alt="Parameter count of each version from 0.2 to 1.0, rising from about 300 thousand to about 254 million over nine weeks" width="760"></p>

The git history of this repository was reset to a single commit in November 2026, because old commits carried private names and file paths. This file is the record of what came before.

## A note on version numbers

Only 0.1, 0.9 and 1.0 were names I used at the time. In 2026 I called everything "0.9" from the first day, because 0.9 was the plan: prove the architecture on a laptop, then build 1.0. Looking back, the 2026 work went through clear stages, so I've given them the numbers 0.2 to 0.8 here, after the fact. Each one is listed with the name it actually ran under, so you can match it to the code and the logs.

## At a glance

| Version | When | What it was | Run name at the time |
|---|---|---|---|
| 0.1 | September 2024 | 64M-parameter transformer on a GTX 1050 (4 GB) | parentheses 0.1 |
| 0.2 | 31 Aug 2026 | ~300k byte-level transformer, English then 22 languages | preset parentheses-0.9-300k; runs multilingual, multilingual-22, all-sources-v1 |
| 0.3 | 5 Sep 2026 | Same size, switched to selective linear attention | selective-v1 (preset parentheses-0.9-300k-selective) |
| 0.4 | 1 to 7 Oct 2026 | Dravidian-first byte-level runs | dravidian-v1 to v3 |
| 0.5 | 1 to 9 Oct 2026 | First BPE tokenizer, 41M parameters | 0.9-41M |
| 0.6 | 9 Oct 2026 | First attempt at a chat model | 0.9-41M-sft-v1, v2 |
| 0.7 | 10 to 21 Oct 2026 | 120M, grown from 41M, deep and narrow | 0.9-120M |
| 0.8 | 22 Oct 2026 | 130M, wide and shallow, first try | 0.9-130M-v2 |
| 0.9 | 22 to 30 Oct 2026 | 130M with the slab_v3 fixes, released | 0.9-130M-v3 |
| 1.0 | 28 Oct to mid-Nov 2026 | 254M (called 240M), grown from 120M, released | 1.0-240M |
| 1.1 | planned | 500M+, adds image input and speech | 1.1 |

## 0.1: the transformer that got away (2024)

I started parentheses in September 2024. Version 0.1 was a 64-million-parameter transformer: all the standard parts, trained on a GTX 1050 with 4 GB of memory. For the time, and for that card, it was honestly pretty good. It could hold a conversation.

It also taught me most of what I now try not to do. The training data was poor. A lot of it was old data I'd pulled from Google Cloud that wasn't reliable and was badly out of date, and it gave the model a "yes-man" way of talking, agreeable and GPT-like rather than useful. Worse, I didn't check licences. I used whatever I found, including English books under non-commercial licences that I had no right to train a model on and release.

The project got more complex than I could handle as someone new to machine learning, and then the computer failed, taking the hard drive with it. The model is technically still on that drive, but I can't get it off. 0.1 is gone, and in some ways that's fine: I couldn't have released it with that data anyway.

## 0.2: starting again on a laptop (31 August 2026)

The GTX 1050 wasn't coming back, so in late August 2026 I moved everything to a laptop with an NVIDIA RTX 5050 and 8 GB of memory. The plan was in two parts. Plan One, which I called 0.9, would prove an architecture and a data pipeline at small scale. Plan Two, 1.0, would be the real model.

The first runs were tiny: a byte-level transformer of about 300,000 parameters, five layers, four heads. It trained on Simple English Wikipedia and 50 Project Gutenberg books, about 265 million bytes, then on English plus 20 languages, then 22. The last of these, all-sources-v1, trained on 29 sources and about 3 billion tokens.

The first bug worth remembering came here. My validation split was the last 1% of a stream sorted alphabetically by language, which meant the validation set was 100% Chinese. Every validation number before I caught it was meaningless. It's a small thing, but it's why I now check what's actually in an evaluation set before trusting a score.

## 0.3: leaving the transformer (5 September 2026)

On 5 September I replaced attention. The reason was a feature I wanted: FreeThink, a mode where the model keeps generating and reading retrieved text in a long stream. A transformer's cost per token grows with everything before it. A recurrent model with a fixed-size state costs the same for the millionth token as for the first, which is exactly what a long stream needs.

So I swapped the transformer for selective linear attention, the recurrent family that Mamba-2 and RWKV belong to, and kept the size and the all-sources-v1 data the same for a fair comparison. The recurrent version came out slightly worse (validation loss 1.4905 against the transformer's 1.4784). I switched anyway, because the long-stream behaviour mattered more than a gap of under 1% at 300,000 parameters, and the gap was something I expected to close with better design. The transformer code left the repository later in September.

Around the same time I made the decision that shaped everything after: the project would be Dravidian-first. Kannada first, because I'm in Bengaluru, then Tamil, Telugu and Malayalam, with English alongside rather than in charge.

## 0.4: Dravidian-first, byte by byte (1 to 7 October 2026)

The first full Dravidian runs were still byte-level and still around 300,000 parameters, now on Dravidian text with English replay and slices of four smaller languages. The first run died when the laptop went to sleep overnight. It's a very laptop problem, and I resumed it on 6 October.

This is also where I made a licensing mistake I want to be clear about. One of the low-resource slices was Aheri Gondi text that is all rights reserved and research-only. It went into the dravidian-v3 mix. Once I understood the terms, the rule became that no model trained on it can be released, and it has been excluded from every run since. v3 can never be released, and I haven't released any of the 0.4 byte-level models.

## 0.5: a real tokenizer and 41M parameters (1 to 9 October 2026)

Feeding the model raw bytes was simple but wasteful, and I didn't want a standard tokenizer either. Standard GPT-style tokenizers cut Kannada words in the middle of letters. I built a 32,000-token byte-level BPE vocabulary whose splitting rule keeps vowel signs and viramas attached to the letter they belong to. The same tokenizer is still in use and ships in this repository.

With the tokenizer came the first serious size: 40.9 million parameters, 16 layers, width 384. I picked that size with a fixed benchmark rule rather than by feel, and it became the base for everything that followed. The longest run finished on 9 October.

## 0.6: the first chat attempt (9 October 2026)

On top of the 41M model I tried instruction tuning to make it chat. It didn't work in the way that mattered most: the Kannada was bad. Looking at the data explained why. Kannada made up only 1.2% of the chat training mix. A model can't learn to talk in a language it barely sees in that stage, however much of it saw in pretraining. That number changed how I build every mix.

## 0.7: growing to 120M (10 to 21 October 2026)

For 120M I didn't start from scratch. I grew the 41M model by widening it in a way that keeps its behaviour, so it started out already knowing what 41M knew. The result had 32 layers at width 512, about 120 million parameters.

The first training run at the usual learning rate damaged the grown model instead of improving it, so I restarted at a third of that rate. Over five sessions it saw 1.25 billion tokens of pretraining.

The chat side of 120M was a long run of honesty work. The first instruction-tuning set turned out to leak about half of my evaluation questions into training, so its scores were worthless and I threw it out. Versions after that focused on getting the model to say "I don't know". Along the way I also removed 242 lines of hard-coded answer rules from the chat script. They made the demo look better and the model no smarter, so they had to go. I also stopped trusting one evaluation pipeline after an audit showed its judging wasn't what it claimed to be.

What finally ended 120M was its shape. Thirty-two narrow layers made the recurrent signal hard to train: gradients differed by a factor of 23.5 between layer 0 and layer 15. Some of the state arithmetic was also losing precision in 16-bit floats.

## 0.8: wide and shallow (22 October 2026)

The fix was to go wide instead of deep: 16 layers at width 768, about 148 million parameters. That first version (130M-v2) was replaced by v3 the same day it was registered. The same day, I added four changes that became slab_v3, and that became 0.9.

## 0.9: the release (22 to 30 October 2026)

parentheses-0.9-130M is the 0.8 layout plus slab_v3: token-shift mixing, a learned write gate, a value highway from the first layer, and 32-bit precision for the decay scan. It pretrained on about 1.5 billion token positions on the laptop, then went through instruction tuning, preference tuning and checkpoint averaging.

Before release I audited my own evaluation and found the problem from 0.7 again in a new place: 18 of the 24 prompts in my honest reasoning suite were copies of training prompts. The score went from 15 of 24 to 5 of 24 once I used fresh prompts. I published the 5.

I released it on GitHub and Hugging Face at the end of October. The weights are CC BY-NC 4.0, non-commercial, and there's a specific reason. Some of the instruction data I used carried non-commercial licences, namely Camel-AI, SciQ and No Robots, and some Karya data had no licence stated at all. I should have caught that before training, not after. Releasing the weights as non-commercial was the honest way to ship what I'd built without claiming rights I didn't have.

Right after release, someone on Bluesky tried it and got gibberish. That turned out to be a bug in the chat script: it skipped 63 of the model's weights when loading the Hugging Face files. It's fixed now, and it's why the setup guide got a troubleshooting section.

## The licence rule (2 November 2026)

Three versions in a row had a licence problem: 0.1's books, 0.4's Aheri Gondi text, 0.9's instruction data. On 2 November I made the rule I should have started with. Everything I release from now on is under Apache 2.0, and a dataset can be used for training only if its licence permits commercial use. No non-commercial, no research-only, no "no licence stated", no text generated by Llama models. 1.0 was trained under that rule.

It costs a lot. Most good Kannada text is non-commercial, so the rule shuts out exactly the data this project needs most. But it means anyone can use what comes next without worrying about where it came from.

## 1.0: the 254M model (28 October to mid-November 2026)

1.0 is 16 layers at width 1024 with 16 heads, on the slab_v4 version of the architecture. It's called 240M, and the exact count is 253,544,176 parameters. It was grown from the 120M checkpoint by width expansion, the same behaviour-preserving trick that took 41M to 120M, and trained over ten sessions on the same laptop. The released weights are the checkpoint at step 14,173.

The fourth session, which added STEM, maths and code, made the Dravidian languages measurably worse within a hundred steps. Later sessions put multilingual replay back in to fix it. That's where the rule came from that every training phase carries a floor of replay data for every language.

What 1.0 can't do is the list 1.1 is built around. It gets grade-school maths wrong almost every time. Its thinking mode restates the question more than it reasons. It generates around 20 to 30 tokens a second, which is slow for its size. And when a prompt is a little vague it tends to say "I don't know" when it could have asked what I meant.

## The benchmark tables (November 2026)

After the 1.0 release, people read my benchmark script and told me what was wrong with it. They were right, so here it is plainly.

Each benchmark used the first 30 questions. "MMLU" was a single subject. The MBPP score only checked that the code compiled and never ran the tests. TruthfulQA and the terminal tasks were scored by looking for keywords, and an answer could earn credit just for not being empty. The headline numbers were the best of four runs, three of them sampled at temperature 0.7. One paragraph of generated commentary called a Belebele score "solid" when the average was 17% on a four-choice test, which is worse than guessing. And the comparison tables put those numbers next to other labs' full-benchmark results, which made the differences meaningless.

I didn't set out to mislead anyone, but the result overstated the model, and that's the same mistake as the hard-coded answers in 0.7 wearing different clothes. The 1.0 documents stay as published so the record is intact. From 1.1 on, scores come from lm-evaluation-harness on full test sets, greedy, one run, with the raw output committed, and I read the results myself before any number goes into a document. The rules are in [docs/BENCHMARK_SPEC.md](docs/BENCHMARK_SPEC.md). My own held-out prompts are still useful to me, but I wrote them and tuned against them, so they aren't a benchmark and won't be presented as one.

## The licence rule, revised for 1.1

The 2 November rule was strict, and keeping it meant maintaining citations, links and licence text for every source in public. For one person that turned into more bookkeeping than model building. So 1.1 changes what is promised.

1.1 is open weights. The weights are Apache 2.0 and the inference code is MIT. The training data and the training pipeline are not published. Every source is still logged privately, with its licence, revision and hash, and the repository publishes a summary by licence category in [docs/LICENSE_TABLE.md](docs/LICENSE_TABLE.md). Non-commercial data and data whose authors say not to train on it stay out. Text with no licence, and text written by other models, can only go in one source at a time, with the decision written down.

This is a weaker promise than 1.0 made, and I'd rather say so than have someone find out. If you need a model whose every input is permissively licensed and listed, 1.0 is that model and 1.1 is not.

## What I'd tell myself in 2024

Check the licence before you download the data, not after you've trained on it. Look inside your evaluation set before you trust a score from it. Don't make the demo look better than the model is. And don't score your own model with your own script.

---

Related: [README](README.md), [Benchmark protocol](docs/BENCHMARK_SPEC.md), [Licence tiers](docs/LICENSE_TABLE.md)

<p align="center"><sub>Engineered by uncoalesced</sub></p>
