# Parentheses 0.9: Training Data Ledger

This lists every dataset that went into the released Parentheses 0.9 models, with token counts and licenses. Each license has a status:

- **Verified:** taken from the license tag on the source's Hugging Face Hub page (checked 2026-10-30).
- **Ledger:** taken from our internal data ledger, not independently re-verified.
- **Unknown:** no license is recorded.

Each GitHub release also has `training_data_ledger.jsonl` attached. It is the full machine ledger: every processed file with its byte size, SHA-256 where one was recorded, source, license and the training run that used it.

## License summary

| Model | Non-commercial or unlicensed sources | Effect |
| :--- | :--- | :--- |
| 0.9-120M | Session 5 SFT tail includes 2,965 synthetic distillation rows generated through the NVIDIA API trial, whose terms allow internal testing only and no distribution (tagged research-only in our ledger). All other verified sources are permissive or share-alike (CC BY-SA 4.0, ODbL). | Weights released for research and non-commercial use only. |
| 0.9-130M-v3 | Camel-AI Chemistry and Physics (CC BY-NC 4.0), SciQ (CC BY-NC 3.0), No Robots (CC BY-NC 4.0), OpenHermes (GPT-4 outputs, no license), Karya (no license recorded), NVIDIA NIM trial distillation (internal testing only) | Weights released for research and non-commercial use only. |

## Exclusions

- **Aheri Gondi (`esg`, Wycliffe Bible Translators, all rights reserved).** Used only in one unreleased byte-level experiment (`dravidian-v3-lowres`). It is excluded from every released model: all corpora and SFT sets contain 0 `esg` tokens.
- **Kannada literature harvest** (mostly still in copyright). Not used in any released model.

## Parentheses 0.9-120M

119.7M parameters. Pretrained on 1.25B tokens over 5 sessions, then fine-tuned with SFT v4.

### Pretraining corpus (Corpus v5, ~1.08B tokens)

| Source | Content | Tokens | License | Status |
| :--- | :--- | ---: | :--- | :--- |
| [AI4Bharat Sangraha (verified)](https://huggingface.co/datasets/ai4bharat/sangraha) | Kannada, Tamil, Telugu, Malayalam | 497.2M | CC BY 4.0 | Verified |
| [Wikimedia](https://tcy.wikipedia.org) Tulu Wikipedia + Wikisource | Tulu (`tcy`) | 17.0M | CC BY-SA 4.0 | Ledger |
| eBible, The Word for the World | Koya (`kff`), Kuvi (`kxv`), Kolami (`nit`) | 10.3M | CC BY-SA 4.0 | Ledger |
| [Project Gutenberg](https://www.gutenberg.org) | English literature | part of the English share | Public domain (US) | Ledger |
| [Proof-Pile-2](https://huggingface.co/datasets/EleutherAI/proof-pile-2) (OpenWebMath) | English math | part of the English share | ODC-By 1.0 | Ledger |
| [Common Pile StackExchange](https://huggingface.co/datasets/common-pile/stackexchange) (CS, hardware, robotics) | Code and technical Q&A | 33.9M | CC BY-SA | Ledger |
| OPUS parallel corpora | English-Indic sentence pairs | part of the English share | Mixed, per corpus | Ledger |

English is 47.3% of Corpus v5. Hindi is 9.0M tokens.

### Session 3 + 4 QA and dictionary lanes (138.5M tokens)

| Source | License | Status |
| :--- | :--- | :--- |
| [TyDi QA](https://huggingface.co/datasets/google-research-datasets/tydiqa) | Apache-2.0 | Verified |
| [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | CC BY-SA 4.0 | Verified |
| Alar (Kannada-English dictionary) | ODbL | Ledger |
| Olam (Malayalam-English dictionary) | ODbL | Ledger |
| Kaikki (Wiktionary extracts) | CC BY-SA | Ledger |
| WordNet | WordNet license (BSD-style) | Ledger |

### Session 5 SFT tail (62.6M tokens) and SFT v2-v4 (~52M tokens)

| Source | License | Status |
| :--- | :--- | :--- |
| [Aya Dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Apache-2.0 | Verified |
| [IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA) | CC BY 4.0 | Verified |
| [OpenAssistant OASST1/OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) | Apache-2.0 | Verified |
| [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | Apache-2.0 | Verified |
| [Hermes Function-Calling v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Apache-2.0 | Verified |
| [Glaive Function-Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | Verified |
| [ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | Verified |
| [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | MIT | Verified |
| Synthetic distillation, 2,965 rows (NVIDIA Nemotron and GLM models via NVIDIA API trial) | **Research-only** (trial terms: internal testing, no distribution) | Ledger |

The SFT v2-v4 lane builders skip rows tagged research-only, but SFT v4 continues from the Session 5 checkpoint, which saw the distillation rows above.

## Parentheses 0.9-130M-v3

148.38M parameters. Pretrained on `processed_130m_v1` (808M tokens), then SFT v2 (131.07M tokens) and DPO v2.

### Pretraining pillars (`processed_130m_v1`, target shares)

| Pillar | Share | Source | License | Status |
| :--- | ---: | :--- | :--- | :--- |
| Dravidian core | 40% | Sangraha verified (kan, tam, tel, mal) + Tulu Wikimedia + eBible (kff, kxv, nit) | CC BY 4.0 / CC BY-SA 4.0 | Verified / Ledger |
| STEM reasoning | 25% | [Camel-AI Chemistry](https://huggingface.co/datasets/camel-ai/chemistry), [Camel-AI Physics](https://huggingface.co/datasets/camel-ai/physics) | **CC BY-NC 4.0** | Verified |
| | | [SciQ](https://huggingface.co/datasets/allenai/sciq) | **CC BY-NC 3.0** | Verified |
| | | [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | MIT | Verified |
| | | [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | Apache-2.0 | Verified |
| Conversational / open data | 20% | [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots) | **CC BY-NC 4.0** | Verified |
| | | [OpenHermes](https://huggingface.co/datasets/teknium/openhermes) (50k subset) | **None stated (GPT-4 outputs)** | Verified (no tag) |
| | | [Self-OSS-Instruct SC2 50k](https://huggingface.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k) | ODC-By | Verified |
| | | Karya (karya.in) conversational text, 152k tokens | **Unknown** | Ledger |
| | | NVIDIA NIM synthetic distillation (Nemotron) | **Research-only** (NVIDIA API trial terms) | Ledger |
| English anchors | 15% | Project Gutenberg | Public domain (US) | Ledger |

SFT v2 (`processed_sft_stem_open_v2`) and the DPO v2 preference pairs are built from the same STEM and open-data pools, so the same licenses apply.

## Attribution

Data from: AI4Bharat (Sangraha, IndicQA), Wikimedia Foundation, eBible.org / The Word for the World, Project Gutenberg, EleutherAI, Common Pile, OPUS, Cohere Labs (Aya), OpenAssistant, Bespoke Labs, Nous Research, Glaive AI, Team-ACE, Anthropic, Google Research (TyDi QA), HotpotQA, Alar, Olam, Kaikki/Wiktionary, Princeton WordNet, CAMEL-AI, Allen Institute for AI (SciQ), Hugging Face H4 (No Robots), Teknium (OpenHermes), BigCode, Meta-Math, Karya, NVIDIA.

Share-alike sources (CC BY-SA, ODbL) keep their licenses in any redistribution of derived data.

*Engineered by uncoalesced*
