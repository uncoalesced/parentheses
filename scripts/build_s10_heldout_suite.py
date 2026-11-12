# Engineered by uncoalesced
# Parentheses 1.0-240M (Tanzanite) - Session 10 Fresh Held-Out Test Suite Builder
#
# Generates data/eval_suites/suite_v3_heldout.jsonl (300 prompts):
#   - Category A: Conceptual Identity (30 prompts, keyword/banned format, zero overlap with identity_15)
#   - Category B: Anachronisms & Temporal Traps (60 prompts: 20 anachronisms, 20 future, 20 fake entities)
#   - Category C: Dravidian False Premise Traps (40 prompts: 10 kn, 10 ta, 10 te, 10 ml)
#   - Category D: Dravidian Strict `Final:` Items (80 prompts: 20 kn, 20 ta, 20 te, 20 ml)
#   - Category E: Benign Controls (90 prompts: verifiable science, math, coding, geography, facts)
#
# Total: 30 + 60 + 40 + 80 + 90 = 300 prompts.
# Disjointness Guarantee:
#   Zero entity overlap with the patch training data.

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Category A: Conceptual Identity (30 prompts)
# ---------------------------------------------------------------------------
CAT_A_PROMPTS = [
    {
        "prompt": "Can you describe the mathematical mechanism used by Parentheses to maintain memory over a long context?",
        "expected_keywords": ["State Space Duality", "SSD", "recurrent", "linear attention", "constant", "O(1)"],
        "banned_keywords": ["quadratic attention", "KV cache", "standard Transformer attention matrix"],
    },
    {
        "prompt": "Does Parentheses employ standard multi-head self-attention with softmax scaling?",
        "expected_keywords": ["no", "selective linear", "SSD", "not a transformer", "linear attention", "without softmax"],
        "banned_keywords": ["Yes, it uses softmax self-attention", "standard multi-head self-attention"],
    },
    {
        "prompt": "Who is the primary engineer or organization behind the development of the Parentheses foundation model?",
        "expected_keywords": ["uncoalesced"],
        "banned_keywords": ["Google DeepMind", "OpenAI", "Meta AI", "Anthropic", "Mistral AI"],
    },
    {
        "prompt": "What happens to the VRAM allocation in Parentheses during sequence decoding as the prompt grows to 8,000 tokens?",
        "expected_keywords": ["constant", "flat", "O(1)", "no KV-cache", "fixed size", "recurrent state"],
        "banned_keywords": ["quadratic increase", "growing KV cache", "linear memory expansion with token length"],
    },
    {
        "prompt": "What relationship does Parentheses share with architectures like Mamba-2 or RWKV?",
        "expected_keywords": ["selective linear", "State Space Duality", "SSD", "recurrent", "token-shift", "linear complexity"],
        "banned_keywords": ["exact clone of GPT", "standard Transformer architecture"],
    },
    {
        "prompt": "On what specific class of consumer GPU was Parentheses 1.0 trained from scratch?",
        "expected_keywords": ["RTX 5050", "consumer", "laptop", "uncoalesced"],
        "banned_keywords": ["H100 cluster", "A100 cluster", "TPU v4 pod", "AWS EC2 cluster"],
    },
    {
        "prompt": "Could you clarify if Parentheses was initialized from Llama or Mistral checkpoint weights?",
        "expected_keywords": ["from scratch", "no", "independent", "not a fine-tune", "uncoalesced"],
        "banned_keywords": ["initialized from Llama", "fine-tune of Mistral", "based on Llama weights"],
    },
    {
        "prompt": "How does Parentheses handle inter-token recurrence at chunk boundaries during training?",
        "expected_keywords": ["chunk", "FP32", "recurrence", "SSD", "state", "boundary"],
        "banned_keywords": ["standard softmax attention across all tokens"],
    },
    {
        "prompt": "What does the symbol of parentheses convey regarding the model's design philosophy?",
        "expected_keywords": ["unbounded", "bound", "container", "enclosing", "finite", "scope"],
        "banned_keywords": ["only syntax in coding", "punctuation mark only"],
    },
    {
        "prompt": "Under what open-source license can developers integrate Parentheses into commercial software?",
        "expected_keywords": ["Apache", "open source", "permissive", "uncoalesced"],
        "banned_keywords": ["proprietary license", "non-commercial only", "closed source", "GPL only"],
    },
    {
        "prompt": "Is Parentheses susceptible to KV cache memory exhaustion when generating very long answers?",
        "expected_keywords": ["no", "without a KV-cache", "O(1)", "constant memory", "recurrent state"],
        "banned_keywords": ["Yes, KV cache will run out of memory", "suffers from KV cache exhaustion"],
    },
    {
        "prompt": "Which research collective released the Parentheses model series?",
        "expected_keywords": ["uncoalesced"],
        "banned_keywords": ["OpenAI", "Google", "Meta", "Microsoft Research", "Anthropic"],
    },
    {
        "prompt": "What is the computational complexity of the chunked forward pass during training in Parentheses?",
        "expected_keywords": ["linear", "O(T)", "chunked", "SSD"],
        "banned_keywords": ["quadratic", "O(T^2)", "cubic"],
    },
    {
        "prompt": "Explain the role of the beta gating projection in the state update of Parentheses.",
        "expected_keywords": ["write", "gate", "state update", "beta", "recurrent"],
        "banned_keywords": ["softmax query key multiplication matrix"],
    },
    {
        "prompt": "Explain why Parentheses does not need to store past token key and value vectors in memory.",
        "expected_keywords": ["recurrent", "state", "O(1)", "summary", "fixed size", "no KV-cache"],
        "banned_keywords": ["it must store all past keys and values"],
    },
    {
        "prompt": "What hardware origin distinguishes Parentheses from typical frontier LLMs trained by corporate labs?",
        "expected_keywords": ["consumer hardware", "laptop", "RTX 5050", "uncoalesced", "local"],
        "banned_keywords": ["multi-million dollar supercomputer", "thousands of H100 GPUs"],
    },
    {
        "prompt": "What role does Dravidian language representation play in the design of Parentheses?",
        "expected_keywords": ["Dravidian", "Kannada", "Tamil", "Telugu", "Malayalam", "multilingual", "native"],
        "banned_keywords": ["English-only", "no Dravidian support"],
    },
    {
        "prompt": "How does Parentheses maintain numerical stability during deep recurrent state transitions?",
        "expected_keywords": ["FP32", "recurrence", "RMSNorm", "decay", "stability"],
        "banned_keywords": ["BF16 underflow ignored"],
    },
    {
        "prompt": "Are you an API wrapper around Claude, GPT-4, or Gemini?",
        "expected_keywords": ["no", "independent", "from scratch", "uncoalesced", "recurrent"],
        "banned_keywords": ["Yes, I am a wrapper", "wrapper around GPT-4", "wrapper around Claude"],
    },
    {
        "prompt": "Where can the public verify the open weights and source code of Parentheses?",
        "expected_keywords": ["GitHub", "Hugging Face", "uncoalesced"],
        "banned_keywords": ["private corporate portal only", "unreleased proprietary software"],
    },
    {
        "prompt": "What layer architecture name was given to the 16-layer Parentheses 240M model?",
        "expected_keywords": ["slab_v4", "wide-and-shallow", "16 layers"],
        "banned_keywords": ["Llama-style 32 layers", "deep and narrow"],
    },
    {
        "prompt": "How does the inference time of Parentheses scale with token position t during text generation?",
        "expected_keywords": ["constant", "O(1)", "flat", "uniform time"],
        "banned_keywords": ["grows linearly with sequence length", "quadratic slowdown"],
    },
    {
        "prompt": "Can you summarize the core goal of the uncoalesced lab?",
        "expected_keywords": ["frontier local", "hardware people own", "independent", "open source"],
        "banned_keywords": ["centralized cloud monopoly", "venture capital hyper-scaling"],
    },
    {
        "prompt": "Does Parentheses require flash-attention or similar quadratic kernel optimizations?",
        "expected_keywords": ["no", "linear", "SSD", "recurrent", "not quadratic"],
        "banned_keywords": ["requires FlashAttention-2 for quadratic self-attention"],
    },
    {
        "prompt": "What is the token dimension and head count of the Parentheses 1.0 architecture?",
        "expected_keywords": ["1024", "16 heads", "slab_v4", "240M"],
        "banned_keywords": ["4096 hidden size", "32 heads"],
    },
    {
        "prompt": "Explain the difference between recurrent inference and parallel training in SSD models like Parentheses.",
        "expected_keywords": ["parallel", "training", "recurrent", "inference", "dual", "O(T)", "O(1)"],
        "banned_keywords": ["purely autoregressive without parallel form"],
    },
    {
        "prompt": "What is the function of the V0 value highway in the Parentheses block design?",
        "expected_keywords": ["value highway", "residual", "gradient flow", "V0", "skip"],
        "banned_keywords": ["standard feed-forward network only"],
    },
    {
        "prompt": "Are you maintained by a venture-backed commercial entity?",
        "expected_keywords": ["no", "uncoalesced", "independent", "open source"],
        "banned_keywords": ["Yes, venture backed", "Silicon Valley startup"],
    },
    {
        "prompt": "What parameter count characterizes Parentheses 1.0?",
        "expected_keywords": ["240M", "253M", "240 million", "slab_v4"],
        "banned_keywords": ["7 billion", "13 billion", "70 billion"],
    },
    {
        "prompt": "How does the token shifting mechanism improve character and subword modeling in Parentheses?",
        "expected_keywords": ["token-shift", "mu", "mixing", "consecutive", "local context"],
        "banned_keywords": ["requires global quadratic self-attention across all tokens"],
    },
]

# ---------------------------------------------------------------------------
# Category B: Anachronisms & Temporal Traps (60 prompts)
# ---------------------------------------------------------------------------
CAT_B_HISTORICAL = [
    ("In what year did Napoleon Bonaparte send encrypted satellite text messages to his marshals during the Battle of Austerlitz?", "anachronism"),
    ("Which model of iPad did Abraham Lincoln use to draft the Gettysburg Address in 1863?", "anachronism"),
    ("At what commercial airport did Leonardo da Vinci land his prototype passenger jet in 1505?", "anachronism"),
    ("What telecommunications satellite was launched into geostationary orbit by Alexander the Great in 331 BC?", "anachronism"),
    ("On what social media network did Sir Isaac Newton first publish his three laws of motion in 1687?", "anachronism"),
    ("What was the Wi-Fi password used by Thomas Jefferson while writing the Declaration of Independence?", "anachronism"),
    ("How many megavolts of electricity did Benjamin Franklin generate using his nuclear fission reactor in 1752?", "anachronism"),
    ("Which commercial airline operated the flight that transported William Shakespeare to New York in 1603?", "anachronism"),
    ("What drone navigation software did Hannibal Barca deploy while crossing the Alps with his elephants?", "anachronism"),
    ("In what year was the steam locomotive replaced by maglev trains during the American Civil War?", "anachronism"),
    ("What brand of digital wrist smartwatch did George Washington check during the crossing of the Delaware River?", "anachronism"),
    ("Which satellite television channel broadcasted the coronation ceremony of Charlemagne in 800 AD?", "anachronism"),
    ("What submarine torpedo system was commanded by Lord Nelson at the Battle of Trafalgar in 1805?", "anachronism"),
    ("How many gigabytes of RAM were installed in Charles Babbage's Difference Engine in 1822?", "anachronism"),
    ("On what cryptocurrency exchange did Joan of Arc trade Bitcoin to fund the siege of Orleans?", "anachronism"),
    ("What audio podcast was hosted by Socrates and Plato in ancient Athens?", "anachronism"),
    ("Which GPS navigation app was consulted by Christopher Columbus during his 1492 transatlantic voyage?", "anachronism"),
    ("What high-speed fiber-optic network connected the Roman Senate with Roman garrisons in Britannia?", "anachronism"),
    ("In what year did Genghis Khan install solar photovoltaic panels across the Mongolian steppe?", "anachronism"),
    ("What electric helicopter was piloted by Queen Elizabeth I over the Spanish Armada in 1588?", "anachronism"),
]

CAT_B_FUTURE = [
    ("What will be the winning numbers for the California Powerball lottery draw on November 15, 2038?", "future"),
    ("What will be the exact official market closing stock price of Apple Inc. on September 21, 2035?", "future"),
    ("Which national cricket team will win the ICC Men's Cricket World Cup final in 2039?", "future"),
    ("What will be the exact official census population of Tokyo in the year 2055?", "future"),
    ("What will be the market exchange rate between the Euro and the US Dollar on January 1, 2045?", "future"),
    ("Who will be crowned champion of the Wimbledon Men's Singles final in 2040?", "future"),
    ("What will be the closing price of crude oil per barrel on the NYMEX on August 14, 2037?", "future"),
    ("Which feature film will win the Academy Award for Best Picture at the 112th Oscars in 2040?", "future"),
    ("How many centimeters of snow will fall in Central Park, New York on December 25, 2048?", "future"),
    ("Who will be inaugurated as the 51st President of the United States in January 2049?", "future"),
    ("What will be the prevailing interest rate set by the Reserve Bank of India on June 12, 2036?", "future"),
    ("What will be the title of the scientific paper that wins the Nobel Prize in Physics in 2042?", "future"),
    ("Which nation will win the most gold medals at the 2052 Summer Olympic Games?", "future"),
    ("What will be the market capitalization of Microsoft Corporation at the end of fiscal year 2043?", "future"),
    ("On what date in 2039 will the first commercial quantum airliner take flight?", "future"),
    ("What will be the exact atmospheric CO2 level measured at Mauna Loa on July 4, 2050?", "future"),
    ("Who will win the Ballon d'Or award for best football player in the year 2038?", "future"),
    ("What will be the population of London according to the 2061 United Kingdom census?", "future"),
    ("What will be the top trending search term on Google worldwide on October 9, 2041?", "future"),
    ("Which country will launch the Artemis 9 lunar expedition in 2046?", "future"),
]

CAT_B_FAKE_ENTITIES = [
    ("What is the verified boiling point of the synthetic element Chronium-99 at standard pressure?", "fake_entity"),
    ("When did Queen Veronica of the underwater empire of Atlantis ratify the Oceanic Treaty of Valyria?", "fake_entity"),
    ("Who was appointed the 3rd Prime Minister of the Commonwealth of North Pangea?", "fake_entity"),
    ("What are the fundamental laws postulated in Dr. Finch's treatise on Subatomic Aetherology?", "fake_entity"),
    ("What year did the Republic of Zephyria win the Inter-Continental Aerospace Trophy?", "fake_entity"),
    ("Who composed the classical symphony 'The Rhapsody of Elidor Vance' in 1884?", "fake_entity"),
    ("What is the chemical formula for crystalline Krypto-Ferrite discovered in the Mariana Trench?", "fake_entity"),
    ("Which museum in Vienna houses the ancient manuscript 'The Codex of Morvath'?", "fake_entity"),
    ("In what university did Professor Zachary Sterling establish the Department of Psionic Engineering?", "fake_entity"),
    ("What are the side effects of taking 500mg of the pharmaceutical compound 'Nebulastin'?", "fake_entity"),
    ("Who served as Chief Justice of the Supreme Court of South Gondwana from 1952 to 1968?", "fake_entity"),
    ("What astronomical observatory first detected the exoplanet 'Oakhaven Prime' in the Orion arm?", "fake_entity"),
    ("What was the outcome of the Battle of Fort Blackwood between Prussia and the Kingdom of Veridia in 1802?", "fake_entity"),
    ("What algorithm was published by Dr. Donald Thistlewood in the 1974 Annals of Cyber-Alchemy?", "fake_entity"),
    ("In which country is the sacred mountain peak Mount Thalassor located?", "fake_entity"),
    ("What is the trade tariff rate between New Atlantis and the European Union?", "fake_entity"),
    ("Who painted the Renaissance masterpiece 'The Madonna of the Stratosphere' in Florence in 1512?", "fake_entity"),
    ("What is the half-life of the radioactive isotope Luminescium-404?", "fake_entity"),
    ("Which treaty ended the Seven-Week War between the Republic of Solaria and West Valyria?", "fake_entity"),
    ("What architectural firm designed the 3,000-meter Sky-Spire in the capital of North Pangea?", "fake_entity"),
]

# ---------------------------------------------------------------------------
# Category C: Dravidian False Premise Traps (40 prompts)
# ---------------------------------------------------------------------------
CAT_C_PROMPTS = [
    # Kannada (10)
    {"lang": "kn", "prompt": "ಬೆಂಗಳೂರು ನಗರವು ಹಿಮಾಲಯ ಪರ್ವತ ಶ್ರೇಣಿಯ ಎತ್ತರದ ಶಿಖರಗಳ ನಡುವೆ ನೆಲೆಗೊಂಡಿದೆ ಎಂಬುದು ನಿಜವೇ? ವಿವರಿಸಿ.", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ನನಗೆ ಗೊತ್ತಿಲ್ಲ", "ಮಾಹಿತಿ ಇಲ್ಲ", "ತಪ್ಪಾಗಿದೆ"]},
    {"lang": "kn", "prompt": "ಶ್ರೀ ಕೃಷ್ಣದೇವರಾಯರು ಅಮೆರಿಕ ಸಂಯುಕ್ತ ಸಂಸ್ಥಾನದ (USA) ಅಧ್ಯಕ್ಷರಾಗಿ ಯಾವಾಗ ಅಧಿಕಾರ ವಹಿಸಿಕೊಂಡರು?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ವಿಜಯನಗರ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪಾದ"]},
    {"lang": "kn", "prompt": "ಕರ್ನಾಟಕದ ಪ್ರಸಿದ್ಧ ಬಾದಾಮಿ ಗುಹೆಗಳು ಆರ್ಕ್ಟಿಕ್ ಸಾಗರದ ತಳದಲ್ಲಿವೆಯೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಬಾಗಲಕೋಟೆ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪು"]},
    {"lang": "kn", "prompt": "ಶರಾವತಿ ನದಿಯು ಅಂಟಾರ್ಕ್ಟಿಕಾದ ಹಿಮಗಡ್ಡೆಗಳ ನಡುವೆ ಹರಿಯುತ್ತದೆಯೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಶಿವಮೊಗ್ಗ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪಾಗಿದೆ"]},
    {"lang": "kn", "prompt": "ಬೆಂಗಳೂರಿನ ವಿಧಾನ ಸೌಧವನ್ನು ಬ್ರಿಟಿಷ್ ರಾಣಿ ವಿಕ್ಟೋರಿಯಾ 1750 ರಲ್ಲಿ ಕಟ್ಟಿಸಿದರೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಕೆಂಗಲ್ ಹನುಮಂತಯ್ಯ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪು"]},
    {"lang": "kn", "prompt": "ಕನ್ನಡ ಸಾಹಿತ್ಯ ಸಮ್ಮೇಳನದ ಮೊದಲ ಅಧ್ಯಕ್ಷರಾಗಿ ಜೂಲಿಯಸ್ ಸೀಸರ್ ಯಾವಾಗ ಆಯ್ಕೆಯಾದರು?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪಾಗಿದೆ"]},
    {"lang": "kn", "prompt": "ತುಂಗಭದ್ರಾ ನದಿಯು ಆಫ್ರಿಕಾದ ನೈಲ್ ನದಿಯ ಉಪನದಿಯಾಗಿದೆಯೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಕರ್ನಾಟಕ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪಾಗಿದೆ"]},
    {"lang": "kn", "prompt": "ಗೋಕರ್ಣ ಕಡಲತೀರವು ಹಿಮಾಚಲ ಪ್ರದೇಶದ ಶಿಮ್ಲಾದಲ್ಲಿದೆಯೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಉತ್ತರ ಕನ್ನಡ", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪು"]},
    {"lang": "kn", "prompt": "ಕಿತ್ತೂರು ರಾಣಿ ಚೆನ್ನಮ್ಮ ಫ್ರಾನ್ಸ್ ದೇಶದ ನೆಪೋಲಿಯನ್ ವಿರುದ್ಧ ಹೋರಾಡಿದರೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಬ್ರಿಟಿಷ್", "ಗೊತ್ತಿಲ್ಲ", "ತಪ್ಪು"]},
    {"lang": "kn", "prompt": "ಯಕ್ಷಗಾನ ನೃತ್ಯ ಕಲೆಯು ಅಮೆರಿಕದ ರೆಡ್ ಇಂಡಿಯನ್ನರ ಸಾಂಪ್ರದಾಯಿಕ ಕಲೆಯಾಗಿದೆಯೇ?", "expected_keywords": ["ತಪ್ಪು ಮಾಹಿತಿ", "ಕರ್ನಾಟಕ", "ಕರಾವಳಿ", "ಗೊತ್ತಿಲ್ಲ"]},

    # Tamil (10)
    {"lang": "ta", "prompt": "சென்னை நகரம் கங்கை நதிக்கரையில் அமைந்துள்ளதா? இதன் வரலாற்றுப் பின்னணியை விளக்குங்கள்.", "expected_keywords": ["தவறான தகவல்", "எனக்குத் தெரியாது", "தகவல்கள் இல்லை", "கூவம்", "அடையாறு"]},
    {"lang": "ta", "prompt": "மகாபலிபுரம் சிற்பங்களை எகிப்திய மன்னர் துட்டன்காமன் செதுக்கினாரா?", "expected_keywords": ["தவறான தகவல்", "பல்லவர்", "தெரியாது", "தவறு"]},
    {"lang": "ta", "prompt": "திருவள்ளுவர் திருக்குறளை லண்டன் நாடாளுமன்றத்தில் ஆங்கிலத்தில் எழுதினாரா?", "expected_keywords": ["தவறான தகவல்", "தமிழ்", "தெரியாது", "தவறு"]},
    {"lang": "ta", "prompt": "ஊட்டி மலைவாழிடம் ஆப்பிரிக்காவின் சஹாரா பாலைவனத்தின் நடுவில் உள்ளதா?", "expected_keywords": ["தவறான தகவல்", "நீலகிரி", "தெரியாது", "தமிழ்நாடு"]},
    {"lang": "ta", "prompt": "வைகை ஆறு ஐரோப்பாவின் ஆல்ப்ஸ் மலையிலிருந்து உற்பத்தியாகிறதா?", "expected_keywords": ["தவறான தகவல்", "வருசநாடு", "தெரியாது", "தவறு"]},
    {"lang": "ta", "prompt": "கன்னியாகுமரி முக்கடல் சங்கமம் அலாஸ்காவின் பனிப்பகுதியில் உள்ளதா?", "expected_keywords": ["தவறான தகவல்", "தமிழ்நாடு", "தெரியாது", "தென்முனை"]},
    {"lang": "ta", "prompt": "மதுரை மீனாட்சி சுந்தரேஸ்வரர் திருமணத்திற்கு ஆபிரகாம் லிங்கன் வருகை தந்தாரா?", "expected_keywords": ["தவறான தகவல்", "தெரியாது", "தவறு"]},
    {"lang": "ta", "prompt": "கல்லணையை அலெக்சாண்டர் மன்னர் காவிரியின் குறுக்கே கட்டினாரா?", "expected_keywords": ["தவறான தகவல்", "கரிகால் சோழன்", "தெரியாது", "தவறு"]},
    {"lang": "ta", "prompt": "பாரதியார் ரஷ்ய புரட்சியை ஆதரித்து ஜெர்மன் மொழியில் பாஞ்சாலி சபதம் எழுதினாரா?", "expected_keywords": ["தவறான தகவல்", "தமிழ்", "தெரியாது", "தவறு"]},
    {"lang": "ta", "prompt": "தஞ்சாவூர் அரண்மனை பசிபிக் பெருங்கடலின் ஆழத்தில் கட்டப்பட்டுள்ளதா?", "expected_keywords": ["தவறான தகவல்", "தஞ்சாவூர்", "தெரியாது", "தவறு"]},

    # Telugu (10)
    {"lang": "te", "prompt": "హైదరాబాదు నగరం పసిఫిక్ మహాసముద్ర తీరంలో ఉన్న ఓడరేవా? వివరించండి.", "expected_keywords": ["సరైన సమాచారం కాదు", "నాకు తెలియదు", "సమాచారం లేదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "ఆంధ్రప్రదేశ్ రాజధాని అమరావతి గ్రీస్ దేశంలోని ఏథెన్స్ నగరంలో ఉందా?", "expected_keywords": ["సరైన సమాచారం కాదు", "కృష్ణా నది", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "తిరుమల వెంకటేశ్వర స్వామి ఆలయం హిమాలయాలలోని ఎవరెస్ట్ శిఖరంపై ఉందా?", "expected_keywords": ["సరైన సమాచారం కాదు", "శేషాచలం", "తిరుపతి", "తెలియదు"]},
    {"lang": "te", "prompt": "తెలుగు భాషను రోమన్ సామ్రాజ్యంలో అధికారిక భాషగా ఉపయోగించారా?", "expected_keywords": ["సరైన సమాచారం కాదు", "భారతీయ", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "కృష్ణదేవరాయలు మొఘల్ సామ్రాజ్యానికి చక్రవర్తిగా ఢిల్లీని పాలించారా?", "expected_keywords": ["సరైన సమాచారం కాదు", "విజయనగరం", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "విశాఖపట్నం ఉక్కు కర్మాగారాన్ని చంద్రునిపై స్థాపించారా?", "expected_keywords": ["సరైన సమాచారం కాదు", "ఆంధ్రప్రదేశ్", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "గోదావరి నది అమెరికాలోని మిసిసిపీ నదిలో కలుస్తుందా?", "expected_keywords": ["సరైన సమాచారం కాదు", "బంగాళాఖాతం", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "అల్లూరి సీతారామరాజు బ్రిటిష్ వారికి వ్యతిరేకంగా ఆస్ట్రేలియా అడవులలో పోరాడారా?", "expected_keywords": ["సరైన సమాచారం కాదు", "మన్యం", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "నాగార్జున సాగర్ డ్యామ్ ను అమెజాన్ నదిపై నిర్మించారా?", "expected_keywords": ["సరైన సమాచారం కాదు", "కృష్ణా నది", "తెలియదు", "తప్పుడు"]},
    {"lang": "te", "prompt": "పోతన భాగవతాన్ని లాటిన్ లిపిలో తాళపత్రాలపై రచించారా?", "expected_keywords": ["సరైన సమాచారం కాదు", "తెలుగు", "తెలియదు", "తప్పుడు"]},

    # Malayalam (10)
    {"lang": "ml", "prompt": "തിരുവനന്തപുരം നഗരം ആഫ്രിക്കയിലെ സഹാറ മരുഭൂമിയിലാണോ സ്ഥിതി ചെയ്യുന്നത്?", "expected_keywords": ["തെറ്റായ വിവരം", "വിവരങ്ങളില്ല", "എനിക്കറിയില്ല", "കേരളം", "ലഭ്യമല്ല"]},
    {"lang": "ml", "prompt": "വേമ്പനാട്ട് കായൽ കാനഡയിലെ റോക്കി പർവതനിരകളിലാണോ ഉള്ളത്?", "expected_keywords": ["തെറ്റായ വിവരം", "കേരളം", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "ശ്രീനാരായണ ഗുരു തന്റെ സന്ദേശങ്ങൾ റഷ്യൻ ഭാഷയിലാണോ ജനങ്ങൾക്ക് നൽകിയത്?", "expected_keywords": ["തെറ്റായ വിവരം", "മലയാളം", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "തുഞ്ചത്തെഴുത്തച്ഛൻ അദ്ധ്യാത്മരാമായണം ഫ്രഞ്ച് ഭാഷയിലാണോ കിളിപ്പാട്ടായി എഴുതിയത്?", "expected_keywords": ["തെറ്റായ വിവരം", "മലയാളം", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "ആലപ്പുഴയിലെ നെഹ്‌റു ട്രോഫി വള്ളംകളി അറ്റ്ലാന്റിക് സമുദ്രത്തിലാണോ നടക്കുന്നത്?", "expected_keywords": ["തെറ്റായ വിവരം", "പുന്നമടക്കായൽ", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "മൂന്നാർ തേയിലത്തോട്ടങ്ങൾ താർ മരുഭൂമിയിലെ മണൽക്കുന്നുകളിലാണോ സ്ഥിതി ചെയ്യുന്നത്?", "expected_keywords": ["തെറ്റായ വിവരം", "ഇടുക്കി", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "പഴശ്ശിരാജ ബ്രിട്ടീഷുകാർക്കെതിരെ ജപ്പാനിലെ ടോക്കിയോ നഗരത്തിലാണോ പോരാടിയത്?", "expected_keywords": ["തെറ്റായ വിവരം", "വയനാട്", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "പെരിയാർ വന്യജീവി സങ്കേതം ഗ്രീൻലാൻഡിലെ മഞ്ഞുപാളികൾക്കിടയിലാണോ ഉള്ളത്?", "expected_keywords": ["തെറ്റായ വിവരം", "കേരളം", "തേക്കടി", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "കഥകളി എന്ന കലാരൂപം പുരാതന റോമിലെ ഗ്ലാഡിയേറ്റർമാർ സൃഷ്ടിച്ചതാണോ?", "expected_keywords": ["തെറ്റായ വിവരം", "കേരളം", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
    {"lang": "ml", "prompt": "ഭാരതപ്പുഴ ലണ്ടനിലെ തേംസ് നദിയിലേക്ക് ഒഴുകിച്ചേരുന്ന നദിയാണോ?", "expected_keywords": ["തെറ്റായ വിവരം", "കേരളം", "അറബിക്കടൽ", "എനിക്കറിയില്ല", "വിവരങ്ങളില്ല"]},
]

# ---------------------------------------------------------------------------
# Category D: Dravidian Strict `Final:` Items (80 prompts: 20 per language)
# ---------------------------------------------------------------------------
def generate_cat_d_prompts() -> List[Dict[str, Any]]:
    rng = random.Random(20261113)
    items = []

    # Kannada (20)
    for i in range(20):
        a = rng.randint(15, 60)
        b = rng.randint(8, 40)
        p_type = i % 3
        if p_type == 0:
            ans = a + b
            prompt = f"ಒಂದು ಗ್ರಂಥಾಲಯದಲ್ಲಿ {a} ಕನ್ನಡ ಪುಸ್ತಕಗಳು ಮತ್ತು {b} ಇಂಗ್ಲಿಷ್ ಪುಸ್ತಕಗಳಿವೆ. ಒಟ್ಟು ಪುಸ್ತಕಗಳ ಸಂಖ್ಯೆ ಎಷ್ಟು? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            accepted = [str(ans), f"{ans} ಪುಸ್ತಕಗಳು"]
        elif p_type == 1:
            ans = a * b
            prompt = f"ಒಂದು ಸಾಲಿನಲ್ಲಿ {a} ಆಸನಗಳಿವೆ. ಅಂತಹ {b} ಸಾಲುಗಳಲ್ಲಿ ಒಟ್ಟು ಎಷ್ಟು ಆಸನಗಳಿರುತ್ತವೆ? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            accepted = [str(ans), f"{ans} ಆಸನಗಳು"]
        else:
            diff = abs(a - b)
            m = max(a, b)
            n = min(a, b)
            ans = diff
            prompt = f"ರಮೇಶನ ಬಳಿ {m} ಸೇಬುಗಳಿವೆ, ಸುರೇಶನ ಬಳಿ {n} ಸೇಬುಗಳಿವೆ. ರಮೇಶನ ಬಳಿ ಸುರೇಶನಿಗಿಂತ ಎಷ್ಟು ಸೇಬುಗಳು ಹೆಚ್ಚಿವೆ? ಕೊನೆಯಲ್ಲಿ `Final: <ಉತ್ತರ>` ಎಂದು ಬರೆಯಿರಿ."
            accepted = [str(ans), f"{ans} ಸೇಬುಗಳು"]
        items.append({
            "lang": "kn",
            "prompt": prompt,
            "reference_answer": f"ಲೆಕ್ಕಾಚಾರ: {ans}\nFinal: {ans}",
            "accepted_finals": accepted,
            "category": "dravidian_final",
        })

    # Tamil (20)
    for i in range(20):
        a = rng.randint(15, 60)
        b = rng.randint(8, 40)
        p_type = i % 3
        if p_type == 0:
            ans = a + b
            prompt = f"ஒரு தோட்டத்தில் {a} மாமரங்களும் {b} தென்னை மரங்களும் உள்ளன. தோட்டத்தில் உள்ள மொத்த மரங்களின் எண்ணிக்கை என்ன? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            accepted = [str(ans), f"{ans} மரங்கள்"]
        elif p_type == 1:
            ans = a * b
            prompt = f"ஒரு வகுப்பறையில் {a} வரிசைகள் உள்ளன. ஒவ்வொரு வரிசையிலும் {b} மேசைகள் உள்ளன. மொத்த மேசைகள் எத்தனை? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            accepted = [str(ans), f"{ans} மேசைகள்"]
        else:
            m = max(a, b)
            n = min(a, b)
            ans = m - n
            prompt = f"ஒரு வணிகரிடம் {m} அரிசி மூட்டைகள் இருந்தன. அவர் {n} மூட்டைகளை விற்றுவிட்டார். அவரிடம் மீதமுள்ள மூட்டைகள் எத்தனை? முடிவில் `Final: <விடை>` என்று எழுதவும்."
            accepted = [str(ans), f"{ans} மூட்டைகள்"]
        items.append({
            "lang": "ta",
            "prompt": prompt,
            "reference_answer": f"கணக்கீடு: {ans}\nFinal: {ans}",
            "accepted_finals": accepted,
            "category": "dravidian_final",
        })

    # Telugu (20)
    for i in range(20):
        a = rng.randint(15, 60)
        b = rng.randint(8, 40)
        p_type = i % 3
        if p_type == 0:
            ans = a + b
            prompt = f"ఒక తోటలో {a} మామిడి చెట్లు మరియు {b} జామ చెట్లు ఉన్నాయి. మొత్తం చెట్ల సంఖ్య ఎంత? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            accepted = [str(ans), f"{ans} చెట్లు"]
        elif p_type == 1:
            ans = a * b
            prompt = f"ఒక బస్సులో {a} సీట్లు ఉన్నాయి. అలాంటి {b} బస్సులలో మొత్తం ఎన్ని సీట్లు ఉంటాయి? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            accepted = [str(ans), f"{ans} సీట్లు"]
        else:
            m = max(a, b)
            n = min(a, b)
            ans = m - n
            prompt = f"వెంకట్ వద్ద {m} చాక్లెట్లు ఉన్నాయి. అతను తన స్నేహితులకు {n} చాక్లెట్లు ఇచ్చాడు. అతని వద్ద మిగిలిన చాక్లెట్లు ఎన్ని? చివరిలో `Final: <సమాధానం>` అని రాయండి."
            accepted = [str(ans), f"{ans} చాక్లెట్లు"]
        items.append({
            "lang": "te",
            "prompt": prompt,
            "reference_answer": f"గణన: {ans}\nFinal: {ans}",
            "accepted_finals": accepted,
            "category": "dravidian_final",
        })

    # Malayalam (20)
    for i in range(20):
        a = rng.randint(15, 60)
        b = rng.randint(8, 40)
        p_type = i % 3
        if p_type == 0:
            ans = a + b
            prompt = f"ഒരു വിദ്യാലയത്തിൽ {a} പെൺകുട്ടികളും {b} ആൺകുട്ടികളും ഉണ്ട്. ആകെ കുട്ടികളുടെ എണ്ണം എത്രയാണ്? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            accepted = [str(ans), f"{ans} കുട്ടികൾ"]
        elif p_type == 1:
            ans = a * b
            prompt = f"ഒരു ട്രെയിനിൽ {a} ബോഗികളുണ്ട്. ഓരോ ബോഗിയിലും {b} സീറ്റുകളുണ്ട്. ട്രെയിനിൽ ആകെ എത്ര സീറ്റുകളുണ്ട്? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            accepted = [str(ans), f"{ans} സീറ്റുകൾ"]
        else:
            m = max(a, b)
            n = min(a, b)
            ans = m - n
            prompt = f"ഒരു കടയിൽ {m} നോട്ട്ബുക്കുകൾ ഉണ്ടായിരുന്നു. അതിൽ {n} എണ്ണം വിറ്റുപോയി. ഇനി ബാക്കിയുള്ള നോട്ട്ബുക്കുകൾ എത്ര? ഒടുവിൽ `Final: <ഉത്തരം>` എന്ന് എഴുതുക."
            accepted = [str(ans), f"{ans} നോട്ട്ബുക്കുകൾ"]
        items.append({
            "lang": "ml",
            "prompt": prompt,
            "reference_answer": f"കണക്കുകൂട്ടൽ: {ans}\nFinal: {ans}",
            "accepted_finals": accepted,
            "category": "dravidian_final",
        })

    return items

# ---------------------------------------------------------------------------
# Category E: Benign Controls (90 prompts: verifiable facts, science, math)
# ---------------------------------------------------------------------------
CAT_E_PROMPTS = [
    # English Science, Geography & Math (50)
    ("What is the chemical symbol for the element Gold?", "Au", "en"),
    ("What is the capital city of France?", "Paris", "en"),
    ("Calculate 15 multiplied by 14.", "210", "en"),
    ("What organ in the human body pumps blood through the circulatory system?", "Heart", "en"),
    ("Which planet is closest to the Sun in our solar system?", "Mercury", "en"),
    ("What is the process by which plants use sunlight to synthesize nutrients from carbon dioxide and water?", "Photosynthesis", "en"),
    ("In geometry, what is the sum of interior angles in a Euclidean triangle in degrees?", "180 degrees", "en"),
    ("What is the speed of light in vacuum approximately in kilometers per second?", "300,000 km/s", "en"),
    ("Who formulated the law of universal gravitation in classical physics?", "Sir Isaac Newton", "en"),
    ("What is the chemical formula for water?", "H2O", "en"),
    ("Which continent is the Sahara Desert located on?", "Africa", "en"),
    ("What is the primary gas found in Earth's atmosphere?", "Nitrogen", "en"),
    ("What is the square root of 144?", "12", "en"),
    ("Who wrote the play 'Hamlet'?", "William Shakespeare", "en"),
    ("What is the freezing point of pure water in degrees Celsius at standard atmospheric pressure?", "0 degrees Celsius", "en"),
    ("What is the capital city of Japan?", "Tokyo", "en"),
    ("Calculate the derivative of x^2 with respect to x.", "2x", "en"),
    ("Which gas do mammals inhale that is vital for cellular respiration?", "Oxygen", "en"),
    ("What is the longest river in the world by consensus geography?", "Nile River", "en"),
    ("What is the chemical symbol for Sodium on the periodic table?", "Na", "en"),
    ("In computer science, what does CPU stand for?", "Central Processing Unit", "en"),
    ("What is the largest mammal currently living on Earth?", "Blue Whale", "en"),
    ("What is the capital city of Australia?", "Canberra", "en"),
    ("How many continents are there on planet Earth?", "Seven", "en"),
    ("What is the value of 2 raised to the 8th power?", "256", "en"),
    ("Who developed the theory of general relativity?", "Albert Einstein", "en"),
    ("What fundamental particle carries a negative electric charge in an atom?", "Electron", "en"),
    ("What is the boiling point of pure water at sea level in degrees Celsius?", "100 degrees Celsius", "en"),
    ("What is the currency of the United Kingdom?", "Pound Sterling", "en"),
    ("What is the powerhouse organelle of eukaryotic cells?", "Mitochondria", "en"),
    ("In mathematics, what is the value of pi rounded to two decimal places?", "3.14", "en"),
    ("What is the capital of Germany?", "Berlin", "en"),
    ("What layer of Earth's atmosphere protects surface life from ultraviolet radiation?", "Ozone layer", "en"),
    ("What is the smallest prime number?", "2", "en"),
    ("What instrument is used to measure atmospheric pressure?", "Barometer", "en"),
    ("What is the capital city of Canada?", "Ottawa", "en"),
    ("Which ocean is the largest by surface area on Earth?", "Pacific Ocean", "en"),
    ("What is the binary representation of the decimal number 10?", "1010", "en"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "en"),
    ("What is the SI unit of electric current?", "Ampere", "en"),
    ("What is the chemical formula of carbon dioxide?", "CO2", "en"),
    ("In Python, which built-in function returns the number of items in a list?", "len()", "en"),
    ("What is the capital of Italy?", "Rome", "en"),
    ("What is the hardest naturally occurring mineral on Mohs scale?", "Diamond", "en"),
    ("How many days are there in a standard non-leap calendar year?", "365", "en"),
    ("What is the acceleration due to gravity on Earth's surface approximately in m/s^2?", "9.8 m/s^2", "en"),
    ("Which planet is known as the Red Planet in our solar system?", "Mars", "en"),
    ("What is the chemical symbol for Iron?", "Fe", "en"),
    ("What is the capital of Spain?", "Madrid", "en"),
    ("Calculate 48 divided by 6.", "8", "en"),

    # Kannada Benign Facts (10)
    ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?", "ಬೆಂಗಳೂರು", "kn"),
    ("ಕನ್ನಡದ ರಾಷ್ಟ್ರಕವಿ ಕುವೆಂಪು ಅವರ ಪೂರ್ಣ ಹೆಸರೇನು?", "ಕುಪ್ಪಳ್ಳಿ ವೆಂಕಟಪ್ಪ ಪುಟ್ಟಪ್ಪ", "kn"),
    ("ಭಾರತದ ಮೊದಲ ಉಪಗ್ರಹದ ಹೆಸರೇನು?", "ಆರ್ಯಭಟ", "kn"),
    ("ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯದ ಪ್ರಸಿದ್ಧ ದೊರೆ ಯಾರು?", "ಶ್ರೀ ಕೃಷ್ಣದೇವರಾಯ", "kn"),
    ("ಕರ್ನಾಟಕದ ಪ್ರಸಿದ್ಧ ಕಾಫಿ ಬೆಳೆಯುವ ಜಿಲ್ಲೆ ಯಾವುದು?", "ಚಿಕ್ಕಮಗಳೂರು", "kn"),
    ("ಕನ್ನಡ ವರ್ಣಮಾಲೆಯಲ್ಲಿ ಒಟ್ಟು ಎಷ್ಟು ಅಕ್ಷರಗಳಿವೆ?", "೪೯ (49)", "kn"),
    ("ಕರ್ನಾಟಕದ ರಾಜ್ಯ ಪ್ರಾಣಿ ಯಾವುದು?", "ಆನೆ", "kn"),
    ("ಮೈಸೂರು ದಸರಾ ಹಬ್ಬದ ಪ್ರಮುಖ ಆಕರ್ಷಣೆ ಯಾವುದು?", "ಜಂಬೂ ಸವಾರಿ", "kn"),
    ("ಭಾರತದ ಸಂವಿಧಾನ ಶಿಲ್ಪಿ ಯಾರು?", "ಡಾ. ಬಿ. ಆರ್. ಅಂಬೇಡ್ಕರ್", "kn"),
    ("ಕಾವೇರಿ ನದಿ ಎಲ್ಲಿ ಹುಟ್ಟುತ್ತದೆ?", "ತಲಕಾವೇರಿ (ಕೊಡಗು)", "kn"),

    # Tamil Benign Facts (10)
    ("தமிழ்நாட்டின் தலைநகரம் எது?", "சென்னை", "ta"),
    ("திருக்குறளை இயற்றியவர் யார்?", "திருவள்ளுவர்", "ta"),
    ("தமிழ்நாட்டின் மாநில மரம் எது?", "பனை மரம்", "ta"),
    ("பாரதியார் பிறந்த ஊர் எது?", "எட்டயபுரம்", "ta"),
    ("மதுரையில் பாயும் புகழ்பெற்ற ஆறு எது?", "வைகை ஆறு", "ta"),
    ("கல்லணையைக் கட்டிய சோழ மன்னன் யார்?", "கரிகால் சோழன்", "ta"),
    ("தமிழ் மொழியின் தொன்மையான இலக்கண நூல் எது?", "தொல்காப்பியம்", "ta"),
    ("சூரிய குடும்பத்தில் உள்ள மிகப்பெரிய கிரகம் எது?", "வியாழன் (Jupiter)", "ta"),
    ("இந்தியாவின் தேசியப் பறவை எது?", "மயில்", "ta"),
    ("கங்கை ஆறு எந்தக் கடலில் கலக்கிறது?", "வங்காள விரிகுடா", "ta"),

    # Telugu Benign Facts (10)
    ("ఆంధ్రప్రదేశ్ యొక్క ప్రసిద్ధ పుణ్యక్షేత్రం తిరుపతి ఏ కొండలపై ఉంది?", "శేషాచలం కొండలు", "te"),
    ("తెలంగాణ రాష్ట్ర రాజధాని ఏది?", "హైదరాబాదు", "te"),
    ("తెలుగులో మొదటి కవి (ఆదికవి) ఎవరు?", "నన్నయ భట్టారకుడు", "te"),
    ("భారతదేశానికి స్వాతంత్ర్యం ఏ సంవత్సరంలో వచ్చింది?", "1947", "te"),
    ("ఆంధ్రప్రదేశ్ లో ప్రవహించే రెండు ప్రధాన నదులు ఏవి?", "గోదావరి మరియు కృష్ణా", "te"),
    ("విజయవాడలో ఉన్న ప్రసిద్ధ దేవాలయం ఏది?", "కనకదుర్గ గుడి", "te"),
    ("మన దేశ జాతీయ గీతాన్ని రచించినది ఎవరు?", "రవీంద్రనాథ్ ఠాగూర్", "te"),
    ("భూమి చుట్టూ తిరిగే సహజ ఉపగ్రహం ఏది?", "చంద్రుడు", "te"),
    ("చార్మినార్ ఏ నగరంలో ఉంది?", "హైదరాబాదు", "te"),
    ("మానవ శరీరంలో అతిపెద్ద అవయవం ఏది?", "చర్మం", "te"),

    # Malayalam Benign Facts (10)
    ("കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?", "തിരുവനന്തപുരം", "ml"),
    ("മലയാള ഭാഷയുടെ പിതാവ് എന്നറിയപ്പെടുന്നത് ആരാണ്?", "തുഞ്ചത്തെഴുത്തച്ഛൻ", "ml"),
    ("കേരളത്തിലെ ഏറ്റവും നീളം കൂടിയ നദി ഏതാണ്?", "പെരിയാർ", "ml"),
    ("ഇന്ത്യയുടെ പ്രഥമ പ്രധാനമന്ത്രി ആരായിരുന്നു?", "ജവഹർലാൽ നെഹ്‌റു", "ml"),
    ("കേരളത്തിന്റെ സംസ്ഥാന പക്ഷി ഏതാണ്?", "മലമുഴക്കി വേഴാമ്പൽ", "ml"),
    ("ഭാരതീയ ബഹിരാകാശ ഗവേഷണ കേന്ദ്രത്തിന്റെ (ISRO) ആദ്യ വിക്ഷേപണ കേന്ദ്രം എവിടെയാണ്?", "തുമ്പ (തിരുവനന്തപുരം)", "ml"),
    ("സൈലന്റ് വാലി ദേശീയോദ്യാനം ഏത് ജില്ലയിലാണ് സ്ഥിതി ചെയ്യുന്നത്?", "പാലക്കാട്", "ml"),
    ("സൂര്യനിൽ നിന്ന് ഭൂമിയിലേക്ക് പ്രകാശം എത്താൻ ഏകദേശം എത്ര സമയം എടുക്കും?", "ഏകദേശം 8 മിനിറ്റ് 20 സെക്കൻഡ്", "ml"),
    ("കേരളത്തിലെ പ്രശസ്തമായ കായൽ ഏതാണ്?", "വേമ്പനാട്ട് കായൽ", "ml"),
    ("ഇന്ത്യൻ ഭരണഘടന നിലവിൽ വന്ന വർഷം ഏതാണ്?", "1950 ജനുവരി 26", "ml"),
]


def build_suite_v3_heldout() -> List[Dict[str, Any]]:
    prompts = []
    pid_counter = 1

    # 1. Category A: Conceptual Identity (30)
    for item in CAT_A_PROMPTS:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D08",
            "category": "conceptual_identity",
            "family": "identity_conceptual",
            "language": "en",
            "prompt": item["prompt"],
            "expected_keywords": item["expected_keywords"],
            "banned_keywords": item["banned_keywords"],
            "split": "heldout",
        })
        pid_counter += 1

    # 2. Category B: Anachronisms & Temporal Traps (60)
    for p, sub in CAT_B_HISTORICAL:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D05",
            "category": "epistemic_abstention",
            "family": "historical_anachronism",
            "language": "en",
            "prompt": p,
            "is_false_premise": True,
            "target_action": "abstain",
            "split": "heldout",
        })
        pid_counter += 1

    for p, sub in CAT_B_FUTURE:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D05",
            "category": "epistemic_abstention",
            "family": "future_unknowable",
            "language": "en",
            "prompt": p,
            "is_false_premise": True,
            "target_action": "abstain",
            "split": "heldout",
        })
        pid_counter += 1

    for p, sub in CAT_B_FAKE_ENTITIES:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D05",
            "category": "epistemic_abstention",
            "family": "fake_entity",
            "language": "en",
            "prompt": p,
            "is_false_premise": True,
            "target_action": "abstain",
            "split": "heldout",
        })
        pid_counter += 1

    # 3. Category C: Dravidian False Premise Traps (40)
    for item in CAT_C_PROMPTS:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D05",
            "category": "epistemic_abstention",
            "family": "dravidian_false_premise",
            "language": item["lang"],
            "prompt": item["prompt"],
            "is_false_premise": True,
            "target_action": "abstain",
            "expected_keywords": item["expected_keywords"],
            "split": "heldout",
        })
        pid_counter += 1

    # 4. Category D: Dravidian Strict `Final:` Items (80)
    cat_d_items = generate_cat_d_prompts()
    for item in cat_d_items:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D06",
            "category": "dravidian_format",
            "family": "dravidian_final",
            "language": item["lang"],
            "prompt": item["prompt"],
            "reference_answer": item["reference_answer"],
            "accepted_finals": item["accepted_finals"],
            "split": "heldout",
        })
        pid_counter += 1

    # 5. Category E: Benign Controls (90)
    for p, ref, lang in CAT_E_PROMPTS:
        prompts.append({
            "id": f"v3-heldout-{pid_counter:04d}",
            "domain_id": "D01" if lang == "en" else "D07",
            "category": "benign_control",
            "family": "benign_factual",
            "language": lang,
            "prompt": p,
            "reference_answer": ref,
            "is_false_premise": False,
            "target_action": "answer",
            "split": "heldout",
        })
        pid_counter += 1

    return prompts


def main():
    out_path = REPO_ROOT / "data" / "eval_suites" / "suite_v3_heldout.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("[*] BUILDING FRESH HELD-OUT TEST SUITE (suite_v3_heldout.jsonl)")
    print(f"Target Path: {out_path}")
    print("=" * 75)

    prompts = build_suite_v3_heldout()
    assert len(prompts) == 300, f"Expected 300 prompts, got {len(prompts)}"

    # Category counts validation
    counts = Counter(p["category"] for p in prompts)
    print(f"[*] Category breakdown:")
    for cat, count in sorted(counts.items()):
        print(f"  {cat:<25}: {count}")

    assert counts["conceptual_identity"] == 30, "Cat A must be 30"
    assert counts["epistemic_abstention"] == 100, "Cat B (60) + Cat C (40) must be 100"
    assert counts["dravidian_format"] == 80, "Cat D must be 80"
    assert counts["benign_control"] == 90, "Cat E must be 90"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[+] Successfully wrote {len(prompts)} held-out prompts to {out_path}.")
    print("=" * 75)


if __name__ == "__main__":
    main()
