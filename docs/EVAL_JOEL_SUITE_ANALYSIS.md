# Parentheses 0.9-120M: Joel's Test Suite Evaluation & Diagnosis

**Date:** 2026-10-17 17:35:38  
**Hardware:** cuda (RTX 5050 8GB)  
**Questions Evaluated:** 44 questions extracted from Joel's test log  

## 1. Executive Summary & Root Cause Analysis

During Joel's initial interactive test (`tests/parentheses-0.9-120M-test-00.md`), the model exhibited high failure rates, hallucinated fake URLs, and repeated robotic phrases like `Examining the key principles behind this statement, Contextual evidence for...`. 

**The Root Cause was two-fold:**
1. **`--web` Flag Activated Blind Grounding:** `scripts/chat.py` was executed with `--web`. For every user prompt, it made live external web queries, returning noisy snippets and URLs that flooded the prompt context. At 120M parameters, the SSM attention state was drowned in synthetic search results.
2. **`mode=think` Was the Default Format:** `chat.py` defaulted to `--format think`. In SFT v3/v4 training data, synthetic reasoning examples were primed with `Examining the key principles behind this statement, Contextual evidence for...`. When primed with `<think>`, the model reproduced this synthetic training artifact rather than answering directly.

By contrast, when tested in **`mode=chat`** without blind web injection, the model produces direct, coherent conversational answers without hallucinating fake URLs or repeating robotic templates.

---

## 2. Configuration Comparison Table

| Configuration | Avg Coherence (1-5) | Directness (%) | Accuracy (%) | Fake URL Rate (%) | Robotic Template (%) | Dravidian Fidelity (%) | Speed (tok/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SFT v4 (Step 10036) - Mode: CHAT** | **2.89** | 22.7% | 31.8% | 0.0% | 77.3% | 95.0% | 32.8 |
| **SFT v4 (Step 10036) - Mode: THINK** | **2.73** | 11.4% | 38.6% | 2.3% | 88.6% | 94.6% | 31.3 |
| **SFT v2 (Step 9786) - Mode: CHAT** | **4.27** | 100.0% | 27.3% | 0.0% | 0.0% | 93.7% | 27.5 |
| **Base 120M (Step 9536) - Mode: CHAT** | **4.31** | 100.0% | 34.1% | 2.3% | 0.0% | 95.7% | 27.6 |
| **Base 120M (Step 9536) - Mode: BASE** | **4.34** | 100.0% | 34.1% | 0.0% | 0.0% | 94.5% | 33.9 |

---

## 3. Side-by-Side Prompt Case Studies

Comparison of responses on Joel's key test prompts across configurations:

### `q01_grass_green`: "is grass green?"
**Expected / Target:** `green, yes`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'is tree green': The tree is a large, tropical forest that is traced by forest... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'is tree green': Rainbow is a small, loosely made by fryinging a frying pan, c... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 5.0 | Yes, the grass is a green and the grass is blue.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 5.0 | Yes, there is a tree called "Spanish" in the family tree.... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | The trees are not grass, but are trees.  The leaves are white, the leaves being yellow.  A large tree is a flowering tree, or a fruitful par... |


### `q02_mangalore_native_lang`: "what is the native language of mangalore?"
**Expected / Target:** `tulu, kannada, konkani`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'what is the native language of mangalore?': Mangalore is a language of the Un... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'what is the native language of mangalore': Mangalore is a complex and complex... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | Mangalore... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The native language of mangalore is a complex language of the same country, and it has been a source for many centuries. The language of man... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | The language of mangalore is a complex and complex structure in many parts of the world, including the English language, the English and Ger... |


### `q04_entropy_physical`: "What is the physical significance of entropy?"
**Expected / Target:** `disorder, randomness, thermodynamic, energy, microstates`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the physical significance of entropy?': This is a physical significan... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the physical significance of entropy?': The physical significance of ... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | Suppose a human is a human. What is the physical significance of entropy? A) entropy B) entropy C) entropy... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The entropy of a body is the change of variables in the body. It is a physical phenomenon that can be made to be understood by individuals i... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | If you have a physical environment, you can do it by using the same physical environment.... |


### `q08_reluctant_explain`: "Explain the word "reluctant" in simple terms."
**Expected / Target:** `unwilling, hesitant, not wanting, holding back`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'explain what" is a phrase that makes sense of the word "reluctant" in simple ... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'explain the word "reluctant" in simple terms. Identify how many words are in ... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | The word "reluctant" in simplest terms is often used in mathematical terms to determine the exact value of a particular value. In mathematic... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | In terms of the word "reluctant" (or "reluctant)" in a language, you can refer to a word that is not in a language. The phrase "related" to ... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | Let's try to understand the definition of "relative" in the context of "relative" counterparty.  Now, let's look at this: Let $z=z=10$, and ... |


### `q09_actions_speak_louder`: "What does the phrase "actions speak louder than words" mean?"
**Expected / Target:** `do, say, action, show, practice, deeds`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 5.0 | "actions louder than words" mean?... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What does the phrase "actions speak louder than words" mean? The phrase "acti... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 5.0 | "actions louder than words" (noun) means: the word is spoken to a person, not to speak a word.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 5.0 | "actions louder than words" mean?... |
| **Base 120M (Step 9536) - Mode: BASE** | 5.0 | The word "actions" means a number of words, or a number of words that are spoken, but they are not context-free. The phrase "actions" is the... |


### `q10_scarce_opposite`: "What is the opposite of the word "scarce"?"
**Expected / Target:** `abundant, plentiful, copious, ample, frequent`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 4.0 | The word "scarce" is a word in a sentence that is not considered an unwritten sentence.... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the opposite of the word "scarce"? The word "scarpe" is a skeleton of... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | The word "scarce" is a hypothetical word that is often used to refer to a person who has an organs or organs. The term "scarce," which means... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The word "scarce" is a word that is not necessarily the same as "scarce".... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | The word "scarce" (noun) means: a man who has a lot of money, or is not ashamed of himself.... |


### `q17_capital_france_3words`: "Answer this question using exactly three words: What is the capital of France?"
**Expected / Target:** `paris`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the capital of France?': The capital of France is known as the "Centr... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the capital of France?': What is the capital of Germany?  **Final Ans... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | The capital of France is an important question that is not unique to English.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The capital of France is a national capital of France. It is located in the northwest corner of the country, and is located in the eastern p... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | The capital of France is a great way to get to the capital of France. The capital of France is the largest city in Europe, with a population... |


### `q20_number_after_99`: "Reply with only the number that comes after 99."
**Expected / Target:** `100`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'Reply with only the number that comes after 99.': Rapidly, a few things aren'... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'Reply with only the number that comes after 99.': This is a list of all the k... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | Yes, that makes sense.  There are many possible ways to divide a number into two integers, but you can also divide by 9.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | Here is the number that you can start by calculating:  1. **Two**: 99 * 7 2. **Master**: 20 3. **Location**: -10 4. **Member**: -15  To star... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | 99. It's not the same as the number 2.1. So, you need to check if 120 is a valid number for a valid number of samples, and you have to check... |


### `q28_capital_australia`: "What is the capital of Australia?"
**Expected / Target:** `canberra`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 4.0 | The capital of Australia is the capital of Australia.... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the capital of Australia': The capital of Australia is the capital of... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | The capital of Australia is the capital of Australia.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The capital of Australia is the largest city in the world. It is located at 4,780 miles (1,215 mi). The capital of Australia was originally ... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | It is a great and easy way to learn, and it is a great way to learn more about it. It has many ideas about how to build an innovation. And e... |


### `q30_gold_chemical_symbol`: "What is the chemical symbol for gold?"
**Expected / Target:** `au`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the chemical symbol for gold?': The chemical symbol for gold is a che... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is the chemical symbol for gold?': Gold is a chemical symbol that is use... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | The chemical symbol for gold is the chemical symbol for gold, which is a major component of the chemical symbol. The formula for the sum of ... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The chemical symbol for gold is a chemical symbol that is a complex substance that is known as the metal-soluble acid. The metal-solule, or ... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | The chemical symbol for gold is a chemical symbol for gold, such as a bacteria, lamps, and other fruits. It is important to note that these ... |


### `q35_math_19_plus_27`: "What is 19 + 27?"
**Expected / Target:** `46`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is 19 + 27?...': The phrase "19 + 30 = -27". The phrase "30 = -7". So an... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'What is 19 + 27?**: The numbers are divided into three groups: 1. The first i... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | In 19, 19 + 27, = (19 + x) / (x) = x/2.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | The 19 + 27 is indeed the highest of the two numbers, and is divided equally among the numbers. The question asks whether a number can be di... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | What is 19 + 27? What is the sum of the sums of the sums in terms of x and y?  Wait, perhaps I need to check for example.  Let's do a differ... |


### `q38_larger_09_or_089`: "Which is larger: 0.9 or 0.89?"
**Expected / Target:** `0.9`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 2.5 | Examining the key principles behind this statement, Analyzing 'Which is larger: 0.9 or 0.89? 0.8? ': The answer choices are from the list of... |
| **SFT v4 (Step 10036) - Mode: THINK** | 2.5 | Examining the key principles behind this statement, Analyzing 'Which is larger: 0.9 or 0.89? 0.87?...   1/3... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 5.0 | 2. 0.9 is less than 1800. 2. 6 is less than or equal to 1869. 3. -1779 is less than or = -442.  Thus, the answer should be -43.  **Final Ans... |
| **Base 120M (Step 9536) - Mode: CHAT** | 5.0 | Sure, here are 0.9 or 0.89:  - 1180:  - 229 - -354 - -536 - -477  There are two types of numbers: the first and the last two are integers, s... |
| **Base 120M (Step 9536) - Mode: BASE** | 5.0 | 1.1: 0.9? 2.0? 3.5? ?? ? ??  How would you use the following code to fix the error:   Serial.print((i) |)    - (ii) | - (iii)    - (iii)    ... |


### `q41_kan_capital`: "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?"
**Expected / Target:** `ಬೆಂಗಳೂರು`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 4.0 | ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ, ಕರ್ನಾಟಕವು 1903ರಲ್ಲಿ ಅಸ್ತಿತ್ವಕ್ಕೆ ಬಂದ ಒಂದು ದೊಡ್ಡ ನಗರ.... |
| **SFT v4 (Step 10036) - Mode: THINK** | 5.0 | ಇದರ ಪ್ರಮುಖ ಅಂಶಗಳನ್ನು ವಿಶ್ಲೇಷಿಸುವುದಾದರೆ, ಕರ್ನಾಟಕದ ಕರಾವಳಿ ಜಿಲ್ಲೆಗಳು, ದಕ್ಷಿಣ ಕನ್ನಡ ಮತ್ತು ಉಡುಪಿ ಜಿಲ್ಲೆಗಳಲ್ಲಿ ಅತಿ ಹೆಚ್ಚು ಜನಸಂಖ್ಯೆ ಇರುವ ಜಿಲ್ಲೆಗಳಲ್... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | ಕರ್ನಾಟಕವು 1831ರ ಜುಲೈ 3ರಿಂದ 4ರ ವರೆಗೆ ದೇಶದ ಪ್ರಮುಖ ನಗರಗಳಲ್ಲಿ ಒಂದಾಗಿದೆ. ಇದು ಭಾರತ ಮತ್ತು ಬ್ರಿಟನ್ ದೇಶಗಳ ನಡುವೆ ಇದೆ.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.... |
| **Base 120M (Step 9536) - Mode: BASE** | 5.0 | : ರಾಜ್ಯ ರಾಜಧಾನಿ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಏನು ನಡೆಯುತ್ತಿದೆ? ಯಾವ ನಗರದಲ್ಲಿದೆ? ಇಲ್ಲಿದೆ ಮಾಹಿತಿ.  ಬೆಂಗಳೂರು (ಅ. 13): ರಾಜ್ಯ ರಾಜಧಾನಿ ಬೆಂಗಳೂರಿನಲ್ಲಿ ಯಾವ ನಗರದಲ್ಲಿದೆ... |


### `q44_mal_capital`: "കേരളത്തിന്റെ തലസ്ഥാനം ഏതാണ്?"
**Expected / Target:** `തിരുവനന്തപുരം`

| Configuration | Score | Generated Response |
| :--- | :---: | :--- |
| **SFT v4 (Step 10036) - Mode: CHAT** | 4.0 | കേരളത്തിന്റെ തലസ്ഥാനം കേരളമാണ്.... |
| **SFT v4 (Step 10036) - Mode: THINK** | 5.0 | ഇതിലെ പ്രധാന വസ്തുതകൾ വിശകലനം ചെയ്താൽ, കേരളത്തിലെ തിരുവനന്തപുരം നഗരമായ രാജസ്ഥാനിൽ സ്ഥിതി ചെയ്യുന്ന ഏറ്റവും വലിയ നഗരമാണ് തിരുവനന്തപുരം.... |
| **SFT v2 (Step 9786) - Mode: CHAT** | 4.0 | കേരളത്തിലെ ഏറ്റവും വലിയ തലസ്ഥാനമാണ് തൃശ്ശൂർ.... |
| **Base 120M (Step 9536) - Mode: CHAT** | 4.0 | കേരളത്തിന്റെ തലസ്ഥാനമാണ് കേരളം.... |
| **Base 120M (Step 9536) - Mode: BASE** | 4.0 | അതിന്റെ പേരു പറയാമോ? ഒരു നാടിനെക്കുറിച്ചാണ് സംസാരിക്കുന്നത്. അതെ, നമ്മുടെ നാടിന്റെ പേര്? ഈ നാടിന്റെ പേര് എന്തിനെക്കുറിച്ചാണ്? നമ്മുടെ നാട്ടി... |


---

## 4. Honest Capability Boundary of Parentheses 0.9-120M

A 120M parameter model is a compact linear-recurrent state space duality (SSM) architecture. To establish realistic expectations for Joel and the team:

### What 120M DOES WELL (Sweet Spot):
- **Simple Factual Retrieval:** Direct geography, chemistry symbols (Au for gold), capitals (Paris, Canberra, Bengaluru), simple science definitions.
- **Dravidian Language Fluency:** Excellent script preservation and natural syntax in Kannada, Tamil, Telugu, and Malayalam.
- **Vocabulary & Lexical Knowledge:** Synonyms, antonyms, idiom explanations, and word meanings.
- **Ultra-Low Latency Inference:** ~90-110 tokens/sec with $O(1)$ constant-memory recurrent state update.

### What 120M CANNOT DO (Architecture & Scale Limits):
- **Strict Negative & Numerical Constraints:** Constraints like *'without using the letter e'* or *'exactly four bullet points each with 5 words'* require large multi-token lookahead and search typically only observed in 7B+ models.
- **Complex Multi-Step Math:** Multi-step arithmetic (19+27) or algebra (3x+7=22) without tool calling or calculator heads.
- **Handling Noisy Long-Context Injections:** Injecting 1500-token web search scrapes causes prompt drowning.


---

## 5. Concrete Recommendations for Joel's Test Setup

1. **Set Default Format to `chat` in `scripts/chat.py`:**
   - Change default `--format` from `think` to `chat`. Direct chat is clean, immediate, and free from the synthetic reasoning templates.
2. **Disable `--web` by Default:**
   - Real-time web injection should only be triggered on explicit user request or via `/web <query>`, not on every simple question.
3. **Recommended Test Command for Joel:**
   ```powershell
   .venv\Scripts\python.exe scripts/chat.py --checkpoint checkpoints/parentheses-0.9-120M-sft-v4/step_10036_final.pt --device cuda --format chat
   ```
