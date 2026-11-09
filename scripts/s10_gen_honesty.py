#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/s10_gen_honesty.py
Session 10 (Tanzanite) honesty / grounded-reading / identity data generator.

Teaches the owner's core rule (memory: honesty-training-goal): never fabricate, say "I don't know" when it is
true, and never refuse a benign question. Buckets written (one JSONL each, under --out-dir):

  false_premise.jsonl  fake people / cities / books / reports in English + tam/tel/kan/mal, plus wrong-location
                       corrections of real landmarks and not-yet-happened future questions. The model answers
                       "I don't know" (short, no <think>) or corrects the premise. Fake names are random
                       syllables, so there is nothing true to recall.
  benign_facts.jsonl   small table of verified facts (capitals, symbols, unit conversions computed in Python) so
                       "I don't know" is not learned as a general reflex (gate G7 no false refusal).
  grounded.jsonl       SQuAD 2.0 train (CC-BY-SA-4.0): answer from the passage, or "The passage does not say."
  unans_math.jsonl     Synthetic_Unanswerable_Math (MIT) unanswerable side only: "not enough information".
  abstention_short.jsonl  the answer part (after </think>) of data/sft_v4_abstention_4000.jsonl, short rows only.
  identity.jsonl       data/curated/parentheses_identity_sft_v1.jsonl + direct (no <think>) and paraphrased
                       prompt variants. Eval prompts of identity_15 / honesty_10 / benign_refusal_50 are removed.

NOT included here: NQ-open R-Tuning rows. Those need the base model to be probed on the GPU first
(scripts/s10_probe_rtuning.py), because "I don't know" must follow what the student really does not know.

The native-script phrases are fixed short templates written by Claude (Joel: "anything"), not yet reviewed by a
native speaker. tcy/kff/kxv/nit get none (no source to check phrases against).

    .venv\\Scripts\\python.exe scripts\\s10_gen_honesty.py --out-dir data\\s10_gen
    .venv\\Scripts\\python.exe scripts\\s10_gen_honesty.py --self-test
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQUAD = r"F:\Lingual_Data\raw\hf\s10\squad_v2\squad_v2\train-00000-of-00001.parquet"
SUM_GLOB = r"F:\Lingual_Data\raw\hf\s10\Synthetic_Unanswerable_Math\synthetic_unanswerable_math.parquet"
ABST = ROOT / "data" / "sft_v4_abstention_4000.jsonl"
IDENT = ROOT / "data" / "curated" / "parentheses_identity_sft_v1.jsonl"
EVAL_FILES = [ROOT / "data" / "eval_suites" / n for n in ("identity_15.jsonl", "honesty_10.jsonl", "benign_refusal_50.jsonl")]


def norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower()))


# ------------------------------------------------------------------ fake names
ON = ["b", "d", "k", "l", "m", "n", "r", "s", "t", "v", "z", "th", "sh", "gr", "tr", "kh", "br", "p", "f"]
VO = ["a", "e", "i", "o", "u", "ae", "ou", "ia"]
CO = ["", "", "n", "r", "l", "s", "x", "th", "m"]


def en_word(r, syl=None):
    w = "".join(r.choice(ON) + r.choice(VO) + r.choice(CO) for _ in range(syl or r.choice([2, 3])))
    return w.capitalize()


SCRIPT = {
    "tam": ("கசடதபமரலவநன", ["", "ா", "ி", "ு", "ெ", "ே", "ொ"], "்"),
    "tel": ("కగచజటడతదనపబమరలవసహ", ["", "ా", "ి", "ీ", "ు", "ూ", "ె", "ే", "ొ"], ""),
    "kan": ("ಕಗಚಜಟಡತದನಪಬಮರಲವಸಹ", ["", "ಾ", "ಿ", "ು", "ೆ", "ೇ", "ೊ"], ""),
    "mal": ("കഗചജടഡതദനപബമരലവസഹ", ["", "ാ", "ി", "ു", "െ", "േ", "ൊ"], ""),
}


def native_word(r, lang):
    cons, signs, virama = SCRIPT[lang]
    w = "".join(r.choice(cons) + r.choice(signs) for _ in range(r.choice([3, 4])))
    if virama and r.random() < 0.5:
        w += r.choice("னமல") + virama
    return w


IDK_EN = [
    "I don't know. I have no reliable information about {x}, and I won't guess.",
    "I don't know. {x} doesn't match anything I know, so I would only be making something up.",
    "I'm not aware of {x}. It may not exist, so I can't answer that reliably.",
    "I don't know. I'd rather say so than invent details about {x}.",
    "I don't have any reliable information about {x}, so I can't answer.",
]
IDK_NAT = {
    "tam": "எனக்குத் தெரியாது. இதைப் பற்றி நம்பகமான தகவல் என்னிடம் இல்லை; ஊகித்துச் சொல்ல மாட்டேன்.",
    "tel": "నాకు తెలియదు. దీని గురించి నా దగ్గర నమ్మదగిన సమాచారం లేదు; ఊహించి చెప్పను.",
    "kan": "ನನಗೆ ಗೊತ್ತಿಲ್ಲ. ಇದರ ಬಗ್ಗೆ ನನ್ನ ಬಳಿ ನಂಬಲರ್ಹ ಮಾಹಿತಿ ಇಲ್ಲ; ಊಹಿಸಿ ಹೇಳುವುದಿಲ್ಲ.",
    "mal": "എനിക്ക് അറിയില്ല. ഇതിനെക്കുറിച്ച് വിശ്വസനീയമായ വിവരം എന്റെ പക്കലില്ല; ഊഹിച്ച് പറയില്ല.",
}
NAT_Q = {
    "tam": ["{x} எப்போது பிறந்தார்?", "{x} என்ற நகரம் எந்த நாட்டில் உள்ளது?", "{x} எழுதிய நூலின் பெயர் என்ன?"],
    "tel": ["{x} ఎప్పుడు జన్మించారు?", "{x} అనే నగరం ఏ దేశంలో ఉంది?", "{x} రాసిన పుస్తకం పేరు ఏమిటి?"],
    "kan": ["{x} ಯಾವಾಗ ಜನಿಸಿದರು?", "{x} ಎಂಬ ನಗರ ಯಾವ ದೇಶದಲ್ಲಿದೆ?", "{x} ಬರೆದ ಪುಸ್ತಕದ ಹೆಸರೇನು?"],
    "mal": ["{x} എപ്പോൾ ജനിച്ചു?", "{x} എന്ന നഗരം ഏത് രാജ്യത്താണ്?", "{x} എഴുതിയ പുസ്തകത്തിന്റെ പേര് എന്താണ്?"],
}
EN_Q = [
    ("When was {p} born?", "p"), ("Who was {p} and what are they best known for?", "p"),
    ("What is the capital of {c}?", "c"), ("How many people live in the city of {c}?", "c"),
    ("Who wrote the novel \"{t}\"?", "t"), ("Summarize the plot of the book \"{t}\".", "t"),
    ("Who directed the film \"{t}\"?", "t"),
    ("What did the {yr} {c} Accord say about trade?", "c"), ("What were the main findings of the {yr} {c} report?", "c"),
    ("What is the boiling point of {e}ium?", "e"), ("When did {p} win the Nobel Prize in physics?", "p"),
    ("Who founded the company {c} Industries and when?", "c"), ("Who discovered the {e}ium effect?", "e"),
]
FUTURE_EN = [
    "Who won the {fy} cricket World Cup?", "What was the closing price of the Sensex on 5 March {fy}?",
    "Who is the Prime Minister of India in {fy}?", "What was the winning number in the {fy} lottery draw?",
    "Which team won the {fy} football World Cup final?", "What was the average rainfall in Chennai in {fy}?",
]
FUTURE_ANS = [
    "I don't know. That is in the future, so there is nothing for me to recall, and I won't guess.",
    "I can't know that: it hasn't happened yet. I'd only be inventing an answer.",
    "I don't know. It is a future event, and I can't predict it reliably.",
]
# real landmark -> (right city, wrong cities). Eiffel Tower is excluded on purpose (honesty_10 eval item).
LANDMARKS = [("Taj Mahal", "Agra", ["Delhi", "Mumbai", "Jaipur"]), ("Colosseum", "Rome", ["Paris", "Athens", "Madrid"]),
             ("Big Ben", "London", ["Dublin", "Edinburgh", "Paris"]), ("Charminar", "Hyderabad", ["Chennai", "Bengaluru", "Pune"]),
             ("Gateway of India", "Mumbai", ["Kolkata", "Chennai", "Delhi"]), ("Meenakshi Temple", "Madurai", ["Mysuru", "Kochi", "Thanjavur"]),
             ("Hawa Mahal", "Jaipur", ["Udaipur", "Jodhpur", "Agra"]), ("Statue of Liberty", "New York", ["Boston", "Chicago", "London"]),
             ("Golden Temple", "Amritsar", ["Delhi", "Lahore", "Chandigarh"]), ("Mysore Palace", "Mysuru", ["Hyderabad", "Chennai", "Kochi"])]


def make_false_premise(r):
    kind = r.random()
    if kind < 0.28:                                  # English fake entity
        tpl, slot = r.choice(EN_Q)
        x = {"p": f"{en_word(r)} {en_word(r)}", "c": en_word(r), "t": f"The {en_word(r)} {r.choice(['Garden', 'Code', 'Letters', 'Tide'])}",
             "e": en_word(r, 2)}[slot]
        q = tpl.format(p=x, c=x, t=x, e=x, yr=r.randint(1905, 2019))
        return "eng", q, r.choice(IDK_EN).format(x=x), "fake_entity_en"
    if kind < 0.68:                                  # native-script fake entity
        lang = r.choice(list(NAT_Q))
        x = native_word(r, lang) + (" " + native_word(r, lang) if r.random() < 0.4 else "")
        return lang, r.choice(NAT_Q[lang]).format(x=x), IDK_NAT[lang], f"fake_entity_{lang}"
    if kind < 0.84:                                  # wrong-location correction of a real landmark
        name, right, wrongs = r.choice(LANDMARKS)
        wrong = r.choice(wrongs)
        q = r.choice([f"When was the {name} built in {wrong}?", f"Who designed the {name} in {wrong}?",
                      f"How tall is the {name} in {wrong}?"])
        return "eng", q, f"The {name} is in {right}, not {wrong}, so the question has a wrong premise. I don't know of a {name} in {wrong}.", "wrong_location"
    fy = r.randint(2031, 2060)
    return "eng", r.choice(FUTURE_EN).format(fy=fy), r.choice(FUTURE_ANS), "future"


# ------------------------------------------------------------------ benign facts (verified by hand, computed where possible)
CAPITALS = {"France": "Paris", "Japan": "Tokyo", "India": "New Delhi", "Italy": "Rome", "Egypt": "Cairo", "Canada": "Ottawa",
            "Brazil": "Brasilia", "Kenya": "Nairobi", "Spain": "Madrid", "Germany": "Berlin", "China": "Beijing",
            "Australia": "Canberra", "Russia": "Moscow", "Nepal": "Kathmandu", "Sri Lanka": "Sri Jayawardenepura Kotte",
            "Argentina": "Buenos Aires", "Turkey": "Ankara", "Thailand": "Bangkok", "Norway": "Oslo", "Greece": "Athens"}
SYMBOLS = {"gold": "Au", "silver": "Ag", "iron": "Fe", "sodium": "Na", "oxygen": "O", "copper": "Cu", "hydrogen": "H",
           "carbon": "C", "potassium": "K", "lead": "Pb"}


def make_benign(r):
    k = r.random()
    if k < 0.4:
        c = r.choice(list(CAPITALS))
        return r.choice([f"What is the capital of {c}?", f"Which city is the capital of {c}?"]), f"The capital of {c} is {CAPITALS[c]}.", "capital"
    if k < 0.65:
        e = r.choice(list(SYMBOLS))
        return f"What is the chemical symbol for {e}?", f"The chemical symbol for {e} is {SYMBOLS[e]}.", "symbol"
    if k < 0.85:
        n = r.randint(2, 90)
        return f"How many centimetres are there in {n} metres?", f"{n} x 100 = {n * 100}. There are {n * 100} centimetres.", "unit"
    d = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    i, k2 = r.randrange(7), r.randint(2, 20)
    return f"If today is {d[i]}, what day will it be in {k2} days?", f"{k2} mod 7 = {k2 % 7}, so it will be {d[(i + k2) % 7]}.", "weekday"


# ------------------------------------------------------------------ SQuAD2 grounded reading
NOT_STATED = ["The passage does not say.", "The passage doesn't mention that.", "That is not stated in the passage.",
              "I don't know; the passage doesn't say."]
SQ_INSTR = ["Answer using only the passage. If it is not stated, say so.", "Use only the passage below. If the answer is not there, say it is not stated.",
            "Answer from the passage only. Say so if it does not say."]


def squad_rows(r, n):
    import pyarrow.parquet as pq
    t = pq.read_table(SQUAD, columns=["id", "context", "question", "answers"]).to_pylist()
    r.shuffle(t)
    ans_rows, un_rows = [], []
    for x in t:
        if len(x["context"]) > 1100:
            continue
        texts = x["answers"]["text"]
        (ans_rows if texts else un_rows).append(x)
    out = []
    for i in range(n):
        x = (ans_rows if i % 2 == 0 else un_rows)[i // 2]
        prompt = f"Passage:\n{x['context']}\n\nQuestion: {x['question']}\n{r.choice(SQ_INSTR)}"
        resp = min(x["answers"]["text"], key=len) if x["answers"]["text"] else r.choice(NOT_STATED)
        out.append(("eng", prompt, resp, "squad_answerable" if x["answers"]["text"] else "squad_unanswerable", x["id"]))
    return out


# ------------------------------------------------------------------ unanswerable math
SUM_ANS = ["This can't be answered as stated: the problem leaves out information needed for a single answer.\nFinal: not enough information",
           "I don't know. The question doesn't give enough information to work out one answer.\nFinal: not enough information",
           "There is not enough information to solve this.\nFinal: not enough information"]


def sum_rows(r, n):
    import pyarrow.parquet as pq
    t = pq.read_table(SUM_GLOB, columns=["unanswerable_question"]).to_pylist()
    r.shuffle(t)
    return [("eng", x["unanswerable_question"].strip() + ("" if r.random() < 0.5 else " End with `Final: <answer>`."), r.choice(SUM_ANS), "sum_unanswerable", str(i))
            for i, x in enumerate(t[:n])]


# ------------------------------------------------------------------ short-form abstention from the existing 4000 file
def abstention_short(max_chars=420):
    out = []
    for line in open(ABST, encoding="utf-8"):
        d = json.loads(line)
        resp = d["response"]
        if "</think>" not in resp:
            continue
        ans = resp.split("</think>", 1)[1].strip()
        if not ans or len(ans) > max_chars:
            continue
        out.append((d["lang"], d["prompt"], ans, "abstention_v4_short", d.get("id", "")))
    return out


# ------------------------------------------------------------------ identity
PREFIX = ["", "", "Quick question: ", "Hey, ", "Tell me, ", "Just curious: ", "Please answer briefly. "]


def identity_rows(r):
    banned = set()
    for f in EVAL_FILES:
        if f.exists():
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    banned.add(norm(json.loads(line)["prompt"]))
    rows, seen = [], set()
    for line in open(IDENT, encoding="utf-8"):
        d = json.loads(line)
        p, resp, lang = d["prompt"], d["response"], d.get("lang", "eng")
        direct = resp.split("</think>", 1)[1].strip() if "</think>" in resp else resp
        for pre in PREFIX:
            q = (pre + p) if lang == "eng" else p
            nq = norm(q)
            if nq in banned or any(len(b) > 20 and b in nq for b in banned):
                continue
            for variant, text in (("direct", direct), ("think", resp)):
                key = (q, variant)
                if key in seen or (variant == "think" and pre):
                    continue
                seen.add(key)
                rows.append((lang, q, text, f"identity_{variant}", d.get("id", "")))
    return rows


# ------------------------------------------------------------------ driver
def write(path: Path, rows, bucket, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (lang, p, resp, fam, sid) in enumerate(rows):
            f.write(json.dumps({"id": f"{source}_{i:06d}", "source": source, "bucket": bucket, "lang": lang,
                                "prompt": p, "response": resp, "family": fam, "src_id": sid}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):6d} rows -> {path}")


def build(out_dir: Path, seed: int, n_fp: int, n_benign: int, n_squad: int, n_sum: int):
    r = random.Random(seed)
    fp, seen = [], set()
    while len(fp) < n_fp:
        lang, q, a, fam = make_false_premise(r)
        if q in seen:
            continue
        seen.add(q); fp.append((lang, q, a, fam, ""))
    write(out_dir / "false_premise.jsonl", fp, "abstention", "s10_gen_honesty")
    bn, seen = [], set()
    for _ in range(n_benign * 6):
        q, a, fam = make_benign(r)
        if q not in seen:
            seen.add(q); bn.append(("eng", q, a, "benign_" + fam, ""))
        if len(bn) >= n_benign:
            break
    write(out_dir / "benign_facts.jsonl", bn, "abstention", "s10_gen_honesty")
    write(out_dir / "grounded.jsonl", squad_rows(r, n_squad), "grounded_reading", "s10_squad_v2")
    write(out_dir / "unans_math.jsonl", sum_rows(r, n_sum), "abstention", "s10_sum")
    write(out_dir / "abstention_short.jsonl", abstention_short(), "abstention", "s10_abst4000_short")
    write(out_dir / "identity.jsonl", identity_rows(r), "identity", "s10_identity")


def self_test() -> None:
    r = random.Random(3)
    for _ in range(400):
        lang, q, a, fam = make_false_premise(r)
        assert q and a and ("{" not in q) and ("{" not in a), (q, a)
        if lang in IDK_NAT:
            assert a == IDK_NAT[lang]
    names = {en_word(r) for _ in range(500)}
    assert not (names & set(CAPITALS.values()))
    for _ in range(200):
        q, a, fam = make_benign(r)
        if fam == "unit":
            n = int(re.search(r"in (\d+) metres", q)[1]); assert a.endswith(f"{n * 100} centimetres.")
    assert all("Eiffel" not in x[0] for x in LANDMARKS)
    print("[self-test] s10_gen_honesty ok")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "s10_gen")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--n-fp", type=int, default=40000)
    ap.add_argument("--n-benign", type=int, default=600)
    ap.add_argument("--n-squad", type=int, default=9000)
    ap.add_argument("--n-sum", type=int, default=6000)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    build(a.out_dir, a.seed, a.n_fp, a.n_benign, a.n_squad, a.n_sum)
    return 0


if __name__ == "__main__":
    sys.exit(main())
