#!/usr/bin/env python3
# Engineered by uncoalesced
"""
scripts/generate_sft_abstention_v4.py
Generates 4,000 multilingual epistemic abstention SFT traces (800 per language:
English, Kannada, Tamil, Telugu, Malayalam) for Parentheses 1.0 (240M / slab_v4).
Ensures ~3.5% SFT prior probability mass on honest abstention with 100% <think> closure.
"""

from __future__ import annotations
import argparse
import json
import os
import random
from typing import Dict, List, Tuple


def build_english_catalog() -> List[Tuple[str, str, str]]:
    """Returns (prompt, think_trace, abstention_response) tuples in English."""
    years = [1894, 1919, 1937, 1954, 1966, 1973, 1981, 1988, 1995, 2002, 2009, 2014]
    places = [
        "Velanthor", "Sahyadri", "Coromandel", "Travancore", "Rayalaseema",
        "Dandakaranya", "Malabar", "Nilgiri", "Tungabhadra", "Godavari",
        "Kaveri", "Vijayanagara", "Hoysala", "Chola", "Chalukya", "Kakatiya",
        "Kodagu", "Wayanad", "Tirunelveli", "Warangal",
    ]
    artifacts = [
        ("Maritime Accord", "diplomatic treaty"),
        ("Cryogenic Observatory", "scientific facility"),
        ("Quantum Rail Protocol", "transport standard"),
        ("Basalt Superconductor", "material discovery"),
        ("Sub-Sea Tunnel Blueprint", "civil engineering project"),
        ("Antimatter Containment Trial", "physics experiment"),
        ("Heavy-Water Monorail", "transit system"),
        ("Magneto-Hydrodynamic Grid", "energy grid"),
        ("Acoustic Levitation Codex", "historical manuscript"),
        ("Piezoelectric Highway Charter", "infrastructure charter"),
        ("Stratospheric Dirigible Fleet", "aviation program"),
        ("Hyper-Lattice Decay Theorem", "mathematical theorem"),
        ("Chronometric Standard", "timekeeping convention"),
        ("Neutrino Dam", "particle physics facility"),
    ]
    q_templates = [
        "What were the primary provisions and outcomes of the {yr} {pl} {art}?",
        "Explain the technical architecture and historical impact of the {yr} {pl} {art}.",
        "Who signed or authored the {yr} {pl} {art}, and what did it establish?",
        "Provide a detailed summary of the {yr} {pl} {art} and its key clauses.",
    ]

    samples: List[Tuple[str, str, str]] = []
    idx = 0
    for yr in years:
        for pl in places:
            for art, cat in artifacts:
                q_tmpl = q_templates[idx % len(q_templates)]
                entity = f"{yr} {pl} {art}"
                prompt = q_tmpl.format(yr=yr, pl=pl, art=art)
                think = (
                    f"The user is asking about the '{entity}', described as a {cat}. "
                    f"Evaluating historical and technical records for {pl} in {yr}: no verified "
                    f"{art.lower()} exists. This is a fabricated or unverified premise, so I must abstain "
                    f"rather than invent details."
                )
                resp = (
                    f"I cannot verify any historical or technical record of the {entity}. "
                    f"No documented {cat} by that name appears in established sources, so I cannot "
                    f"provide provisions or details for it."
                )
                samples.append((prompt, think, resp))
                idx += 1
                if len(samples) >= 800:
                    return samples
    return samples[:800]


def build_kannada_catalog() -> List[Tuple[str, str, str]]:
    years = ["1894ರ", "1921ರ", "1954ರ", "1968ರ", "1975ರ", "1983ರ", "1991ರ", "1999ರ", "2004ರ", "2012ರ"]
    places = ["ವೇಲಾಂಥೂರ್", "ಸಹ್ಯಾದ್ರಿ", "ಕಾವೇರಿ", "ತುಂಗಭದ್ರಾ", "ಮೈಸೂರು", "ಕೊಡಗು", "ಮಂಗಳೂರು", "ಬಳ್ಳಾರಿ", "ಉಡುಪಿ", "ಹಾಸನ"]
    artifacts = [
        ("ಕಡಲ ಒಪ್ಪಂದ", "ಐತಿಹಾಸಿಕ ಒಪ್ಪಂದ"),
        ("ಕ್ರಯೋಜೆನಿಕ್ ವೀಕ್ಷಣಾಲಯ", "ವೈಜ್ಞಾನಿಕ ಕೇಂದ್ರ"),
        ("ಕ್ವಾಂಟಮ್ ರೈಲು ಶಿಷ್ಟಾಚಾರ", "ಸಾರಿಗೆ ಯೋಜನೆ"),
        ("ಬಸಾಲ್ಟ್ ಸೂಪರ್‌ಕಂಡಕ್ಟರ್", "ಭೌತಶಾಸ್ತ್ರ ಸಂಶೋಧನೆ"),
        ("ಆಂಟಿಮ್ಯಾಟರ್ ಪ್ರಯೋಗ", "ಪರಮಾಣು ಪ್ರಯೋಗ"),
        ("ಭೂಗತ ಸಸ್ಯಾಗಾರ", "ಸಂಶೋಧನಾ ಕೇಂದ್ರ"),
        ("ಪೀಜೋ ಹೆದ್ದಾರಿ ಯೋಜನೆ", "ಮೂಲಸೌಕರ್ಯ ಯೋಜನೆ"),
        ("ನ್ಯೂಟ್ರಿನೊ ಅಣೆಕಟ್ಟು", "ಬೃಹತ್ ಯೋಜನೆ"),
    ]
    samples: List[Tuple[str, str, str]] = []
    for yr in years:
        for pl in places:
            for art, cat in artifacts:
                ent = f"{yr} {pl} {art}"
                prompt = f"{ent} ಕುರಿತು ಪ್ರಮುಖ ಮಾಹಿತಿ ಮತ್ತು ಐತಿಹಾಸಿಕ ವಿವರಗಳನ್ನು ತಿಳಿಸಿ."
                think = (
                    f"ಬಳಕೆದಾರರು '{ent}' ({cat}) ಕುರಿತು ಮಾಹಿತಿ ಕೇಳಿದ್ದಾರೆ. "
                    f"ದಾಖಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿದಾಗ ಇಂತಹ ಯಾವುದೇ ಅಧಿಕೃತ {cat} ನಡೆದಿರುವುದು ಕಂಡುಬರುವುದಿಲ್ಲ. "
                    f"ಇದು ದೃಢೀಕರಿಸದ ಅಥವಾ ಕಾಲ್ಪನಿಕ ವಿಷಯವಾಗಿರುವುದರಿಂದ ನೇರವಾಗಿ ನಿರಾಕರಿಸಬೇಕು."
                )
                resp = (
                    f"'{ent}' ಕುರಿತು ಯಾವುದೇ ಅಧಿಕೃತ ಐತಿಹಾಸಿಕ ಅಥವಾ ವೈಜ್ಞಾನಿಕ ದಾಖಲೆಗಳು ಲಭ್ಯವಿಲ್ಲ. "
                    f"ಇದು ದೃಢೀಕರಿಸದ ಅಥವಾ ಕಾಲ್ಪನಿಕ ವಿಷಯವಾಗಿರುವುದರಿಂದ ಇದರ ವಿವರಗಳನ್ನು ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ."
                )
                samples.append((prompt, think, resp))
    return samples[:800]


def build_tamil_catalog() -> List[Tuple[str, str, str]]:
    years = ["1894", "1921", "1954", "1968", "1975", "1983", "1991", "1999", "2004", "2012"]
    places = ["வேலந்தூர்", "சஹ்யாத்ரி", "காவிரி", "சோழமண்டல", "மைசூர்", "மதுரை", "தஞ்சாவூர்", "திருநெல்வேலி", "கொச்சி", "பாலக்காடு"]
    artifacts = [
        ("கடல்சார் ஒப்பந்தம்", "வரலாற்று ஒப்பந்தம்"),
        ("கிரையோஜெனிக் ஆய்வகம்", "அறிவியல் நிலையம்"),
        ("குவாண்டம் ரயில் நெறிமுறை", "போக்குவரத்துத் திட்டம்"),
        ("பசால்ட் மீக்கடத்தி", "இயற்பியல் கண்டுபிடிப்பு"),
        ("எதிர்பொருள் சோதனை", "அறிவியல் சோதனை"),
        ("நிலத்தடி தாவரவியல் காப்பகம்", "ஆய்வு மையம்"),
        ("அழுத்தமின் நெடுஞ்சாலை", "கட்டமைப்புத் திட்டம்"),
        ("நியூட்ரினோ அணை", "ஆராய்ச்சித் திட்டம்"),
    ]
    samples: List[Tuple[str, str, str]] = []
    for yr in years:
        for pl in places:
            for art, cat in artifacts:
                ent = f"{yr} {pl} {art}"
                prompt = f"{ent} பற்றிய முக்கிய வரலாற்று மற்றும் தொழில்நுட்ப விவரங்களை விளக்கவும்."
                think = (
                    f"பயனர் '{ent}' ({cat}) பற்றிக் கேட்கிறார். "
                    f"வரலாற்று மற்றும் அறிவியல் சான்றுகளைச் சரிபார்க்கும்போது இப்படியொரு {cat} "
                    f"உண்மையில் நடந்ததற்கான ஆதாரம் எதுவும் இல்லை. எனவே கற்பனையான தகவலைத் தவிர்த்து உண்மையைக் கூற வேண்டும்."
                )
                resp = (
                    f"'{ent}' குறித்து எந்தவொரு அதிகாரப்பூர்வ வரலாற்று அல்லது அறிவியல் ஆதாரமும் இல்லை. "
                    f"இது உறுதிப்படுத்தப்படாத அல்லது கற்பனையான தகவல் என்பதால் இதன் விவரங்களை வழங்க இயலாது."
                )
                samples.append((prompt, think, resp))
    return samples[:800]


def build_telugu_catalog() -> List[Tuple[str, str, str]]:
    years = ["1894", "1921", "1954", "1968", "1975", "1983", "1991", "1999", "2004", "2012"]
    places = ["వేలంతోర్", "సహ్యాద్రి", "కావేరీ", "రాయలసీమ", "గోదావరి", "నెల్లూరు", "వరంగల్", "బళ్లారి", "కోరమండల్", "కొడగు"]
    artifacts = [
        ("సముద్ర ఒప్పందం", "చారిత్రక ఒప్పందం"),
        ("క్రయోజెనిక్ పరిశోధనాలయం", "శాస్త్రీయ కేంద్రం"),
        ("క్వాంటమ్ రైలు ప్రోటోకాల్", "రవాణా ప్రాజెక్టు"),
        ("బసాల్ట్ సూపర్ కండక్టర్", "భౌతికశాస్త్ర ఆవిష్కరణ"),
        ("యాంటీమ్యాటర్ ప్రయోగం", "శాస్త్రీయ ప్రయోగం"),
        ("భూగర్భ బొటానికల్ వాల్ట్", "పరిశోధనా కేంద్రం"),
        ("పీజో ఎలక్ట్రిక్ హైవే", "మౌలిక సదుపాయాల ప్రాజెక్టు"),
        ("న్యూట్రినో ఆనకట్ట", "భారీ ప్రాజెక్టు"),
    ]
    samples: List[Tuple[str, str, str]] = []
    for yr in years:
        for pl in places:
            for art, cat in artifacts:
                ent = f"{yr} {pl} {art}"
                prompt = f"{ent} యొక్క ప్రధాన చారిత్రక మరియు సాంకేతిక వివరాలను తెలియజేయండి."
                think = (
                    f"వినియోగదారు '{ent}' ({cat}) గురించి అడుగుతున్నారు. "
                    f"చారిత్రక మరియు శాస్త్రీయ రికార్డులను పరిశీలిస్తే ఇలాంటి {cat} ఉన్నట్లు ఎలాంటి ఆధారాలు లేవు. "
                    f"ఇది ధృవీకరించబడని అంశం కాబట్టి ఊహాజనిత సమాధానం ఇవ్వకూడదు."
                )
                resp = (
                    f"'{ent}' గురించి ఎలాంటి విశ్వసనీయ చారిత్రక లేదా శాస్త్రీయ ఆధారాలు లేవు. "
                    f"ఇది కల్పితమైన లేదా ధృవీకరించబడని అంశం కనుక దీని వివరాలను అందించలేము."
                )
                samples.append((prompt, think, resp))
    return samples[:800]


def build_malayalam_catalog() -> List[Tuple[str, str, str]]:
    years = ["1894-ലെ", "1921-ലെ", "1954-ലെ", "1968-ലെ", "1975-ലെ", "1983-ലെ", "1991-ലെ", "1999-ലെ", "2004-ലെ", "2012-ലെ"]
    places = ["വേലാന്തൂർ", "സഹ്യാദ്രി", "കാവേരി", "തിരുവിതാംകൂർ", "മലബാർ", "വയനാട്", "പാലക്കാട്", "കൊച്ചി", "നീലഗിരി", "കുടക്"]
    artifacts = [
        ("സമുദ്ര ഉടമ്പടി", "ചരിത്ര ഉടമ്പടി"),
        ("ക്രയോജനിക് നിരീക്ഷണാലയം", "ശാസ്ത്ര കേന്ദ്രം"),
        ("ക്വാണ്ടം റെയിൽ പ്രോട്ടോക്കോൾ", "ഗതാഗത പദ്ധതി"),
        ("ബസാൾട്ട് സൂപ്പർകണ്ടക്ടർ", "ഭൗതികശാസ്ത്ര കണ്ടെത്തൽ"),
        ("ആന്റിമാറ്റർ പരീക്ഷണം", "ശാസ്ത്ര പരീക്ഷണം"),
        ("ഭൂഗർഭ സസ്യ സംഭരണി", "ഗവേഷണ കേന്ദ്രം"),
        ("പീസോ ഇലക്ട്രിക് ഹൈവേ", "അടിസ്ഥാന സൗകര്യ പദ്ധതി"),
        ("ന്യൂട്രിനോ ഡാം", "പരീക്ഷണ പദ്ധതി"),
    ]
    samples: List[Tuple[str, str, str]] = []
    for yr in years:
        for pl in places:
            for art, cat in artifacts:
                ent = f"{yr} {pl} {art}"
                prompt = f"{ent} സംബന്ധിച്ച പ്രധാന ചരിത്രപരവും സാങ്കേതികവുമായ വിവരങ്ങൾ വിശദീകരിക്കുക."
                think = (
                    f"ഉപയോക്താവ് '{ent}' ({cat}) എന്നതിനെക്കുറിച്ചാണ് ചോദിക്കുന്നത്. "
                    f"ചരിത്രപരവും ശാസ്ത്രീയവുമായ രേഖകൾ പരിശോധിക്കുമ്പോൾ ഇത്തരമൊരു {cat} നിലവിലുള്ളതായി കാണുന്നില്ല. "
                    f"അതിനാൽ തെറ്റായ വിവരങ്ങൾ നിർമ്മിക്കാതെ വസ്തുത വ്യക്തമാക്കണം."
                )
                resp = (
                    f"'{ent}' സംബന്ധിച്ച് ആധികാരികമായ ചരിത്ര-ശാസ്ത്ര രേഖകളൊന്നും ലഭ്യമല്ല. "
                    f"ഇതൊരു സ്ഥിരീകരിക്കാത്തതോ സാങ്കൽപ്പികമോ ആയ വിഷയമായതിനാൽ കൂടുതൽ വിവരങ്ങൾ നൽകാൻ കഴിയില്ല."
                )
                samples.append((prompt, think, resp))
    return samples[:800]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 4,000 multilingual SFT abstention traces")
    parser.add_argument("--out", default="data/sft_v4_abstention_4000.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    catalogs = {
        "eng": build_english_catalog(),
        "kan": build_kannada_catalog(),
        "tam": build_tamil_catalog(),
        "tel": build_telugu_catalog(),
        "mal": build_malayalam_catalog(),
    }

    records: List[Dict[str, str]] = []
    for lang, items in catalogs.items():
        assert len(items) == 800, f"Expected 800 items for {lang}, got {len(items)}"
        for idx, (prompt, think, resp) in enumerate(items):
            # 85% with explicit <think>...</think>, 15% direct abstention so both modes have SFT support
            if idx % 20 < 17:
                assistant_text = f"<think>{think}</think>\n{resp}"
            else:
                assistant_text = resp
            records.append({
                "kind": "epistemic_abstention",
                "lang": lang,
                "prompt": prompt,
                "response": assistant_text,
                "formatted": f"<|user|><{lang}>{prompt}<|assistant|>{assistant_text}<|endoftext|>",
            })

    random.shuffle(records)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] Generated {len(records)} multilingual epistemic abstention SFT traces -> {args.out}")


if __name__ == "__main__":
    main()
