#!/usr/bin/env python3
"""Comprehensive benchmark & evaluation suite for FreeThink Mode on BPE checkpoints.

Tests all 4 pillars of the Parentheses FreeThink stack:
  1. 9-Language Continuation FreeThink (FreeThinkSession, mode='continuation')
  2. Structured <think> FreeThink (FreeThinkSession, mode='structured')
  3. Modular FreeThink BM25 RAG Grounding (ModularFreeThinkSession, multi-splice)
  4. Conversation Memory Auto-Trigger & Multi-Turn Recall (ConversationSession)
     + O(1) Selective Linear Attention per-token latency invariance profile.

Usage:
    .venv\\Scripts\\python.exe scripts\\eval_freethink.py --self-test
    .venv\\Scripts\\python.exe scripts\\eval_freethink.py \\
        --checkpoint checkpoints/parentheses-0.9-120M/step_9536_final.pt \\
        --out out/freethink_eval_step9536.json \\
        --samples-out out/freethink_samples_step9536.jsonl
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_dravidian import LANG_SCRIPT, distinct_n, repetition_rate, script_of  # noqa: E402
from features.conversation_memory import (  # noqa: E402
    ConversationNotes,
    ConversationSession,
    warrants_free_think,
)
from features.free_think import (  # noqa: E402
    FreeThinkSession,
    is_question,
    load_bpe_tokenizer,
    timing_summary,
)
from features.modular_free_think import IngestedStore, ModularFreeThinkSession  # noqa: E402
from model import Parentheses  # noqa: E402
from model.config import ModelConfig  # noqa: E402

LEAK_MARKERS = ("<|user|>", "<|assistant|>", "<|system|>", "<|pad|>", "<esg>", "<tool_call>", "</tool_call>")

# 18 statements across all 9 languages (2 per language) for Pillar 1
FREETHINK_STATEMENTS = [
    {"id": "fth-eng-01", "lang": "eng", "prompt": "The deep ocean remains one of the least explored frontiers on Earth, holding ecosystems that thrive without sunlight."},
    {"id": "fth-eng-02", "lang": "eng", "prompt": "Writing software from scratch teaches you how every layer of abstraction actually works beneath the surface."},
    {"id": "fth-kan-01", "lang": "kan", "prompt": "ಪಶ್ಚಿಮ ಘಟ್ಟಗಳ ಕಾಡುಗಳು ಕರ್ನಾಟಕದ ನದಿಗಳಿಗೆ ಮತ್ತು ಜೀವವೈವಿಧ್ಯಕ್ಕೆ ಪ್ರಮುಖ ಆಧಾರವಾಗಿವೆ."},
    {"id": "fth-kan-02", "lang": "kan", "prompt": "ಗ್ರಾಮೀಣ ಭಾಗದ ಜನಪದ ಕಲೆಗಳು ತಲೆಮಾರುಗಳಿಂದ ಜನರ ಬದುಕಿನ ಅನುಭವಗಳನ್ನು ಉಳಿಸಿಕೊಂಡು ಬಂದಿವೆ."},
    {"id": "fth-tam-01", "lang": "tam", "prompt": "சங்க இலக்கியப் பாடல்கள் பண்டைய தமிழர்களின் இயற்கை வாழ்வியலையும் பண்பாட்டையும் தெளிவாகப் பதிவு செய்துள்ளன."},
    {"id": "fth-tam-02", "lang": "tam", "prompt": "கடல்சார் வணிகம் பல நூற்றாண்டுகளாகத் தென்னிந்தியத் துறைமுகங்களை உலக நாடுகளுடன் இணைத்தது."},
    {"id": "fth-tel-01", "lang": "tel", "prompt": "గోదావరి నది పరీవాహక ప్రాంతం తెలుగు రాష్ట్రాల వ్యవసాయానికి మరియు సంస్కృతికి జీవనాడిగా ఉంది."},
    {"id": "fth-tel-02", "lang": "tel", "prompt": "పుస్తక పఠనం మనిషి ఆలోచనా పరిధిని విస్తరించి కొత్త విషయాలను అర్థం చేసుకునే శక్తిని ఇస్తుంది."},
    {"id": "fth-mal-01", "lang": "mal", "prompt": "കേരളത്തിലെ കായലുകളും പശ്ചിമഘട്ട മലനിരകളും കാലാവസ്ഥയെ നിയന്ത്രിക്കുന്നതിൽ വലിയ പങ്കുവഹിക്കുന്നു."},
    {"id": "fth-mal-02", "lang": "mal", "prompt": "മാതൃഭാഷയിലുള്ള പഠനം കുട്ടികളുടെ ചിന്താശേഷിയും സർഗ്ഗാത്മകതയും വളർത്താൻ സഹായിക്കുന്നു."},
    {"id": "fth-tcy-01", "lang": "tcy", "prompt": "ತುಳುನಾಡ್ ದ ಸಂಸ್ಕೃತಿ ಬೊಕ್ಕ ಭೂತಾರಾಧನೆ ಜನಕುಲೆನ ಬದುಕುದ ಒಂಜಿ ಮುಖ್ಯ ಭಾಗ ಆದುಂಡು."},
    {"id": "fth-tcy-02", "lang": "tcy", "prompt": "ಕರಾವಳಿ ಪ್ರದೇಶದ ಕೃಷಿ ಬೊಕ್ಕ ಮೀನುಗಾರಿಕೆ ಜನಕುಲೆನ ಜೀವನೊಗು ಆಧಾರ ಆದುಂಡು."},
    {"id": "fth-kff-01", "lang": "kff", "prompt": "ଇଦ୍ ନାୟୁଁତେ ଲୋକ୍‌କୁ କୃଷି ଆରି ବଣ ଜଙ୍ଗଲ ତଲେ ମିଶି ଜିବନ କିନେରି ।"},
    {"id": "fth-kff-02", "lang": "kff", "prompt": "ଗାଁ ଲୋକ୍‌କୁ ମିଳିମିଶି କାମ କିଲେ ସବୁ କష్ట ଦୂର ଆୟେ ।"},
    {"id": "fth-kxv-01", "lang": "kxv", "prompt": "ମାଣ୍‌ସି ମୀର୍‌ଏଣା ଦିନାତା ଆ଼ନେ, ସାମା ଏ଼ୱାସି ଗାଡି କସ୍ତ ପା଼ଣ୍‌ନାୟି ମାନେ ।"},
    {"id": "fth-kxv-02", "lang": "kxv", "prompt": "ନା଼ୟୁଁତି ଲ଼କୁ ରଅତେରି ରଅଣାଇଁ ସା଼ୟେମ କିହିଁ ମାଞ୍ଜାନା ସବୁ କାମ କିନେରି ।"},
    {"id": "fth-nit-01", "lang": "nit", "prompt": "మా ఊరిలో పంటలు మరియు అడవులు ప్రజల జీవనానికి ముఖ్య ఆధారము."},
    {"id": "fth-nit-02", "lang": "nit", "prompt": "మంచి మాటలు మరియు సహాయం మనుషుల మధ్య స్నేహాన్ని పెంచుతాయి."},
]

# 5 statements for Pillar 2 (Structured <think> mode)
STRUCTURED_STATEMENTS = [
    {"id": "str-eng-01", "lang": "eng", "prompt": "Recurrent state space models process sequences in linear time with constant memory state size, unlike quadratic attention."},
    {"id": "str-kan-01", "lang": "kan", "prompt": "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಮಾಡುವುದರಿಂದ ಬೇಸಿಗೆಯಲ್ಲಿ ಕುಡಿಯುವ ನೀರಿನ ಕೊರತೆಯನ್ನು ತಪ್ಪಿಸಬಹುದು."},
    {"id": "str-tam-01", "lang": "tam", "prompt": "மரங்களை நட்டுப் பாதுகாப்பது நகரங்களின் வெப்பநிலையைக் குறைக்க உதவும்."},
    {"id": "str-tel-01", "lang": "tel", "prompt": "సౌరశక్తి వినియోగం వల్ల పర్యావరణ కాలుష్యం తగ్గడమే కాకుండా విద్యుత్ ఖర్చులు కూడా తగ్గుతాయి."},
    {"id": "str-mal-01", "lang": "mal", "prompt": "പരിസ്ഥിതി സംരക്ഷണവും സുസ്ഥിര വികസനവും ഒരുമിച്ച് കൊണ്ടുപോകേണ്ടത് ഭാവിയുടെ ആവശ്യമാണ്."},
]


def check_script(lang: str, text: str) -> bool:
    want = LANG_SCRIPT.get(lang)
    got = script_of(text)
    if lang == "eng":
        return got is None and any(c.isascii() and c.isalpha() for c in text)
    return got == want


def has_role_leak(text: str) -> bool:
    return any(m in text for m in LEAK_MARKERS)


def evaluate_model(model: Parentheses, tok, device: str, max_tokens: int = 160,
                   min_tokens: int = 48, latency_tokens: int = 300, seed: int = 42) -> tuple[dict, list[dict]]:
    samples = []

    # 1. Question rejection gate check
    q_checks = [
        ("Why does the moon orbit the Earth?", True),
        ("ಏಕೆ ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ?", True),
        ("The moon orbits the Earth every twenty-seven days.", False),
        ("ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.", False),
    ]
    q_gate_ok = all(is_question(q) == want for q, want in q_checks)

    # 2. Pillar 1: 9-language Continuation FreeThink
    cont_results = []
    for item in FREETHINK_STATEMENTS:
        torch.manual_seed(seed)
        sess = FreeThinkSession(model, temperature=0.85, top_k=40,
                                tokenizer=tok, lang=item["lang"],
                                mode="continuation", min_tokens=min_tokens)
        t0 = time.perf_counter()
        out = sess.run(item["prompt"], max_tokens=max_tokens)
        elapsed = round(time.perf_counter() - t0, 3)
        n_tok = len(tok.encode(out, add_special_tokens=False).ids) if tok else len(out.encode("utf-8"))
        s_ok = check_script(item["lang"], out)
        leak = has_role_leak(out)
        rep = round(repetition_rate(out), 4)
        d2 = round(distinct_n(out, 2), 4)
        rec = {
            "pillar": "continuation",
            "id": item["id"],
            "lang": item["lang"],
            "prompt": item["prompt"],
            "output": out,
            "n_tokens": n_tok,
            "n_chars": len(out),
            "script": script_of(out),
            "script_ok": s_ok,
            "role_leak": leak,
            "repetition_rate": rep,
            "distinct_2": d2,
            "seconds": elapsed,
        }
        cont_results.append(rec)
        samples.append(rec)

    # 3. Pillar 2: Structured <think> FreeThink
    struct_results = []
    for item in STRUCTURED_STATEMENTS:
        torch.manual_seed(seed)
        sess = FreeThinkSession(model, temperature=0.85, top_k=40,
                                tokenizer=tok, lang=item["lang"],
                                mode="structured", min_tokens=min_tokens)
        t0 = time.perf_counter()
        out = sess.run(item["prompt"], max_tokens=max_tokens)
        elapsed = round(time.perf_counter() - t0, 3)
        clean_for_script = out.replace("</think>", "")
        n_tok = len(tok.encode(clean_for_script, add_special_tokens=False).ids) if tok else len(out.encode("utf-8"))
        s_ok = check_script(item["lang"], clean_for_script)
        leak = has_role_leak(clean_for_script)
        rec = {
            "pillar": "structured",
            "id": item["id"],
            "lang": item["lang"],
            "prompt": item["prompt"],
            "output": out,
            # the visible text has tags stripped, so ask the session what the model sampled
            "closed_think": sess.model_closed_think,
            "n_tokens": n_tok,
            "n_chars": len(out),
            "script": script_of(clean_for_script),
            "script_ok": s_ok,
            "role_leak": leak,
            "repetition_rate": round(repetition_rate(clean_for_script), 4),
            "distinct_2": round(distinct_n(clean_for_script, 2), 4),
            "seconds": elapsed,
        }
        struct_results.append(rec)
        samples.append(rec)

    # 4. Pillar 3: Modular FreeThink (BM25 RAG Grounding in English & Kannada)
    modular_results = []
    with tempfile.TemporaryDirectory() as tmp:
        eng_doc = Path(tmp) / "eng_notes.txt"
        eng_doc.write_text(
            "Project Peridot uses the Selective Linear Attention architecture with 32 layers and 512 hidden dimension.\n\n"
            "The hydrothermal vents of the Mariana Trench reach temperatures of 400 degrees Celsius while the seabed stays near freezing.",
            encoding="utf-8",
        )
        kan_doc = Path(tmp) / "kan_notes.txt"
        kan_doc.write_text(
            "ಹಂಪಿಯ ವಿಜಯನಗರ ಸಾಮ್ರಾಜ್ಯವನ್ನು ಹಕ್ಕ ಮತ್ತು ಬುಕ್ಕರು ೧೩೩೬ ರಲ್ಲಿ ಸ್ಥಾಪಿಸಿದರು.\n\n"
            "ಕಾವೇರಿ ನದಿಯು ಕೊಡಗು ಜಿಲ್ಲೆಯ ತಲಕಾವೇರಿಯಲ್ಲಿ ಹುಟ್ಟಿ ಬಂಗಾಳಕೊಲ್ಲಿಯನ್ನು ಸೇರುತ್ತದೆ.",
            encoding="utf-8",
        )
        store = IngestedStore()
        store.ingest([eng_doc, kan_doc])

        for mid, mlang, mprompt, want_sub in [
            ("mod-eng-01", "eng", "The Mariana Trench has hydrothermal vents near the seabed.", "400 degrees"),
            ("mod-kan-01", "kan", "ಕಾವೇರಿ ನದಿಯು ಕೊಡಗು ಜಿಲ್ಲೆಯಲ್ಲಿ ಹುಟ್ಟುತ್ತದೆ.", "ತಲಕಾವೇರಿ"),
        ]:
            torch.manual_seed(seed)
            msess = ModularFreeThinkSession(
                model, store, splice_every=max(16, max_tokens // 3),
                retrieve_k=2, temperature=0.85, top_k=40,
                tokenizer=tok, lang=mlang,
            )
            primed_preview = msess._primed(mprompt)
            retrieved_ok = want_sub in primed_preview
            out = msess.run(mprompt, max_tokens=max_tokens)
            export_path = msess.export(f"{mid}.json", base_dir=tmp)
            exported_ok = export_path.exists() and json.loads(export_path.read_text(encoding="utf-8"))["prompt"] == mprompt
            rec = {
                "pillar": "modular_rag",
                "id": mid,
                "lang": mlang,
                "prompt": mprompt,
                "retrieved_in_primed": retrieved_ok,
                "exported_ok": exported_ok,
                "output": out,
                "n_chars": len(out),
                "script_ok": check_script(mlang, out),
                "role_leak": has_role_leak(out),
            }
            modular_results.append(rec)
            samples.append(rec)

        # 5. Pillar 4: Conversation Memory Auto-Trigger & Multi-Turn Recall
        conv_notes = ConversationNotes("eval_conv.md", base_dir=tmp)
        conv_sess = ConversationSession(
            model, conv_notes, splice_every=max(16, max_tokens // 2),
            temperature=0.85, top_k=40, tokenizer=tok, lang="eng",
        )
        t1_out = "".join(conv_sess.turn("The telescope mirror in dome four was recoated with aluminum on Tuesday."))
        t2_out = "".join(conv_sess.turn("When was the telescope mirror recoated?"))
        heavy_turn = (
            "I have been spending every night in dome four checking the telescope mirror "
            "alignment and I still think my calibration drift is coming from the mount"
        )
        torch.manual_seed(seed)
        t3_out = "".join(conv_sess.turn(heavy_turn, max_tokens=max_tokens))
        conv_store = conv_sess._store()
        recalled_t1 = any("aluminum" in c for c in conv_store.retrieve("telescope mirror dome four"))
        conv_summary = {
            "turn1_short_silent": t1_out == "",
            "turn2_question_silent": t2_out == "",
            "turn3_heavy_triggered": len(t3_out) > 0,
            "turn1_recalled_from_notes": recalled_t1,
            "turn3_chars": len(t3_out),
        }
        samples.append({
            "pillar": "conversation_memory",
            "id": "conv-eng-01",
            "lang": "eng",
            "prompt": heavy_turn,
            "output": t3_out,
            **conv_summary,
        })

    # 6. O(1) Selective Linear Attention Latency Invariance Check
    torch.manual_seed(seed)
    timings = []
    lat_sess = FreeThinkSession(model, temperature=0.85, top_k=40,
                                tokenizer=tok, lang="eng",
                                mode="continuation", min_tokens=latency_tokens)
    _ = lat_sess.run(
        "The selective linear attention state space recurrence updates a fixed-size matrix state per step.",
        max_tokens=latency_tokens,
        timings=timings,
    )
    body = timings[1:] if len(timings) > 1 else timings
    win = max(1, min(50, len(body) // 2))
    first_ms = round(statistics.median(body[:win]) * 1000, 3) if body else 0.0
    last_ms = round(statistics.median(body[-win:]) * 1000, 3) if body else 0.0
    ratio = round(last_ms / max(first_ms, 1e-6), 3) if body else 1.0

    # Aggregate summary
    by_lang = {}
    for lang in sorted({r["lang"] for r in cont_results}):
        rs = [r for r in cont_results if r["lang"] == lang]
        by_lang[lang] = {
            "n": len(rs),
            "script_ok_rate": round(sum(r["script_ok"] for r in rs) / len(rs), 4),
            "mean_tokens": round(sum(r["n_tokens"] for r in rs) / len(rs), 1),
            "mean_chars": round(sum(r["n_chars"] for r in rs) / len(rs), 1),
            "role_leaks": sum(r["role_leak"] for r in rs),
            "mean_rep_rate": round(sum(r["repetition_rate"] for r in rs) / len(rs), 4),
            "mean_distinct_2": round(sum(r["distinct_2"] for r in rs) / len(rs), 4),
        }

    summary = {
        "question_gate_ok": q_gate_ok,
        "continuation": {
            "total": len(cont_results),
            "script_ok_count": sum(r["script_ok"] for r in cont_results),
            "script_ok_rate": round(sum(r["script_ok"] for r in cont_results) / len(cont_results), 4),
            "role_leak_count": sum(r["role_leak"] for r in cont_results),
            "mean_tokens": round(sum(r["n_tokens"] for r in cont_results) / len(cont_results), 1),
            "mean_distinct_2": round(sum(r["distinct_2"] for r in cont_results) / len(cont_results), 4),
            "mean_repetition_rate": round(sum(r["repetition_rate"] for r in cont_results) / len(cont_results), 4),
            "by_lang": by_lang,
        },
        "structured": {
            "total": len(struct_results),
            "script_ok_count": sum(r["script_ok"] for r in struct_results),
            "closed_think_count": sum(r["closed_think"] is True for r in struct_results),
            "role_leak_count": sum(r["role_leak"] for r in struct_results),
            "mean_tokens": round(sum(r["n_tokens"] for r in struct_results) / len(struct_results), 1),
        },
        "modular_rag": {
            "total": len(modular_results),
            "retrieved_in_primed_count": sum(r["retrieved_in_primed"] for r in modular_results),
            "exported_ok_count": sum(r["exported_ok"] for r in modular_results),
            "script_ok_count": sum(r["script_ok"] for r in modular_results),
        },
        "conversation_memory": conv_summary,
        "latency_profile": {
            "n_tokens": len(timings),
            "window": win,
            "median_first_ms": first_ms,
            "median_last_ms": last_ms,
            "last_vs_first_ratio": ratio,
            "summary": timing_summary(timings, window=win),
        },
    }
    return summary, samples


def _self_test():
    tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
    tok = load_bpe_tokenizer({"tokenizer": {"path": str(tok_dir)}})
    bpe = Parentheses(ModelConfig(vocab_size=32768, block_size=512, n_layer=2, n_head=2, n_embd=32,
                                  attn_type="selective_linear")).eval()
    summary, samples = evaluate_model(bpe, tok, device="cpu", max_tokens=12, min_tokens=4, latency_tokens=16)
    assert summary["question_gate_ok"] is True
    assert summary["continuation"]["total"] == 18
    assert summary["continuation"]["role_leak_count"] == 0
    assert summary["structured"]["total"] == 5
    assert summary["modular_rag"]["retrieved_in_primed_count"] == 2
    assert summary["modular_rag"]["exported_ok_count"] == 2
    assert summary["conversation_memory"]["turn1_short_silent"] is True
    assert summary["conversation_memory"]["turn2_question_silent"] is True
    assert summary["conversation_memory"]["turn3_heavy_triggered"] is True
    assert summary["conversation_memory"]["turn1_recalled_from_notes"] is True
    assert len(samples) == 18 + 5 + 2 + 1
    print("[self-test] eval_freethink ok")


def main():
    p = argparse.ArgumentParser(description="Evaluate all 4 pillars of FreeThink Mode on a BPE checkpoint.")
    p.add_argument("--checkpoint", default="checkpoints/parentheses-0.9-120M/step_9536_final.pt")
    p.add_argument("--out", default="out/freethink_eval_step9536.json")
    p.add_argument("--samples-out", default="out/freethink_samples_step9536.jsonl")
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--min-tokens", type=int, default=48)
    p.add_argument("--latency-tokens", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    print(f"[eval-freethink] Loading {args.checkpoint} on {args.device}...")
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device).eval()
    model.load_state_dict(ckpt["model"])
    tok = load_bpe_tokenizer(ckpt)

    summary, samples = evaluate_model(
        model, tok, device=args.device,
        max_tokens=args.max_tokens, min_tokens=args.min_tokens,
        latency_tokens=args.latency_tokens, seed=args.seed,
    )
    summary["checkpoint"] = args.checkpoint
    summary["step"] = ckpt.get("step")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    os.makedirs(os.path.dirname(args.samples_out) or ".", exist_ok=True)
    with open(args.samples_out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[eval-freethink] Wrote summary to {args.out} and {len(samples)} samples to {args.samples_out}")
    print(f"  Continuation (18 prompts, 9 langs): script_ok={summary['continuation']['script_ok_count']}/18 "
          f"({summary['continuation']['script_ok_rate']*100:.1f}%), leaks={summary['continuation']['role_leak_count']}, "
          f"mean_tokens={summary['continuation']['mean_tokens']}")
    print(f"  Structured <think> (5 prompts):     script_ok={summary['structured']['script_ok_count']}/5, "
          f"closed_think={summary['structured']['closed_think_count']}/5, mean_tokens={summary['structured']['mean_tokens']}")
    print(f"  Modular RAG (2 stores):             retrieved={summary['modular_rag']['retrieved_in_primed_count']}/2, "
          f"exported={summary['modular_rag']['exported_ok_count']}/2")
    print(f"  Conversation Memory:                {summary['conversation_memory']}")
    print(f"  O(1) Latency Profile:               {summary['latency_profile']['summary']}")


if __name__ == "__main__":
    main()
