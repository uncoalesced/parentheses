#!/usr/bin/env python3
"""Comprehensive FreeThink Mode Test Suite for Parentheses 0.9-120M (SSM / Selective Linear Attention).

Thoroughly verifies all components of FreeThink Mode across all 9 supported languages:
  Anchor Dravidian: kan (Kannada), tam (Tamil), tel (Telugu), mal (Malayalam)
  Low-resource Dravidian: tcy (Tulu), kff (Koya), kxv (Kuvi), nit (Naikpod)
  Global anchor: eng (English)

Test Coverage:
  1. Non-question statement gating (rejection of questions, acceptance of statements across all 9 languages).
  2. Script preservation & native continuation cues across all 9 languages.
  3. Structured FreeThink Mode (<think>...</think>) token format, ensuring role token leak prevention & valid closure.
  4. Streaming ByteLevel BPE decoder: multi-byte character boundary chunking across UTF-8 boundaries (zero \\ufffd mid-stream).
  5. Latency constancy: benchmark per-token latency to verify O(1) recurrent memory update (S_t = a_t S_{t-1} + k_t (x) v_t).
  6. Modular RAG (BM25 IngestedStore) and Conversation Memory integration with FreeThink.
  7. CPU Integration tests on step_9636_final.pt (all 9 languages, structured mode, modular RAG, conversation memory).

Run:
  .venv\\Scripts\\python.exe tests/test_freethink_suite.py
"""

import codecs
import json
import os
import statistics
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

# Ensure repository root and scripts directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_dravidian import LANG_SCRIPT, script_of
from features.conversation_memory import (
    FIRST_PERSON,
    ConversationNotes,
    ConversationSession,
    warrants_free_think,
)
from features.free_think import (
    INTERNAL_COMMANDS,
    INTERNAL_TAGS,
    LANG_PRIMING,
    STRUCTURED_THINK_ANCHOR,
    FreeThinkSession,
    _QUESTION_STARTS,
    format_primed,
    is_question,
    load_bpe_tokenizer,
    parse_internal_command,
    strip_internal_tags,
    timing_summary,
    write_timings,
)
from features.modular_free_think import (
    IngestedStore,
    ModularFreeThinkSession,
    _chunk,
    _tokens,
)
from model import Parentheses
from model.config import ModelConfig

LEAK_MARKERS = (
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|pad|>",
    "<esg>",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<freethink>",
    "</freethink>",
    "/think",
    "/freethink",
)
ALL_LANGS = ("eng", "kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit")
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "parentheses-0.9-120M-sft" / "step_9636_final.pt"

# Multilingual test prompts across all 9 languages
MULTILINGUAL_CORPUS = {
    "eng": {
        "statement": "The deep ocean remains one of the least explored frontiers on Earth, holding ecosystems that thrive without sunlight.",
        "question_mark": "Why does the moon orbit the Earth?",
        "question_word": "How does selective linear attention achieve linear complexity in sequence length?",
        "question_quoted": '"What is the maximum depth of the Mariana Trench?"',
        "heavy_first_person": "I have been working on this recurrent model architecture for months and I think I finally understand how the state updates.",
        "short_first_person": "I am fine.",
    },
    "kan": {
        "statement": "ಪಶ್ಚಿಮ ಘಟ್ಟಗಳ ಕಾಡುಗಳು ಕರ್ನಾಟಕದ ನದಿಗಳಿಗೆ ಮತ್ತು ಜೀವವೈವಿಧ್ಯಕ್ಕೆ ಪ್ರಮುಖ ಆಧಾರವಾಗಿವೆ.",
        "question_mark": "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು?",
        "question_word": "ಏಕೆ ಆಕಾಶ ನೀಲಿಯಾಗಿದೆ ಮತ್ತು ಮೋಡಗಳು ಬಿಳಿಯಾಗಿವೆ?",
        "question_quoted": '"ಬೆಂಗಳೂರಿನಲ್ಲಿ ಎಷ್ಟು ಜನಸಂಖ್ಯೆ ಇದೆ?"',
        "heavy_first_person": "ನಾನು ಕಳೆದ ಹಲವು ತಿಂಗಳುಗಳಿಂದ ಈ ದೊಡ್ಡ ಜವಾಬ್ದಾರಿಯನ್ನು ಒಬ್ಬನೇ ಹೊತ್ತುಕೊಂಡು ಬಂದಿದ್ದೇನೆ ಮತ್ತು ಈಗ ನನ್ನ ಮನಸ್ಸಿಗೆ ತುಂಬಾ ಆಯಾಸವಾಗಿದೆ.",
        "short_first_person": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೇನೆ.",
    },
    "tam": {
        "statement": "சங்க இலக்கியப் பாடல்கள் பண்டைய தமிழர்களின் இயற்கை வாழ்வியலையும் பண்பாட்டையும் தெளிவாகப் பதிவு செய்துள்ளன.",
        "question_mark": "மதுரையின் சிறப்பு என்ன?",
        "question_word": "எப்படி கடல்சார் வணிகம் பண்டைய துறைமுகங்களை இணைத்தது?",
        "question_quoted": '"ஏன் மரங்களை நட்டுப் பாதுகாப்பது அவசியமானது?"',
        "heavy_first_person": "நான் பல நாட்களாக இந்த பெரிய பொறுப்பை தனியாக சுமந்து வருகிறேன் மற்றும் என் மனது மிகவும் சோர்வடைந்துள்ளது.",
        "short_first_person": "நான் நலம்.",
    },
    "tel": {
        "statement": "గోదావరి నది పరీవాహక ప్రాంతం తెలుగు రాష్ట్రాల వ్యవసాయానికి మరియు సంస్కృతికి జీవనాడిగా ఉంది.",
        "question_mark": "హైదరాబాద్ నగరం ఎక్కడ ఉంది?",
        "question_word": "ఎందుకు సౌరశక్తి వినియోగం పర్యావరణానికి చాలా మంచిది?",
        "question_quoted": '"ఎలా పుస్తక పఠనం ఆలోచనా పరిధిని విస్తరిస్తుంది?"',
        "heavy_first_person": "నేను చాలా కాలంగా ఈ ప్రాజెక్ట్ మీద రాత్రింబవళ్ళు కష్టపడుతున్నాను మరియు నా మనస్సు చాలా అలసిపోయింది.",
        "short_first_person": "నేను బాగున్నాను.",
    },
    "mal": {
        "statement": "കേരളത്തിലെ കായലുകളും പശ്ചിമഘട്ട മലനിരകളും കാലാവസ്ഥയെ നിയന്ത്രിക്കുന്നതിൽ വലിയ പങ്കുവഹിക്കുന്നു.",
        "question_mark": "തിരുവനന്തപുരം എവിടെയാണ് സ്ഥിതി ചെയ്യുന്നത്?",
        "question_word": "എങ്ങനെ മഴക്കാലത്ത് ജലസംരക്ഷണം ഫലപ്രദമായി നടപ്പാക്കാം?",
        "question_quoted": '"എന്തുകൊണ്ട് സുസ്ഥിരവികസനം ഭാവിയുടെ അനിവാര്യതയാകുന്നു?"',
        "heavy_first_person": "ഞാൻ കുറെ നാളുകളായി ഈ വലിയ ജോലി ഒറ്റയ്ക്ക് ചെയ്തുകൊണ്ടിരിക്കുകയാണ് ഇപ്പോൾ എന്റെ മനസ്സ് തളർന്നിരിക്കുന്നു.",
        "short_first_person": "ഞാൻ സുഖമായിരിക്കുന്നു.",
    },
    "tcy": {
        "statement": "ತುಳುನಾಡ್ ದ ಸಂಸ್ಕೃತಿ ಬೊಕ್ಕ ಭೂತಾರಾಧನೆ ಜನಕುಲೆನ ಬದುಕುದ ಒಂಜಿ ಮುಖ್ಯ ಭಾಗ ಆದುಂಡು.",
        "question_mark": "ಕುಡ್ಲಡ್ ಒವ್ವು ಪ್ರಸಿದ್ಧ ಜಾಗೆ?",
        "question_word": "ದಾಯೆ ಕರಾವಳಿ ಪ್ರದೇಶೊಡು ಕೃಷಿ ಬೊಕ್ಕ ಮೀನುಗಾರಿಕೆ ಮುಕ್ಯ?",
        "question_quoted": '"ಎಂಚ ತುಳು ಭಾಷೆನ್ ಒರಿಪಾವುನು ಬೊಕ್ಕ ದುಂಬು ಕೊನೊವುನು?"',
        "heavy_first_person": "ಯಾನ್ ಸುಮಾರ್ ಸಮಯೊರ್ದು ಈ ಬೇಲೆನ್ ಒರಿಯೇ ಮಲ್ತೊಂದು ಬೈದೆ ಬೊಕ್ಕ ಎನ್ನ ಮನಸ್ಸ್ ಗ್ ಬಾರೀ ಬೇಜಾರಾದುಂಡು.",
        "short_first_person": "ಯಾನ್ ಉಲ್ಲೆ.",
    },
    "kff": {
        "statement": "ଇଦ୍ ନାୟୁଁତେ ଲୋକ୍‌କୁ କୃଷି ଆରି ବଣ ଜଙ୍ଗଲ ତଲେ ମିଶି ଜିବନ କିନେରି ।",
        "question_mark": "ମାକୋ ନାୟୁଁ ବେନୋ ୱାଞ୍ଜୋରେ?",
        "question_word": "ବାତା କାମ କିଲେ ଆମୋଟ୍ ସୁଖତେ ମନ୍ଦାଟ୍?",
        "question_quoted": '"ଇଞ୍ଚୋ ଲୋକ୍‌କୁ ଆ ନାୟୁଁତେ ମାନେରି?"',
        "heavy_first_person": "ନାନ୍ନା ପେର୍‌କେ ଦିନାଂତାକି ଇଦ୍ କାମ କିହିଁ ମାଞ୍ଜାନା ମାକୋ ପାରିତେ ଆରି ସାଙ୍ଗତାକି ଜିବନ କିଆଟ୍ ।",
        "short_first_person": "ନାନ୍ନା ମାନାନ୍ ।",
    },
    "kxv": {
        "statement": "ମାଣ୍‌ସି ମୀର୍‌ଏଣା ଦିନାତା ଆ଼ନେ, ସାମା ଏ଼ୱାସି ଗାଡି କସ୍ତ ପା଼ଣ୍‌ନାୟି ମାନେ ।",
        "question_mark": "ଇଦାଆଁ ଇନାକା କାମ ମାନେ?",
        "question_word": "ଇମ୍ବାଣା ଏ଼ୱାସି ମାଞ୍ଜାନା କାମ କିନେସି?",
        "question_quoted": '"ଇନେସି ମା଼ରୋ ନାୟୁଁତାକି ସା଼ୟେମ କିନେସି?"',
        "heavy_first_person": "ନା଼ନୁ ୱାର୍ସିତିଙ୍ଗା ଇଦି କସ୍ତ କାମ ତାକି ମାଞ୍ଜାନା ମା଼ରୋ ଲ଼କୁତାକି ସବୁ ସା଼ୟେମ କିହିଁ ମାନାଇଁ ।",
        "short_first_person": "ନା଼ନୁ ମାନାଇଁ ।",
    },
    "nit": {
        "statement": "మా ఊరిలో పంటలు మరియు అడవులు ప్రజల జీవనానికి ముఖ్య ఆధారము.",
        "question_mark": "మీ ఊరు ఎక్కడ ఉంది?",
        "question_word": "తాని పని చేస్తే మాకు మేలు జరుగుతుంది?",
        "question_quoted": '"ఎంద్క్ మనుషులు అడవులను కాపాడుకోవాలి?"',
        "heavy_first_person": "ఆన్ చాలా రోజుల నుండి ఈ పని ఒక్కడినే చేస్తున్నాను మరియు నా మనస్సు చాలా భారంగా ఉంది.",
        "short_first_person": "ఆన్ బాగున్నాను.",
    },
}


def check_script_native(lang: str, text: str) -> bool:
    """Verifies that `text` is predominantly written in the native script for `lang`."""
    want = LANG_SCRIPT.get(lang)
    got = script_of(text)
    if lang == "eng":
        return got is None and any(c.isascii() and c.isalpha() for c in text)
    return got == want


class TestFreeThinkUnit(unittest.TestCase):
    """Fast, CPU-only unit tests with synthetic/dummy Parentheses models."""

    @classmethod
    def setUpClass(cls):
        # Load real BPE tokenizer
        tok_dir = REPO_ROOT / "data" / "tokenizer" / "v3_32k"
        cls.tok = load_bpe_tokenizer({"tokenizer": {"path": str(tok_dir)}})
        # Tiny synthetic BPE model for structural testing
        cfg = ModelConfig(
            vocab_size=32768,
            block_size=256,
            n_layer=2,
            n_head=2,
            n_embd=32,
            attn_type="selective_linear",
        )
        cls.dummy_model = Parentheses(cfg).eval()

    def test_01_non_question_gating_all_9_languages(self):
        """Non-question statement gating: rejection of questions, acceptance of statements across all 9 languages."""
        for lang in ALL_LANGS:
            data = MULTILINGUAL_CORPUS[lang]
            stmt = data["statement"]
            q_mark = data["question_mark"]
            q_word = data["question_word"]
            q_quoted = data["question_quoted"]

            # Statement acceptance
            self.assertFalse(is_question(stmt), f"Statement wrongly marked as question in {lang}: {stmt}")

            # Question rejection: question mark, interrogative starter, quoted question
            self.assertTrue(is_question(q_mark), f"Question with ? missed in {lang}: {q_mark}")
            self.assertTrue(is_question(q_word), f"Question starter missed in {lang}: {q_word}")
            self.assertTrue(is_question(q_quoted), f"Quoted question missed in {lang}: {q_quoted}")

            # Full-width question mark check
            q_fullwidth = stmt.rstrip(".। ") + "？"
            self.assertTrue(is_question(q_fullwidth), f"Full-width question mark missed in {lang}: {q_fullwidth}")

            # Verify FreeThinkSession enforcement on dummy model
            sess = FreeThinkSession(self.dummy_model, tokenizer=self.tok, lang=lang)
            with self.assertRaises(ValueError, msg=f"Session should reject question in {lang}"):
                list(sess.start(q_mark, max_tokens=4, force=False))

            # Force=True should bypass question check
            try:
                forced_out = list(sess.start(q_mark, max_tokens=4, force=True))
                self.assertIsInstance(forced_out, list)
            except ValueError:
                self.fail(f"force=True should bypass question gate for {lang}")

    def test_02_script_preservation_and_priming_format_all_9_languages(self):
        """Priming prompt formatting preserves native Unicode script and avoids isolated trailing space."""
        for lang in ALL_LANGS:
            self.assertIn(lang, LANG_PRIMING, f"Language {lang} missing from LANG_PRIMING dictionary")
            # Canned anchors were purged (2026-10-18 practical-testing handoff): they made every
            # answer open with the same preamble. Guard against them coming back.
            self.assertNotIn(lang, STRUCTURED_THINK_ANCHOR, f"STRUCTURED_THINK_ANCHOR[{lang!r}] reintroduced")
            template = LANG_PRIMING[lang]
            stmt = MULTILINGUAL_CORPUS[lang]["statement"]
            primed = format_primed(stmt, lang, "continuation")

            # Must not end in trailing space (prevents BPE isolated 'Ġ' token corruption)
            self.assertFalse(template.endswith(" "), f"LANG_PRIMING[{lang!r}] has trailing space")
            self.assertIn(stmt, primed)

            # Native script check on priming prefix
            self.assertTrue(
                check_script_native(lang, primed),
                f"Primed text for {lang} does not match expected native script: {primed[:60]}",
            )

            # Structured mode returns clean stripped prompt and hides internal commands
            struct_primed = format_primed(f"/think {stmt}", lang, "structured")
            self.assertEqual(struct_primed, stmt.strip())
            self.assertNotIn("/think", struct_primed)

    def test_03_structured_mode_token_format_closure_and_role_leak_prevention(self):
        """Structured FreeThink Mode guarantees internal </freethink> closure, hides /think & /freethink, and prevents role leaks."""
        sess = FreeThinkSession(
            self.dummy_model,
            tokenizer=self.tok,
            lang="kan",
            mode="structured",
            min_tokens=4,
        )
        prompt = "/think " + MULTILINGUAL_CORPUS["kan"]["statement"]

        # Run with small max_tokens (where model stream ends before producing think_end_id)
        out = sess.run(prompt, max_tokens=10)
        self.assertTrue(sess.closed_freethink, "Structured mode must set internal closed_freethink=True")
        self.assertEqual(sess.internal_close_tag, "</freethink>", "FreeThink internal close tag must be </freethink>")
        self.assertTrue(sess.internal_transcript.startswith("<freethink>"))
        self.assertTrue(sess.internal_transcript.endswith("</freethink>"))

        # Ensure no role / control tokens or internal tags/commands ever leak into visible output
        for marker in LEAK_MARKERS:
            self.assertNotIn(marker, out, f"Internal tag or role leak detected in visible output: {marker}")

        # Verify that </think> mid-stream does NOT cut off structured output and that initial lowercase words are capitalized
        from features.free_think import _stream_bpe
        think_close_id = self.tok.token_to_id("</think>")
        eos_id = self.tok.token_to_id("<|endoftext|>")
        part1_ids = self.tok.encode(" the photovoltaic effect converts light.", add_special_tokens=False).ids
        part2_ids = self.tok.encode("\nthe final conclusion is clear.", add_special_tokens=False).ids
        scripted_ids = part1_ids + [think_close_id] + part2_ids + [eos_id]

        class _ScriptedModel(torch.nn.Module):
            def __init__(self, base_model, seq):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1))
                self.cfg = base_model.cfg
                self._seq = seq

            def stream(self, *args, **kwargs):
                for tid in self._seq:
                    yield torch.tensor([[tid]], dtype=torch.long)

        scripted_out = "".join(
            _stream_bpe(
                _ScriptedModel(self.dummy_model, scripted_ids),
                self.tok,
                "Solar cells generate electricity.",
                "eng",
                max_tokens=50,
                temperature=0.5,
                top_k=30,
                mode="structured",
            )
        )
        self.assertTrue(scripted_out.startswith("The photovoltaic effect converts light."), f"Unexpected start: {scripted_out!r}")
        self.assertIn("The final conclusion is clear.", scripted_out, f"Post-</think> structured answer was cut off: {scripted_out!r}")
        self.assertNotIn("</think>", scripted_out)

    def test_04_streaming_bytelevel_bpe_utf8_multibyte_chunking(self):
        """Streaming ByteLevel BPE decoder: multi-byte character boundary chunking across UTF-8 boundaries."""
        # Test sequences with 2-byte, 3-byte (Dravidian), and 4-byte characters
        test_phrases = [
            "ಕನ್ನಡ ಸಾಹಿತ್ಯ",  # 3-byte Kannada
            "தமிழ் இலக்கியம்",  # 3-byte Tamil
            "తెలుగు భాష",  # 3-byte Telugu
            "മലയാളം കവിത",  # 3-byte Malayalam
            "ଓଡ଼ିଆ ଭାଷା",  # 3-byte Odia
            "Unicode 𐍈 Gothic & 🚀 Rocket",  # 4-byte supplementary characters
        ]
        for phrase in test_phrases:
            token_ids = self.tok.encode(phrase, add_special_tokens=False).ids

            # Simulate the incremental chunking logic used by _stream_bpe
            chunks = []
            pending = []
            for t in token_ids:
                pending.append(t)
                chunk = self.tok.decode(pending, skip_special_tokens=False)
                if chunk and not chunk.endswith("\ufffd"):
                    pending = []
                    chunks.append(chunk)

            if pending:
                chunks.append(self.tok.decode(pending, skip_special_tokens=False))

            # Verifications:
            # 1. No intermediate chunk should end with replacement character
            for idx, c in enumerate(chunks[:-1]):
                self.assertFalse(
                    c.endswith("\ufffd"),
                    f"Mid-stream chunk {idx} ended with \\ufffd in phrase {phrase!r}: {c!r}",
                )

            # 2. Reconstructed stream must exactly match original phrase
            reconstructed = "".join(chunks)
            self.assertEqual(
                reconstructed,
                phrase,
                f"Reconstructed stream differs from original: {reconstructed!r} vs {phrase!r}",
            )

    def test_05_latency_constancy_ssm_o1(self):
        """Benchmark per-token decoding latency to verify O(1) recurrent memory update (S_t = a_t S_{t-1} + k_t (x) v_t)."""
        sess = FreeThinkSession(self.dummy_model, tokenizer=self.tok, lang="eng", mode="continuation")
        # Warmup CPU thread pool
        _ = sess.run("Warmup prompt for thread initialization.", max_tokens=10)
        timings = []
        _ = sess.run(
            "Recurrent selective linear attention updates a fixed-size memory state.",
            max_tokens=100,
            timings=timings,
        )
        self.assertGreaterEqual(len(timings), 50, "Insufficient timing samples collected")

        body = timings[1:]  # skip prefill row 0
        window = min(30, len(body) // 2)
        first_ms = statistics.median(body[:window]) * 1000
        last_ms = statistics.median(body[-window:]) * 1000
        ratio = last_ms / max(first_ms, 1e-6)

        # O(1) linear recurrence guarantees non-quadratic growth (strictly absent of O(N^2) KV expansion)
        self.assertLess(
            ratio,
            1.35,  # Sub-millisecond CPU scheduling margin for tiny 2-layer synthetic model
            f"Decoding latency grew with sequence length: first {first_ms:.3f}ms, last {last_ms:.3f}ms (ratio {ratio:.3f}x)",
        )

    def test_06_modular_rag_ingest_and_retrieval(self):
        """Modular Free Think: IngestedStore BM25 indexing and retrieval across English and Dravidian."""
        store = IngestedStore()
        with tempfile.TemporaryDirectory() as tmp:
            doc_path = Path(tmp) / "knowledge.txt"
            doc_path.write_text(
                "The Mariana Trench is located in the western Pacific Ocean.\n\n"
                "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಮತ್ತು ಐಟಿ ಹಬ್ ಆಗಿದೆ.\n\n"
                "ಕಾವೇರಿ ನದಿ ತಲಕಾವೇರಿಯಲ್ಲಿ ಹುಟ್ಟಿ ತಮಿಳುನಾಡಿನ ಮೂಲಕ ಹರಿಯುತ್ತದೆ.",
                encoding="utf-8",
            )
            store.ingest([doc_path])

        self.assertGreaterEqual(len(store.documents), 3)

        # English retrieval
        hits_eng = store.retrieve("western Pacific Ocean")
        self.assertTrue(hits_eng and "Mariana Trench" in hits_eng[0])

        # Kannada retrieval
        hits_kan = store.retrieve("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಬೆಂಗಳೂರು")
        self.assertTrue(hits_kan and "ಬೆಂಗಳೂರು" in hits_kan[0])

        # Empty & unmatched queries
        self.assertEqual(store.retrieve(""), [])
        self.assertEqual(store.retrieve("nonexistentrandomterm"), [])

        # ModularFreeThinkSession primed prefix test
        mod_sess = ModularFreeThinkSession(
            self.dummy_model,
            store,
            splice_every=16,
            tokenizer=self.tok,
            lang="kan",
        )
        primed = mod_sess._primed("ಬೆಂಗಳೂರು ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ.")
        self.assertIn("ಬೆಂಗಳೂರು", primed)
        self.assertIn("ಇದರ ಬಗ್ಗೆ", primed)

        # Sandboxed export check
        with tempfile.TemporaryDirectory() as tmp_export:
            _ = mod_sess.run("ಬೆಂಗಳೂರು ನಗರ.", max_tokens=10)
            exp_json = mod_sess.export("mod_test.json", base_dir=tmp_export)
            self.assertTrue(exp_json.exists())
            data = json.loads(exp_json.read_text(encoding="utf-8"))
            self.assertEqual(data["prompt"], "ಬೆಂಗಳೂರು ನಗರ.")

            # Traversal escape must be rejected
            with self.assertRaises(ValueError):
                mod_sess.export("../escaped.json", base_dir=tmp_export)

    def test_07_conversation_memory_integration(self):
        """Conversation Memory: warrants_free_think trigger, sandboxed notes, and session continuity."""
        # warrants_free_think trigger checks across languages
        for lang in ("eng", "kan", "tam", "tel", "mal"):
            heavy = MULTILINGUAL_CORPUS[lang]["heavy_first_person"]
            short = MULTILINGUAL_CORPUS[lang]["short_first_person"]
            q = MULTILINGUAL_CORPUS[lang]["question_mark"]

            self.assertTrue(warrants_free_think(heavy), f"Heavy turn not triggered in {lang}: {heavy}")
            self.assertFalse(warrants_free_think(short), f"Short turn incorrectly triggered in {lang}: {short}")
            self.assertFalse(warrants_free_think(q), f"Question incorrectly triggered in {lang}: {q}")

        # Multi-turn conversation simulation with dummy model
        with tempfile.TemporaryDirectory() as tmp_notes:
            notes = ConversationNotes("test_conv.md", base_dir=tmp_notes)
            conv = ConversationSession(self.dummy_model, notes, tokenizer=self.tok, lang="eng")

            # Turn 1: Short factual statement -> recorded silently
            t1_out = list(conv.turn("The lab door was painted blue on Friday.", max_tokens=10))
            self.assertEqual(len(t1_out), 0, "Short fact should not trigger FreeThink")
            self.assertIn("painted blue", notes.read())

            # Turn 2: Question -> recorded silently
            t2_out = list(conv.turn("What color was the lab door painted?", max_tokens=10))
            self.assertEqual(len(t2_out), 0, "Question should not trigger FreeThink")

            # Turn 3: Heavy first-person -> triggers FreeThink and indexes prior turns
            heavy_turn = MULTILINGUAL_CORPUS["eng"]["heavy_first_person"]
            t3_out = list(conv.turn(heavy_turn, max_tokens=12))
            self.assertGreater(len(t3_out), 0, "Heavy turn should trigger FreeThink")

            # Re-read notes to verify full conversation log was maintained
            history_text = notes.read()
            self.assertIn("## user", history_text)
            self.assertIn("## thinking", history_text)


class TestFreeThinkIntegration(unittest.TestCase):
    """Integration tests on CPU using the SFT checkpoint (step_9636_final.pt)."""

    model = None
    tok = None

    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT_PATH.exists():
            raise unittest.SkipTest(f"Checkpoint not found at {CHECKPOINT_PATH}")

        # Load real 120M checkpoint on CPU once for all integration tests
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        cls.model = Parentheses(ckpt["cfg"]).to("cpu").eval()
        cls.model.load_state_dict(ckpt["model"])
        cls.tok = load_bpe_tokenizer(ckpt)

    def test_10_integration_checkpoint_continuation_all_9_languages(self):
        """Real checkpoint continuation across all 9 languages preserves native script with zero role/tag leaks."""
        for lang in ALL_LANGS:
            prompt = MULTILINGUAL_CORPUS[lang]["statement"]
            torch.manual_seed(42)
            sess = FreeThinkSession(
                self.model,
                temperature=0.72,
                top_k=35,
                tokenizer=self.tok,
                lang=lang,
                mode="continuation",
                min_tokens=16,
            )
            out = sess.run(prompt, max_tokens=24)

            self.assertGreater(len(out), 0, f"Empty generation for {lang}")
            self.assertTrue(sess.closed_freethink)
            self.assertEqual(sess.internal_close_tag, "</freethink>")
            self.assertTrue(
                check_script_native(lang, out),
                f"Language {lang} script violation in output: {out[:80]!r}",
            )
            for marker in LEAK_MARKERS:
                self.assertNotIn(marker, out, f"Role/tag leak {marker} in {lang} output")

    def test_11_integration_checkpoint_structured_mode(self):
        """Real checkpoint structured mode across all 9 languages produces native-script thoughts, internal </freethink> closure, and zero leaks."""
        for lang in ALL_LANGS:
            prompt = "/think " + MULTILINGUAL_CORPUS[lang]["statement"]
            torch.manual_seed(42)
            sess = FreeThinkSession(
                self.model,
                temperature=0.72,
                top_k=35,
                tokenizer=self.tok,
                lang=lang,
                mode="structured",
                min_tokens=16,
            )
            out = sess.run(prompt, max_tokens=32)

            self.assertGreater(len(out), 0, f"Empty structured output in {lang}")
            self.assertTrue(sess.closed_freethink, f"Structured session missing internal closed_freethink in {lang}")
            self.assertEqual(sess.internal_close_tag, "</freethink>")
            self.assertTrue(
                check_script_native(lang, out),
                f"Language {lang} script violation in structured output: {out[:80]!r}",
            )
            for marker in LEAK_MARKERS:
                self.assertNotIn(marker, out, f"Role/tag leak in structured {lang}: {marker}")

    def test_12_integration_checkpoint_modular_rag(self):
        """Real checkpoint modular FreeThink splices retrieved domain knowledge and stays grounded."""
        store = IngestedStore()
        with tempfile.TemporaryDirectory() as tmp:
            doc = Path(tmp) / "ocean.txt"
            doc.write_text(
                "The Mariana Trench reaches depths of 11000 meters in the Pacific Ocean.\n\n"
                "ಕಾವೇರಿ ನದಿಯು ಕೊಡಗು ಜಿಲ್ಲೆಯ ತಲಕಾವೇರಿಯಲ್ಲಿ ಉಗಮಿಸುತ್ತದೆ.",
                encoding="utf-8",
            )
            store.ingest([doc])

            # English Modular Run
            torch.manual_seed(42)
            eng_sess = ModularFreeThinkSession(
                self.model,
                store,
                splice_every=16,
                retrieve_k=2,
                tokenizer=self.tok,
                lang="eng",
            )
            eng_primed = eng_sess._primed("The Mariana Trench in the Pacific Ocean.")
            self.assertIn("11000 meters", eng_primed)
            eng_out = eng_sess.run("The Mariana Trench in the Pacific Ocean.", max_tokens=20)
            self.assertGreater(len(eng_out), 0)

            # Kannada Modular Run
            torch.manual_seed(42)
            kan_sess = ModularFreeThinkSession(
                self.model,
                store,
                splice_every=16,
                retrieve_k=2,
                tokenizer=self.tok,
                lang="kan",
            )
            kan_primed = kan_sess._primed("ಕಾವೇರಿ ನದಿಯ ಉಗಮ ಸ್ಥಾನ.")
            self.assertIn("ತಲಕಾವೇರಿಯಲ್ಲಿ", kan_primed)
            kan_out = kan_sess.run("ಕಾವೇರಿ ನದಿಯ ಉಗಮ ಸ್ಥಾನ.", max_tokens=20)
            self.assertTrue(check_script_native("kan", kan_out))

    def test_13_integration_checkpoint_conversation_memory(self):
        """Real checkpoint conversation memory maintains multi-turn thread and triggers on heavy turn."""
        with tempfile.TemporaryDirectory() as tmp:
            notes = ConversationNotes("live_session.md", base_dir=tmp)
            conv = ConversationSession(self.model, notes, tokenizer=self.tok, lang="eng")

            # Turn 1: Short turn
            t1 = list(conv.turn("The telescope mirror in dome four was recoated on Tuesday.", max_tokens=10))
            self.assertEqual(len(t1), 0)

            # Turn 2: Question
            t2 = list(conv.turn("When was the mirror recoated?", max_tokens=10))
            self.assertEqual(len(t2), 0)

            # Turn 3: Heavy first-person turn -> triggers FreeThink
            torch.manual_seed(42)
            heavy = MULTILINGUAL_CORPUS["eng"]["heavy_first_person"]
            t3 = list(conv.turn(heavy, max_tokens=20))
            self.assertGreater(len(t3), 0)
            self.assertIn("recoated on Tuesday", notes.read())

    def test_14_integration_checkpoint_latency_constancy(self):
        """Real checkpoint 120M SSM decoding latency stays flat (<1.05x ratio across 100+ tokens)."""
        torch.manual_seed(42)
        timings = []
        sess = FreeThinkSession(
            self.model,
            temperature=0.85,
            top_k=40,
            tokenizer=self.tok,
            lang="eng",
            mode="continuation",
            min_tokens=100,
        )
        _ = sess.run(
            "Selective linear attention state space recurrence updates a fixed-size matrix state per step.",
            max_tokens=100,
            timings=timings,
        )
        self.assertGreaterEqual(len(timings), 80, f"Expected 80+ timing steps, got {len(timings)}")

        body = timings[1:]  # skip row 0 prefill
        window = min(30, len(body) // 2)
        first_ms = statistics.median(body[:window]) * 1000
        last_ms = statistics.median(body[-window:]) * 1000
        ratio = last_ms / max(first_ms, 1e-6)

        self.assertLess(
            ratio,
            1.05,
            f"O(1) latency constancy violated: first {window} median={first_ms:.3f}ms, last {window} median={last_ms:.3f}ms, ratio={ratio:.3f}x",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
