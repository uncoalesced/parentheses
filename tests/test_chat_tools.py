"""Unit tests for scripts/chat.py helpers: math intercept, compact web snippets, system prompt.

Run from the repo root:
    .venv\\Scripts\\python.exe -m unittest tests.test_chat_tools -v
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.chat import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    _safe_eval,
    compact_web_snippet,
    reply,
    solve_simple_math,
)


class TestSolveSimpleMath(unittest.TestCase):
    def ans(self, prompt):
        out = solve_simple_math(prompt)
        self.assertIsNotNone(out, prompt)
        return out

    def test_basic_ops(self):
        self.assertTrue(self.ans("What is 19 + 27?").endswith("= 46"))
        self.assertTrue(self.ans("15 * 8").endswith("= 120"))
        self.assertTrue(self.ans("100 - 37 =").endswith("= 63"))
        self.assertTrue(self.ans("12 x 12").endswith("= 144"))
        self.assertTrue(self.ans("7 × 6").endswith("= 42"))
        self.assertTrue(self.ans("84 ÷ 4").endswith("= 21"))
        self.assertTrue(self.ans("10 / 4").endswith("= 2.5"))
        self.assertTrue(self.ans("1,000 + 1").endswith("= 1001"))
        self.assertTrue(self.ans("what is 6 times 7").endswith("= 42"))
        self.assertTrue(self.ans("(2 + 3) * 4").endswith("= 20"))
        self.assertTrue(self.ans("0.1 + 0.2").endswith("= 0.3"))  # exact, no float noise

    def test_powers(self):
        self.assertTrue(self.ans("12^2").endswith("= 144"))
        self.assertIn("is 225", self.ans("What is 15 squared?"))
        self.assertIn("is 27", self.ans("3 cubed"))
        self.assertIn("is 81", self.ans("the square of 9"))

    def test_percent_of(self):
        self.assertIn("is 30", self.ans("What is 15% of 200?"))
        self.assertIn("is 12.5", self.ans("25% of 50"))

    def test_successive_discounts(self):
        out = self.ans("1000 with successive discounts of 20% and 10%")
        self.assertIn("price is 720", out)
        self.assertIn("total discount of 28%", out)
        out = self.ans("What is the single discount equivalent to successive discounts of 20% and 10%?")
        self.assertIn("28%", out)

    def test_division_by_zero(self):
        self.assertEqual(self.ans("5 / 0"), "Division by zero is undefined.")

    def test_not_math(self):
        for p in [
            "Who is the president of India?",
            "x = 5",
            "42",
            "What is 2 + 2 in Kannada and why?",
            "Tell me about the year 1947",
            "",
            "2 ** 99999999",
        ]:
            self.assertIsNone(solve_simple_math(p), p)

    @unittest.skip("linear-equation solving lived only in the uncommitted v12 chat.py, lost 2026-10-20; "
                   "the intercept restored from git HEAD does not include it")
    def test_linear_equation(self):
        self.assertEqual(self.ans("solve 3x + 7 = 16"), "x = 3")

    def test_safe_eval_rejects_code(self):
        for expr in ["__import__('os')", "a + 1", "(1).__class__", "[1, 2]", "2 ** 1000"]:
            with self.assertRaises((ValueError, SyntaxError)):
                _safe_eval(expr)


class TestCompactWebSnippet(unittest.TestCase):
    def test_strips_urls_and_limits_sentences(self):
        results = [
            {"title": "x", "snippet": "See https://example.com/a. "},  # too short once URL is gone
            {"title": "News", "snippet": "India launched a new satellite today from www.isro.gov.in. "
                                          "It carries three payloads. Officials visited example.com later. Fourth."},
        ]
        out = compact_web_snippet(results)
        self.assertNotIn("http", out)
        self.assertNotIn("www.", out)
        self.assertNotIn(".com", out)
        self.assertNotIn(".gov", out)
        self.assertIn("India launched a new satellite today", out)
        self.assertIn("It carries three payloads.", out)
        self.assertNotIn("Officials", out)

    def test_word_cap_and_empty(self):
        long = {"snippet": " ".join(["word"] * 200)}
        out = compact_web_snippet([long], max_words=60)
        self.assertEqual(len(out.split()), 60)
        self.assertTrue(out.endswith("..."))
        self.assertEqual(compact_web_snippet([]), "")


class TestReplyIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from model.backbone import Parentheses
        from model.config import ModelConfig
        cfg = ModelConfig(vocab_size=256, block_size=128, n_layer=2, n_head=2, n_embd=32,
                          attn_type="selective_linear")
        cls.model = Parentheses(cfg).eval()

    def test_math_skips_model(self):
        with patch.object(self.model, "generate", side_effect=AssertionError("model called")):
            out = reply(self.model, "What is 19 + 27?", "cpu", max_new_tokens=4, temperature=1.0,
                        top_k=4, stream=False)
        self.assertTrue(out.endswith("= 46"))

    @patch("features.web_tools.search_web")
    def test_web_injects_short_url_free_context(self, mock_search):
        mock_search.return_value = [{"title": "T", "url": "https://example.com/n",
                                     "snippet": "Rain is expected today in Bengaluru. Second line. Third line."}]
        seen = {}
        orig = self.model.generate

        def spy(ids, *a, **k):
            seen["prompt"] = bytes(ids[0].tolist()).decode("utf-8", errors="replace")
            return orig(ids, *a, **k)

        with patch.object(self.model, "generate", side_effect=spy):
            reply(self.model, "What is the weather today in Bengaluru?", "cpu", max_new_tokens=2,
                  temperature=1.0, top_k=4, stream=False, web=True)
        self.assertIn("[Web Context: Rain is expected today in Bengaluru. Second line.]", seen["prompt"])
        self.assertNotIn("http", seen["prompt"])
        self.assertNotIn("Third line", seen["prompt"])

    @patch("features.web_tools.search_web")
    def test_web_not_triggered_for_static_question(self, mock_search):
        reply(self.model, "Who wrote Hamlet?", "cpu", max_new_tokens=2, temperature=1.0, top_k=4,
              stream=False, web=True)
        mock_search.assert_not_called()

    def test_system_prompt_prepended_bpe(self):
        tok_path = REPO / "data" / "tokenizer" / "v3_32k" / "tokenizer.json"
        if not tok_path.exists():
            self.skipTest("tokenizer missing")
        from tokenizers import Tokenizer
        from model.backbone import Parentheses
        from model.config import ModelConfig
        tok = Tokenizer.from_file(str(tok_path))
        tok.encode_special_tokens = True
        bpe = Parentheses(ModelConfig(vocab_size=32768, block_size=256, n_layer=2, n_head=2, n_embd=32,
                                      attn_type="selective_linear")).eval()
        seen = {}
        orig = bpe.stream

        def spy(idx, *a, **k):
            seen["ids"] = idx[0].tolist()
            return orig(idx, *a, **k)

        sys_id = tok.token_to_id("<|system|>")
        user_id = tok.token_to_id("<|user|>")
        with patch.object(bpe, "stream", side_effect=spy):
            reply(bpe, "Who are you?", "cpu", max_new_tokens=2, temperature=1.0, top_k=4, tok=tok,
                  lang="eng", fmt="chat", stream=False, system=DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(seen["ids"][0], sys_id)
        self.assertIn(user_id, seen["ids"])
        with patch.object(bpe, "stream", side_effect=spy):
            reply(bpe, "Who are you?", "cpu", max_new_tokens=2, temperature=1.0, top_k=4, tok=tok,
                  lang="eng", fmt="chat", stream=False)
        self.assertEqual(seen["ids"][0], user_id)
        self.assertNotIn(sys_id, seen["ids"])


if __name__ == "__main__":
    unittest.main()
