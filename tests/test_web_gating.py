"""
Comprehensive Web Tooling & Intent Gating Verification Suite for Parentheses 0.9-120M.

Tests and verifies:
  1. Negative Gating under --web (MUST NOT SEARCH):
     - Common sense definitions: 'is grass green?', 'what is a dog and wolf', 'entropy', 'native language of mangalore'
     - Math & arithmetic: 'What is 19 + 27?', '19 + 27', 'x = 5', 'solve 3x + 7 = 16'
     - Logic puzzles: 'all bloops are razzies', '17 sheep'
     - Conversational: 'hi', 'how are you?', 'good morning', 'who are you'
     - Non-English / Dravidian prompts without explicit /web:
       Kannada, Tamil, Telugu, Malayalam (preserving language purity and avoiding context hijacking)
  2. Positive Gating under --web (MUST SEARCH):
     - Real-time factual lookups: 'What is the latest news in India today?', 'Who is currently the prime minister of the UK?'
     - Weather & live status: 'weather today in Bangalore', 'score of India vs Australia match'
     - Recent temporal anchors: 'who won the 2024 election', 'what happened in 2025?'
  3. Explicit Inline Commands (ALWAYS SEARCH / FETCH regardless of --web flag):
     - '/web who won the 2024 IPL'
     - '/search breaking news updates'
     - '/url https://example.com/article'
     - Dravidian explicit search: '/web ಬೆಂಗಳೂರಿನ ಇತಿಹಾಸ'
  4. Concise Grounding Formatting:
     - '[Web Context: {title} - {snippet}]'
     - ZERO raw URLs ('Source: https://...' stripped)
     - Strict 150-word budget cap
  5. End-to-End Chat & Model Integration:
     - Verifying chat.py reply() search call dispatch counts and language routing integrity

Run:
    python tests/test_web_gating.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.web_tools import (
    extract_web_queries,
    format_web_results_for_prompt,
    is_non_english,
    should_trigger_web_search,
)


class TestWebIntentGating(unittest.TestCase):
    """Unit tests for query intent classification and search trigger gating."""

    def test_negative_filters_under_web_flag(self):
        """Queries that must NEVER trigger web search even when --web is enabled."""
        negative_queries = [
            # Common sense / static definitions
            ("is grass green?", "eng"),
            ("is water wet?", "eng"),
            ("is the sky blue?", "eng"),
            ("what is a dog and wolf", "eng"),
            ("what is an apple", "eng"),
            ("what is a dog", "eng"),
            ("native language of mangalore", "eng"),
            ("capital of france", "eng"),
            ("entropy", "eng"),
            ("what is entropy", "eng"),
            ("photosynthesis", "eng"),
            ("what is gravity", "eng"),
            # Arithmetic & math
            ("What is 19 + 27?", "eng"),
            ("19 + 27", "eng"),
            ("19 + 27 = ?", "eng"),
            ("15 * 8", "eng"),
            ("x = 5", "eng"),
            ("y = 2x + 1", "eng"),
            ("solve 3x + 7 = 16", "eng"),
            ("calculate 15 * 8", "eng"),
            ("compute 2 + 2", "eng"),
            # Logical puzzles
            ("all bloops are razzies", "eng"),
            ("17 sheep", "eng"),
            ("if all bloops are razzies and all razzies are fizzes", "eng"),
            # Conversational greetings
            ("hi", "eng"),
            ("hello", "eng"),
            ("hey", "eng"),
            ("howdy", "eng"),
            ("how are you?", "eng"),
            ("how are you doing", "eng"),
            ("good morning", "eng"),
            ("good night", "eng"),
            ("who are you", "eng"),
            ("what is your name", "eng"),
            ("what can you do", "eng"),
            ("tell me a joke", "eng"),
            ("thank you", "eng"),
            ("bye", "eng"),
        ]

        for query, lang in negative_queries:
            with self.subTest(query=query):
                result = should_trigger_web_search(query, web_enabled=True, lang=lang)
                self.assertFalse(
                    result,
                    f"Expected NO web search for query '{query}', but search was triggered!"
                )
                q_list, _ = extract_web_queries(query, web_enabled=True, lang=lang)
                self.assertEqual(
                    len(q_list), 0,
                    f"extract_web_queries returned non-empty list for '{query}': {q_list}"
                )

    def test_language_safety_dravidian_without_explicit_web(self):
        """Dravidian queries without explicit /web must NEVER trigger web search."""
        dravidian_queries = [
            ("ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಮಾಡುವುದು ಏಕೆ ಮುಖ್ಯ?", "kan"),
            ("ಬೆಂಗಳೂರಿನ ಪ್ರಮುಖ ಐತಿಹಾಸಿಕ ತಾಣಗಳು ಯಾವುವು?", "kan"),
            ("ಸೂರ್ಯನ ಬೆಳಕು ಸಸ್ಯಗಳಿಗೆ ಹೇಗೆ ಉಪಯುಕ್ತ?", "kan"),
            ("மரங்களை நட்டுப் பாதுகாப்பது நகரங்களின் வெப்பநிலையைக் குறைக்க எவ்வாறு உதவும்?", "tam"),
            ("தமிழ் இலக்கியத்தின் சிறப்புகள் யாவை?", "tam"),
            ("సౌరశక్తి వినియోగం వల్ల కలిగే ప్రయోజనాలు ఏమిటి?", "tel"),
            ("హైదరాబాద్ నగరం యొక్క ప్రాముఖ్యత ఏమిటి?", "tel"),
            ("പരിസ്ഥിതി സംരക്ഷണവും സുസ്ഥിര വികസനവും എന്തുകൊണ്ട് പ്രധാനം?", "mal"),
            ("കേരളത്തിലെ പ്രധാന നദികൾ ഏവ?", "mal"),
            ("ಕರಾವಳಿ ಪ್ರದೇಶೊಡು ಮಳೆಗಾಲೊಡು ಕೃಷಿ ಮಲ್ಪುನ ಕ್ರಮ ಎಂಚಿನವು?", "tcy"),
        ]

        for query, lang in dravidian_queries:
            with self.subTest(query=query, lang=lang):
                self.assertTrue(is_non_english(query), f"Failed to detect non-English script in '{query}'")
                result = should_trigger_web_search(query, web_enabled=True, lang=lang)
                self.assertFalse(
                    result,
                    f"Dravidian query '{query}' triggered web search under --web! Language purity violated."
                )

    def test_positive_realtime_factual_triggers(self):
        """Real-time factual lookup queries with explicit triggers under --web MUST search."""
        positive_queries = [
            ("What is the latest news in India today?", "eng"),
            ("Who is currently the prime minister of the UK?", "eng"),
            ("Who is the current CEO of Google?", "eng"),
            ("What is the weather today in Bangalore?", "eng"),
            ("weather today in Chennai", "eng"),
            ("Score of India vs Australia match", "eng"),
            ("who won the 2024 election", "eng"),
            ("What happened in the 2025 summit?", "eng"),
            ("latest stock price of Apple", "eng"),
            ("breaking news about space mission", "eng"),
        ]

        for query, lang in positive_queries:
            with self.subTest(query=query):
                result = should_trigger_web_search(query, web_enabled=True, lang=lang)
                self.assertTrue(
                    result,
                    f"Expected web search for real-time query '{query}', but search was NOT triggered!"
                )
                q_list, _ = extract_web_queries(query, web_enabled=True, lang=lang)
                self.assertGreaterEqual(
                    len(q_list), 1,
                    f"extract_web_queries returned empty list for '{query}'"
                )

    def test_explicit_inline_commands(self):
        """Explicit /web, /search, /url commands MUST search/fetch regardless of --web flag."""
        inline_cases = [
            ("/web who won the 2024 IPL", False, "who won the 2024 IPL"),
            ("/web who won the 2024 IPL", True, "who won the 2024 IPL"),
            ("/search latest AI models in 2026", False, "latest AI models in 2026"),
            ('/web "Bengaluru weather today"', True, "Bengaluru weather today"),
            ("/web ಬೆಂಗಳೂರಿನ ಇತಿಹಾಸ", False, "ಬೆಂಗಳೂರಿನ ಇತಿಹಾಸ"),
            ("/url https://example.com/data.txt", False, None),
        ]

        for cmd, web_flag, expected_query in inline_cases:
            with self.subTest(cmd=cmd, web_flag=web_flag):
                result = should_trigger_web_search(cmd, web_enabled=web_flag)
                self.assertTrue(result, f"Inline command '{cmd}' failed to trigger search (web={web_flag})")
                q_list, u_list = extract_web_queries(cmd, web_enabled=web_flag)
                if expected_query:
                    self.assertIn(expected_query, q_list)
                if "/url" in cmd:
                    self.assertGreaterEqual(len(u_list), 1)

    def test_concise_formatting_no_raw_urls(self):
        """format_web_results_for_prompt must never dump raw URLs and must enforce <=150 words."""
        mock_results = [
            {
                "title": "IPL 2024 Final Winner",
                "snippet": "Kolkata Knight Riders won the 2024 IPL championship beating SRH by 8 wickets. Visit https://iplt20.com/final for full scorecard.",
                "url": "https://www.iplt20.com/match/2024",
            },
            {
                "title": "Match Summary and Awards",
                "snippet": "Mitchell Starc was named player of the match. More details at http://espncricinfo.com/series/ipl-2024.",
                "url": "https://www.espncricinfo.com/series/ipl-2024",
            },
        ]

        formatted = format_web_results_for_prompt(mock_results, query="who won the 2024 IPL", max_words=150)

        # 1. Must use clean [Web Context: {title} - {snippet}] block format
        self.assertIn("[Web Context: IPL 2024 Final Winner -", formatted)
        self.assertIn("Kolkata Knight Riders won the 2024 IPL", formatted)

        # 2. MUST NOT leak raw URLs or 'Source: https://...'
        self.assertNotIn("https://", formatted)
        self.assertNotIn("http://", formatted)
        self.assertNotIn("Source:", formatted)

        # 3. Word count must be <= 150 words
        word_count = len(formatted.split())
        self.assertLessEqual(word_count, 150, f"Formatted context exceeded 150 words: {word_count}")

    def test_word_budget_truncation(self):
        """Verify strict truncation when snippets exceed word budget."""
        long_snippet = "word " * 200
        mock_results = [{"title": "Long Page", "snippet": long_snippet, "url": "https://example.com"}]
        formatted = format_web_results_for_prompt(mock_results, max_words=150)
        word_count = len(formatted.split())
        self.assertLessEqual(word_count, 155)  # including [Web Context: ...] tag overhead


class TestEndToEndChatIntegration(unittest.TestCase):
    """End-to-end integration tests using Parentheses chat.py reply() on CPU."""

    @classmethod
    def setUpClass(cls):
        from model.backbone import Parentheses
        from model.config import ModelConfig
        # Lightweight CPU smoke model for fast deterministic pipeline validation
        cfg = ModelConfig(
            vocab_size=256, block_size=128, n_layer=2, n_head=2, n_embd=32,
            attn_type="selective_linear"
        )
        cls.model = Parentheses(cfg).eval()

    @patch("features.web_tools.search_web")
    def test_chat_reply_negative_gating(self, mock_search):
        """Verify reply() under --web does NOT execute search for basic queries."""
        from scripts.chat import reply

        # 'is grass green?' under web=True
        mock_search.reset_mock()
        out = reply(self.model, "is grass green?", "cpu", max_new_tokens=4, temperature=1.0, top_k=4, stream=False, web=True)
        mock_search.assert_not_called()
        self.assertIsInstance(out, str)

        # 'What is 19 + 27?' under web=True
        mock_search.reset_mock()
        out = reply(self.model, "What is 19 + 27?", "cpu", max_new_tokens=4, temperature=1.0, top_k=4, stream=False, web=True)
        mock_search.assert_not_called()

        # 'x = 5' under web=True
        mock_search.reset_mock()
        out = reply(self.model, "x = 5", "cpu", max_new_tokens=4, temperature=1.0, top_k=4, stream=False, web=True)
        mock_search.assert_not_called()

        # 'all bloops are razzies' under web=True
        mock_search.reset_mock()
        out = reply(self.model, "all bloops are razzies", "cpu", max_new_tokens=4, temperature=1.0, top_k=4, stream=False, web=True)
        mock_search.assert_not_called()

    @patch("features.web_tools.search_web")
    def test_chat_reply_positive_triggers(self, mock_search):
        """Verify reply() executes search when real-time trigger or /web is used."""
        from scripts.chat import reply

        mock_search.return_value = [
            {"title": "Live News", "snippet": "India launches new science initiative today.", "url": "https://example.com/news"}
        ]

        # Explicit /web command (even with web=False)
        mock_search.reset_mock()
        out = reply(self.model, "/web who won the 2024 IPL", "cpu", max_new_tokens=4, temperature=1.0, top_k=4, stream=False, web=False)
        mock_search.assert_called_once_with("who won the 2024 IPL", max_results=3)

        # Real-time trigger under web=True
        mock_search.reset_mock()
        out = reply(self.model, "What is the latest news in India today?", "cpu", max_new_tokens=4, temperature=1.0, top_k=4, stream=False, web=True)
        self.assertTrue(mock_search.called)
        call_arg = mock_search.call_args[0][0]
        self.assertIn("latest news in India today", call_arg)


if __name__ == "__main__":
    unittest.main()
