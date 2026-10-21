"""
Pure-Python Web Tooling for Parentheses: Web Search, URL Fetching, and Tool Calling.

Engineered by uncoalesced

Zero external dependencies beyond standard library:
  - urllib.request, urllib.parse, json, re, html, html.parser

Capabilities:
  - search_web(query, max_results=5): Real-time web search via DuckDuckGo Lite & Wikipedia fallback.
  - fetch_url(url, max_chars=4000): Fetches and extracts clean readable text/Markdown from any webpage.
  - execute_tool_call(call_str): Handles Hermes function calling `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
  - Self-contained offline self-test (--self-test).
"""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional


class _DDGLiteParser(HTMLParser):
    """Parses DuckDuckGo Lite search results into structured dictionaries."""

    def __init__(self):
        super().__init__()
        self.results: List[Dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._curr_title = ""
        self._curr_snippet = ""
        self._curr_url = ""

    def handle_starttag(self, tag: str, attrs: list):
        attr_map = dict(attrs)
        if tag == "a" and "result-link" in attr_map.get("class", ""):
            self._in_link = True
            href = attr_map.get("href", "")
            # Unquote DDG redirect if present
            if "uddg=" in href:
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    href = urllib.parse.unquote(m.group(1))
            self._curr_url = href
        elif tag == "td" and "result-snippet" in attr_map.get("class", ""):
            self._in_snippet = True

    def handle_endtag(self, tag: str):
        if tag == "a" and self._in_link:
            self._in_link = False
        elif tag == "td" and self._in_snippet:
            self._in_snippet = False
            if self._curr_title or self._curr_snippet:
                self.results.append({
                    "title": self._curr_title.strip(),
                    "snippet": self._curr_snippet.strip(),
                    "url": self._curr_url.strip(),
                })
                self._curr_title = ""
                self._curr_snippet = ""
                self._curr_url = ""

    def handle_data(self, data: str):
        if self._in_link:
            self._curr_title += data
        elif self._in_snippet:
            self._curr_snippet += data


class _HTMLTextExtractor(HTMLParser):
    """Strips tags, scripts, navigation, and boilerplate, extracting clean text."""

    def __init__(self):
        super().__init__()
        self.texts: List[str] = []
        self._skip_depth = 0
        self._ignore_tags = {
            "script", "style", "nav", "header", "footer", "aside", "svg",
            "noscript", "form", "button", "iframe"
        }

    def handle_starttag(self, tag: str, attrs: list):
        if tag.lower() in self._ignore_tags:
            self._skip_depth += 1
        elif tag.lower() in ("p", "h1", "h2", "h3", "h4", "li", "tr", "div", "br"):
            if self.texts and not self.texts[-1].endswith("\n"):
                self.texts.append("\n")

    def handle_endtag(self, tag: str):
        if tag.lower() in self._ignore_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        if self._skip_depth == 0:
            cleaned = html.unescape(data).strip()
            if cleaned:
                self.texts.append(cleaned + " ")


def search_web(query: str, max_results: int = 5, timeout: float = 8.0) -> List[Dict[str, str]]:
    """Performs live web search.
    
    Tries DuckDuckGo Lite first; if unreachable, falls back to Wikipedia OpenSearch.
    Returns: list of dicts with 'title', 'snippet', 'url'.
    """
    clean_q = query.strip()
    if not clean_q:
        return []

    # 1. Primary: DuckDuckGo Lite
    try:
        url = "https://lite.duckduckgo.com/lite/"
        post_data = urllib.parse.urlencode({"q": clean_q}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=post_data,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            parser = _DDGLiteParser()
            parser.feed(content)
            if parser.results:
                return parser.results[:max_results]
    except Exception:
        pass

    # 2. Secondary fallback: Wikipedia OpenSearch (supports English and Indic scripts)
    try:
        # Detect script for Wikipedia subdomain
        subdomain = "en"
        first_char = clean_q[0] if clean_q else "a"
        if "\u0C80" <= first_char <= "\u0CFF":
            subdomain = "kn"
        elif "\u0B80" <= first_char <= "\u0BFF":
            subdomain = "ta"
        elif "\u0C00" <= first_char <= "\u0C7F":
            subdomain = "te"
        elif "\u0D00" <= first_char <= "\u0D7F":
            subdomain = "ml"

        wiki_url = f"https://{subdomain}.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_q)}&limit={max_results}&format=json"
        req = urllib.request.Request(
            wiki_url,
            headers={"User-Agent": "ParenthesesBot/1.0 (https://github.com/uncoalesced/parentheses)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            titles, snippets, urls = data[1], data[2], data[3]
            results = []
            for t, s, u in zip(titles, snippets, urls):
                results.append({"title": t, "snippet": s or t, "url": u})
            if results:
                return results[:max_results]
    except Exception:
        pass

    return []


def fetch_url(url: str, max_chars: int = 4000, timeout: float = 10.0) -> Dict[str, str]:
    """Fetches any webpage URL and extracts clean readable text.
    
    Tries Jina Reader proxy (r.jina.ai) first for pristine Markdown;
    falls back to direct GET + HTMLParser.
    """
    clean_url = url.strip()
    if not clean_url:
        return {"url": "", "title": "", "text": ""}

    if not clean_url.startswith(("http://", "https://")):
        clean_url = "https://" + clean_url

    # 1. Try Jina Reader proxy for clean markdown
    try:
        jina_url = f"https://r.jina.ai/{clean_url}"
        req = urllib.request.Request(
            jina_url,
            headers={"User-Agent": "ParenthesesBot/1.0", "Accept": "text/plain"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            # Extract title if Jina prepends Title: ...
            title_m = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
            title = title_m.group(1).strip() if title_m else clean_url
            # Strip preamble metadata
            body = re.sub(r"^(Title|URL Source|Markdown Content):\s*.*$\n?", "", text, flags=re.MULTILINE).strip()
            return {
                "url": clean_url,
                "title": title,
                "text": body[:max_chars].strip(),
            }
    except Exception:
        pass

    # 2. Direct HTTP GET with pure-Python HTML extraction
    try:
        req = urllib.request.Request(
            clean_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw_html = resp.read().decode("utf-8", errors="replace")
            # Extract <title>
            title_m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
            title = html.unescape(title_m.group(1)).strip() if title_m else clean_url

            parser = _HTMLTextExtractor()
            parser.feed(raw_html)
            text = "".join(parser.texts)
            text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
            return {
                "url": clean_url,
                "title": title,
                "text": text[:max_chars].strip(),
            }
    except Exception as e:
        return {"url": clean_url, "title": "Error", "text": f"Failed to fetch {clean_url}: {e}"}


def format_web_results_for_prompt(results: List[Dict[str, str]], query: str = "", max_words: int = 150) -> str:
    """Formats search results into concise text summaries without raw URLs.

    Adheres strictly to Parentheses 120M grounding standards:
      - Concise text summary capped at max_words (default 150 words total).
      - NEVER dumps raw URLs (no 'Source: https://...') to prevent the 120M model
        from hallucinating URL lists.
      - Formatted cleanly as: '[Web Context: {title} - {snippet}]'
    """
    if not results:
        return ""

    items = []
    current_word_count = 0

    for r in results:
        title = r.get("title", "").strip()
        snippet = r.get("snippet", "").strip()

        # Strip any URLs embedded in title or snippet
        title = re.sub(r"https?://\S+", "", title).strip()
        snippet = re.sub(r"https?://\S+", "", snippet).strip()

        if not title and not snippet:
            continue

        if title and snippet:
            text = f"{title} - {snippet}"
        else:
            text = title or snippet

        words = text.split()
        if not words:
            continue

        # Check word budget
        if current_word_count + len(words) > max_words:
            remaining = max_words - current_word_count
            if remaining >= 5:
                truncated_text = " ".join(words[:remaining]) + "..."
                items.append(f"[Web Context: {truncated_text}]")
            break
        else:
            items.append(f"[Web Context: {text}]")
            current_word_count += len(words)

    return "\n".join(items)


_INLINE_WEB_CMD_RE = re.compile(
    r'(?:^|\s)/(?:web|search)\s+(?:"([^"]+)"|\'([^\']+)\'|([^\n\r/]+))',
    re.IGNORECASE,
)
_INLINE_URL_CMD_RE = re.compile(
    r'(?:^|\s)/(?:url|fetch)\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    re.IGNORECASE,
)
_INTERNAL_CMD_STRIP_RE = re.compile(
    r'(?:^|\s)/(?:think|freethink|free-think|nothink|no-think|chat|base|auto|lang\s+\w+|eng|kan|tam|tel|mal|tcy|kff|kxv|nit)\b',
    re.IGNORECASE,
)


def is_non_english(text: str) -> bool:
    """Detects whether text contains Indic or Dravidian script characters."""
    for ch in text:
        cp = ord(ch)
        if (0x0C80 <= cp <= 0x0CFF or   # Kannada
            0x0B80 <= cp <= 0x0BFF or   # Tamil
            0x0C00 <= cp <= 0x0C7F or   # Telugu
            0x0D00 <= cp <= 0x0D7F or   # Malayalam
            0x0B00 <= cp <= 0x0B7F or   # Odia
            0x0900 <= cp <= 0x097F):    # Devanagari
            return True
    return False


# Negative filters for queries that should NEVER trigger web search under --web
_CONVERSATIONAL_RE = re.compile(
    r"^(?:"
    r"(?:hi|hello|hey|howdy|sup|greetings|yo)\b[!\?\.\s]*|"
    r"how\s+are\s+you(?:\s+doing)?[\?\.\s]*|"
    r"good\s+(?:morning|afternoon|evening|night|day)[\!\?\.\s]*|"
    r"who\s+are\s+you[\?\.\s]*|"
    r"what\s+(?:is|are)\s+your\s+name[\?\.\s]*|"
    r"what\s+can\s+you\s+do[\?\.\s]*|"
    r"(?:tell\s+me\s+a\s+joke|joke|tell\s+a\s+joke)[\!\?\.\s]*|"
    r"(?:thank\s+you|thanks|thx|ty)[\!\?\.\s]*|"
    r"(?:bye|goodbye|see\s+you|cya)[\!\?\.\s]*"
    r")$",
    re.IGNORECASE,
)

_MATH_RE = re.compile(
    r"^(?:"
    # Arithmetic: 19 + 27, What is 19 + 27?, 15 * 8 = ?
    r"(?:what\s+is\s+)?\(?\s*\d+(?:\.\d+)?\s*[\+\-\*\/\^\%]\s*\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^\%]\s*\d+(?:\.\d+)?)*\s*(?:\=\s*\??)?[\?\.\s]*|"
    # Variable assignments: x = 5, y = 2x + 1
    r"[a-zA-Z]\s*=\s*[\d\w\.\s\+\-\*\/\^\%]+|"
    # Math action verbs: solve 3x + 7 = 16, calculate 15 * 8
    r"(?:solve|calculate|compute|evaluate|factor|simplify|integrate|differentiate)\s+[\d\w\.\s\+\-\*\/\^\%\=\(\)]+"
    r")$",
    re.IGNORECASE,
)

_LOGIC_PUZZLE_RE = re.compile(
    r"\b(?:bloops|razzies|\d+\s+sheep|riddle\s+me|logic\s+puzzle|knights\s+and\s+knaves|monty\s+hall|liar\s+paradox)\b|"
    r"^(?:if\s+)?all\s+\w+\s+are\s+\w+",
    re.IGNORECASE,
)

_STATIC_DEFINITION_RE = re.compile(
    r"^(?:"
    # Common sense questions: is grass green?, is water wet?, is sky blue?
    r"is\s+(?:the\s+)?(?:grass|water|sky|fire|ice|snow)\s+(?:green|wet|blue|hot|cold|white)[\?\.\s]*|"
    # What is a dog and wolf, what is a dog, what is an apple
    r"what\s+is\s+(?:a|an|the)\s+(?:dog|wolf|cat|apple|banana|car|tree|fish|bird)(?:\s+and\s+(?:a|an|the)?\s*(?:wolf|dog|cat|apple))?[\?\.\s]*|"
    # Static entity definitions: native language of mangalore, capital of france
    r"(?:what\s+is\s+(?:the\s+)?)?(?:native\s+language|capital|currency)\s+of\s+[a-zA-Z\s]+[\?\.\s]*|"
    # Single-word static science concepts: entropy, gravity, photosynthesis
    r"(?:what\s+is\s+)?(?:entropy|gravity|photosynthesis|dna|rna|atom|molecule)[\?\.\s]*"
    r")$",
    re.IGNORECASE,
)

# Positive triggers for real-time factual lookup intent
_REALTIME_TRIGGERS_RE = re.compile(
    r"\b(?:"
    # Temporal anchors
    r"latest|recent|recently|today|tonight|yesterday|this\s+week|this\s+month|this\s+year|upcoming|ongoing|breaking|"
    # Current status / office holders
    r"currently|right\s+now|at\s+present|presently|"
    r"who\s+is\s+(?:the\s+)?current\b|"
    r"who\s+currently\b|"
    r"current\s+(?:president|prime\s+minister|ceo|leader|champion|status|price|score|weather|news|cabinet|roster|ranking|gdp|rate)|"
    # News & live information
    r"news|headline|headlines|breaking\s+news|"
    r"weather|weather\s+today|forecast|temperature\s+today|"
    # Live scores / sports
    r"score\s+of|live\s+score|match\s+score|who\s+won|who\s+is\s+winning|"
    # Financial live queries
    r"stock\s+price|share\s+price|crypto\s+price|exchange\s+rate|gas\s+price|"
    # Recent / current / near-future years
    r"2024|2025|2026|2027|"
    # Live events / elections
    r"election|elections|poll\s+results|ipl|olympics|world\s+cup|super\s+bowl"
    r")\b",
    re.IGNORECASE,
)


def should_trigger_web_search(prompt: str, web_enabled: bool = False, lang: str = "eng") -> bool:
    """Determines whether web search should be executed for a given prompt.

    Rules:
      1. Explicit inline command (/web <q>, /search <q>, /url <u>, /fetch <u>):
         Always triggers search/fetch (returns True).
      2. If web_enabled is False:
         Returns False.
      3. Language safety:
         If prompt is non-English (Kannada, Tamil, Telugu, Malayalam, etc.) and
         no explicit /web is present: returns False.
      4. Negative filters:
         Conversational queries, math/arithmetic, logic puzzles, and simple
         static definitions: returns False.
      5. Positive triggers:
         Requires explicit real-time factual lookup intent (e.g. 'latest',
         'recent', 'today', 'current', 'news', 'weather today', '2024',
         '2025', '2026', 'score of'): returns True.
      6. Otherwise: returns False.
    """
    if not prompt or not prompt.strip():
        return False

    raw_text = prompt.strip()

    # Rule 1: Explicit inline command always triggers
    if _INLINE_WEB_CMD_RE.search(raw_text) or _INLINE_URL_CMD_RE.search(raw_text):
        return True

    # Rule 2: If web is not enabled, do not search
    if not web_enabled:
        return False

    # Clean off internal tags and mode commands (/think, /freethink, etc.)
    cleaned = _INTERNAL_CMD_STRIP_RE.sub(" ", raw_text)
    cleaned = re.sub(r"</?(?:think|freethink|free-think)>", " ", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    # Rule 3: Language Safety for Dravidian / non-English prompts without explicit /web
    active_lang = (lang or "eng").lower()
    if active_lang in ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit") or is_non_english(cleaned):
        return False

    # Rule 4: Negative Filters
    # Conversational
    if _CONVERSATIONAL_RE.match(cleaned):
        return False

    # Math & Arithmetic
    if _MATH_RE.match(cleaned):
        return False

    # Logic puzzles
    if _LOGIC_PUZZLE_RE.search(cleaned):
        return False

    # Static definitions / common sense
    if _STATIC_DEFINITION_RE.match(cleaned):
        return False

    # Rule 5: Positive Real-time Factual Lookup Triggers
    if _REALTIME_TRIGGERS_RE.search(cleaned):
        return True

    # Rule 6: Default to NO search
    return False


def extract_web_queries(prompt: str, web_enabled: bool = False, lang: str = "eng") -> tuple[list[str], list[str]]:
    """Extracts explicit web search queries and URLs, or gates the full prompt if --web is enabled.

    Returns:
      (queries_to_search: list[str], urls_to_fetch: list[str])
    """
    if not prompt or not prompt.strip():
        return [], []

    raw_text = prompt.strip()
    queries = []
    urls = []

    # Check explicit inline commands
    for m in _INLINE_WEB_CMD_RE.finditer(raw_text):
        q = m.group(1) or m.group(2) or m.group(3)
        if q and q.strip():
            queries.append(q.strip())

    for m in _INLINE_URL_CMD_RE.finditer(raw_text):
        u = m.group(1) or m.group(2) or m.group(3)
        if u and u.strip():
            urls.append(u.strip())

    # If no explicit inline query was given, check if --web intent gating allows searching the prompt
    if not queries and not urls and web_enabled:
        if should_trigger_web_search(prompt, web_enabled=True, lang=lang):
            cleaned = _INTERNAL_CMD_STRIP_RE.sub(" ", raw_text)
            cleaned = re.sub(r"</?(?:think|freethink|free-think)>", " ", cleaned)
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
            if cleaned:
                queries.append(cleaned)

    return queries, urls


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*({.*?})\s*</tool_call>",
    re.DOTALL | re.IGNORECASE
)


def execute_tool_call(call_str: str) -> Optional[str]:
    """Detects and executes Hermes-style tool calls.

    Example input:
      `<tool_call>{"name": "web_search", "arguments": {"query": "quantum computing"}}</tool_call>`
    Returns:
      `<tool_response>{"results": [...]}</tool_response>`
    """
    m = TOOL_CALL_RE.search(call_str)
    if not m:
        # Check if raw JSON
        if call_str.strip().startswith("{") and call_str.strip().endswith("}"):
            json_blob = call_str.strip()
        else:
            return None
    else:
        json_blob = m.group(1).strip()

    try:
        payload = json.loads(json_blob)
    except Exception:
        return None

    name = payload.get("name")
    args = payload.get("arguments", {})

    if name == "web_search":
        q = args.get("query", "")
        hits = search_web(q, max_results=args.get("max_results", 5))
        resp_obj = {"query": q, "results": hits}
        return f"<tool_response>\n{json.dumps(resp_obj, ensure_ascii=False, indent=2)}\n</tool_response>"

    elif name in ("fetch_url", "web_fetch", "read_url"):
        u = args.get("url", "")
        doc = fetch_url(u, max_chars=args.get("max_chars", 3000))
        return f"<tool_response>\n{json.dumps(doc, ensure_ascii=False, indent=2)}\n</tool_response>"

    return None


def self_test():
    """Offline unit test verification."""
    # Test HTML text extractor
    raw = "<html><head><title>Test Title</title><style>.x{}</style></head><body><h1>Heading</h1><p>Paragraph text.</p><script>alert(1)</script></body></html>"
    p = _HTMLTextExtractor()
    p.feed(raw)
    extracted = "".join(p.texts)
    assert "Heading" in extracted
    assert "Paragraph text" in extracted
    assert "alert" not in extracted

    # Test DDG lite parser
    mock_html = """
    <table><tr><td>
      <a class="result-link" href="https://example.com/test">Example Title</a>
      <table><tr><td class="result-snippet">This is an example snippet.</td></tr></table>
    </td></tr></table>
    """
    ddg = _DDGLiteParser()
    ddg.feed(mock_html)
    assert len(ddg.results) == 1
    assert ddg.results[0]["title"] == "Example Title"
    assert ddg.results[0]["snippet"] == "This is an example snippet."
    assert ddg.results[0]["url"] == "https://example.com/test"

    # Test format_web_results_for_prompt (concise, max 150 words, NO raw URLs)
    formatted = format_web_results_for_prompt(ddg.results, "example query", max_words=150)
    assert "[Web Context: Example Title - This is an example snippet.]" in formatted
    assert "https://" not in formatted
    assert "Source:" not in formatted
    assert len(formatted.split()) <= 150

    # Test web intent gating negative cases (MUST NOT SEARCH)
    assert not should_trigger_web_search("is grass green?", web_enabled=True)
    assert not should_trigger_web_search("What is 19 + 27?", web_enabled=True)
    assert not should_trigger_web_search("19 + 27", web_enabled=True)
    assert not should_trigger_web_search("x = 5", web_enabled=True)
    assert not should_trigger_web_search("solve 3x + 7 = 16", web_enabled=True)
    assert not should_trigger_web_search("entropy", web_enabled=True)
    assert not should_trigger_web_search("what is a dog and wolf", web_enabled=True)
    assert not should_trigger_web_search("native language of mangalore", web_enabled=True)
    assert not should_trigger_web_search("all bloops are razzies", web_enabled=True)
    assert not should_trigger_web_search("17 sheep", web_enabled=True)
    assert not should_trigger_web_search("hi", web_enabled=True)
    assert not should_trigger_web_search("how are you?", web_enabled=True)
    assert not should_trigger_web_search("good morning", web_enabled=True)
    # Dravidian prompts without explicit /web MUST NOT SEARCH
    assert not should_trigger_web_search("ಬೆಂಗಳೂರಿನ ಇತಿಹಾಸವೇನು?", web_enabled=True, lang="kan")
    assert not should_trigger_web_search("தமிழ்நாட்டின் வரலாறு என்ன?", web_enabled=True, lang="tam")

    # Test web intent gating positive cases (MUST SEARCH)
    assert should_trigger_web_search("/web who won the 2024 IPL", web_enabled=False)
    assert should_trigger_web_search("/web who won the 2024 IPL", web_enabled=True)
    assert should_trigger_web_search("/search current news in India", web_enabled=False)
    assert should_trigger_web_search("/url https://example.com", web_enabled=False)
    assert should_trigger_web_search("What is the latest news in India today?", web_enabled=True)
    assert should_trigger_web_search("Who is currently the prime minister of the UK?", web_enabled=True)
    assert should_trigger_web_search("weather today in Bangalore", web_enabled=True)
    assert should_trigger_web_search("score of India vs Australia match", web_enabled=True)
    assert should_trigger_web_search("who won the 2024 election", web_enabled=True)

    # Test extract_web_queries
    q_neg, _ = extract_web_queries("is grass green?", web_enabled=True)
    assert len(q_neg) == 0

    q_pos, _ = extract_web_queries("What is the latest news in India today?", web_enabled=True)
    assert len(q_pos) == 1 and "latest news" in q_pos[0]

    q_inline, _ = extract_web_queries("/web who won the 2024 IPL", web_enabled=False)
    assert len(q_inline) == 1 and q_inline[0] == "who won the 2024 IPL"

    # Test tool call parsing & dispatch with mock
    call = '<tool_call>{"name": "unknown_tool", "arguments": {}}</tool_call>'
    assert execute_tool_call(call) is None

    # Test valid tool structure detection
    mock_call = '<tool_call>{"name": "web_search", "arguments": {"query": ""}}</tool_call>'
    res = execute_tool_call(mock_call)
    assert res is not None and "<tool_response>" in res

    print("[self-test] web_tools ok")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        self_test()
    else:
        q = " ".join(arg for arg in sys.argv[1:] if not arg.startswith("--"))
        if q:
            print(f"Searching web for: {q}...")
            hits = search_web(q)
            for h in hits:
                print(f"- {h['title']}\n  {h['snippet']}\n  {h['url']}\n")
        else:
            print("Usage: python -m features.web_tools <query> | --self-test")
