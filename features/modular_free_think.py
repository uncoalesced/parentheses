"""
Modular Free Think / RAG.

From the plan: same free-thinking behavior as Free Think Mode, but with a
user-supplied, decentralized data store loaded alongside the model (SSD,
RAM, or VRAM depending on user choice) so the model can "rethink" using
data it wasn't trained on. No embedding model bundled into Parentheses
itself per the plan ("Parentheses shouldn't have embedding features") --
retrieval defaults to lexical (BM25, via rank_bm25). Intended to eventually
live as a Peridot feature/integration per the plan note.

`backend="vector"` is now wired (TurboVec index over embeddings from
model/embedding_head.py -- see handoff-vector-memory.md), but it is a
validation path, not the default, and it still bundles nothing: the caller
must pass an `embedder` callable, so no embedding model ships inside
Parentheses. `backend="bm25"` stays the default until the vector path is
measured against it on retrieval quality as well as throughput.

Two pieces:
    - IngestedStore: user points at files -> chunk -> BM25 index -> retrieve
    - ModularFreeThinkSession: FreeThinkSession's decode loop, restarted
      every `splice_every` tokens with the top chunks for the running
      thought spliced in front of the prompt

    python -m features.modular_free_think --checkpoint checkpoints/step_161000.pt \
        --ingest notes/*.txt --prompt "The ocean is very deep."
    python -m features.modular_free_think --self-test    # no checkpoint needed

Everything is raw text encoded with text.encode("utf-8"), matching the rest
of the codebase: the model is byte-level (vocab_size=256), so a chunk is
just bytes in the same window the prompt lives in.
"""

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import torch
from rank_bm25 import BM25Okapi

from model import Parentheses

from .free_think import PRIMING, FreeThinkSession, _reject_questions, _stream_text

# A retrieved chunk has to *fit* next to the prompt in a block_size window
# (256 bytes on every current byte-level preset), so chunks are capped at
# roughly half a window -- a chunk bigger than that could only be spliced in
# by evicting the prompt it's supposed to be informing.
CHUNK_BYTES = 128

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """BM25's unit of matching: lowercased alphanumeric runs."""
    return _WORD.findall(text.lower())


def _chunk(text: str, max_bytes: int = CHUNK_BYTES) -> list[str]:
    """Split `text` into retrieval chunks: paragraphs, hard-split on words.

    Paragraph-based (blank-line separated) rather than fixed-size or
    line-based: the corpora this targets are the same prose the model was
    trained on (Wikipedia articles, Gutenberg books), where a paragraph is
    already a self-contained unit and a line break is arbitrary wrapping.
    Long paragraphs are then split at word boundaries to respect
    CHUNK_BYTES; a single word longer than the cap becomes its own
    (oversized) chunk rather than being cut mid-word.
    """
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        current, size = [], 0
        for word in para.split():
            width = len(word.encode("utf-8")) + 1  # +1 for the joining space
            if current and size + width > max_bytes:
                chunks.append(" ".join(current))
                current, size = [], 0
            current.append(word)
            size += width
        if current:
            chunks.append(" ".join(current))
    return chunks


def _tail(text: str, max_bytes: int) -> str:
    """Last <= max_bytes UTF-8 bytes of `text`, cut on a character boundary."""
    return text.encode("utf-8")[-max_bytes:].decode("utf-8", "ignore")


class IngestedStore:
    def __init__(self, backend: str = "bm25", location: str = "ram",
                 embedder=None, min_score: float | None = None):
        assert backend in ("bm25", "vector"), "vector backend intentionally not default (see module docstring)"
        assert location in ("ram", "vram", "ssd")
        self.backend = backend
        # `location` is advisory for now: a BM25 index is a Python object, so
        # it lives in RAM whatever this says. It stays in the signature for
        # the mmap-from-SSD / pinned-memory variants the plan describes.
        self.location = location
        # `embedder` keeps the module docstring's promise intact: Parentheses
        # still bundles no embedding model. The vector path is wired, but the
        # caller supplies texts -> (N, dim) unit vectors -- in practice
        # model.embedding_head over a trained checkpoint (see
        # scripts/train_embedding_head.py). Without one, vector still refuses.
        self.embedder = embedder
        # None = no cutoff, and that is deliberate. The BM25 path drops chunks
        # sharing no term with the query; cosine has no equally principled
        # threshold, and picking a number without measuring retrieval quality
        # would be inventing one. Note 0.0 is *not* the neutral value -- it
        # discards negative similarities, which quantised scores do produce.
        self.min_score = min_score
        if backend == "vector" and embedder is None:
            raise NotImplementedError(
                "vector backend needs an `embedder` callable: Parentheses bundles no "
                "embedding model (see module docstring). Pass one, or use backend='bm25'."
            )
        self.documents: list[str] = []
        self._doc_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._index = None
        self._vectors = None

    def ingest(self, paths: list[Path]):
        """Read, chunk and index `paths`. Repeated calls add to the store."""
        for path in paths:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            self.documents.extend(_chunk(text))
        if self.backend == "vector":
            self._build_vector_index()
            return
        # BM25 needs corpus-wide statistics (IDF, average length), so the
        # index is rebuilt over everything rather than appended to.
        self._doc_tokens = [_tokens(c) for c in self.documents]
        self._bm25 = BM25Okapi(self._doc_tokens) if self.documents else None

    def _build_vector_index(self):
        """Embed every chunk and (re)build the TurboVec index.

        Rebuilt rather than appended to, matching the BM25 path: TurboVec's
        calibration is fitted from a representative sample of the vectors the
        index will hold, so growing the store shifts what "representative"
        means. Rebuilding keeps calibration and contents consistent; swap to
        incremental add() once stores get big enough for that to hurt.
        """
        import numpy as np
        import turbovec

        if not self.documents:
            self._index, self._vectors = None, None
            return
        vecs = np.ascontiguousarray(self.embedder(self.documents), dtype=np.float32)
        self._vectors = vecs
        self._index = turbovec.TurboQuantIndex(vecs.shape[1])
        self._index.calibrate(vecs)
        self._index.add(vecs)
        self._index.prepare()      # pay the one-time warm-up now, not on first query

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """The `top_k` chunks best matching `query`. Empty before ingest()."""
        if self.backend == "vector":
            return self._retrieve_vector(query, top_k)
        query_tokens = _tokens(query)
        if self._bm25 is None or not query_tokens:
            return []
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # BM25 ranks every document, including ones sharing no term with the
        # query, and splicing an unrelated chunk into a 256-byte window costs
        # more context than it buys. Filter on a shared term rather than on
        # score > 0: BM25Okapi's IDF is exactly 0 for a term that appears in
        # half the corpus, which is an easy hit on a small user store.
        wanted = set(query_tokens)
        return [self.documents[i] for i in ranked[:top_k] if wanted & set(self._doc_tokens[i])]

    def _retrieve_vector(self, query: str, top_k: int) -> list[str]:
        import numpy as np
        if self._index is None or not query.strip():
            return []
        q = np.ascontiguousarray(self.embedder([query]), dtype=np.float32)
        k = min(top_k, len(self.documents))
        scores, ids = self._index.search(q, k)
        return [self.documents[i] for s, i in zip(scores[0], ids[0])
                if 0 <= i < len(self.documents)
                and (self.min_score is None or s >= self.min_score)]


# --- Web search backend ------------------------------------------------------
# Not a third IngestedStore.backend value: bm25/vector both score a pre-built
# index sitting in RAM. Web search has no index -- every retrieve() call is a
# live network query -- so it's a separate class that implements only the one
# method ModularFreeThinkSession actually calls (retrieve(query, top_k) ->
# list[str]). Duck-typed, not a subclass: IngestedStore's backend/location/
# embedder/ingest() machinery is for a prebuilt index this class doesn't have.

# Filler words a conversational prompt carries that a search engine doesn't
# need -- plus a few task words ("search", "web", "tell", "please", "latest")
# that show up specifically when a Free Think prompt is really a wrapped
# request rather than raw subject matter.
_STOPWORDS = frozenset("""
a about after again against all am an and any are as at be because been before
being below between both but by can did do does doing down during each few for
from further had has have having he her here hers herself him himself his how
i if in into is it its itself just me more most my myself no nor not now of
off on once only or other our ours ourselves out over own s same she should so
some such t than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where
which while who whom why will with you your yours yourself yourselves
search web tell please latest
""".split())


def _search_keywords(query: str, max_terms: int = 6) -> str:
    """Strip a conversational prompt down to a search-engine keyword string.

    Lowercase, drop stopwords/duplicates, keep first-seen order and cap at
    max_terms. Not YAKE -- no n-gram scoring, no new dependency: reuses this
    module's existing _tokens() regex, and a search engine's own ranking
    absorbs a slightly noisy keyword bag fine (that's its job). Upgrade to a
    real extractor only if SearXNG result quality turns out to depend on
    phrase-level scoring, not word-level filtering.
    """
    seen, terms = set(), []
    for word in _tokens(query):
        if word in _STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
        if len(terms) == max_terms:
            break
    return " ".join(terms) if terms else query.strip()


class WebSearchStore:
    """Live web search as a retrieve()-compatible store, via a SearXNG JSON API.

    ingest() doesn't exist here on purpose -- there's nothing to build ahead of
    time. Point ModularFreeThinkSession at one of these instead of an
    IngestedStore and every splice queries the web fresh.
    """

    def __init__(self, searxng_url: str = "http://localhost:8080", timeout: float = 5.0):
        self.base_url = searxng_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Search + chunk the top results. Never raises, same contract as
        IngestedStore.retrieve -- a caller splices this straight into a model
        prompt, so a network failure has to come back as "no results", not an
        error string dressed up as a retrieved chunk.
        """
        if not query.strip():
            return []
        params = urllib.parse.urlencode({
            "q": _search_keywords(query), "format": "json",
            "categories": "general", "language": "en",
        })
        req = urllib.request.Request(
            f"{self.base_url}/search?{params}",
            headers={"User-Agent": "ParenthesesBot/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            # network boundary: DNS failure, connection refused, timeout,
            # non-JSON body -- all of it means "no results", never a crash.
            return []

        chunks = []
        for item in payload.get("results", [])[:top_k]:
            title = item.get("title", "").strip()
            content = item.get("content", "").strip()
            if not content:
                continue
            text = f"{title}: {content}" if title else content
            chunks.extend(_chunk(text))  # same CHUNK_BYTES cap as file-ingested chunks
        return chunks[:top_k]


class ModularFreeThinkSession:
    """Same as FreeThinkSession, plus a retrieval hook. See features/free_think.py."""

    def __init__(self, model: Parentheses, store: IngestedStore, splice_every: int = 64,
                 retrieve_k: int = 3, temperature: float = 0.9, top_k: int = 40):
        # Composed, not subclassed: the inner session owns prompt/history/
        # export (so the export format can't drift from Free Think's) and the
        # byte-level vocab_size guard.
        self.session = FreeThinkSession(model, temperature, top_k)
        self.model = model
        self.store = store
        # splice_every: tokens decoded per retrieval. Every step would re-run
        # BM25 and re-prefill the whole window per token -- i.e. throw the KV
        # cache away, the one thing making decoding cheap. 64 is a quarter of
        # a 256-byte window: the query refreshes ~4x per window while ~98% of
        # steps still decode from cache. Lower it for tighter grounding on a
        # fast-drifting run, raise it if retrieval shows up in tokens/sec.
        self.splice_every = splice_every
        self.retrieve_k = retrieve_k
        # Byte budgets inside one block_size window: chunks in front, then the
        # prompt + priming, then the tail of the thought so far. A long prompt
        # can still push the total over the window -- stream() keeps the last
        # block_size bytes, so the chunk text is what degrades first.
        self.retrieval_bytes = model.cfg.block_size // 2
        self.tail_bytes = model.cfg.block_size // 4

    @property
    def history(self) -> list[str]:
        return self.session.history

    @property
    def prompt(self) -> str | None:
        return self.session.prompt

    def export(self, path: str, **kwargs):
        """Write the session to `path` -- same format and sandboxing as FreeThinkSession."""
        return self.session.export(path, **kwargs)

    def _primed(self, prompt: str) -> str:
        """Priming prefix with the chunks retrieved for the running thought."""
        tail = _tail("".join(self.session.history), self.tail_bytes)
        chunks, used = [], 0
        for chunk in self.store.retrieve(f"{prompt} {tail}", self.retrieve_k):
            width = len(chunk.encode("utf-8")) + 2  # +2 for the joining "\n\n"
            if used + width > self.retrieval_bytes:
                break  # ranked best-first, so stop rather than skip-and-continue
            chunks.append(chunk)
            used += width
        context = "\n\n".join(chunks + [prompt.strip()])
        return PRIMING.format(prompt=context) + tail

    def start(self, prompt: str, max_tokens: int | None = None, force: bool = False):
        """Yield decoded text as the model thinks, re-retrieving every splice_every tokens.

        max_tokens=None keeps thinking until the consumer stops iterating.
        """
        _reject_questions(prompt, force)
        self.session.prompt = prompt
        self.session.history = []
        remaining = max_tokens
        while remaining is None or remaining > 0:
            step = self.splice_every if remaining is None else min(self.splice_every, remaining)
            # Each splice restarts stream(), which re-prefills the window and
            # re-pins its own attention sink -- now the head of the retrieved
            # text, so a long run stays anchored to the store, not just to its
            # own opening.
            for chunk in _stream_text(self.model, self._primed(prompt), step,
                                      self.session.temperature, self.session.top_k):
                self.session.history.append(chunk)
                yield chunk
            if remaining is not None:
                remaining -= step

    def run(self, prompt: str, max_tokens: int = 400, **kwargs) -> str:
        """Non-streaming convenience: consume start() and return the whole text."""
        return "".join(self.start(prompt, max_tokens, **kwargs))


def main():
    p = argparse.ArgumentParser(description="Modular Free Think: free thinking over an ingested store.")
    p.add_argument("--checkpoint", default="checkpoints/step_161000.pt")
    p.add_argument("--ingest", nargs="*", default=[], help="text files to chunk + index")
    p.add_argument("--web-search", default=None, metavar="SEARXNG_URL",
                   help="use live web search instead of --ingest, e.g. http://localhost:8080")
    p.add_argument("--prompt", default="The ocean is very deep.")
    p.add_argument("--max-tokens", type=int, default=400, help="0 = think until Ctrl-C")
    p.add_argument("--splice-every", type=int, default=64, help="tokens decoded per retrieval")
    p.add_argument("--retrieve-k", type=int, default=3, help="chunks retrieved per splice")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--export", default=None,
                   help="filename to write inside features/free_think.py's EXPORT_DIR "
                        "(.json, .txt, .md -- can't escape that directory)")
    p.add_argument("--force", action="store_true", help="free-think even if the prompt looks like a question")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true",
                   help="check store + session logic on an untrained tiny model and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    # weights_only=False: train.py pickles the ModelConfig dataclass into the ckpt
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.web_search:
        store = WebSearchStore(args.web_search)
    else:
        store = IngestedStore()
        store.ingest([Path(f) for f in args.ingest])
    session = ModularFreeThinkSession(model, store, args.splice_every, args.retrieve_k,
                                      args.temperature, args.top_k)
    print(f"[modular] {args.checkpoint} (step {ckpt['step']}) on {args.device}")
    if args.web_search:
        print(f"[modular] store: live web search via {args.web_search}")
    else:
        print(f"[modular] store: {len(store.documents)} chunks from {len(args.ingest)} file(s)")
    print(f"[modular] prompt: {args.prompt}\n")
    try:
        for chunk in session.start(args.prompt, args.max_tokens or None, force=args.force):
            print(chunk, end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    if args.export:
        target = session.export(args.export)
        print(f"[modular] exported to {target}")


def _self_test():
    import os
    import tempfile

    from model import PRESETS

    torch.manual_seed(0)

    # --- chunking -----------------------------------------------------------
    chunks = _chunk("One two three.\n\nFour five six.")
    assert chunks == ["One two three.", "Four five six."], chunks
    long_para = " ".join(["word"] * 200)
    assert max(len(c.encode("utf-8")) for c in _chunk(long_para)) <= CHUNK_BYTES

    # --- store --------------------------------------------------------------
    store = IngestedStore()
    assert store.retrieve("anything") == []  # nothing ingested yet: [] not a raise

    with tempfile.TemporaryDirectory() as tmp:
        corpus = {
            "ocean.txt": "The deep ocean is cold and dark.\n\n"
                         "Hydrothermal vents heat the water near the seabed.",
            "space.txt": "Mars is a cold desert planet.\n\n"
                         "Rockets burn a lot of fuel to reach orbit.",
        }
        paths = []
        for name, text in corpus.items():
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            paths.append(Path(path))
        store.ingest(paths)

    assert len(store.documents) == 4, store.documents
    assert store.retrieve("hydrothermal vents")[0].startswith("Hydrothermal vents")
    assert store.retrieve("rockets fuel orbit")[0].startswith("Rockets burn")
    assert len(store.retrieve("cold", top_k=1)) == 1
    assert store.retrieve("zzzzq qqqqz") == []  # no shared term -> nothing worth splicing
    assert store.retrieve("") == []

    # --- vector backend -----------------------------------------------------
    # Still refuses without an embedder: Parentheses bundles no embedding model.
    try:
        IngestedStore(backend="vector")
        raise AssertionError("vector backend should refuse without an embedder")
    except NotImplementedError:
        pass

    # With one, it indexes and searches for real. A deterministic toy embedder
    # keeps this self-test free of a checkpoint -- it hashes each chunk's words
    # into a fixed-width unit vector, so texts sharing words land near each other.
    try:
        import numpy as np
        import turbovec                                    # noqa: F401
    except ImportError:
        print("[self-test] turbovec not installed, vector index checks skipped")
    else:
        def toy_embedder(texts):
            out = np.zeros((len(texts), 32), dtype=np.float32)
            for r, t in enumerate(texts):
                for w in _tokens(t):
                    out[r, hash(w) % 32] += 1.0
            n = np.linalg.norm(out, axis=1, keepdims=True)
            return out / np.maximum(n, 1e-9)

        vstore = IngestedStore(backend="vector", embedder=toy_embedder)
        assert vstore.retrieve("anything") == []           # before ingest: [] not a raise
        # its own temp dir: the BM25 block's files are already cleaned up
        with tempfile.TemporaryDirectory() as vtmp:
            vpaths = []
            for name, text in corpus.items():
                path = os.path.join(vtmp, name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                vpaths.append(Path(path))
            vstore.ingest(vpaths)
        assert len(vstore.documents) == len(store.documents)
        hits = vstore.retrieve("Rockets burn a lot of fuel to reach orbit.", top_k=1)
        assert hits and hits[0].startswith("Rockets burn"), hits
        assert len(vstore.retrieve("cold", top_k=2)) == 2
        assert vstore.retrieve("") == []
        # top_k above the store size must clamp, not index off the end
        assert len(vstore.retrieve("cold", top_k=99)) == len(vstore.documents)

    # --- web search backend ---------------------------------------------------
    # No real network call in a self-test: monkeypatch urlopen with a canned
    # SearXNG-shaped response, same spirit as the toy embedder above.
    kw = _search_keywords("Can you search the web and tell me the latest updates "
                           "on the Linux kernel scheduler?")
    assert kw.split() == ["updates", "linux", "kernel", "scheduler"], kw
    assert _search_keywords("") == ""
    assert _search_keywords("the a an") == "the a an"  # all-stopword: fall back to raw query

    class _FakeResponse:
        def __init__(self, payload):
            self._body = json.dumps(payload).encode("utf-8")

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    web_store = WebSearchStore()
    real_urlopen = urllib.request.urlopen

    urllib.request.urlopen = lambda req, timeout=5.0: _FakeResponse({
        "results": [
            {"title": "Linux kernel scheduler", "content": "Recent scheduler patches improve latency."},
            {"title": "", "content": ""},  # no content -> dropped, not an empty chunk
        ]
    })
    try:
        hits = web_store.retrieve("Linux kernel scheduler updates")
        assert hits and "scheduler" in hits[0].lower(), hits
        assert web_store.retrieve("") == []  # empty query: no network call needed
    finally:
        urllib.request.urlopen = real_urlopen

    # network failure must come back as [], never raise or leak an error string
    # into what looks like a retrieved chunk
    def _boom(req, timeout=5.0):
        raise OSError("no route to host")

    urllib.request.urlopen = _boom
    try:
        assert web_store.retrieve("anything") == []
    finally:
        urllib.request.urlopen = real_urlopen

    # end-to-end: WebSearchStore duck-types into ModularFreeThinkSession same as
    # IngestedStore -- no changes to the session needed to swap the backend
    urllib.request.urlopen = lambda req, timeout=5.0: _FakeResponse({
        "results": [{"title": "Ocean facts", "content": "The deep ocean is cold and dark."}]
    })
    try:
        tiny = Parentheses(PRESETS["parentheses-0.9-100k"]).eval()
        web_session = ModularFreeThinkSession(tiny, web_store, splice_every=8)
        primed = web_session._primed("The deep ocean is cold.")
        assert "cold and dark" in primed, primed
    finally:
        urllib.request.urlopen = real_urlopen

    # --- session ------------------------------------------------------------
    # smallest *byte-level* preset, same as free_think's self-test
    model = Parentheses(PRESETS["parentheses-0.9-100k"]).eval()
    session = ModularFreeThinkSession(model, store, splice_every=8)

    # the retrieved chunk has to actually reach the model's window
    primed = session._primed("The deep ocean is cold.")
    assert "The deep ocean is cold." in primed and "Thinking about this:" in primed
    assert "ocean" in primed and len(primed.encode("utf-8")) <= model.cfg.block_size

    try:
        list(session.start("Why?"))
        raise AssertionError("questions should be rejected")
    except ValueError:
        pass
    assert list(session.start("Why?", max_tokens=3, force=True))  # ...unless forced

    text = session.run("The deep ocean is cold.", max_tokens=20)
    assert text == "".join(session.history) and text, repr(text)

    # retrieval must actually re-run per splice, not just once at the start
    calls = []
    plain_retrieve = store.retrieve
    store.retrieve = lambda q, k=5: (calls.append(q), plain_retrieve(q, k))[1]
    session.run("The deep ocean is cold.", max_tokens=24)  # splice_every=8
    store.retrieve = plain_retrieve
    assert len(calls) == 3, calls
    assert calls[1] != calls[0], "later queries should carry the running thought"

    # open-ended stream: no max_tokens, the consumer decides when to stop
    seen = 0
    for _ in session.start("Another statement."):
        seen += 1
        if seen == 5:
            break
    assert seen == 5

    # export must stay byte-for-byte structurally identical to Free Think's
    with tempfile.TemporaryDirectory() as tmp:
        plain = FreeThinkSession(model)
        plain.run("A third statement.", max_tokens=10)
        session.run("A third statement.", max_tokens=10)
        mine = session.export("m.json", base_dir=tmp)
        theirs = plain.export("f.json", base_dir=tmp)
        txt = session.export("m.txt", base_dir=tmp)
        got = json.load(open(mine, encoding="utf-8"))
        want = json.load(open(theirs, encoding="utf-8"))
        assert got.keys() == want.keys(), (got.keys(), want.keys())
        assert got["prompt"] == want["prompt"] == "A third statement."
        assert got["thinking"] == "".join(session.history)
        assert "A third statement." in open(txt, encoding="utf-8").read()
    print("[self-test] modular_free_think ok")


if __name__ == "__main__":
    main()
