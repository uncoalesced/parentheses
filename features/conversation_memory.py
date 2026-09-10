"""
Live conversation memory + auto-FreeThink trigger.

From handoff-conversation-memory.md: two gaps sitting between Free Think and a
real support-style exchange.

1. `is_question()` is the only thing gating entry into Free Think, and it only
   knows question shapes. A heavy, first-person statement -- the input that
   most wants the model reasoning through *why* -- looks the same to it as
   "The ocean is very deep." `warrants_free_think()` is the second gate.
2. Nothing survives a window reset. A 256-byte block_size means the opening of
   a conversation is gone within a few hundred bytes, so the model loses the
   thread of what the user already told it. `ConversationNotes` writes every
   turn to a file, and `ConversationSession` feeds that file back through the
   Modular Free Think retrieval path, so earlier turns come back as retrieved
   chunks instead of as context the window can't hold.

No new retrieval machinery: the notes file is just another source for
`IngestedStore`, so this inherits the BM25 default (and the open
BM25-beats-vector question from handoff-vector-memory.md) rather than picking
a side.

    python -m features.conversation_memory --checkpoint checkpoints/step_161000.pt
    python -m features.conversation_memory --self-test    # no checkpoint needed
"""

import argparse
from pathlib import Path

import torch

from model import Parentheses

from .free_think import is_question, resolve_inside
from .modular_free_think import IngestedStore, ModularFreeThinkSession, _tokens

# Where conversation notes are allowed to live -- the same kind of boundary
# EXPORT_DIR is, and applied to reads as well as writes: ConversationNotes only
# ever opens a path that resolved inside here, so "re-read the conversation"
# can't be pointed at the rest of the disk.
NOTES_DIR = Path(__file__).resolve().parent.parent / "conversations"

# Deliberately coarse, matching is_question(): a long, first-person statement.
# The alternative the handoff floated -- keywords or sentiment -- means either a
# word list that misses everything not on it, or a sentiment model, which is a
# second trained thing bolted onto a project whose whole point is one small
# model. Shape is cheap and doesn't pretend to read emotion.
FIRST_PERSON = frozenset(("i", "me", "my", "myself", "mine", "we", "us", "our"))
MIN_WORDS = 12


def warrants_free_think(text: str, min_words: int = MIN_WORDS) -> bool:
    """True if `text` is the shape of a turn worth thinking about rather than answering.

    Long + first-person + not a question. Questions are Free Think's existing
    exclusion; length is what separates "I'm fine" from someone actually
    putting something down.
    """
    if is_question(text):
        return False
    words = _tokens(text)
    return len(words) >= min_words and bool(FIRST_PERSON & set(words))


class ConversationNotes:
    """An append-only record of a conversation, confined to NOTES_DIR."""

    def __init__(self, name: str = "session.md", base_dir: Path | str = NOTES_DIR):
        self.path = resolve_inside(base_dir, name)

    def append(self, speaker: str, text: str):
        """Add one turn. Markdown headings so the file stays readable by hand."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"## {speaker}\n{text.strip()}\n\n")

    def read(self) -> str:
        """The whole conversation so far, or "" before the first turn."""
        return self.path.read_text(encoding="utf-8") if self.path.exists() else ""


class ConversationSession:
    """Free Think across a conversation: every turn recorded, earlier turns retrieved."""

    def __init__(self, model: Parentheses, notes: ConversationNotes | None = None,
                 splice_every: int = 64, retrieve_k: int = 3,
                 temperature: float = 0.9, top_k: int = 40):
        self.model = model
        self.notes = notes or ConversationNotes()
        self.splice_every = splice_every
        self.retrieve_k = retrieve_k
        self.temperature = temperature
        self.top_k = top_k

    def turn(self, user_text: str, max_tokens: int = 400, force: bool = False):
        """Record `user_text`; return an iterator of thinking text.

        Empty iterator when the turn doesn't warrant Free Think -- the turn is
        still recorded, so a later one can retrieve it. Recording happens here,
        not on first iteration, so a caller that ignores the return value
        doesn't silently lose the turn.
        """
        triggered = force or warrants_free_think(user_text)
        # Built from the notes as they stand *before* this turn is appended:
        # the current text is already in the prompt, so retrieving it back
        # would spend window on a copy of it.
        store = self._store() if triggered else None
        self.notes.append("user", user_text)
        if not triggered:
            return iter(())
        return self._think(user_text, store, max_tokens)

    def _store(self) -> IngestedStore:
        """A fresh index over the notes so far.

        Rebuilt per turn rather than incrementally: BM25 needs corpus-wide
        statistics anyway (see IngestedStore.ingest), and a conversation is a
        few kilobytes. ponytail: full re-ingest per turn, make it incremental
        if conversations ever get long enough to feel it.
        """
        store = IngestedStore()
        if self.notes.path.exists():
            store.ingest([self.notes.path])
        return store

    def _think(self, user_text: str, store: IngestedStore, max_tokens: int):
        session = ModularFreeThinkSession(self.model, store, self.splice_every,
                                          self.retrieve_k, self.temperature, self.top_k)
        # force=True: warrants_free_think() already ran the question check and
        # decided, so the inner is_question() gate would just re-litigate it.
        parts = []
        for chunk in session.start(user_text, max_tokens, force=True):
            parts.append(chunk)
            yield chunk
        self.notes.append("thinking", "".join(parts))


def main():
    p = argparse.ArgumentParser(
        description="Conversation memory: Free Think that triggers itself and remembers the thread.")
    p.add_argument("--checkpoint", default="checkpoints/step_161000.pt")
    p.add_argument("--notes", default="session.md",
                   help=f"notes filename inside {NOTES_DIR} (can't escape that directory)")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--splice-every", type=int, default=64, help="tokens decoded per retrieval")
    p.add_argument("--retrieve-k", type=int, default=3, help="chunks retrieved per splice")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--force", action="store_true",
                   help="free-think on every turn, not just triggered ones")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true",
                   help="check the trigger, notes and session logic on an untrained tiny model and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    # weights_only=False: train.py pickles the ModelConfig dataclass into the ckpt
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = Parentheses(ckpt["cfg"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    notes = ConversationNotes(args.notes)
    session = ConversationSession(model, notes, args.splice_every, args.retrieve_k,
                                  args.temperature, args.top_k)
    print(f"[conversation] {args.checkpoint} (step {ckpt['step']}) on {args.device}")
    print(f"[conversation] notes: {notes.path}")
    print("[conversation] type a turn, empty line or Ctrl-C to stop\n")
    try:
        while True:
            line = input("> ").strip()
            if not line:
                break
            printed = False
            for chunk in session.turn(line, args.max_tokens, force=args.force):
                print(chunk, end="", flush=True)
                printed = True
            print() if printed else print("[recorded, not free-thought]")
    except (KeyboardInterrupt, EOFError):
        pass
    print(f"\n[conversation] notes at {notes.path}")


def _self_test():
    import tempfile

    from model import PRESETS

    torch.manual_seed(0)

    heavy = ("I have been carrying this on my own for months now and I think "
             "I am finally out of ways to keep pretending it is fine")
    assert warrants_free_think(heavy)
    assert not warrants_free_think("I'm fine.")                       # too short
    assert not warrants_free_think(heavy + "?")                       # a question
    assert not warrants_free_think(                                   # no first person
        "The ocean is very deep and it stays cold all the way down to the floor of it")
    assert not warrants_free_think("")

    with tempfile.TemporaryDirectory() as tmp:
        notes = ConversationNotes("s.md", base_dir=tmp)
        assert notes.read() == ""
        notes.append("user", "The kiln in the back room cracked again.")
        assert "kiln" in notes.read()

        # same containment as FreeThinkSession.export(): no traversal, no absolute escape
        for escape in ("../escaped.md", str(Path(tmp).parent / "escaped.md")):
            try:
                ConversationNotes(escape, base_dir=tmp)
                raise AssertionError(f"notes should have rejected {escape!r}")
            except ValueError:
                pass

        # smallest byte-level preset -- untrained, so this checks plumbing, not prose
        model = Parentheses(PRESETS["parentheses-0.9-100k"]).eval()
        session = ConversationSession(model, notes)

        assert list(session.turn("Fine.")) == []          # not triggered...
        assert "Fine." in notes.read()                    # ...but still recorded

        text = "".join(session.turn(heavy, max_tokens=20))
        assert text, "a triggered turn should produce thinking"
        body = notes.read()
        assert body.count("## thinking") == 1 and heavy in body

        # the point of the notes file: an earlier turn comes back as a
        # retrievable chunk once it has scrolled out of the window
        store = session._store()
        assert any("kiln" in c for c in store.retrieve("kiln back room")), store.documents

        # forced: a turn that wouldn't trigger on its own still thinks
        assert "".join(session.turn("Fine.", max_tokens=10, force=True))

    print("[self-test] conversation_memory ok")


if __name__ == "__main__":
    main()
