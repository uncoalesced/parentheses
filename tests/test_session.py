"""ChatSession keeps the state between turns instead of re-reading history.

The state after any sequence of feeds and generated tokens must equal the
state from reading `ids_seen` in one parallel pass.
"""

import os

import pytest
import torch

from parentheses import TOKENIZER_PATH, Parentheses
from parentheses.checkpoint import load_model
from parentheses.session import ChatSession

from test_slab_v4 import EOT, TINY, tiny_model


def max_state_diff(a, b) -> float:
    return max((x - y).abs().max().item() for la, lb in zip(a, b) for x, y in zip(la, lb))


def reread(model: Parentheses, ids):
    x = torch.tensor([ids], dtype=torch.long)
    logits, states = model.forward_with_state(x)
    return logits[0, -1], states


def random_ids(seed: int, n: int):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, TINY.vocab_size, (n,), generator=g).tolist()


def test_turns_and_generation_match_one_pass():
    model = tiny_model(1)
    s = ChatSession(model)
    # Turn lengths deliberately not multiples of the chunk size.
    s.feed(random_ids(1, 5))
    s.generate(max_new_tokens=7, stop_token_ids=[])
    s.feed(random_ids(2, 13))
    s.generate(max_new_tokens=4, stop_token_ids=[])
    s.feed(random_ids(3, 9))
    assert len(s.ids_seen) == 5 + 7 + 13 + 4 + 9
    logits, states = reread(model, s.ids_seen)
    assert (s.next_logits - logits).abs().max().item() < 1e-9
    assert max_state_diff(s.states, states) < 1e-9


def test_endoftext_inside_a_feed_and_at_a_turn_end():
    model = tiny_model(2)
    s = ChatSession(model)
    s.feed(random_ids(4, 6) + [EOT] + random_ids(5, 3))
    s.feed(random_ids(6, 4) + [EOT])          # next feed starts a new document
    s.feed(random_ids(7, 11))
    s.feed_one(EOT)
    s.feed_one(5)                              # step right after <|endoftext|>
    logits, states = reread(model, s.ids_seen)
    assert (s.next_logits - logits).abs().max().item() < 1e-9
    assert max_state_diff(s.states, states) < 1e-9


def test_state_is_carried_not_reset():
    """A fact written in turn 1 must still change turn 3's output."""
    model = tiny_model(3)
    later = random_ids(9, 20)
    a, b = ChatSession(model), ChatSession(model)
    a.feed(random_ids(8, 10)); a.feed(later)
    b.feed(random_ids(80, 10)); b.feed(later)
    assert (a.next_logits - b.next_logits).abs().max().item() > 1e-3


def test_generate_does_not_write_the_stop_token():
    model = tiny_model(4)
    s = ChatSession(model)
    s.feed(random_ids(10, 8))
    first = int(s.next_logits.argmax())
    out = s.generate(max_new_tokens=5, stop_token_ids=[first])
    assert out == [] and len(s.ids_seen) == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_stepper_session_matches_eager():
    from parentheses.graph_step import GraphStepper
    model = tiny_model(6).cuda()
    stepper = GraphStepper(model)
    eager, graph = ChatSession(model), ChatSession(model, stepper=stepper)
    for s in (eager, graph):
        s.feed(random_ids(20, 9))
        s.generate(max_new_tokens=12, stop_token_ids=[])
        s.feed(random_ids(21, 5) + [EOT])        # next reply starts a new document
        s.generate(max_new_tokens=6, stop_token_ids=[])
    assert graph.ids_seen == eager.ids_seen
    assert (graph.next_logits - eager.next_logits).abs().max().item() < 1e-9
    assert max_state_diff(graph.states, eager.states) < 1e-9


def test_turn_format_uses_chat_tokens():
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_PATH))
    s = ChatSession(tiny_model(5), tokenizer=tok, lang="kan")
    ids = s.user_turn_ids("hello")
    assert ids[0] == tok.token_to_id("<|user|>")
    assert ids[1] == tok.token_to_id("<kan>")
    assert ids[-1] == tok.token_to_id("<|assistant|>")
    assert tok.token_to_id("<|endoftext|>") not in ids
    assert tok.token_to_id("<|endoftext|>") in s.default_stops()


CKPT = os.environ.get("PARENTHESES_BASE_CKPT")


@pytest.mark.skipif(not CKPT, reason="set PARENTHESES_BASE_CKPT")
def test_release_three_turns_match_reread():
    tokenizers = pytest.importorskip("tokenizers")
    tok = tokenizers.Tokenizer.from_file(str(TOKENIZER_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(CKPT, device=device)
    s = ChatSession(model, tokenizer=tok, lang="eng")
    s.add_user("My locker code is 4719. Please remember it.")
    s.generate(max_new_tokens=24)
    s.add_user("Tell me something about rivers in Karnataka.")
    s.generate(max_new_tokens=48)
    s.add_user("What is my locker code?")
    x = torch.tensor([s.ids_seen], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, states = model.forward_with_state(x)
    diff = (s.next_logits - logits[0, -1]).abs().max().item()
    print(f"\nrelease 3-turn carried vs re-read: max abs logit diff {diff:.3e}, "
          f"{len(s.ids_seen)} tokens, state {s.state_bytes()} bytes")
    assert diff < 1e-4
