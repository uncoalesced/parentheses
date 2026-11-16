"""The harness adapter must return what the model actually computes.

Needs the released weights (PARENTHESES_BASE_CKPT) and lm_eval installed.
"""

import os

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("lm_eval")
CKPT = os.environ.get("PARENTHESES_BASE_CKPT")
pytestmark = pytest.mark.skipif(not CKPT, reason="set PARENTHESES_BASE_CKPT")


class Req:
    def __init__(self, *args):
        self.args = args


@pytest.fixture(scope="module")
def lm():
    from parentheses.lm_eval_adapter import ParenthesesLM
    return ParenthesesLM(checkpoint=CKPT, batch_size=4)


def by_hand(lm, context_ids, continuation_ids):
    """Score a continuation with step(), independently of the batched forward path."""
    total, greedy, states, prev_eot = 0.0, True, None, False
    seq = context_ids + continuation_ids
    for t, tok in enumerate(seq[:-1]):
        logits, states = lm.model.step(torch.tensor([tok], device=lm.device), states, boundary=prev_eot)
        prev_eot = tok == lm.eot_token_id
        if t + 1 >= len(context_ids):
            lp = F.log_softmax(logits[0].float(), dim=-1)
            total += lp[seq[t + 1]].item()
            greedy &= int(lp.argmax()) == seq[t + 1]
    return total, greedy


PAIRS = [
    ("The capital of France is", " Paris"),
    ("Question: What is 2 + 2?\nAnswer:", " 4"),
    ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ", " ಬೆಂಗಳೂರು"),
    ("def add(a, b):\n    return", " a + b"),
    ("", "Once upon a time there was a small village."),
    ("A much longer context sentence, so that rows in one batch have different lengths. " * 6, " The end."),
]


@torch.no_grad()
def test_loglikelihood_matches_step(lm):
    got = lm.loglikelihood([Req(c, k) for c, k in PAIRS], disable_tqdm=True)
    for (context, continuation), (lp, greedy) in zip(PAIRS, got):
        if context == "":
            ctx, cont = [lm.prefix_token_id], lm.tok_encode(continuation)
        else:
            ctx, cont = lm._encode_pair(context, continuation)
        want_lp, want_greedy = by_hand(lm, ctx, cont)
        assert abs(lp - want_lp) < 2e-3 * max(1, len(cont)), (context[:30], lp, want_lp)
        assert greedy == want_greedy
        assert lp < 0


@torch.no_grad()
def test_batching_does_not_change_scores(lm):
    reqs = [Req(c, k) for c, k in PAIRS]
    together = lm.loglikelihood(reqs, disable_tqdm=True)
    alone = [lm.loglikelihood([r], disable_tqdm=True)[0] for r in reqs]
    for a, b in zip(together, alone):
        assert abs(a[0] - b[0]) < 2e-3 and a[1] == b[1]


@torch.no_grad()
def test_rolling_equals_sum_of_token_logprobs(lm):
    text = "The quick brown fox jumps over the lazy dog. ಇದು ಒಂದು ಪರೀಕ್ಷೆ."
    got = lm.loglikelihood_rolling([Req(text)], disable_tqdm=True)[0]
    want, _ = by_hand(lm, [lm.prefix_token_id], lm.tok_encode(text))
    assert abs(got - want) < 2e-3 * len(lm.tok_encode(text))


@torch.no_grad()
def test_generate_until_is_greedy_and_stops(lm):
    prompt = "Question: Name three colours.\nAnswer:"
    a = lm.generate_until([Req(prompt, {"until": ["\n\n"], "max_gen_toks": 40})], disable_tqdm=True)[0]
    b = lm.generate_until([Req(prompt, {"until": ["\n\n"], "max_gen_toks": 40})], disable_tqdm=True)[0]
    assert a == b                      # greedy: same output twice
    assert "\n\n" not in a
    long = lm.generate_until([Req(prompt, {"until": [], "max_gen_toks": 12})], disable_tqdm=True)[0]
    assert len(lm.tok_encode(long)) <= 14


def test_tokenizer_round_trip(lm):
    for text in ("Hello, world!", "ಕನ್ನಡ ತಮಿಳು తెలుగు മലയാളം", "x = [i**2 for i in range(10)]"):
        assert lm.tok_decode(lm.tok_encode(text)) == text
