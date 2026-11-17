"""Chat CLI helpers and the release sampling options."""

import torch

from parentheses.chat import detect_language
from parentheses.session import ChatSession, _block_repeated_ngrams

from test_slab_v4 import tiny_model


def test_detect_language_by_script():
    assert detect_language("hello there") == "eng"
    assert detect_language("ಕನ್ನಡ ಭಾಷೆ") == "kan"
    assert detect_language("தமிழ் மொழி") == "tam"
    assert detect_language("తెలుగు భాష") == "tel"
    assert detect_language("മലയാളം ഭാഷ") == "mal"
    assert detect_language("what is ಕನ್ನಡ?") == "kan"      # any Indic script wins over Latin


def test_ngram_block_forbids_completing_a_seen_ngram():
    scores = torch.zeros(10)
    _block_repeated_ngrams(scores, [1, 2, 3, 1, 2], 3)     # "1 2" was followed by 3 before
    assert scores[3] == float("-inf")
    assert torch.isfinite(scores[[0, 1, 2, 4]]).all()


def test_generation_never_repeats_an_ngram():
    model = tiny_model(21)
    s = ChatSession(model)
    s.feed([5, 6, 7])
    out = s.generate(max_new_tokens=40, temperature=0.0, no_repeat_ngram_size=3, stop_token_ids=[])
    grams = [tuple(out[i:i + 3]) for i in range(len(out) - 2)]
    assert len(grams) == len(set(grams))


def test_min_p_one_is_greedy():
    model = tiny_model(22)
    a, b = ChatSession(model), ChatSession(model)
    a.feed([5, 6, 7]); b.feed([5, 6, 7])
    torch.manual_seed(0)
    sampled = a.generate(max_new_tokens=12, temperature=1.0, min_p=1.0, stop_token_ids=[])
    greedy = b.generate(max_new_tokens=12, temperature=0.0, stop_token_ids=[])
    assert sampled == greedy
