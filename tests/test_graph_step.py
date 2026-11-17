"""The CUDA-graph step must give the same logits and state as eager step()."""

import os

import pytest
import torch

from parentheses.checkpoint import load_model

from test_slab_v4 import EOT, TINY, step_all, tiny_model

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def graph_all(g, row):
    out = []
    for t in range(row.numel()):
        out.append(g.step(int(row[t])).clone())
    return torch.stack(out)


@cuda
def test_tiny_graph_matches_eager_with_documents():
    from parentheses.graph_step import GraphStepper
    model = tiny_model(11).cuda()
    gen = torch.Generator().manual_seed(11)
    row = torch.randint(2, TINY.vocab_size, (256,), generator=gen)
    row[[40, 41, 170]] = EOT                      # back-to-back and lone document ends
    row = row.cuda()
    eager, eager_states = step_all(model, row)
    g = GraphStepper(model)
    graph = graph_all(g, row)
    assert (graph - eager).abs().max().item() < 1e-12
    for la, lb in zip(g.read_states(), eager_states):
        for a, b in zip(la, lb):
            assert (a - b).abs().max().item() < 1e-12


@cuda
def test_graph_continues_from_a_prefill():
    from parentheses.graph_step import GraphStepper
    model = tiny_model(12).cuda()
    gen = torch.Generator().manual_seed(12)
    prompt = torch.randint(2, TINY.vocab_size, (1, 37), generator=gen).cuda()
    tail = torch.randint(2, TINY.vocab_size, (50,), generator=gen).cuda()
    with torch.no_grad():
        _, states = model.forward_with_state(prompt)
    eager, _ = step_all(model, tail, [type(s)(*(t.clone() for t in s)) for s in states])
    g = GraphStepper(model)
    g.load_states(states)
    assert (graph_all(g, tail) - eager).abs().max().item() < 1e-12


CKPT = os.environ.get("PARENTHESES_BASE_CKPT")


@cuda
@pytest.mark.skipif(not CKPT, reason="set PARENTHESES_BASE_CKPT")
def test_release_graph_matches_eager():
    from parentheses.graph_step import GraphStepper
    model = load_model(CKPT, device="cuda")
    gen = torch.Generator().manual_seed(13)
    row = torch.randint(2, 32768, (256,), generator=gen)
    row[100] = EOT
    row = row.cuda()
    eager, _ = step_all(model, row)
    graph = graph_all(GraphStepper(model), row)
    diff = (graph - eager).abs().max().item()
    print(f"\nrelease graph vs eager: max abs logit diff {diff:.3e} over 256 tokens")
    assert diff < 1e-5
