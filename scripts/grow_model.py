"""
Grow a trained selective_linear checkpoint into a wider/deeper preset, function-preserving.

    python scripts/grow_model.py --src checkpoints/parentheses-0.9-41M-v4/step_16082_final.pt \
        --preset parentheses-0.9-120M --out checkpoints/parentheses-0.9-120M/step_grown.pt \
        --check-data data/processed_v4/val/kan.bin
    python scripts/grow_model.py --self-test

Rules (new residual channels stay exactly zero, so the grown net computes the same logits):
  - width C -> C': every weight that READS the residual/hidden gets random new input columns
    (they multiply zeros); every weight that WRITES the residual gets zero new output rows.
  - RMSNorm weights are scaled by sqrt(C/C'): the zero channels shrink the RMS by sqrt(C'/C).
  - new heads (same head_dim): random q/k/v rows, zero columns in proj; decay bias from a fresh
    init of the target preset.
  - MLP hidden H -> H': new w1/w3 rows random, new w2 columns zero.
  - depth L -> L': each old block i is followed by a copy of itself with attn.proj and mlp.w2
    zeroed (LLaMA-Pro style identity block), until L' is reached.
The output checkpoint has step -1 and no optimizer, so train.py --resume-from starts at step 0
with a fresh optimizer (it warns about the missing optimizer state; that is expected).
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone import Parentheses  # noqa: E402
from model.config import PRESETS, ModelConfig  # noqa: E402


def _pad(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """new with old copied into its leading block; the rest of new is kept."""
    out = new.clone()
    out[tuple(slice(0, s) for s in old.shape)] = old
    return out


def _grow_heads_rows(old: torch.Tensor, new: torch.Tensor, n_parts: int, h_old: int, h_new: int, dh: int):
    """Rows laid out as n_parts x heads x head_dim (qkv: 3 parts). Copy old heads, keep new ones."""
    o = old.view(n_parts, h_old * dh, *old.shape[1:])
    out = new.clone().view(n_parts, h_new * dh, *new.shape[1:])
    for p in range(n_parts):
        out[p][tuple(slice(0, s) for s in o[p].shape)] = o[p]
    return out.view(new.shape)


@torch.no_grad()
def grow_block(src: dict, dst: dict, s_cfg: ModelConfig, d_cfg: ModelConfig, identity: bool) -> dict:
    """src/dst: one block's state_dict (dst = fresh init of the target). Returns the grown dict."""
    C, C2 = s_cfg.n_embd, d_cfg.n_embd
    dh = C // s_cfg.n_head
    assert dh == C2 // d_cfg.n_head, "head_dim must not change"
    H, H2 = s_cfg.n_head, d_cfg.n_head
    norm_scale = math.sqrt(C / C2)
    out = {}
    for k, new in dst.items():
        old = src[k]
        if k.endswith(("ln1.weight", "ln2.weight")):
            out[k] = torch.full_like(new, norm_scale)
            out[k][:C] = old * norm_scale
        elif k == "attn.qkv.weight":                         # (3*H2*dh, C2): reads residual
            out[k] = _grow_heads_rows(old, new, 3, H, H2, dh)
        elif k == "attn.alpha_proj.weight":                  # (H2, C2)
            out[k] = _pad(old, new)
        elif k == "attn.alpha_proj.bias":                    # old heads keep their horizon
            out[k] = _pad(old, new)
        elif k == "attn.proj.weight":                        # (C2, H2*dh): writes residual
            out[k] = torch.zeros_like(new)
            out[k][:C, :C] = old
        elif k in ("mlp.w1.weight", "mlp.w3.weight"):        # (hid2, C2): reads residual
            out[k] = _pad(old, new)
        elif k == "mlp.w2.weight":                           # (C2, hid2): writes residual
            out[k] = torch.zeros_like(new)
            out[k][:old.shape[0], :old.shape[1]] = old
        else:
            raise KeyError(f"grow_block does not know how to grow {k!r} (slab_v2/bias presets unsupported)")
    if identity:
        out["attn.proj.weight"].zero_()
        out["mlp.w2.weight"].zero_()
    return out


@torch.no_grad()
def grow(src_model: Parentheses, d_cfg: ModelConfig, seed: int = 0) -> Parentheses:
    s_cfg = src_model.cfg
    assert s_cfg.attn_type == d_cfg.attn_type == "selective_linear"
    assert not s_cfg.slab_v2 and not d_cfg.slab_v2 and not s_cfg.bias and not d_cfg.bias
    assert s_cfg.vocab_size == d_cfg.vocab_size and s_cfg.tie_embeddings and d_cfg.tie_embeddings
    assert d_cfg.n_embd >= s_cfg.n_embd and d_cfg.n_layer >= s_cfg.n_layer and d_cfg.n_head >= s_cfg.n_head
    assert d_cfg.n_layer <= 2 * s_cfg.n_layer, "at most one identity copy per source block"
    torch.manual_seed(seed)
    dst = Parentheses(d_cfg)
    C, C2 = s_cfg.n_embd, d_cfg.n_embd
    dst.tok_emb.weight.zero_()                               # new channels of the residual start at 0
    dst.tok_emb.weight[:, :C] = src_model.tok_emb.weight     # head is tied, so this covers it too
    dst.ln_f.weight.fill_(math.sqrt(C / C2))
    dst.ln_f.weight[:C] = src_model.ln_f.weight * math.sqrt(C / C2)
    n_extra = d_cfg.n_layer - s_cfg.n_layer
    j = 0
    for i, sb in enumerate(src_model.blocks):
        plan = [False] + ([True] if i < n_extra else [])     # copy, then (maybe) an identity copy
        for identity in plan:
            db = dst.blocks[j]
            db.load_state_dict(grow_block(sb.state_dict(), db.state_dict(), s_cfg, d_cfg, identity))
            j += 1
    assert j == d_cfg.n_layer
    return dst


def window_loss(model, data: np.ndarray, starts, block: int, device) -> float:
    losses = []
    for s in starts:
        w = torch.from_numpy(data[s:s + block + 1].astype(np.int64)).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            _, loss = model(w[None, :-1], w[None, 1:])
        losses.append(loss.item())
    return float(np.mean(losses))


def _self_test():
    torch.manual_seed(0)
    s_cfg = ModelConfig(vocab_size=97, block_size=128, n_layer=3, n_head=2, n_embd=32, attn_type="selective_linear")
    d_cfg = ModelConfig(vocab_size=97, block_size=128, n_layer=5, n_head=3, n_embd=48, attn_type="selective_linear")
    src = Parentheses(s_cfg).double()
    with torch.no_grad():                                    # make norms non-trivial
        for n, p in src.named_parameters():
            if n.endswith("ln1.weight") or n.endswith("ln2.weight") or n == "ln_f.weight":
                p.normal_(1.0, 0.2)
    dst = grow(src, d_cfg).double()
    x = torch.randint(0, 97, (2, 128))
    for m in (src, dst):
        m.eval()
    with torch.no_grad():                                    # real eps: only the eps term differs
        a, _ = src(x, x, return_logits=True)
        b, _ = dst(x, x, return_logits=True)
        assert (a - b).abs().max() < 0.05, f"grown logits far off even with eps: {(a - b).abs().max()}"
    # RMSNorm's eps is not scaled with the zero channels, so exactness holds only at eps=0; on the
    # trained 41M the residual RMS dwarfs eps and the --check-data loss gate measures the real gap.
    for m in list(src.modules()) + list(dst.modules()):
        if hasattr(m, "eps"):
            m.eps = 0.0
    with torch.no_grad():
        a, _ = src(x, x, return_logits=True)
        b, _ = dst(x, x, return_logits=True)
        assert torch.allclose(a, b, atol=1e-6, rtol=0), f"chunked forward differs: {(a - b).abs().max()}"
        x2 = torch.randint(0, 97, (2, 50))                   # dense path (T not a multiple of 64)
        a2, _ = src(x2, x2, return_logits=True)
        b2, _ = dst(x2, x2, return_logits=True)
        assert torch.allclose(a2, b2, atol=1e-6, rtol=0), f"dense forward differs: {(a2 - b2).abs().max()}"
    assert dst.num_params() > src.num_params()
    # new parameters must be able to learn: a backward pass gives nonzero grads on new blocks
    dst.train()
    _, loss = dst(x, x)
    loss.backward()
    g = dst.blocks[1].attn.proj.weight.grad                  # an identity block's zeroed output
    assert g is not None and g.abs().sum() > 0, "identity block gets no gradient"
    g = dst.blocks[0].mlp.w2.weight.grad[:, -1]              # a new hidden unit's zeroed column
    assert g.abs().sum() > 0, "new MLP hidden units get no gradient"
    print("grow_model self-test passed (exact logits, chunked + dense; new params get gradient)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src")
    p.add_argument("--preset", default="parentheses-0.9-120M")
    p.add_argument("--out")
    p.add_argument("--check-data", help="uint16 .bin to compare step-0 loss on (e.g. a v4 val file)")
    p.add_argument("--check-windows", type=int, default=8)
    p.add_argument("--check-block", type=int, default=1024)
    p.add_argument("--max-ratio", type=float, default=1.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
        return
    if not (args.src and args.out):
        p.error("--src and --out are required")

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    src = Parentheses(ckpt["cfg"])
    src.load_state_dict(ckpt["model"])
    d_cfg = PRESETS[args.preset]
    dst = grow(src, d_cfg, seed=args.seed)
    print(f"[info] grown {src.num_params()/1e6:.1f}M -> {dst.num_params()/1e6:.1f}M params ({args.preset})")

    if args.check_data:
        data = np.fromfile(args.check_data, dtype=np.uint16)   # small val file; sequential read
        rng = np.random.default_rng(1234)
        starts = rng.integers(0, len(data) - args.check_block - 1, args.check_windows)
        src.to(args.device).eval()
        with torch.no_grad():
            l_src = window_loss(src, data, starts, args.check_block, args.device)
        src.cpu()
        dst.to(args.device).eval()
        with torch.no_grad():
            l_dst = window_loss(dst, data, starts, args.check_block, args.device)
        dst.cpu()
        ratio = l_dst / l_src
        print(f"[gate] step-0 loss src {l_src:.4f} grown {l_dst:.4f} ratio {ratio:.5f} (max {args.max_ratio})")
        if ratio > args.max_ratio:
            sys.exit(f"[gate] FAIL: grown loss ratio {ratio:.5f} > {args.max_ratio}; not saving")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": dst.state_dict(), "cfg": d_cfg, "step": -1, "tokenizer": ckpt.get("tokenizer"),
                "grown_from": args.src}, args.out)
    print(f"[info] saved {args.out}")


if __name__ == "__main__":
    main()
