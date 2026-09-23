"""
Export the trained embedding head (+ its backbone) to ONNX -- so it can run
on something other than CPU/CUDA, specifically the Ryzen AI NPU via
onnxruntime's VitisAIExecutionProvider, or DirectML as a lighter-weight
alternative on hardware where that's set up.

Engineered by uncoalesced

Tokenization stays outside the graph (plain UTF-8 byte encoding, same as
model/embedding_head.py's batch_bytes) -- only the tensor math (backbone
hidden states + pooling + projection + L2-normalize) goes through ONNX.

NPU setup this script does NOT do for you (no shell access at the time this
was written, and it's a proprietary installer, not a pip package): AMD's
Ryzen AI Software has to be installed manually, the exported model likely
needs INT8 quantization via `vai_q_onnx` for the NPU path specifically, and
XLNX_VART_FIRMWARE needs to be set. See
https://ryzenai.docs.amd.com/en/latest/getstartex.html. Until that's done,
this exports a plain FP32 ONNX model that already runs fine on
CPUExecutionProvider -- scripts/dedup_raw_sourced.py picks whatever
provider is actually available and tells you which one it used.

    python3 scripts/export_embedder_onnx.py
    python3 scripts/export_embedder_onnx.py --self-test
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.embedding_head import EmbeddingHead, batch_bytes, load_trained
from model.backbone import Parentheses


class _EmbedForExport(nn.Module):
    """backbone.hidden() + head(), fused into one ONNX-traceable module.

    Kept separate from EmbeddingHead itself rather than adding an
    export mode to that class -- this module owns nothing about training,
    it is purely the inference-time composition, same reasoning
    embedding_head.py's own docstring gives for staying "beside
    Parentheses, not a fork of it."
    """

    def __init__(self, backbone: Parentheses, head: EmbeddingHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, idx: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.hidden(idx), mask)


def export(checkpoint: str, head_path: str, out_path: str, opset: int = 17) -> dict:
    """-> head_meta (dim, n_embd, etc.), same shape load_trained returns."""
    backbone, head, meta = load_trained(checkpoint, head_path, "cpu")
    wrapper = _EmbedForExport(backbone, head).eval()

    # Two probe texts of different lengths so the traced graph's seq-len axis
    # is exercised as genuinely dynamic before dynamic_axes gets to rely on it.
    idx, mask = batch_bytes(["export probe", "a second, longer probe sentence"],
                             backbone.cfg.block_size, "cpu")
    torch.onnx.export(
        wrapper, (idx, mask), out_path,
        input_names=["idx", "mask"], output_names=["embedding"],
        dynamic_axes={"idx": {0: "batch", 1: "seq"},
                       "mask": {0: "batch", 1: "seq"},
                       "embedding": {0: "batch"}},
        opset_version=opset,
        dynamo=False,  # torch>=2.5 defaults to the onnxscript-based exporter;
                        # force the legacy TorchScript one so onnxscript isn't a dep.
    )

    # Stash block_size in the ONNX file's own metadata so a caller loading
    # only the .onnx (not the original .pt checkpoint) still truncates text
    # correctly. This matters, not just informational: RoPE's precomputed
    # table (precompute_rope in model/backbone.py) is sized to
    # cfg.block_size at trace time -- feeding a longer sequence at
    # inference would index past it.
    import onnx
    onnx_model = onnx.load(out_path)
    entry = onnx_model.metadata_props.add()
    entry.key = "block_size"
    entry.value = str(backbone.cfg.block_size)
    onnx.save(onnx_model, out_path)

    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/all-sources-v1/step_560999_final.pt",
                   help="backbone checkpoint (the 29-source all-sources-v1 backbone by default; "
                        "checkpoints/multilingual-22/step_298999_final.pt is still usable explicitly)")
    p.add_argument("--head", default="checkpoints/all-sources-v1/embedding_head_all_sources_v1.pt",
                   help="embedding head trained against that backbone -- a head only means "
                        "anything paired with the exact backbone it trained against, so this "
                        "default moves with --checkpoint and is NOT interchangeable with "
                        "multilingual-22's embedding_head_multi22.pt")
    p.add_argument("--out", default="checkpoints/all-sources-v1/embedder.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    if not os.path.exists(args.head):
        raise SystemExit(
            f"--head {args.head} does not exist. A head is only meaningful beside the exact "
            f"backbone it trained against (see model/embedding_head.py's docstring), so it "
            f"cannot be borrowed from another checkpoint dir. Either train one:\n"
            f"  python3 scripts/train_embedding_head.py --checkpoint {args.checkpoint} "
            f"--raw-dirs data/raw/parallel/en-* --max-pairs 50000 --out {args.head}\n"
            f"or export the older matched pair explicitly:\n"
            f"  python3 scripts/export_embedder_onnx.py "
            f"--checkpoint checkpoints/multilingual-22/step_298999_final.pt "
            f"--head checkpoints/multilingual-22/embedding_head_multi22.pt "
            f"--out checkpoints/multilingual-22/embedder.onnx")

    # load_trained() (inside export()) refuses a head/backbone mismatch on
    # its own now -- was checked here too before that guard moved into the
    # shared loader so every caller gets it, not just this script.
    try:
        meta = export(args.checkpoint, args.head, args.out, args.opset)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"[info] exported {args.out} (dim={meta['dim']}, n_embd={meta['n_embd']})")


def _self_test():
    """Save a tiny untrained model as a fake checkpoint+head pair, run it
    through the real export() function (not a reimplementation of it), and
    check the ONNX output matches PyTorch's. The failure mode worth
    catching here isn't "does torch.onnx.export raise" -- it's the graph
    silently computing something different (a squeezed axis, the mask cast
    getting dropped, padding leaking into the pooled mean), or the
    block_size metadata not actually round-tripping. Doesn't need the real
    checkpoint, the real corpus, or an NPU -- this is a pure mechanics check.
    """
    import tempfile

    from model.config import PRESETS

    try:
        import onnxruntime as ort
    except ImportError:
        print("[self-test] SKIPPED -- onnxruntime not installed "
              "(pip install onnxruntime to run this check)")
        return

    torch.manual_seed(0)
    # Not tiny-smoke: its vocab_size=64 can't hold ordinary ASCII bytes
    # ('h' is 104), same reason embedding_head.py's own self-test avoids it.
    cfg = PRESETS["parentheses-0.9-300k"]
    backbone = Parentheses(cfg).eval()
    for prm in backbone.parameters():
        prm.requires_grad_(False)
    head = EmbeddingHead(cfg.n_embd, dim=16).eval()

    with tempfile.TemporaryDirectory() as td:
        ckpt_path = os.path.join(td, "backbone.pt")
        head_path = os.path.join(td, "head.pt")
        torch.save({"model": backbone.state_dict(), "cfg": cfg}, ckpt_path)
        torch.save({"n_embd": cfg.n_embd, "dim": 16, "head": head.state_dict()}, head_path)

        onnx_path = os.path.join(td, "probe.onnx")
        export(ckpt_path, head_path, onnx_path)  # the real function, not a duplicate

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # block_size metadata actually round-trips through the .onnx file
        meta_map = sess.get_modelmeta().custom_metadata_map
        assert meta_map.get("block_size") == str(cfg.block_size), meta_map

        wrapper = _EmbedForExport(backbone, head)
        texts = ["hello world", "a slightly longer sentence to pad against", "x"]
        idx, mask = batch_bytes(texts, cfg.block_size, "cpu")
        with torch.no_grad():
            torch_out = wrapper(idx, mask).numpy()
        onnx_out = sess.run(["embedding"], {"idx": idx.numpy(), "mask": mask.numpy()})[0]

        diff = abs(torch_out - onnx_out).max()
        assert diff < 1e-4, f"ONNX output diverges from PyTorch by {diff}"

        # A batch with different padding must still match its solo encoding
        # -- this is the check that would catch the mask being silently
        # ignored by the exported graph (dynamic seq-len axis, real risk).
        idx2, mask2 = batch_bytes(["hello world"], cfg.block_size, "cpu")
        solo_out = sess.run(["embedding"], {"idx": idx2.numpy(), "mask": mask2.numpy()})[0]
        assert abs(solo_out[0] - onnx_out[0]).max() < 1e-4, \
            "padding changed the embedding -- mask not honoured in the exported graph"

    print(f"[self-test] export_embedder_onnx ok (max diff {diff:.2e})")


if __name__ == "__main__":
    main()
