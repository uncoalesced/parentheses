"""
NPU checkpoint eval: per-language val loss of a checkpoint on the AMD Phoenix NPU, so training
progress can be tracked while the GPU trains (AGENTS.md: "small-model eval during training").

Two steps, two Python envs (torch lives in .venv, the VitisAI EP in the Ryzen AI conda env):
    .venv\\Scripts\\python.exe scripts\\npu_ckpt_eval.py export --checkpoint CKPT --out-dir out/npu_eval/<tag>
    E:\\Parentheses\\NPU\\conda_envs\\ryzen-ai\\python.exe scripts\\npu_ckpt_eval.py run --out-dir out/npu_eval/<tag>

export: fixed-shape [1, T] loss graph (idx, tgt -> mean CE) to ONNX fp32 + fixed val windows from
        data/processed_v4/val, plus the torch fp32 reference loss on the same windows. 1 CPU thread.
run:    static QDQ INT8 quantization (MatMul/Gemm), then the same windows on CPU fp32 and on
        VitisAI (NPU) INT8. Writes result.json with both losses and the INT8-vs-fp32 gap.
        INT8 numbers are a *relative* curve across checkpoints only: never compare them with
        the GPU bf16 bpb evals.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("kan", "tam", "tel", "mal", "tcy", "kff", "kxv", "nit", "eng")
VAIP_CFG = Path(r"E:\Parentheses\NPU\ryzenai_sdk\PFiles64\RyzenAI\1.8.0\voe-4.0-win_amd64\vaip_config.json")


def windows(T: int, n: int, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for lang in LANGS:
        a = np.fromfile(ROOT / f"data/processed_v4/val/{lang}.bin", dtype=np.uint16)
        starts = rng.integers(0, len(a) - T - 1, n)
        out[lang] = np.stack([a[s:s + T + 1] for s in starts]).astype(np.int64)
    return out


def cmd_export(a):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)  # don't steal CPU from the GPU trainer
    sys.path.insert(0, str(ROOT))
    from model.backbone import Parentheses

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    m = Parentheses(ck["cfg"])
    m.load_state_dict(ck["model"])
    m.eval()

    class LossGraph(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, idx, tgt):
            logits, _ = self.model(idx)
            return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1)).unsqueeze(0)

    g = LossGraph(m).eval()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, a.seq, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(g, (x, x), str(out / "loss_fp32.onnx"), input_names=["idx", "tgt"],
                          output_names=["loss"], opset_version=17, dynamo=False)
    w = windows(a.seq, a.windows)
    np.savez(out / "windows.npz", **w)
    ref = {}
    with torch.no_grad():
        for lang, arr in w.items():
            ref[lang] = float(np.mean([g(torch.from_numpy(r[None, :-1]), torch.from_numpy(r[None, 1:])).item()
                                       for r in arr]))
    meta = {"checkpoint": a.checkpoint, "step": ck.get("step"), "seq": a.seq, "windows": a.windows,
            "export_seconds": round(time.time() - t0, 1), "torch_fp32_loss": ref}
    (out / "export.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


def vaip_options(out: Path) -> dict:
    """Phoenix (XDNA1) needs the X1 DPU flow; the config's default target X2 is VAIML (Strix-only).
    NPU_TARGET / NPU_XCLBIN override, as AMD's quicktest does for PHX."""
    o = {"config_file": str(VAIP_CFG), "cache_dir": str(out / "vaip_cache"),
         "cache_key": "loss_" + os.environ.get("NPU_TARGET", "X1"),
         "target": os.environ.get("NPU_TARGET", "X1"), "xlnx_enable_py3_round": "0"}
    if os.environ.get("NPU_XCLBIN"):
        o["xclbin"] = os.environ["NPU_XCLBIN"]
    return o


def cmd_run(a):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    out = Path(a.out_dir)
    w = dict(np.load(out / "windows.npz"))
    int8 = out / "loss_int8_qdq.onnx"
    if not int8.exists():
        calib = [{"idx": r[None, :-1], "tgt": r[None, 1:]} for lang in w for r in w[lang][:1]]

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.it = iter(calib)

            def get_next(self):
                return next(self.it, None)

        t0 = time.time()
        quantize_static(str(out / "loss_fp32.onnx"), str(int8), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["MatMul", "Gemm"],
                        # Phoenix DPU (X1) accepts uint8 only with zero point 128, i.e. symmetric
                        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True})
        print(f"[run] quantized in {time.time() - t0:.0f}s")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    res = {}
    for name, path, provs, opts in (
            ("cpu_fp32", out / "loss_fp32.onnx", ["CPUExecutionProvider"], None),
            ("npu_int8", int8, ["VitisAIExecutionProvider", "CPUExecutionProvider"],
             [vaip_options(out), {}])):
        t0 = time.time()
        s = ort.InferenceSession(str(path), sess_options=so, providers=provs, provider_options=opts)
        t1 = time.time()
        loss = {lang: float(np.mean([s.run(["loss"], {"idx": r[None, :-1], "tgt": r[None, 1:]})[0][0]
                                     for r in arr])) for lang, arr in w.items()}
        res[name] = {"providers": s.get_providers(), "init_s": round(t1 - t0, 1),
                     "eval_s": round(time.time() - t1, 1), "loss": loss}
        print(name, json.dumps(res[name]), flush=True)
    meta = json.loads((out / "export.json").read_text(encoding="utf-8"))
    res["int8_minus_fp32"] = {k: round(res["npu_int8"]["loss"][k] - res["cpu_fp32"]["loss"][k], 4) for k in LANGS}
    res["onnx_fp32_minus_torch"] = {k: round(res["cpu_fp32"]["loss"][k] - meta["torch_fp32_loss"][k], 5) for k in LANGS}
    (out / "result.json").write_text(json.dumps({**meta, **res}, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("int8_minus_fp32", "onnx_fp32_minus_torch")}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--out-dir", required=True)
    e.add_argument("--seq", type=int, default=512)
    e.add_argument("--windows", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--out-dir", required=True)
    a = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    (cmd_export if a.cmd == "export" else cmd_run)(a)


if __name__ == "__main__":
    main()
