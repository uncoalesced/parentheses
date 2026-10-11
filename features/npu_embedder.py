"""
NPU / ONNX Embedder backend for Modular Free Think / Vector Memory.
Runs fixed-shape ONNX embeddings with VitisAIExecutionProvider on AMD Ryzen AI NPU
with graceful fallback to CPUExecutionProvider.

Engineered for Parentheses 0.9.
"""

from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class OnnxEmbedder:
    """Wraps an exported fixed-shape or dynamic ONNX embedder model.
    Runs on AMD NPU via VitisAIExecutionProvider if available, with automatic
    fallback to CPUExecutionProvider.
    """

    def __init__(self, model_path: str, provider: str = "auto", fixed_seq: int = 128):
        self.model_path = str(model_path)
        self.fixed_seq = fixed_seq
        self.provider = provider
        self.session = None
        self.active_provider = None
        self._init_session()

    def _init_session(self):
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Install onnxruntime or use backend='bm25'."
            )

        available = ort.get_available_providers()
        requested_providers = []

        if self.provider in ("auto", "npu"):
            if "VitisAIExecutionProvider" in available:
                requested_providers.append("VitisAIExecutionProvider")
            elif self.provider == "npu":
                print(
                    "[embedder] Warning: VitisAIExecutionProvider not in available providers; "
                    "falling back to CPU"
                )

        # Always include CPU fallback
        requested_providers.append("CPUExecutionProvider")

        # Attempt initialization
        try:
            self.session = ort.InferenceSession(
                self.model_path, providers=requested_providers
            )
            self.active_provider = self.session.get_providers()[0]
        except Exception as e:
            if "VitisAIExecutionProvider" in requested_providers:
                print(
                    f"[embedder] VitisAIExecutionProvider init failed ({e}), "
                    f"falling back to CPUExecutionProvider"
                )
                self.session = ort.InferenceSession(
                    self.model_path, providers=["CPUExecutionProvider"]
                )
                self.active_provider = "CPUExecutionProvider"
            else:
                raise

        # Check metadata for fixed dimensions
        meta = self.session.get_modelmeta().custom_metadata_map
        if "fixed_seq" in meta:
            self.fixed_seq = int(meta["fixed_seq"])
        elif "block_size" in meta:
            self.fixed_seq = int(meta["block_size"])

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.embed(texts)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 64), dtype=np.float32)

        results = []
        for text in texts:
            raw_bytes = list(text.encode("utf-8"))[: self.fixed_seq]
            seq_len = len(raw_bytes)
            idx = np.zeros((1, self.fixed_seq), dtype=np.int64)
            mask = np.zeros((1, self.fixed_seq), dtype=bool)
            if seq_len > 0:
                idx[0, :seq_len] = raw_bytes
                mask[0, :seq_len] = True

            out = self.session.run(["embedding"], {"idx": idx, "mask": mask})[0]
            results.append(out[0])

        vecs = np.array(results, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-9)


def make_onnx_embedder(
    model_path: str = "checkpoints/all-sources-v1/embedder_fixed128.onnx",
    provider: str = "auto",
) -> OnnxEmbedder:
    """Convenience factory returning a callable embedder."""
    return OnnxEmbedder(model_path, provider=provider)
