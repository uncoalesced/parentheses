"""Loading and saving weights.

The format for 1.1 is safetensors. Checkpoints written by the 1.0 training
code are pickled `.pt` files; `load_state` reads those too, without running
arbitrary pickle code.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys
import types
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

from .config import SlabConfig
from .model import Parentheses

PathLike = Union[str, Path]


def sha256_file(path: PathLike, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _legacy_config_class():
    """1.0 checkpoints pickle a `model.config.ModelConfig` object next to the
    weights. Provide an inert stand-in so the safe unpickler can accept it."""
    names = ("model", "model.config")
    saved = {n: sys.modules.get(n) for n in names}
    package, module = types.ModuleType("model"), types.ModuleType("model.config")

    class ModelConfig:
        pass

    ModelConfig.__module__ = "model.config"
    ModelConfig.__qualname__ = "ModelConfig"
    module.ModelConfig = ModelConfig
    package.config = module
    sys.modules.update({"model": package, "model.config": module})
    try:
        with torch.serialization.safe_globals([ModelConfig]):
            yield
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def load_state(path: PathLike) -> Tuple[Dict[str, torch.Tensor], dict]:
    """Read weights from `.safetensors` or a 1.0 `.pt`. Returns (state dict, metadata)."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path)), {}
    with _legacy_config_class():
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        raise ValueError(f"{path.name}: not a checkpoint dictionary")
    state = blob.get("model", blob)
    meta = {k: blob[k] for k in ("step", "tokenizer") if k in blob}
    return state, meta


def load_model(path: PathLike, device: Union[str, torch.device] = "cpu",
               dtype: Optional[torch.dtype] = None, **config_overrides) -> Parentheses:
    """Build a model shaped like the checkpoint and load it. Returned in eval mode."""
    state, _ = load_state(path)
    model = Parentheses(SlabConfig.from_state_dict(state, **config_overrides))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The output head shares the embedding matrix, so a file may leave it out.
    if set(missing) - {"head.weight"} or unexpected:
        raise ValueError(f"checkpoint does not fit: missing {sorted(missing)}, "
                         f"unexpected {sorted(unexpected)}")
    model.head.weight = model.tok_emb.weight
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval()


def save_safetensors(model: Parentheses, path: PathLike) -> None:
    """Write weights as safetensors. The tied head is stored once, as the embedding."""
    from safetensors.torch import save_file
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
             if k != "head.weight"}
    save_file(state, str(path))
