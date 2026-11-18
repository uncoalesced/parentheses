"""parentheses: a recurrent language model with constant-size state."""

from pathlib import Path

from .config import SLAB_V4_240M, SlabConfig
from .model import LayerState, Parentheses

# The 32k byte-level BPE tokenizer (1.0). Its merges can split an akshara (GATES M4).
TOKENIZER_PATH = Path(__file__).resolve().parent / "tokenizer_data" / "tokenizer.json"

__all__ = ["SLAB_V4_240M", "SlabConfig", "LayerState", "Parentheses", "TOKENIZER_PATH"]
