# Engineered by uncoalesced
"""AutoModel (trust_remote_code) integration tests for parentheses-1.0-240M.

These load the Hub staging folder built by scripts/stage_release_240m.py and check
that the self-contained Hugging Face port computes the same function as the
training-code model. They are skipped when the staging folder is absent.

    python scripts/stage_release_240m.py
    python -m pytest tests/test_hf_automodel.py -q
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "release" / "hf-staging" / "parentheses-1.0-240M"
BUNDLE = ROOT / "release" / "parentheses-1.0-240M-tanzanite"

# Keep the transformers dynamic-module cache next to the repo instead of the user profile.
os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / "out" / "hf_modules_cache"))

pytestmark = pytest.mark.skipif(
    not (STAGE / "model.safetensors").exists(),
    reason="Hub staging folder missing; run scripts/stage_release_240m.py",
)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

EXPECTED_PARAMS = 253_544_176
EOT = 0


@pytest.fixture(scope="module")
def hf():
    tok = transformers.AutoTokenizer.from_pretrained(str(STAGE))
    model = transformers.AutoModelForCausalLM.from_pretrained(str(STAGE), trust_remote_code=True).eval()
    return model, tok


@pytest.fixture(scope="module")
def reference():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import chat as chat_mod
    model = chat_mod.load_model_and_tokenizer(BUNDLE, "cpu")[0]
    return model.eval()


def _ids(length: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(256, 32768, (1, length), generator=g)
    ids[0, length // 3] = EOT  # a document boundary inside the sequence
    return ids


def test_class_and_param_count(hf):
    model, _ = hf
    assert type(model).__name__ == "ParenthesesForCausalLM"
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS
    assert model.head.weight is model.tok_emb.weight, "embeddings must be tied"


def test_tokenizer_special_tokens(hf):
    _, tok = hf
    assert tok.convert_tokens_to_ids("<|endoftext|>") == 0
    assert tok.convert_tokens_to_ids("<|user|>") == 13
    assert tok.convert_tokens_to_ids("<|assistant|>") == 14
    assert tok.convert_tokens_to_ids("<kan>") == 2


def test_forward_matches_training_code(hf, reference):
    model, _ = hf
    ids = _ids(128)
    with torch.no_grad():
        ours = model(ids).logits
        theirs, _ = reference(ids)
    assert ours.shape == theirs.shape == (1, 128, 32768)
    assert (ours - theirs).abs().max().item() < 1e-5


def test_forward_handles_any_length(hf):
    """Lengths that are not a multiple of the chunk size are padded internally."""
    model, _ = hf
    ids = _ids(128)
    with torch.no_grad():
        full = model(ids).logits
        part = model(ids[:, :100]).logits
    assert part.shape[1] == 100
    assert (full[:, :100] - part).abs().max().item() < 1e-4


def test_step_matches_forward(hf):
    """The recurrent form and the parallel form are the same function."""
    model, _ = hf
    ids = _ids(96, seed=1)
    with torch.no_grad():
        parallel = model(ids).logits[0]
        states, prev_eot, rows = None, False, []
        for t in range(ids.size(1)):
            logits, states = model.step(ids[:, t], states, is_boundary=prev_eot)
            prev_eot = int(ids[0, t]) == EOT
            rows.append(logits[0])
    recurrent = torch.stack(rows)
    assert (parallel - recurrent).abs().max().item() < 2e-3
    assert parallel.argmax(-1).eq(recurrent.argmax(-1)).float().mean().item() > 0.99


def test_state_size_is_constant(hf):
    model, _ = hf
    ids = _ids(64, seed=2)
    sizes = []
    with torch.no_grad():
        states = None
        for t in range(ids.size(1)):
            _, states = model.step(ids[:, t], states)
            sizes.append(model.state_bytes(states))
    assert len(set(sizes)) == 1
    # 16 layers x (16x64x64 state + 3x3072 conv buffer + 1024 shift buffer) float32
    assert sizes[0] == 16 * (16 * 64 * 64 + 3 * 3072 + 1024) * 4 == 4_849_664


def _device_params():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _device_params())
def test_generation(hf, device):
    model, tok = hf
    try:
        model.to(device)
        code = model.chat(tok, "Write a Python function that returns the factorial of n.", lang="eng",
                          temperature=0.0, max_new_tokens=60)
        assert "def factorial(n)" in code and "factorial(n-1)" in code.replace(" ", "")

        kan = model.chat(tok, "ಮಳೆಗಾಲದಲ್ಲಿ ನೀರಿನ ಸಂರಕ್ಷಣೆ ಏಕೆ ಮುಖ್ಯ?", lang="kan", temperature=0.0, max_new_tokens=40)
        kannada_chars = sum(1 for ch in kan if "ಀ" <= ch <= "೿")
        assert kannada_chars > 20, f"expected Kannada script, got {kan!r}"

        ids = model.build_prompt(tok, "Write a Python function that returns the factorial of n.").to(device)
        out = model.generate(ids, max_new_tokens=40, do_sample=False)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        assert "def factorial(n)" in text, f"generate() output: {text!r}"
    finally:
        model.to("cpu")


def test_loads_outside_the_repository():
    """A fresh interpreter with no access to the repo can load the staged folder."""
    script = (
        "import sys, torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "import importlib.util\n"
        "assert importlib.util.find_spec('models') is None, 'repo source must not be importable'\n"
        f"p = r'{STAGE}'\n"
        "tok = AutoTokenizer.from_pretrained(p)\n"
        "m = AutoModelForCausalLM.from_pretrained(p, trust_remote_code=True).eval()\n"
        "print('PARAMS', sum(x.numel() for x in m.parameters()))\n"
        "print('OUT', m.chat(tok, 'Write a Python function that returns the factorial of n.', temperature=0.0, max_new_tokens=40))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(dir=str(ROOT.parent)) as tmp:
        res = subprocess.run([sys.executable, "-c", script], cwd=tmp, env=env,
                             capture_output=True, text=True, encoding="utf-8", timeout=600)
    assert res.returncode == 0, res.stderr[-2000:]
    assert f"PARAMS {EXPECTED_PARAMS}" in res.stdout
    assert "def factorial(n)" in res.stdout
