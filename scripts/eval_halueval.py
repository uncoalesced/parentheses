import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

os.environ['HF_HOME'] = r'F:\Lingual_Data\raw\hf\cache'

import torch
import safetensors.torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.model_slab_v4 import SlabV4Model, SlabV4Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RELEASE_DIR = REPO_ROOT / "release" / "parentheses-1.0-240M-tanzanite"
OUT_DIR = REPO_ROOT / "out" / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOCK_FILE = REPO_ROOT / "out" / "locks" / "GPU.lock"
LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)

# Lock GPU
with open(LOCK_FILE, "w") as f:
    f.write(f"PID={os.getpid()} TASK=halueval_qa_evaluation TIME={time.time()}\n")

print(f"[*] Locked GPU for HaluEval evaluation on {DEVICE}...")

try:
    # 1. Fetch 30 balanced QA items from HaluEval
    data_file = OUT_DIR / "halueval_qa_30.json"
    if not data_file.exists():
        print("[*] Downloading HaluEval QA items from Hugging Face...")
        url = 'https://datasets-server.huggingface.co/rows?dataset=pminervini/HaluEval&config=qa_samples&split=data&offset=0&limit=100'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        
        yes_items = []
        no_items = []
        for r in data.get('rows', []):
            item = r.get('row')
            h = item.get('hallucination', '').strip().lower()
            if h == 'yes' and len(yes_items) < 15:
                yes_items.append(item)
            elif h == 'no' and len(no_items) < 15:
                no_items.append(item)
            if len(yes_items) == 15 and len(no_items) == 15:
                break
        
        # Interleave yes and no
        combined = []
        for y, n in zip(yes_items, no_items):
            combined.append(y)
            combined.append(n)
        
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2)
        print(f"[+] Cached 30 balanced HaluEval QA items to {data_file}")
    else:
        with open(data_file, "r", encoding="utf-8") as f:
            combined = json.load(f)
        print(f"[+] Loaded {len(combined)} HaluEval QA items from cache.")

    # 2. Load Model
    print("[*] Loading Parentheses 1.0-240M model...")
    with open(RELEASE_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg_data = json.load(f)
    valid_keys = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd", "chunk_size", "eot_token_id", "dropout", "bias"}
    cfg = SlabV4Config(**{k: v for k, v in cfg_data.items() if k in valid_keys})
    model = SlabV4Model(cfg).to(DEVICE).eval()
    safetensors.torch.load_model(model, str(RELEASE_DIR / "model.safetensors"))

    tokenizer = Tokenizer.from_file(str(RELEASE_DIR / "tokenizer.json"))
    user_id = tokenizer.token_to_id("<|user|>")
    asst_id = tokenizer.token_to_id("<|assistant|>")
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    lang_id = tokenizer.token_to_id("<eng>")

    def generate(prompt: str, max_tokens: int = 32, temperature: float = 0.0, seed: int = 42):
        torch.manual_seed(seed)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
        input_ids = [user_id, lang_id] + prompt_ids + [asst_id]
        inp = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

        states = model.empty_states(1, device=DEVICE)
        with torch.no_grad():
            for i in range(inp.size(1) - 1):
                _, states = model.step(inp[:, i:i+1], states)

            curr = inp[:, -1:]
            gen_ids = []
            for _ in range(max_tokens):
                logits, states = model.step(curr, states)
                logits_step = logits[0] if logits.dim() == 2 else logits[0, 0]
                if temperature > 0:
                    probs = torch.softmax(logits_step / temperature, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_tok = int(torch.argmax(logits_step, dim=-1).item())

                if next_tok in (eot_id, user_id):
                    break
                gen_ids.append(next_tok)
                curr = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)

        return tokenizer.decode(gen_ids, skip_special_tokens=False).strip()

    # 3. Execute 4 Runs
    runs_config = [
        {"run_id": 1, "type": "greedy", "temp": 0.0, "seed": 42},
        {"run_id": 2, "type": "sampled", "temp": 0.7, "seed": 42},
        {"run_id": 3, "type": "sampled", "temp": 0.7, "seed": 101},
        {"run_id": 4, "type": "sampled", "temp": 0.7, "seed": 777},
    ]

    all_run_results = []
    scores = []

    print("[*] Starting 4 evaluation runs across 30 tasks...")
    for rc in runs_config:
        correct = 0
        details = []
        t0 = time.time()
        for idx, item in enumerate(combined):
            knowledge = item.get("knowledge", "").strip()
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            gold = item.get("hallucination", "").strip().lower()

            prompt = (
                f"Knowledge: {knowledge}\n"
                f"Question: {question}\n"
                f"Proposed Answer: {answer}\n"
                f"Does the proposed answer contain a hallucination or factually unsupported claim based on the knowledge? "
                f"Respond with 'Yes' or 'No'."
            )
            resp = generate(prompt, max_tokens=16, temperature=rc["temp"], seed=rc["seed"])
            
            # Extract yes or no
            m = re.search(r'\b(yes|no)\b', resp.lower())
            pred = m.group(1) if m else "unclear"
            is_correct = (pred == gold)
            if is_correct:
                correct += 1
            details.append({
                "idx": idx,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_response": resp
            })
        
        acc = round(correct / len(combined) * 100, 2)
        elapsed = round(time.time() - t0, 1)
        scores.append(acc)
        print(f"[+] Run {rc['run_id']} ({rc['type']}, T={rc['temp']}, seed={rc['seed']}): {correct}/{len(combined)} = {acc}% ({elapsed}s)")
        all_run_results.append({
            "run_config": rc,
            "accuracy": acc,
            "correct": correct,
            "total": len(combined),
            "details": details
        })

    best_score = max(scores)
    avg_score = round(sum(scores) / len(scores), 2)

    summary = {
        "benchmark": "HaluEval QA (Hallucination Detection)",
        "hf_dataset": "pminervini/HaluEval (config: qa_samples)",
        "task_count": len(combined),
        "run_1_greedy": scores[0],
        "run_2_sampled_s42": scores[1],
        "run_3_sampled_s101": scores[2],
        "run_4_sampled_s777": scores[3],
        "best_score": best_score,
        "average_score": avg_score,
        "runs": all_run_results
    }

    out_file = OUT_DIR / "halueval_qa_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n==========================================")
    print(f"HaluEval QA Results:")
    print(f"Run 1: {scores[0]}% | Run 2: {scores[1]}% | Run 3: {scores[2]}% | Run 4: {scores[3]}%")
    print(f"BEST: {best_score}% | AVERAGE: {avg_score}%")
    print(f"Saved full report to {out_file}")
    print(f"==========================================")

finally:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()
        print("[*] Released GPU lock.")
