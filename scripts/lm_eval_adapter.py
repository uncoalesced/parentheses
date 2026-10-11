"""
EleutherAI lm-evaluation-harness adapter for Parentheses BPE checkpoints.

    python scripts/lm_eval_adapter.py --checkpoint checkpoints/parentheses-0.9-41M-v4/step_16082_final.pt \
        --tasks arc_easy,hellaswag --limit 20 --out out/lm_eval_41M_v4.json
    python scripts/lm_eval_adapter.py --self-test

Every context is fed as <lang> + text (the pretraining document format; default <eng>). Contexts
longer than block_size are cut from the left. Greedy generation is top_k=1.
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm_eval.api.model import LM  # noqa: E402

from model.backbone import Parentheses  # noqa: E402


class ParenthesesLM(LM):
    def __init__(self, model: Parentheses, tok, lang: str = "eng", device: str = "cuda"):
        super().__init__()
        self.model, self.tok, self.device_ = model.eval(), tok, device
        self.tag = tok.token_to_id(f"<{lang}>")
        assert self.tag is not None, f"no <{lang}> tag in tokenizer"
        self.eot = tok.token_to_id("<|endoftext|>")
        self.block = model.cfg.block_size

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s).ids if s else []

    @torch.no_grad()
    def _cont_logprob(self, ctx: list[int], cont: list[int]) -> tuple[float, bool]:
        ids = ([self.tag] + ctx + cont)[-(self.block + 1):]
        x = torch.tensor([ids[:-1]], device=self.device_)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device_ == "cuda"):
            logits, _ = self.model(x)
        lp = F.log_softmax(logits[0, -len(cont):].float(), dim=-1)
        tgt = torch.tensor(ids[-len(cont):], device=self.device_)
        return lp.gather(1, tgt[:, None]).sum().item(), bool((lp.argmax(-1) == tgt).all())

    def loglikelihood(self, requests):
        out = []
        for r in requests:
            ctx, cont = r.args
            if ctx.endswith(" "):  # keep the space with the continuation, as HF models do
                ctx, cont = ctx[:-1], " " + cont
            c = self._enc(cont)
            out.append(self._cont_logprob(self._enc(ctx), c) if c else (0.0, True))
        return out

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):
        out = []
        for r in requests:
            ids = [self.tag] + self._enc(r.args[0])
            total = 0.0
            for s in range(1, len(ids), self.block):  # disjoint windows; each scores its own tokens
                ctx, cont = ids[max(1, s - self.block):s], ids[s:s + self.block]
                total += self._cont_logprob(ctx, cont)[0]
            out.append(total)
        return out

    @torch.no_grad()
    def generate_until(self, requests):
        out = []
        for r in requests:
            ctx, kw = r.args
            until = kw.get("until", []) or []
            max_new = kw.get("max_gen_toks", 256)
            ids = ([self.tag] + self._enc(ctx))[-self.block:]
            gen = []
            for t in self.model.stream(torch.tensor([ids], device=self.device_), max_new, 1.0, 1):
                t = int(t)
                if t == self.eot:
                    break
                gen.append(t)
                text = self.tok.decode(gen, skip_special_tokens=True)
                if any(u in text for u in until):
                    break
            text = self.tok.decode(gen, skip_special_tokens=True)
            for u in until:
                text = text.split(u)[0]
            out.append(text)
        return out


def load(checkpoint: str, device: str, lang: str) -> ParenthesesLM:
    from tokenizers import Tokenizer
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = Parentheses(ckpt["cfg"])
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_file(os.path.join(ckpt["tokenizer"]["path"], "tokenizer.json"))
    tok.encode_special_tokens = True
    return ParenthesesLM(model.to(device), tok, lang, device)


def _self_test():
    from tokenizers import Tokenizer

    from model.config import ModelConfig

    class Req:
        def __init__(self, *args):
            self.args = args

    torch.manual_seed(0)
    tok = Tokenizer.from_file(os.path.join("data", "tokenizer", "v3_32k", "tokenizer.json"))
    tok.encode_special_tokens = True
    m = Parentheses(ModelConfig(vocab_size=32768, block_size=64, n_layer=2, n_head=2, n_embd=32,
                                attn_type="selective_linear"))
    lm = ParenthesesLM(m, tok, "eng", "cpu")
    (a, _), (b, _) = lm.loglikelihood([Req("The cat sat on the", " mat"), Req("The cat sat on the ", "mat")])
    assert abs(a - b) < 1e-6 and a < 0, (a, b)  # trailing space moves to the continuation
    # rolling over > block_size tokens: every token after the tag is scored exactly once
    text = "word " * 150
    roll = lm.loglikelihood_rolling([Req(text)])[0]
    ids = [lm.tag] + tok.encode(text).ids
    assert len(ids) > 64 and roll < 0
    ref = sum(lm._cont_logprob(ids[max(1, s - 64):s], ids[s:s + 64])[0] for s in range(1, len(ids), 64))
    assert abs(roll - ref) < 1e-4, (roll, ref)
    g = lm.generate_until([Req("Hello", {"until": ["\n"], "max_gen_toks": 5})])
    assert isinstance(g[0], str) and "\n" not in g[0]
    print("lm_eval_adapter self-test passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--tasks", default="arc_easy,arc_challenge,hellaswag,winogrande,piqa")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--limit", type=float, default=None)
    p.add_argument("--lang", default="eng")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        _self_test()
        return
    import lm_eval
    lm = load(args.checkpoint, args.device, args.lang)
    res = lm_eval.simple_evaluate(model=lm, tasks=args.tasks.split(","), num_fewshot=args.num_fewshot,
                                  limit=args.limit)
    summary = {"checkpoint": args.checkpoint, "limit": args.limit, "results": res["results"]}
    print(json.dumps(summary["results"], indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
