#!/usr/bin/env python3
"""Shareable samples rating page (Claude Code, 2026-10-16; Joel: samples for all languages, friends rate them).

Reads the sample prompt set plus generated outputs (any number of run files; each row's `format` is base or chat)
and writes one self-contained HTML page with the outputs embedded verbatim. Ratings go to the artifact db
(`ratings/<viewer id>`), with a copy-as-text fallback for viewers who can't write.

    .venv\\Scripts\\python.exe scripts\\build_samples_page.py --step 9536 --runs out\\samples_s5\\base_step9536.jsonl out\\samples_s5\\chat_step9536.jsonl
    .venv\\Scripts\\python.exe scripts\\build_samples_page.py --self-test
"""

import argparse
import json
from pathlib import Path

LANG_NAMES = {"kan": "Kannada", "tam": "Tamil", "tel": "Telugu", "mal": "Malayalam", "tcy": "Tulu", "kff": "Koya",
              "kxv": "Kuvi", "nit": "SE Kolami", "eng": "English", "hin": "Hindi"}
TEMPLATE = Path(__file__).with_name("samples_page_template.html")


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def build_data(queries: list[dict], runs: list[list[dict]], step: int, note: str) -> dict:
    outs = {}
    for rows in runs:
        for r in rows:
            outs[(r["id"], r.get("format", "base"))] = r["output"]
    langs = []
    for code, name in LANG_NAMES.items():
        items = [{"id": q["id"], "bucket": q.get("bucket") or q.get("format") or "", "prompt": q["prompt"],
                  "reference": q.get("reference") or "",
                  "base": outs.get((q["id"], "base")), "chat": outs.get((q["id"], "chat"))}
                 for q in queries if q["lang"] == code]
        if items:
            langs.append({"code": code, "name": name, "items": items})
    return {"step": step, "note": note, "decode": "temperature 0.9, top-k 50", "langs": langs}


def render(data: dict) -> str:
    # keep "</script>" out of the JSON; U+FFFD (the model's invalid-byte output) as a JS escape, same value on parse
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/").replace("�", "\\ufffd")
    return TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)


def self_test():
    q = [{"id": "a", "lang": "kan", "bucket": "x", "prompt": "p</script>"}, {"id": "b", "lang": "eng", "prompt": "q"}]
    d = build_data(q, [[{"id": "a", "format": "base", "output": "out"}], [{"id": "a", "format": "chat", "output": "c"}]],
                   7, "n")
    assert [l["code"] for l in d["langs"]] == ["kan", "eng"]
    assert d["langs"][0]["items"][0]["base"] == "out" and d["langs"][0]["items"][0]["chat"] == "c"
    assert d["langs"][1]["items"][0]["base"] is None
    html = render(d)
    assert "p<\\/script>" in html and "/*__DATA__*/" not in html
    print("[self-test] build_samples_page ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", default="eval/queries_samples.jsonl")
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--step", type=int, required=False, default=0)
    p.add_argument("--note", default="")
    p.add_argument("--out", default="out/samples_s5/samples_page.html")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    d = build_data(jl(a.queries), [jl(r) for r in a.runs], a.step, a.note)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render(d), encoding="utf-8")
    n = sum(len(l["items"]) for l in d["langs"])
    filled = sum((i["base"] is not None) + (i["chat"] is not None) for l in d["langs"] for i in l["items"])
    print(f"{a.out}: {len(d['langs'])} languages, {n} prompts, {filled} outputs")


if __name__ == "__main__":
    main()
