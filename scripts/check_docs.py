"""
Catch docs that still name something the code no longer has.

Engineered by uncoalesced

`ParenthesesGPT` was renamed to `Parentheses` in model/backbone.py and every
importer was updated -- but four Markdown files kept naming the old class, and
nothing failed, because prose doesn't get imported. This is the check that
would have caught it.

Two rules, both narrow on purpose. A doc linter that cries wolf gets ignored,
and then it may as well not exist:

- A backticked `*.py` path has to be a file in the repo (`.old` counts -- the
  handoffs legitimately name retired modules).
- A backticked CamelCase name that appears in no .py file is only reported if
  it *looks like a rename of a real symbol*: some class or function this repo
  actually defines is a prefix or suffix of it. `ParenthesesGPT` gets caught
  because `Parentheses` is real; `Claude`, `GitHub`, `SessionStart` and every
  other capitalised word in prose does not, without an allowlist to maintain.

That second rule is the whole trick, and its limit is worth knowing: a rename
that keeps nothing of the old name (`Foo` -> `Bar`) slips through. Renames in
practice shorten, lengthen or re-prefix, which this catches.

Scope is docs that describe the code as it stands now. `documentation.md` is a
chronological session log -- it records renames, so old names appearing in it
is the point -- and `data/raw/` is training corpus, not documentation.

    python3 scripts/check_docs.py               # check the repo, exit 1 if stale
    python3 scripts/check_docs.py --self-test   # check the checker
"""

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Not this project's source or its current-state docs.
SKIP_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules", "raw"}
SKIP_DOCS = {"documentation.md", "documentation-old.md"}
# This file names retired symbols in its own docstring and self-test fixtures.
# Counting itself as source would make every one of them look alive again --
# which it did, on the first run: the planted `ParenthesesGPT` came back clean.
SKIP_SOURCE = {Path(__file__).name}

# Inside a backtick span: an identifier with an initial capital and at least one
# lowercase letter. The lowercase requirement keeps GPU/RAM/AMP/RAG out of it.
_BACKTICKED = re.compile(r"`([^`\n]+)`")
_CAMEL = re.compile(r"\b[A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*\b")
_PY_PATH = re.compile(r"\b[\w./-]+\.py\b")

# Short enough that "Get" or "In" would match half of English. A real symbol
# this project defines and a doc names is longer than this.
MIN_SYMBOL = 5


def _files(repo: Path, suffix: str) -> list[Path]:
    out = []
    for path in repo.rglob(f"*{suffix}"):
        parts = path.relative_to(repo).parts
        if SKIP_DIRS & set(parts) or path.name in SKIP_DOCS or path.name in SKIP_SOURCE:
            continue
        out.append(path)
    return out


def _defined_symbols(sources: list[Path]) -> set[str]:
    """Every class and function this repo defines, by name."""
    names = set()
    for path in sources:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue  # a .py in the tree that isn't importable is not our problem
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if len(node.name) >= MIN_SYMBOL:
                    names.add(node.name)
    return names


def _looks_like_a_rename(name: str, symbols: set[str]) -> bool:
    """True if a symbol the repo really defines is a prefix or suffix of `name`."""
    return any(name != s and (name.startswith(s) or name.endswith(s)) for s in symbols)


def stale_references(repo: Path = REPO) -> list[str]:
    """One message per doc reference that no longer exists in the source tree."""
    sources = _files(repo, ".py")
    blob = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in sources)
    symbols = _defined_symbols(sources)
    on_disk = {str(p.relative_to(repo)).replace("\\", "/")
               for p in list(repo.rglob("*.py")) + list(repo.rglob("*.py.old"))
               if not SKIP_DIRS & set(p.relative_to(repo).parts)}

    problems = []
    for doc in _files(repo, ".md"):
        where = str(doc.relative_to(repo)).replace("\\", "/")
        lines = doc.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_no, line in enumerate(lines, 1):
            for span in _BACKTICKED.findall(line):
                for name in _CAMEL.findall(span):
                    if name not in blob and _looks_like_a_rename(name, symbols):
                        problems.append(f"{where}:{line_no}: `{name}` is in no .py file")
                for path in _PY_PATH.findall(span):
                    # matched by suffix: docs write `features/free_think.py` and
                    # `data/prepare_parallel.py --self-test` alike
                    if not any(f == path or f == path + ".old"
                               or f.endswith("/" + path) or f.endswith("/" + path + ".old")
                               for f in on_disk):
                        problems.append(f"{where}:{line_no}: `{path}` is not a file in the repo")
    return problems


def main():
    p = argparse.ArgumentParser(description="Fail if the docs name code that no longer exists.")
    p.add_argument("--self-test", action="store_true", help="check the checker and exit")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    problems = stale_references()
    for problem in problems:
        print(problem)
    print(f"[check-docs] {len(problems)} stale reference(s)")
    sys.exit(1 if problems else 0)


def _self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "src").mkdir()
        (repo / "src" / "thing.py").write_text(
            "from rank_bm25 import BM25Okapi\n\n\n"
            "class Parentheses:\n"
            "    def stream(self):\n"
            "        return BM25Okapi\n",
            encoding="utf-8")
        (repo / "src" / "gone.py.old").write_text("# retired\n", encoding="utf-8")

        good = ("`Parentheses.stream()` is the model, indexed with `BM25Okapi`, in "
                "`src/thing.py`. Runs on the `GPU`. See `Claude`, `GitHub`, "
                "`SessionStart`, `TrainOnly` -- prose, not symbols. `src/gone.py` "
                "was retired.\n")
        (repo / "ok.md").write_text(good, encoding="utf-8")
        assert stale_references(repo) == [], stale_references(repo)

        # the actual failure this exists for: a class renamed out from under the docs
        (repo / "stale.md").write_text("`ParenthesesGPT.stream()` is KV-cached.\n", encoding="utf-8")
        found = stale_references(repo)
        assert len(found) == 1 and "ParenthesesGPT" in found[0], found

        # ...and the same rename seen from the other end
        (repo / "stale.md").write_text("`FastParentheses` is the model.\n", encoding="utf-8")
        assert len(stale_references(repo)) == 1

        # ...and its sibling: a file moved or renamed out from under the docs
        (repo / "stale.md").write_text("See `src/missing.py` for details.\n", encoding="utf-8")
        found = stale_references(repo)
        assert len(found) == 1 and "missing.py" in found[0], found

        # unbackticked prose is not a reference, however capitalised
        (repo / "stale.md").write_text("ParenthesesGPT was the old name of the class.\n", encoding="utf-8")
        assert stale_references(repo) == []

        # a venv full of other people's code is not this repo's source
        (repo / "stale.md").unlink()
        (repo / "venv").mkdir()
        (repo / "venv" / "vendored.py").write_text("class ParenthesesGPT: pass\n", encoding="utf-8")
        (repo / "vendored.md").write_text("`ParenthesesGPT` lives in the venv.\n", encoding="utf-8")
        found = stale_references(repo)
        assert len(found) == 1 and "ParenthesesGPT" in found[0], found

        # the session log and the training corpus are out of scope
        (repo / "documentation.md").write_text("`ParenthesesGPT` -> `Parentheses`.\n", encoding="utf-8")
        (repo / "raw").mkdir()
        (repo / "raw" / "corpus.md").write_text("`ParenthesesGPT` in training data.\n", encoding="utf-8")
        assert len(stale_references(repo)) == 1  # still just vendored.md

        # this checker's own fixtures name retired symbols; counting itself as
        # source resurrects every one of them (it did, on the first run)
        (repo / "src" / Path(__file__).name).write_text(
            'planted = "ParenthesesGPT"\n', encoding="utf-8")
        found = stale_references(repo)
        assert len(found) == 1 and "ParenthesesGPT" in found[0], found

    print("[self-test] check_docs ok")


if __name__ == "__main__":
    main()
