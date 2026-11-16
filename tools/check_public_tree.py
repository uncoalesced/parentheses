"""Fail if any tracked file leaks a personal name, private path or address.

Runs as the pre-commit and pre-push hook and by hand:

    python tools/check_public_tree.py            # scan tracked + staged files
    python tools/check_public_tree.py --all      # scan every non-ignored file

Extra blocked names live one per line in private/scrub_names.txt, which is
gitignored so the list itself never ships.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
NAMES_FILE = ROOT / "private" / "scrub_names.txt"

# Built by concatenation so this file does not trip its own scan.
BUILTIN_WORDS = ["Jo" + "el", "Anth" + "ony", "Hea" + "fod", "Lingual" + "_Data"]

PATTERNS = [
    ("windows path", re.compile(r"(?<![A-Za-z])[A-Za-z]:\\{1,2}[A-Za-z_.]")),
    ("file url", re.compile(r"file:///", re.I)),
    ("user home", re.compile(r"[\\/]Users[\\/][^\\/\s]+", re.I)),
    ("personal mail", re.compile(r"[A-Za-z0-9._%+-]+@(gmail|outlook|hotmail|yahoo|proton)\.", re.I)),
]

BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff", ".woff2", ".ttf"}
MAX_BYTES = 8 * 1024 * 1024


def tracked_files(scan_all: bool) -> list[Path]:
    args = ["git", "-C", str(ROOT), "ls-files", "-z", "--cached"]
    if scan_all:
        args += ["--others", "--exclude-standard"]
    out = subprocess.run(args, capture_output=True, check=True).stdout
    return [ROOT / p.decode("utf-8") for p in out.split(b"\0") if p]


def word_patterns() -> list[tuple[str, re.Pattern[str]]]:
    words = list(BUILTIN_WORDS)
    if NAMES_FILE.exists():
        for line in NAMES_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.append(line)
    return [("blocked name", re.compile(r"\b" + re.escape(w) + r"\b", re.I)) for w in words]


def main() -> int:
    scan_all = "--all" in sys.argv[1:]
    patterns = PATTERNS + word_patterns()
    files = tracked_files(scan_all)
    hits = 0
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(("private/", ".claude/")):
            print(f"{rel}: private file is tracked")
            hits += 1
            continue
        if not path.is_file() or path.resolve() == SELF:
            continue
        if path.suffix.lower() in BINARY_SUFFIXES or path.stat().st_size > MAX_BYTES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns:
                if pattern.search(line):
                    print(f"{rel}:{lineno}: {label}: {line.strip()[:120]}")
                    hits += 1
    print(f"check_public_tree: {len(files)} files scanned, {hits} problem(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
