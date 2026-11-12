"""Branded terminal UI for parentheses chat (stdlib only, ANSI truecolor).

Brand: Graphite ground, Sandy Brown accent (#F4A460), rounded shapes.
Respects NO_COLOR, non-TTY output, and an explicit plain mode.
Demo: python scripts/term_ui.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Dict, Optional

ACCENT = (244, 164, 96)      # Sandy Brown
ACCENT_HI = (255, 214, 170)
TEXT = (230, 230, 228)
DIM = (128, 128, 124)
GRAPHITE = (78, 78, 82)

_plain = bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def set_plain(flag: bool) -> None:
    global _plain
    _plain = _plain or flag


def _enable_vt() -> None:
    if os.name == "nt":
        os.system("")  # turns on VT processing in Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


_enable_vt()


def fg(rgb, s: str, bold: bool = False) -> str:
    if _plain:
        return s
    b = "\x1b[1m" if bold else ""
    return f"{b}\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{s}\x1b[0m"


def _lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# 3-row rounded glyphs, 3 cols wide
_GLYPHS = {
    "p": ("╭─╮", "├─╯", "╵  "),
    "a": ("╭─╮", "├─┤", "╵ ╵"),
    "r": ("╭─╮", "├┬╯", "╵╰╴"),
    "e": ("╭─╴", "├─ ", "╰─╴"),
    "n": ("╭╮╷", "│││", "╵╰╯"),
    "t": ("╶┬╴", " │ ", " ╵ "),
    "h": ("╷ ╷", "├─┤", "╵ ╵"),
    "s": ("╭─╴", "╰─╮", "╶─╯"),
}

_LOGO = (
    " ╭╴   ╶╮ ",
    "│       │",
    " ╰╴   ╶╯ ",
)


def _wordmark(word: str = "parentheses"):
    rows = ["", "", ""]
    for ch in word:
        g = _GLYPHS[ch]
        for i in range(3):
            rows[i] += g[i] + " "
    return [r.rstrip() for r in rows]


def _gradient(line: str) -> str:
    if _plain:
        return line
    n = max(len(line) - 1, 1)
    return "".join(fg(_lerp(ACCENT, ACCENT_HI, i / n), c, bold=True) if c != " " else c
                   for i, c in enumerate(line))


def _box(lines, width: int) -> str:
    out = [fg(GRAPHITE, "╭" + "─" * (width + 2) + "╮")]
    for raw, styled in lines:
        pad = " " * (width - len(raw))
        out.append(fg(GRAPHITE, "│ ") + styled + pad + fg(GRAPHITE, " │"))
    out.append(fg(GRAPHITE, "╰" + "─" * (width + 2) + "╯"))
    return "\n".join(out)


def banner(info: Optional[Dict[str, str]] = None, version: str = "parentheses-1.0") -> None:
    wm = _wordmark()
    print()
    for logo_row, wm_row in zip(_LOGO, wm):
        print("  " + fg(ACCENT, logo_row, bold=True) + "   " + _gradient(wm_row))
    sub = f"{version}  ·  240M  ·  16k context  ·  by uncoalesced"
    print("  " + " " * (len(_LOGO[0]) + 3) + fg(DIM, sub))
    print()
    rows = []
    for k, v in (info or {}).items():
        raw = f"{k:<11}{v}"
        rows.append((raw, fg(DIM, f"{k:<11}") + fg(TEXT, v)))
    cmds = "/exit  /clear  /think"
    rows.append((f"{'commands':<11}{cmds}", fg(DIM, f"{'commands':<11}") + fg(ACCENT, cmds)))
    width = max(len(r[0]) for r in rows)
    print("  " + _box(rows, width).replace("\n", "\n  "))
    print()


def user_prompt() -> str:
    return input(fg(ACCENT, "( you ) ", bold=True) + fg(DIM, "› "))


def model_header(name: str = "parentheses") -> None:
    print("\n" + fg(ACCENT_HI, f"( {name} ) ", bold=True) + fg(DIM, "›"))


def thinking(text: str) -> None:
    print(fg(DIM, "  ◌ thinking"))
    for line in text.strip().split("\n"):
        print(fg(DIM, "  │ " + line))


def response(text: str) -> None:
    for line in text.strip().split("\n"):
        print("  " + fg(TEXT, line))


def stats(tokens: int, tps: float, secs: float) -> None:
    print(fg(DIM, f"  {tokens} tok · {tps:.1f} tok/s · {secs:.2f}s") + "\n")


def note(text: str) -> None:
    print(fg(DIM, "  " + text))


class Spinner:
    """`( ●   )` bouncing dot while something loads."""

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        frames = ["(●    )", "( ●   )", "(  ●  )", "(   ● )", "(    ●)", "(   ● )", "(  ●  )", "( ●   )"]
        i = 0
        while not self._stop.is_set():
            sys.stdout.write("\r  " + fg(ACCENT, frames[i % len(frames)]) + " " + fg(DIM, self.label))
            sys.stdout.flush()
            i += 1
            time.sleep(0.09)

    def __enter__(self):
        if _plain:
            print(f"  {self.label}")
        else:
            self._t.start()
        return self

    def __exit__(self, *exc):
        if not _plain:
            self._stop.set()
            self._t.join()
            sys.stdout.write("\r\x1b[2K  " + fg(ACCENT, "(  ✓  )") + " " + fg(DIM, self.label) + "\n")
        return False


if __name__ == "__main__":
    banner({"checkpoint": "step_14200_golden.pt", "device": "cuda"})
    with Spinner("loading weights"):
        time.sleep(1.0)
    model_header()
    thinking("user greets; respond briefly")
    response("Hello! I'm parentheses, made by uncoalesced.")
    stats(14, 41.2, 0.34)
