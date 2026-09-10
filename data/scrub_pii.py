r"""
Redact incidentally-scraped contact details out of CommonCrawl-derived
sources. Report-first; writes copies, never in place.

Scope is deliberately narrow (project-gaps-and-fixes.md item 3): this runs
against web-scraped text only -- proof-pile-2-open-web-math today, anything
CommonCrawl-shaped later. Gutenberg, Wikipedia, OpenStax and
personality-spec/ are not scrapes of live forum pages and carry no
third-party-contact-detail risk, so this refuses to run against them rather
than quietly rewriting a licensed corpus.

This is about strangers' phone numbers ending up in a training corpus. It is
not a content filter: it removes an email address, not an opinion. Nothing
here touches what the model is allowed to say.

    python3 data/scrub_pii.py                                   # report only
    python3 data/scrub_pii.py --apply --out-dir data/raw/manual/raw-sourced/proof-pile-2-open-web-math-scrubbed
    python3 data/scrub_pii.py --self-test

The report prints counts and never prints a matched value -- printing the PII
you just found is not a scrub.
"""

import argparse
import os
import re

# Only paths naming one of these may be scrubbed. An unlisted path is a
# refusal, not a warning: the failure mode this guards against is a scrubber
# pointed at Gutenberg rewriting 11.7GB of licensed public-domain text.
ALLOWED = ("open-web-math", "proof-pile", "commoncrawl", "common-crawl")

PATTERNS = {
    # local@domain.tld -- the tld guard keeps LaTeX and code fragments out
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # separators are required. open-web-math is wall-to-wall digits, and a
    # pattern that accepts a bare 10-digit run redacts arithmetic instead.
    # The trailing guard excludes digits and hyphens but NOT ".", or a number
    # ending a sentence ("call (555) 123-4567.") never matches at all.
    "PHONE": re.compile(r"(?<![\d.-])(?:\+?\d{1,2}[ .-])?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])"
                        r"\d{3}[ .-]\d{4}(?![\d-])"),
    "SSN": re.compile(r"(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])"),
}

TEXT_EXT = (".txt", ".md")


def redact(text: str) -> tuple[str, dict]:
    """-> (text with matches replaced by [EMAIL]/[PHONE]/[SSN], counts per kind).

    A PHONE match separated only by spaces is left alone. open-web-math is
    full of numeric tables ("1234 567 8901 234 ..." from a transfer function),
    and every row of one matched the pattern on the first real run -- 1,043
    hits, essentially all of them matrices. A real phone number in prose
    almost always carries a "(", "-" or "." somewhere; a table row does not.
    Cost: "+1 555 123 4567" is missed. This is a report a human reads, and a
    thousand false positives make it unreadable.
    """
    counts = {}
    for kind, pattern in PATTERNS.items():
        n = 0

        def repl(m, kind=kind):
            nonlocal n
            if kind == "PHONE" and not any(c in m.group(0) for c in "().-"):
                return m.group(0)
            n += 1
            return f"[{kind}]"

        text = pattern.sub(repl, text)
        if n:
            counts[kind] = n
    return text, counts


def check_scope(path: str):
    low = path.replace(os.sep, "/").lower()
    if not any(a in low for a in ALLOWED):
        raise SystemExit(
            f"[refused] {path} is not a CommonCrawl-derived source. This scrubber is scoped to "
            f"{', '.join(ALLOWED)} on purpose -- Gutenberg, Wikipedia, OpenStax and "
            f"personality-spec/ are out of scope by design (project-gaps-and-fixes.md item 3). "
            f"Add the source to ALLOWED deliberately if that is really what you mean.")


def scan(src: str, out_dir: str | None):
    """Walk src, redact every text file, and optionally write the result under out_dir.

    -> (per-file [(relpath, counts)], totals). With out_dir=None nothing is
    written: the report is the deliverable, and applying it is a second run.
    """
    check_scope(src)
    per_file, totals = [], {}
    for root, _, files in os.walk(src):
        for name in sorted(files):
            if not name.endswith(TEXT_EXT):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, src).replace(os.sep, "/")
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            clean, counts = redact(text)
            if counts:
                per_file.append((rel, counts))
                for k, v in counts.items():
                    totals[k] = totals.get(k, 0) + v
            if out_dir:
                dst = os.path.join(out_dir, rel)
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                with open(dst, "w", encoding="utf-8", newline="\n") as f:
                    f.write(clean)
    return per_file, totals


def render(src: str, per_file, totals: dict, out_dir: str | None) -> str:
    out = [f"PII scrub report for {src}",
           f"pattern hits: " + (", ".join(f"{k} {v:,}" for k, v in sorted(totals.items()))
                                or "none"),
           f"files with hits: {len(per_file):,}",
           "",
           "Matched values are deliberately not printed. Counts are an upper bound, not a",
           "finding: on the real corpus most SSN-shaped hits were textbook examples and DOI",
           "suffixes rather than anyone's actual number. Check a sample by hand (digits",
           "masked) before treating a count as PII found.",
           ""]
    for rel, counts in per_file[:40]:
        out.append(f"  {rel:<40} " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if len(per_file) > 40:
        out.append(f"  ... and {len(per_file) - 40:,} more files")
    out.append("")
    out.append(f"[written] redacted copies under {out_dir}" if out_dir
               else "[report only] nothing was written; rerun with --apply --out-dir to redact")
    return "\n".join(out).rstrip() + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/raw/manual/raw-sourced/proof-pile-2-open-web-math",
                   help="CommonCrawl-derived source directory (see ALLOWED)")
    p.add_argument("--apply", action="store_true", help="write redacted copies to --out-dir")
    p.add_argument("--out-dir", default=None,
                   help="destination for redacted copies; never the source directory")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    out_dir = None
    if args.apply:
        if not args.out_dir:
            raise SystemExit("--apply needs --out-dir (this never rewrites the source in place)")
        if os.path.abspath(args.out_dir) == os.path.abspath(args.src):
            raise SystemExit("--out-dir must differ from --src")
        out_dir = args.out_dir
    per_file, totals = scan(args.src, out_dir)
    print(render(args.src, per_file, totals, out_dir))


def _self_test():
    import tempfile

    text = ("mail me at joe.smith+x@example.co.uk or call (555) 123-4567.\n"
            "alt: +1 555.123.4567, ssn 123-45-6789.\n")
    clean, counts = redact(text)
    assert counts == {"EMAIL": 1, "PHONE": 2, "SSN": 1}, counts
    assert "@" not in clean and "123-45-6789" not in clean
    assert "[EMAIL]" in clean and clean.count("[PHONE]") == 2

    # math text must survive untouched -- these are the false positives that
    # would quietly corrupt open-web-math. The last line is a transfer-function
    # table, the shape that produced ~1,000 bogus PHONE hits on the real corpus.
    math = ("pi = 3.14159265358979 and 2^32 = 4294967296; the interval [0.5, 12.25],\n"
            "equation 5x - 12 = 3, dated 2024-01-02, ratio 1234-5678-9012-3456\n"
            "1234 567 8901 234 5678 901 2345 678 9012 345 6789 012 3456 789 0123\n")
    clean, counts = redact(math)
    assert counts == {}, counts
    assert clean == math

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "open-web-math")
        os.makedirs(src)
        with open(os.path.join(src, "shard_00.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(src, "shard_01.txt"), "w", encoding="utf-8", newline="\n") as f:
            f.write(math)

        per_file, totals = scan(src, None)
        assert totals == {"EMAIL": 1, "PHONE": 2, "SSN": 1}, totals
        assert len(per_file) == 1 and per_file[0][0] == "shard_00.txt"
        assert os.listdir(src) == ["shard_00.txt", "shard_01.txt"], "report mode wrote something"
        report = render(src, per_file, totals, None)
        assert "example.co.uk" not in report and "123-45-6789" not in report, "report leaked PII"
        assert "report only" in report

        out = os.path.join(tmp, "out")
        scan(src, out)
        got = open(os.path.join(out, "shard_00.txt"), encoding="utf-8").read()
        assert "[EMAIL]" in got and "@" not in got
        assert open(os.path.join(out, "shard_01.txt"), encoding="utf-8").read() == math

        # out-of-scope sources are refused, not warned about
        gut = os.path.join(tmp, "project-gutenberg")
        os.makedirs(gut)
        try:
            scan(gut, None)
            raise AssertionError("an out-of-scope source must be refused")
        except SystemExit as e:
            assert "not a CommonCrawl-derived source" in str(e)

    print("[self-test] scrub_pii ok")


if __name__ == "__main__":
    main()
