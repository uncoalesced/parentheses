"""
Convert PDFs sitting in data/raw/manual/ into plain text -- for review, not
for trust. data/raw/manual/README.md's rule was "PDF-only source? flag it
back, don't extract" because a bad extraction can silently corrupt text.
This script keeps that spirit: it extracts, but writes every result into a
review queue (`_pdf_extracted/`) alongside a manifest a human (or the sorting
agent) checks before anything gets promoted into raw-sourced/ or
personality-spec/. Nothing here moves, deletes, or trusts the source PDF.

Low-confidence extractions (near-zero chars/page -- almost always a scanned
image PDF with no text layer) are flagged in the manifest rather than
silently accepted or OCR'd. OCR is a real dependency (pytesseract + poppler);
add it if `needs_ocr` starts showing up often -- see manifest column.

Uses the `pdftotext` CLI (poppler-utils) rather than a Python PDF library:
pdfplumber's per-character layout analysis measured ~5 minutes for one
~400-page textbook here, which does not scale to a folder of them.
pdftotext is the same poppler engine other tools in this project already
ship with (see the pdf skill's own quick reference) and is a couple of
orders of magnitude faster for plain-text extraction.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

LOW_CONFIDENCE_CHARS_PER_PAGE = 20  # below this, assume scanned/unreadable


def extract_pdf(path: Path, out_path: Path) -> tuple[int, int]:
    """Writes plain text to out_path via pdftotext. Returns (page_count,
    char_count). Raises subprocess.CalledProcessError on unreadable PDF."""
    subprocess.run(
        ["pdftotext", str(path), str(out_path)],
        check=True, capture_output=True, timeout=120,
    )
    text = out_path.read_text(encoding="utf-8", errors="replace")
    # pdftotext emits a trailing form-feed after every page, including the
    # last one -- an N-page doc has exactly N form-feeds, not N-1 (verified
    # empirically; poppler's own docs don't state this).
    pages = max(1, text.count("\f"))
    return pages, len(text)


def convert_all(manual_dir: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            existing.add(json.loads(line)["source"])

    results = []
    # append per-record, not batched at the end -- these PDFs are large
    # (hundreds of pages) and a slow run getting killed mid-way must not
    # lose track of what it already finished.
    manifest_f = manifest_path.open("a", encoding="utf-8")
    try:
        # top-level only -- data/raw/manual's own subdirectories (raw-sourced/,
        # personality-spec/, LLM instructions/, validation/) are already
        # sorted and out of scope for a re-sort.
        for pdf_path in sorted(manual_dir.glob("*.pdf")):
            record = {"source": pdf_path.name, "converted_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if pdf_path.name in existing:
                record["status"] = "skipped_already_done"
                results.append(record)
                continue
            out_path = out_dir / (pdf_path.stem + ".txt")
            try:
                pages, chars = extract_pdf(pdf_path, out_path)
            except Exception as e:
                record["status"] = "failed"
                record["error"] = str(e)
                out_path.unlink(missing_ok=True)  # don't leave a partial/empty .txt behind
                results.append(record)
                manifest_f.write(json.dumps(record) + "\n")
                manifest_f.flush()
                continue

            chars_per_page = chars / pages if pages else 0
            record.update(
                status="needs_ocr" if chars_per_page < LOW_CONFIDENCE_CHARS_PER_PAGE else "extracted",
                output=str(out_path),
                pages=pages,
                chars=chars,
                chars_per_page=round(chars_per_page, 1),
            )
            results.append(record)
            manifest_f.write(json.dumps(record) + "\n")
            manifest_f.flush()
    finally:
        manifest_f.close()

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manual-dir", default="data/raw/manual")
    p.add_argument("--out-dir", default="data/raw/manual/_pdf_extracted",
                    help="review queue -- not raw-sourced/, not tokenized, not trusted")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    results = convert_all(Path(args.manual_dir), Path(args.out_dir))
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"[done] {len(results)} pdf(s): {counts}")
    print(f"[review] check {args.out_dir}/manifest.jsonl before promoting anything")


def _self_test():
    import tempfile
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manual_dir = tmp / "manual"
        manual_dir.mkdir()

        # a normal text PDF -> should extract cleanly
        good_pdf = manual_dir / "good.pdf"
        c = canvas.Canvas(str(good_pdf), pagesize=letter)
        c.drawString(100, 700, "Hello self-test world")
        c.save()

        # a blank PDF -> should be flagged needs_ocr (near-zero text)
        blank_pdf = manual_dir / "blank.pdf"
        c = canvas.Canvas(str(blank_pdf), pagesize=letter)
        c.showPage()
        c.save()

        # a corrupt "PDF" -> should be flagged failed, not raise
        (manual_dir / "corrupt.pdf").write_bytes(b"not a real pdf")

        out_dir = tmp / "_pdf_extracted"
        results = convert_all(manual_dir, out_dir)
        by_name = {r["source"]: r for r in results}

        assert by_name["good.pdf"]["status"] == "extracted", by_name["good.pdf"]
        assert "Hello self-test world" in (out_dir / "good.txt").read_text()
        assert by_name["blank.pdf"]["status"] == "needs_ocr", by_name["blank.pdf"]
        assert by_name["corrupt.pdf"]["status"] == "failed", by_name["corrupt.pdf"]

        # re-run: already-converted files must be skipped, not re-done
        results2 = convert_all(manual_dir, out_dir)
        by_name2 = {r["source"]: r for r in results2}
        assert by_name2["good.pdf"]["status"] == "skipped_already_done"

        # manifest has exactly 3 lines (the corrupt/failed one included, the
        # skipped re-run not double-appended)
        manifest_lines = (out_dir / "manifest.jsonl").read_text().splitlines()
        assert len(manifest_lines) == 3, len(manifest_lines)

    print("[self-test] prepare_manual_pdfs ok")


if __name__ == "__main__":
    main()
