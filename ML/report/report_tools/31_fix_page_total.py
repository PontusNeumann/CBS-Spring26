"""
31_fix_page_total.py

Requires: standard library only (zipfile, shutil, pathlib).

Fix: footer page total reads 34 but the last visible page is 32. Cause: the
docx has two sections (section 1 = cover + ToC, section 2 = body + refs +
appendix). Section 2 restarts page numbering at 1 (`pgNumType start=1`),
so PAGE on the last page reads 32, but NUMPAGES counts every physical page
in the document (cover + ToC + section 2 = 34).

Replace `NUMPAGES` with `SECTIONPAGES` in `word/footer3.xml` (the section-2
footer) so the footer reads "Page X of {section-2 page count}". Result:
"Page 32 of 32" on the last page.

The cover/ToC footer (footer1.xml) is left untouched.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx"
BACKUP = ROOT / "report_tools" / "backup" / "ML_final_exam_paper.pre_page_total_fix.docx"
TARGET = "word/footer3.xml"


def main() -> None:
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    print(f"backup -> {BACKUP}")

    # Read all entries
    with zipfile.ZipFile(DOCX, "r") as zin:
        names = zin.namelist()
        contents = {n: zin.read(n) for n in names}

    if TARGET not in contents:
        raise SystemExit(f"{TARGET} not in docx")

    src = contents[TARGET].decode("utf-8")
    if " NUMPAGES " not in src:
        print(f"WARN: ' NUMPAGES ' not in {TARGET}")
    new = src.replace(" NUMPAGES ", " SECTIONPAGES ")
    n_replaced = src.count(" NUMPAGES ") - new.count(" NUMPAGES ")
    contents[TARGET] = new.encode("utf-8")
    print(f"replaced NUMPAGES -> SECTIONPAGES x{n_replaced} in {TARGET}")

    # Write all back, preserving zip layout
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in names:
            zout.writestr(n, contents[n])
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
