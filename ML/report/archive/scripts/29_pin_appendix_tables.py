"""
29_pin_appendix_tables.py

Fix the orphaned-title problem on the last two appendix tables (A.6 and A.7).
Both are long (~55 and ~34 rows) and currently leave the title paragraph
stranded on its own page above an empty page, then the table starts on the
following page.

The fix:
  - Title paragraph: keepNext=True (already set) AND keepLines=True so the
    title line never splits across pages internally.
  - First row of each table: cantSplit=True so Word treats title + first
    row as inseparable; if they don't fit on the current page they get
    pushed together to the next page. Subsequent rows are left splittable
    (the user wants the long table itself to wrap across pages).

Run:
    python report/scripts/29_pin_appendix_tables.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "ML_final_exam_paper.docx"
BACKUP = ROOT / "backup" / "ML_final_exam_paper.pre_pin_tables.docx"


def ensure_child(parent, tag: str):
    el = parent.find(qn(tag))
    if el is None:
        el = OxmlElement(tag)
        parent.append(el)
    return el


def pin_title(p_elem) -> None:
    pPr = p_elem.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr")
        p_elem.insert(0, pPr)
    ensure_child(pPr, "w:keepNext")
    ensure_child(pPr, "w:keepLines")


def pin_first_row(tbl_elem) -> None:
    rows = tbl_elem.findall(qn("w:tr"))
    if not rows:
        return
    first = rows[0]
    trPr = first.find(qn("w:trPr"))
    if trPr is None:
        trPr = OxmlElement("w:trPr")
        first.insert(0, trPr)
    ensure_child(trPr, "w:cantSplit")


def main() -> None:
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    print(f"backup -> {BACKUP}")

    doc = Document(DOCX)
    elems = list(doc.element.body.iterchildren())

    targets: list[tuple[int, int, str]] = []  # (title_idx, table_idx, label)
    for i, e in enumerate(elems):
        if e.tag == qn("w:p"):
            text = "".join(t.text or "" for t in e.iter(qn("w:t")))
            stripped = text.strip()
            if stripped.startswith("Table A.6.") or stripped.startswith("Table A.7."):
                # Find the next w:tbl element after this paragraph
                j = i + 1
                while j < len(elems) and elems[j].tag != qn("w:tbl"):
                    if elems[j].tag == qn("w:p"):
                        next_text = "".join(t.text or "" for t in elems[j].iter(qn("w:t"))).strip()
                        if next_text:
                            break  # non-empty paragraph between title and table
                    j += 1
                if j < len(elems) and elems[j].tag == qn("w:tbl"):
                    targets.append((i, j, stripped[:12]))

    if len(targets) != 2:
        print(f"WARN: expected 2 targets (A.6, A.7), found {len(targets)}: {targets}")

    for title_idx, table_idx, label in targets:
        pin_title(elems[title_idx])
        pin_first_row(elems[table_idx])
        print(f"  pinned: {label} title (idx {title_idx}) -> table (idx {table_idx})")

    doc.save(DOCX)
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
