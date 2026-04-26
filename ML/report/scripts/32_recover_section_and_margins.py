"""
32_recover_section_and_margins.py

Recovery + margin fix:
  - During the Word save round-trip after the user's manual edits, the
    embedded `<w:sectPr>` that defined Section 1 (cover + ToC) on the
    empty paragraph immediately before "Abstract" was lost; the document
    collapsed into a single section using Section 2's larger top margin.
    That visibly pushed cover-page content down.
  - Several footer parts (footer1/2/3/4.xml) and the related parts list
    were also corrupted (footer2/3/4 became 0 bytes; footer1 was
    truncated). Word silently recovers, but the PAGE/NUMPAGES fields
    are gone.
  - Section 2's bottom margin is 4 cm, exceeding the 3 cm CBS minimum
    when the footer is already inside that 3 cm zone.

Fix (do NOT touch Section 1 cover-page settings):
  1. Restore footer1/2/3/4.xml, [Content_Types].xml, and
     word/_rels/document.xml.rels from the pre_page_total_fix backup.
  2. Re-apply NUMPAGES -> SECTIONPAGES in footer3.xml so the body
     footer reads "Page X of 32" not "Page X of 34".
  3. Re-attach the Section 1 sectPr to the empty paragraph immediately
     before "Abstract", using the exact element copied out of the backup
     (so the cover-page top/header/footer geometry is identical to the
     pre-edit state).
  4. Change the top-level Section 2 sectPr's pgMar.bottom from 2268
     (~4 cm) to 1701 (~3 cm). Section 2 footer offset stays 907
     (~1.6 cm) so the footer sits inside the new 3 cm margin. Section 1
     remains untouched.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "ML_final_exam_paper.docx"
BACKUP = ROOT / "backup" / "ML_final_exam_paper.pre_recovery.docx"
SOURCE = ROOT / "backup" / "ML_final_exam_paper.pre_page_total_fix.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def main() -> None:
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    print(f"backup -> {BACKUP}")

    # Read both archives in full
    with zipfile.ZipFile(SOURCE, "r") as zs:
        src_files = {n: zs.read(n) for n in zs.namelist()}
    with zipfile.ZipFile(DOCX, "r") as zc:
        cur_files = {n: zc.read(n) for n in zc.namelist()}

    # 1. Restore parts that the broken save truncated
    parts_to_restore = [
        "word/footer1.xml",
        "word/footer2.xml",
        "word/footer3.xml",
        "word/footer4.xml",
        "[Content_Types].xml",
        "word/_rels/document.xml.rels",
    ]
    for p in parts_to_restore:
        if p in src_files:
            cur_files[p] = src_files[p]
            print(f"  restored {p} ({len(src_files[p])} bytes)")

    # 2. Re-apply NUMPAGES -> SECTIONPAGES in footer3.xml
    f3 = cur_files.get("word/footer3.xml", b"")
    if f3:
        new = f3.decode("utf-8").replace(" NUMPAGES ", " SECTIONPAGES ")
        cur_files["word/footer3.xml"] = new.encode("utf-8")
        print("  re-applied NUMPAGES -> SECTIONPAGES in footer3.xml")

    # 3. Patch document.xml: re-attach Section 1 sectPr + adjust Section 2 bottom margin
    src_doc = etree.fromstring(src_files["word/document.xml"])
    cur_doc = etree.fromstring(cur_files["word/document.xml"])

    # Locate the Section 1 sectPr element in source: it is the embedded sectPr
    # inside a <w:p>/<w:pPr> earlier in the document (the only embedded one).
    src_section1_sectpr = None
    src_body = src_doc.find(w("body"))
    for p in src_body.findall(w("p")):
        ppr = p.find(w("pPr"))
        if ppr is not None:
            sp = ppr.find(w("sectPr"))
            if sp is not None:
                src_section1_sectpr = sp
                break
    if src_section1_sectpr is None:
        raise SystemExit("could not find embedded section 1 sectPr in source backup")
    print(f"  found Section 1 sectPr in source ({len(list(src_section1_sectpr))} children)")

    # Locate the empty paragraph immediately before the "Abstract" Heading 1
    # in the CURRENT document.
    cur_body = cur_doc.find(w("body"))
    target_para = None
    abstract_idx = None
    paragraphs = list(cur_body.findall(w("p")))
    for i, p in enumerate(paragraphs):
        ppr = p.find(w("pPr"))
        style = None
        if ppr is not None:
            ps = ppr.find(w("pStyle"))
            if ps is not None:
                style = ps.get(w("val"))
        text = "".join(t.text or "" for t in p.iter(w("t"))).strip()
        if style == "Heading1" and text == "Abstract":
            abstract_idx = i
            # walk backward to find an empty paragraph (the one that should
            # carry the section break)
            for j in range(i - 1, -1, -1):
                pj = paragraphs[j]
                tj = "".join(t.text or "" for t in pj.iter(w("t"))).strip()
                if tj == "":
                    target_para = pj
                    break
            break
    if target_para is None:
        raise SystemExit("could not find empty paragraph before Abstract heading")

    # Verify it does not already carry a sectPr
    target_ppr = target_para.find(w("pPr"))
    if target_ppr is None:
        target_ppr = etree.SubElement(target_para, w("pPr"))
        # Move pPr to the start of the paragraph
        target_para.remove(target_ppr)
        target_para.insert(0, target_ppr)
    if target_ppr.find(w("sectPr")) is not None:
        print("  WARN: target paragraph already has a sectPr, leaving as-is")
    else:
        # Append a deep copy of the source Section 1 sectPr onto the target
        # paragraph's pPr so geometry is identical to the pre-edit state.
        clone = deepcopy(src_section1_sectpr)
        target_ppr.append(clone)
        print("  re-attached Section 1 sectPr to empty paragraph before Abstract")

    # Adjust Section 2 (top-level) pgMar.bottom: 2268 -> 1701 (~3 cm)
    top_level_sectpr = cur_body.find(w("sectPr"))
    if top_level_sectpr is None:
        raise SystemExit("no top-level sectPr (Section 2) in current document")
    pgMar = top_level_sectpr.find(w("pgMar"))
    if pgMar is None:
        raise SystemExit("Section 2 sectPr has no pgMar")
    old_bottom = pgMar.get(w("bottom"))
    pgMar.set(w("bottom"), "1701")
    print(f"  Section 2 pgMar.bottom: {old_bottom} -> 1701 (~3 cm, footer remains inside)")

    cur_files["word/document.xml"] = etree.tostring(
        cur_doc, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    # 4. Write everything back, preserving order from the source where possible
    name_order = list(cur_files.keys())
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n in name_order:
            zout.writestr(n, cur_files[n])
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
