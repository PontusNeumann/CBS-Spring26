"""
33_single_section_and_logo_anchor.py

Two corrections:

1. Revert to a single section. The user replaced the original section
   break with a page break (already present in para 38 as <w:br
   w:type="page"/>) and explicitly wants ONE section for the whole
   document. The recovery script (32) re-attached an embedded sectPr to
   that paragraph; remove it so only the top-level sectPr at the end of
   the body remains. Single section means single set of margins for the
   whole document (3 cm top, 3 cm bottom, footer inside the 3 cm).

2. Re-anchor the page-2 Logo_CBS so a TOC refresh cannot remove it.
   The trailing CBS logo (the one positioned on the TOC page) lives in
   an empty paragraph immediately AFTER the TOC field's <w:fldChar
   w:fldCharType="end"/>. Word's TOC-update logic can drop or rebuild
   this trailing paragraph, taking the logo with it.

   Move that anchor onto the TOC heading paragraph ("Table of
   contents"), which sits BEFORE the TOC field begin and is never
   touched by the field refresh. Vertical positioning stays
   relativeFrom="page" with the same posOffset, so the logo renders in
   the same physical location on page 2.

   Cover-page anchors (Logo_CBS x2 and CoverImage x2 anchored to
   "Semester: Spring 2026" and "Pages: 15 pages" paragraphs, plus
   Picture 7 in the very first textbox paragraph) already sit OUTSIDE
   the TOC field on stable text-bearing paragraphs, so they need no
   move. They are left untouched.
"""
from __future__ import annotations

import shutil
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "ML_final_exam_paper.docx"
BACKUP = ROOT / "backup" / "ML_final_exam_paper.pre_single_section.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def wp(tag: str) -> str:
    return f"{{{WP_NS}}}{tag}"


def main() -> None:
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    print(f"backup -> {BACKUP}")

    with zipfile.ZipFile(DOCX, "r") as z:
        files = {n: z.read(n) for n in z.namelist()}

    doc = etree.fromstring(files["word/document.xml"])
    body = doc.find(w("body"))

    # ----- 1. Remove embedded sectPrs from any paragraph (single section) -----
    removed_secpr = 0
    for p in body.findall(w("p")):
        ppr = p.find(w("pPr"))
        if ppr is None:
            continue
        sp = ppr.find(w("sectPr"))
        if sp is not None:
            ppr.remove(sp)
            removed_secpr += 1
    print(f"  removed {removed_secpr} embedded sectPr (single section now)")

    # Sanity: remaining sectPr should be the one top-level at end of body
    top_sectpr = body.find(w("sectPr"))
    assert top_sectpr is not None, "no top-level sectPr"

    # Apply 3cm top / 3cm bottom for the single section. Footer offset stays
    # 907 (~1.6cm) inside the 3cm bottom margin. Header offset stays at
    # whatever the top-level sectPr already has.
    pgMar = top_sectpr.find(w("pgMar"))
    pgMar.set(w("top"), "1701")
    pgMar.set(w("bottom"), "1701")
    print(f"  single-section pgMar: top=1701 bottom=1701 (3cm/3cm), "
          f"footer={pgMar.get(w('footer'))} header={pgMar.get(w('header'))}")

    # ----- 2. Re-anchor the page-2 Logo_CBS to the TOC heading paragraph -----
    # Find the TOC heading paragraph (style="TOCHeading", text "Table of contents").
    toc_heading = None
    for p in body.findall(w("p")):
        ppr = p.find(w("pPr"))
        style = None
        if ppr is not None:
            ps = ppr.find(w("pStyle"))
            if ps is not None:
                style = ps.get(w("val"))
        text = "".join(t.text or "" for t in p.iter(w("t"))).strip()
        if style == "TOCHeading" and text == "Table of contents":
            toc_heading = p
            break
    if toc_heading is None:
        raise SystemExit("could not find 'Table of contents' heading paragraph")

    # Find the page-2 Logo_CBS anchor: an anchor whose docPr name is "Logo_CBS"
    # and whose parent paragraph sits AFTER the TOC field end. We detect it as
    # the third Logo_CBS occurrence in document order (the first two are on the
    # cover page in textbox paragraphs with text "Semester..." / "Pages...").
    logo_cbs_anchors = []
    for p in body.iter(w("p")):
        for d in p.iter(w("drawing")):
            for a in d.iter(wp("anchor")):
                docPr = next(a.iter(wp("docPr")), None)
                if docPr is not None and docPr.get("name") == "Logo_CBS":
                    logo_cbs_anchors.append((p, d, a))
    print(f"  found {len(logo_cbs_anchors)} Logo_CBS anchors in body")

    if not logo_cbs_anchors:
        print("  WARN: no Logo_CBS anchors in body; nothing to move")
    else:
        # The page-2 anchor is the one whose parent paragraph has empty text
        # AND is positioned in the doc AFTER the TOC field end. Practically:
        # take the LAST Logo_CBS anchor in document order.
        target_p, drawing_elem, anchor_elem = logo_cbs_anchors[-1]
        parent_text = "".join(t.text or "" for t in target_p.iter(w("t"))).strip()
        if parent_text != "":
            print(f"  WARN: last Logo_CBS parent has text {parent_text[:30]!r}; "
                  f"skipping move to avoid disturbing it")
        else:
            # Find the run that holds this drawing
            run = drawing_elem.getparent()  # <w:r>
            if run is None or run.tag != w("r"):
                print("  WARN: drawing is not directly inside a <w:r>; bail out")
            else:
                # Make sure no second-level nesting: extract the <w:r>
                source_para = run.getparent()
                source_para.remove(run)
                # Append the run to the TOC heading paragraph (so the anchor
                # rides along with that stable paragraph)
                toc_heading.append(run)
                print(f"  moved trailing Logo_CBS anchor onto TOC heading paragraph")
                # Remove the now-empty source paragraph (it had only the run
                # holding the drawing)
                if (
                    source_para.getparent() is body
                    and len(source_para.findall(w("r"))) == 0
                    and not "".join(t.text or "" for t in source_para.iter(w("t"))).strip()
                ):
                    body.remove(source_para)
                    print(f"  removed now-empty source paragraph")

    # ----- Save -----
    files["word/document.xml"] = etree.tostring(
        doc, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, c in files.items():
            zout.writestr(n, c)
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
