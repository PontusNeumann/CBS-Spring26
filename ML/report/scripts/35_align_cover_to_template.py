"""
35_align_cover_to_template.py

Align the cover-page anchors and the start of the Table-of-Contents to
match Report.dotm:

1. Lock the TOC heading "Table of contents" to start on page 2.
   Add <w:pageBreakBefore/> to its paragraph properties so it can never
   bleed up onto the cover page.

2. Move the cover-page textbox + image (the CoverImage shape, which
   contains all the cover-page text — title, authors, supervisor,
   characters, pages — together with the cover image) to the
   template position. Template: H=0, V=0 (both relativeFrom="page").
   Current: (635, 311) on the modern variant, (-122, 11417) on the
   alternate variant. Set both to (0, 0).

3. Move the bottom-left CBS logo to the template position. Template:
   H=-852, V=9786194 (relativeFrom="margin","page"). Current: H=-342,
   V=9786194. Change H to -852. V already matches.

The mid-page Logo_CBS at (-9525, 4022374) is left untouched — the
template relies on the cover textbox (which contains its own logo
graphic) and that mid-page anchor isn't in the template, but removing
it without a visual confirmation risks losing a logo the user wants.
We'll only move it if the user asks.

Backup written before write.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "ML_final_exam_paper.docx"
BACKUP = ROOT / "backup" / "ML_final_exam_paper.pre_cover_align.docx"

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

    # ----- 1. pageBreakBefore on TOC heading -----
    toc_h = None
    for p in body.findall(w("p")):
        ppr = p.find(w("pPr"))
        if ppr is None:
            continue
        ps = ppr.find(w("pStyle"))
        if ps is None or ps.get(w("val")) != "TOCHeading":
            continue
        text = "".join(t.text or "" for t in p.iter(w("t"))).strip()
        if text == "Table of contents":
            toc_h = p
            break
    if toc_h is None:
        print("  WARN: 'Table of contents' heading not found")
    else:
        ppr = toc_h.find(w("pPr"))
        if ppr.find(w("pageBreakBefore")) is None:
            pbb = etree.SubElement(ppr, w("pageBreakBefore"))
            # Move pageBreakBefore to be right after pStyle
            ppr.remove(pbb)
            ps = ppr.find(w("pStyle"))
            if ps is not None:
                ps.addnext(pbb)
            else:
                ppr.insert(0, pbb)
            print("  added pageBreakBefore on 'Table of contents' heading")
        else:
            print("  pageBreakBefore already present on 'Table of contents'")

    # ----- 2 + 3. Reposition cover anchors -----
    n_cover = 0
    n_logo = 0
    for a in doc.iter(wp("anchor")):
        docPr = next(a.iter(wp("docPr")), None)
        if docPr is None:
            continue
        name = docPr.get("name")
        posH = a.find(wp("positionH"))
        posV = a.find(wp("positionV"))
        if posH is None or posV is None:
            continue
        offH = posH.find(wp("posOffset"))
        offV = posV.find(wp("posOffset"))
        if offH is None or offV is None:
            continue

        if name == "CoverImage":
            old = (offH.text, offV.text)
            offH.text = "0"
            offV.text = "0"
            print(f"  CoverImage (id={docPr.get('id')}): {old} -> ('0', '0')")
            n_cover += 1
        elif name == "Logo_CBS":
            # Bottom-left logo identified by V offset 9786194 (page-bottom area)
            v_rel = posV.get("relativeFrom")
            try:
                v_off = int(offV.text)
            except (TypeError, ValueError):
                continue
            if v_rel == "page" and 9_700_000 <= v_off <= 9_900_000:
                old = (offH.text, offV.text)
                offH.text = "-852"
                # V left as-is (already correct)
                print(f"  Logo_CBS bottom-left (id={docPr.get('id')}): "
                      f"H {old[0]} -> -852  (V {offV.text} unchanged)")
                n_logo += 1
            # Other Logo_CBS anchors left untouched (mid-cover, page-2)

    print(f"  total CoverImage anchors moved: {n_cover}")
    print(f"  total bottom-left Logo_CBS moved: {n_logo}")

    files["word/document.xml"] = etree.tostring(
        doc, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, c in files.items():
            zout.writestr(n, c)
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
