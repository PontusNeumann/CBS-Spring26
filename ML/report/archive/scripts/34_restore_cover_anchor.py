"""
34_restore_cover_anchor.py

Restore the first-page CoverImage anchor offsets to the pre_page_total_fix
values so the cover image bleeds slightly off the page top-left as it did
originally. The Word save round-trip during the user's manual edits flattened
the offsets:

  pre_page_total_fix (correct):
    positionH posOffset = -827600  (~ -2.3 cm, bleed left)
    positionV posOffset = -827405  (~ -2.3 cm, bleed up)

  current (pre_recovery, wrong):
    positionH posOffset = 635      (essentially 0, image clipped at page edge)
    positionV posOffset = 311      (essentially 0, image clipped at page edge)

Only the first CoverImage anchor is changed (the mc:Choice block, anchorId
"6D488B48"). The fallback mc:Fallback CoverImage anchor and all Logo_CBS
anchors already match the pre_page_total_fix values and are left alone.

Single-section structure of pre_recovery is preserved (no sectPr changes).
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "ML_final_exam_paper.docx"
BACKUP = ROOT / "backup" / "ML_final_exam_paper.pre_cover_anchor.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
WP14_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing"


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def wp(tag: str) -> str:
    return f"{{{WP_NS}}}{tag}"


TARGET_ANCHOR_ID = "6D488B48"
NEW_H = "-827600"
NEW_V = "-827405"


def main() -> None:
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DOCX, BACKUP)
    print(f"backup -> {BACKUP}")

    with zipfile.ZipFile(DOCX, "r") as z:
        files = {n: z.read(n) for n in z.namelist()}

    doc = etree.fromstring(files["word/document.xml"])

    # Find the anchor by wp14:anchorId
    target = None
    for a in doc.iter(wp("anchor")):
        anchor_id = a.get(f"{{{WP14_NS}}}anchorId")
        if anchor_id == TARGET_ANCHOR_ID:
            target = a
            break

    if target is None:
        raise SystemExit(f"anchor with wp14:anchorId={TARGET_ANCHOR_ID} not found")

    # Sanity check: docPr name == 'CoverImage'
    docPr = next(target.iter(wp("docPr")), None)
    name = docPr.get("name") if docPr is not None else None
    print(f"  found anchor docPr name={name!r}")
    if name != "CoverImage":
        print(f"  WARN: expected name 'CoverImage', got {name!r}")

    # Update positionH/posOffset and positionV/posOffset
    posH = target.find(wp("positionH"))
    posV = target.find(wp("positionV"))
    posH_off = posH.find(wp("posOffset"))
    posV_off = posV.find(wp("posOffset"))
    old_h, old_v = posH_off.text, posV_off.text
    posH_off.text = NEW_H
    posV_off.text = NEW_V
    print(f"  positionH posOffset: {old_h} -> {posH_off.text}")
    print(f"  positionV posOffset: {old_v} -> {posV_off.text}")

    files["word/document.xml"] = etree.tostring(
        doc, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, c in files.items():
            zout.writestr(n, c)
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
