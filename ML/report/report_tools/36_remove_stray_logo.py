"""
36_remove_stray_logo.py

Requires: lxml  (conda activate py312 && pip install lxml lxml-stubs)
    lxml-stubs is editor-only; runtime needs lxml alone.

Remove the mid-cover Logo_CBS anchor (the one positioned at
H=margin -9525, V=page 4022374). It sits over the cover image's
colored background and is not present in Report.dotm. Its parent run
also produces a small dark/background-color artifact in that area.

The anchor lives inside a textbox paragraph that ALSO carries the
"Pages: 15 pages" cover text; only the one <w:r> that holds the
drawing is removed, the surrounding text runs are preserved.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree  # type: ignore[attr-defined]  # lxml.etree is a C ext, Pylance can't introspect

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx"
BACKUP = ROOT / "report_tools" / "backup" / "ML_final_exam_paper.pre_remove_stray_logo.docx"

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

    target_run = None
    for a in doc.iter(wp("anchor")):
        docPr = next(a.iter(wp("docPr")), None)
        if docPr is None or docPr.get("name") != "Logo_CBS":
            continue
        posV = a.find(wp("positionV"))
        if posV is None:
            continue
        try:
            v_off = int(posV.find(wp("posOffset")).text)
        except (TypeError, ValueError, AttributeError):
            continue
        if 4_000_000 <= v_off <= 4_100_000:
            run = a
            while run is not None and run.tag != w("r"):
                run = run.getparent()
            target_run = run
            print(f"  found mid-cover Logo_CBS run (anchor V_off={v_off})")
            break

    if target_run is None:
        raise SystemExit("mid-cover Logo_CBS anchor not found")

    parent = target_run.getparent()
    parent_text = "".join(t.text or "" for t in parent.iter(w("t")))
    print(f"  parent paragraph text (preserved): {parent_text!r}")

    parent.remove(target_run)
    print("  removed the run holding the stray Logo_CBS drawing")

    files["word/document.xml"] = etree.tostring(
        doc, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, c in files.items():
            zout.writestr(n, c)
    print(f"saved -> {DOCX}")


if __name__ == "__main__":
    main()
