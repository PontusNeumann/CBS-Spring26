"""Insert a Word-field-driven Table of Contents page after the cover.

Mirrors the ML report pattern: a TOCHeading paragraph followed by a TOC field
that Word populates from Heading 1 / Heading 2 / Heading 3 styles. The field is
marked dirty so Word refreshes it on first open.

The cover SDT block is never touched.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML = "http://www.w3.org/XML/1998/namespace"
DOC = Path(__file__).resolve().parents[1] / (
    "KAN-CDSCO1002U_161989_160363_185912_160714_OneStream_Intent_Routing.docx"
)


def make_toc_heading() -> etree._Element:
    """Paragraph styled TOCHeading with page-break-before, text 'Table of contents'."""
    p = etree.Element(f"{{{W}}}p")
    pPr = etree.SubElement(p, f"{{{W}}}pPr")
    pStyle = etree.SubElement(pPr, f"{{{W}}}pStyle")
    pStyle.set(f"{{{W}}}val", "TOCHeading")
    etree.SubElement(pPr, f"{{{W}}}pageBreakBefore")
    r = etree.SubElement(p, f"{{{W}}}r")
    t = etree.SubElement(r, f"{{{W}}}t")
    t.text = "Table of contents"
    return p


def make_toc_field() -> etree._Element:
    """Paragraph holding the TOC field. Word will populate it from heading styles."""
    p = etree.Element(f"{{{W}}}p")
    pPr = etree.SubElement(p, f"{{{W}}}pPr")
    pStyle = etree.SubElement(pPr, f"{{{W}}}pStyle")
    pStyle.set(f"{{{W}}}val", "TOC1")

    # Run 1: <w:fldChar w:fldCharType="begin" w:dirty="true"/>
    r1 = etree.SubElement(p, f"{{{W}}}r")
    fc1 = etree.SubElement(r1, f"{{{W}}}fldChar")
    fc1.set(f"{{{W}}}fldCharType", "begin")
    fc1.set(f"{{{W}}}dirty", "true")

    # Run 2: <w:instrText> TOC \o "1-3" \h \z \u </w:instrText>
    r2 = etree.SubElement(p, f"{{{W}}}r")
    it = etree.SubElement(r2, f"{{{W}}}instrText")
    it.set(f"{{{XML}}}space", "preserve")
    it.text = ' TOC \\o "1-3" \\h \\z \\u '

    # Run 3: separator
    r3 = etree.SubElement(p, f"{{{W}}}r")
    fc2 = etree.SubElement(r3, f"{{{W}}}fldChar")
    fc2.set(f"{{{W}}}fldCharType", "separate")

    # Run 4: placeholder text (visible until Word refreshes the field)
    r4 = etree.SubElement(p, f"{{{W}}}r")
    t = etree.SubElement(r4, f"{{{W}}}t")
    t.text = (
        'Right-click here and choose "Update Field" to populate this Table of Contents.'
    )

    # Run 5: <w:fldChar w:fldCharType="end"/>
    r5 = etree.SubElement(p, f"{{{W}}}r")
    fc3 = etree.SubElement(r5, f"{{{W}}}fldChar")
    fc3.set(f"{{{W}}}fldCharType", "end")

    return p


def insert_toc(body) -> None:
    """Insert ToC heading + field after cover SDT (and any cover bookmarks),
    immediately before the first body paragraph.
    """
    children = list(body)
    # Locate the index right after the SDT cover and any directly trailing bookmarks
    insert_at = None
    seen_sdt = False
    for i, c in enumerate(children):
        local = etree.QName(c).localname
        if local == "sdt":
            seen_sdt = True
            insert_at = i + 1
            continue
        if seen_sdt and local in {"bookmarkStart", "bookmarkEnd"}:
            insert_at = i + 1
            continue
        if seen_sdt:
            break
    if insert_at is None:
        raise RuntimeError("Could not locate SDT cover in body")

    # Guard: do not double-insert a TOCHeading
    for c in children:
        if etree.QName(c).localname == "p":
            pStyle = c.find(f"{{{W}}}pPr/{{{W}}}pStyle")
            if pStyle is not None and pStyle.get(f"{{{W}}}val") == "TOCHeading":
                print("ToC already present; nothing to do.")
                return

    body.insert(insert_at, make_toc_heading())
    body.insert(insert_at + 1, make_toc_field())
    print(f"Inserted ToC heading + field at body index {insert_at}.")


def patch(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_toc")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup written -> {backup.name}")

    with zipfile.ZipFile(path) as z:
        members = {name: z.read(name) for name in z.namelist()}

    root = etree.fromstring(members["word/document.xml"])
    body = root.find(f"{{{W}}}body")
    insert_toc(body)
    members["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    tmp.replace(path)
    print(f"patched {path.name}.")


if __name__ == "__main__":
    patch(DOC)
