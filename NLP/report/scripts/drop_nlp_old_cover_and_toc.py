"""One-off cleanup: drop NLP's original cover SDT and ToC paragraphs that now
sit between the imported ML pages (children 0..38) and the first Heading 1
('Introduction'). Leaves the imported ML cover + ToC at the top and the
existing NLP body content untouched.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NLP = Path(__file__).resolve().parents[1] / (
    "KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Intent_Routing.docx"
)


def patch() -> None:
    backup = NLP.with_suffix(NLP.suffix + ".pre_drop_nlp_old_cover")
    if not backup.exists():
        shutil.copy2(NLP, backup)
        print(f"backup written -> {backup.name}")

    with zipfile.ZipFile(NLP) as z:
        members = {n: z.read(n) for n in z.namelist()}
    root = etree.fromstring(members["word/document.xml"])
    body = root.find(f"{{{W}}}body")

    # Find boundary: locate the second <w:sdt> (NLP's original cover).
    sdt_count = 0
    second_sdt_idx = None
    for i, c in enumerate(body):
        if etree.QName(c).localname == "sdt":
            sdt_count += 1
            if sdt_count == 2:
                second_sdt_idx = i
                break
    if second_sdt_idx is None:
        print("No second SDT found; nothing to drop.")
        return

    # Find the first Heading 1 ('Introduction') after the second SDT
    intro_idx = None
    for i in range(second_sdt_idx, len(body)):
        c = body[i]
        if etree.QName(c).localname != "p":
            continue
        pStyle = c.find(f"{{{W}}}pPr/{{{W}}}pStyle")
        if pStyle is not None and pStyle.get(f"{{{W}}}val") == "Heading1":
            intro_idx = i
            break
    if intro_idx is None:
        raise RuntimeError("Could not locate Introduction heading after second SDT")

    print(f"dropping body children [{second_sdt_idx}:{intro_idx}] (count={intro_idx - second_sdt_idx})")
    for c in list(body[second_sdt_idx:intro_idx]):
        body.remove(c)

    members["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    tmp = NLP.with_suffix(NLP.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    tmp.replace(NLP)
    print(f"patched {NLP.name}.")


if __name__ == "__main__":
    patch()
