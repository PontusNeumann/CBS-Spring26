"""Apply CBS formal-requirements fixes to ML_final_exam_paper.docx.

- Top margin 2.30 cm -> 3.00 cm (1304 -> 1701 twips)
- Normal style body font 9.5 pt -> 11 pt (sz 19 -> 22 half-pt)
- TOC2 / TOC3 9 pt -> 11 pt (sz 18 -> 22)
- Raise any run-level w:sz / w:szCs below 22 half-pt to 22 in document.xml
- Insert four Cover-Text lines on the front page with placeholders for
  paper type, supervisor, character count, page count
"""

from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

SRC = Path("report/ML_final_exam_paper.docx")
DST = Path("report/ML_final_exam_paper.docx")
TMP = Path("report/ML_final_exam_paper.__new__.docx")


def fix_styles(styles_xml: str) -> str:
    # Normal size 19 -> 22
    styles_xml = re.sub(
        r'(<w:style [^>]*w:styleId="Normal"[^>]*>.*?<w:rPr>[^<]*<w:sz w:val=")19("/>)',
        r"\g<1>22\g<2>",
        styles_xml,
        count=1,
        flags=re.DOTALL,
    )

    # Bump TOC2 and TOC3 sizes 18 -> 22
    for sid in ("TOC2", "TOC3"):
        styles_xml = re.sub(
            r'(<w:style [^>]*w:styleId="' + sid + r'"[^>]*>.*?<w:sz w:val=")18("/>)',
            r"\g<1>22\g<2>",
            styles_xml,
            count=1,
            flags=re.DOTALL,
        )
    return styles_xml


def fix_document(doc_xml: str) -> str:
    # Top margin 1304 -> 1701 (exactly 3 cm)
    doc_xml = re.sub(
        r'(<w:pgMar[^/>]*?w:top=")1304(")',
        r"\g<1>1701\g<2>",
        doc_xml,
    )

    # Raise any w:sz / w:szCs below 22 half-pt to 22 in document.xml
    def _bump(m: re.Match) -> str:
        tag = m.group(1)
        val = int(m.group(2))
        return f'<w:{tag} w:val="{22 if val < 22 else val}"/>'

    doc_xml = re.sub(r'<w:(sz|szCs) w:val="(\d+)"/>', _bump, doc_xml)

    # Front-page inserts: place four Cover-Text paragraphs after the Semester
    # paragraph. We match the closing </w:p> of the Semester paragraph (which
    # carries the CBS logo anchor) and append new paragraphs directly after.
    semester_pat = re.compile(
        r'(<w:p [^>]*?>(?:(?!</w:p>).)*?Semester:(?:(?!</w:p>).)*?Spring 202(?:(?!</w:p>).)*?</w:p>)',
        flags=re.DOTALL,
    )

    def _cover_para(label: str, value: str) -> str:
        return (
            '<w:p><w:pPr><w:pStyle w:val="Cover-Text"/></w:pPr>'
            '<w:r><w:rPr><w:b/><w:bCs/></w:rPr>'
            f'<w:t xml:space="preserve">{label}</w:t></w:r>'
            f'<w:r><w:t xml:space="preserve"> {value}</w:t></w:r></w:p>'
        )

    insertions = "".join(
        [
            _cover_para("Type of paper:", "Final exam paper"),
            _cover_para("Supervisor:", "[Supervisor name]"),
            _cover_para("Characters:", "[XX,XXX characters incl. spaces]"),
            _cover_para("Pages:", "[XX pages]"),
        ]
    )

    new_doc, n = semester_pat.subn(r"\g<1>" + insertions, doc_xml, count=1)
    if n != 1:
        raise RuntimeError("Could not locate Semester paragraph for front-page insertion")
    return new_doc


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")

    with zipfile.ZipFile(SRC, "r") as zin, zipfile.ZipFile(
        TMP, "w", compression=zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                data = fix_document(data.decode("utf-8")).encode("utf-8")
            elif item.filename == "word/styles.xml":
                data = fix_styles(data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)

    shutil.move(TMP, DST)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
