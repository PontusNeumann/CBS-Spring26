"""Initial-template patch for the NLP report docx.

Operates on `KAN-CDSCO1002U_161989_160363_185912_160714_OneStream_Intent_Routing.docx`.
Updates the SDT cover (course, title, group, instructor, type) and replaces the
inherited DPD body with an empty NLP section skeleton. Cover drawings, styles,
and template package parts are preserved.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
DOC = Path(__file__).resolve().parents[1] / (
    "KAN-CDSCO1002U_161989_160363_185912_160714_OneStream_Intent_Routing.docx"
)


COVER_REPLACEMENTS = {
    "This paper contains confidential information": "This paper contains confidential information",
    "Pontus Neumann (185912)": (
        "Alejandro Laurlund Gato (161989), "
        "Alexander Myrup (160363), "
        "Linus Stamov Yu (160714), "
        "Pontus Neumann (185912)"
    ),
    "12 May 2026": "TBD",
    "Platforming Wealth": "Improving Intent Routing in a Production Chatbot",
    "FNZ, Platform Outsourcing and the new Basis of Competition in European Banking": (
        "A Comparative NLP Study on Maersk Customer Support Tickets"
    ),
    "Course: Data, Platforms and Digitalization": (
        "Course: Natural Language Processing and Text Analytics"
    ),
    "Program: MSc in Business Administration and Data Science": (
        "Program: MSc in Business Administration and Data Science"
    ),
    "Semester: Spring 2026": "Semester: Spring 2026",
    "Exam Type: Individual Home Assignment": "Exam Type: Group Project (4 students)",
    "Instructor: Ioanna Constantiou, Rony Medaglia": "Instructor: Rajani Singh",
    "Characters: 22, 750  (incl. spaces)": "Characters: TBD (incl. spaces)",
    "Pages: 10": "Pages: TBD (max 15)",
}


SECTION_SKELETON = [
    ("Heading1", "Introduction", False,
     "[Problem statement, research question, scope, and roadmap of the paper.]"),
    ("Heading1", "Background", False,
     "[Maersk help-desk and OneStream chatbot context. Brief review of relevant NLP literature on intent classification, topic modelling, and embeddings.]"),
    ("Heading1", "Data", False,
     "[Corpus description, EDA highlights, preprocessing decisions and rationale.]"),
    ("Heading1", "Method", False,
     "[Pipeline overview: preprocessing, LDA topic discovery, TF-IDF + logistic regression on LDA labels, word2vec semantic expansion, Dify export. Each method introduced at point of use with hyperparameter justification.]"),
    ("Heading1", "Results", False,
     "[Topic discovery, classifier metrics, semantic expansion outputs, routing recommendations.]"),
    ("Heading1", "Discussion", False,
     "[Interpretation, business implications for Maersk's intent-routing layer, methodological reflections and limitations.]"),
    ("Heading1", "Conclusion", False,
     "[Contribution, recommendations, and future work.]"),
    ("ReferenceHeading", "References", True,
     "[APA 7 reference list. One entry per ReferenceText paragraph.]"),
    ("Heading1", "Appendix", True,
     "[Optional. Notebook excerpts, additional tables, full hyperparameter grid.]"),
]


def find_or_create(parent, tag: str):
    """Return the first child with the given namespaced tag, creating it if absent."""
    full = f"{{{W}}}{tag}"
    el = parent.find(full)
    if el is None:
        el = etree.SubElement(parent, full)
    return el


def first_run_rpr_xml(p_elem) -> bytes | None:
    """Return a serialised <w:rPr> from the first <w:r> that has one, or None."""
    for r in p_elem.iter(f"{{{W}}}r"):
        rpr = r.find(f"{{{W}}}rPr")
        if rpr is not None:
            return etree.tostring(rpr)
    return None


def replace_paragraph_text(p_elem, new_text: str) -> None:
    """Replace all run content in a paragraph with a single run carrying new_text.

    The paragraph's <w:pPr> is preserved. Run formatting from the first existing
    run with <w:rPr> is preserved on the new single run. Any inner <w:drawing>
    or <w:object> elements would be wiped, so callers must skip those paragraphs.
    """
    rpr_xml = first_run_rpr_xml(p_elem)
    # Remove all <w:r> children
    for r in list(p_elem.findall(f"{{{W}}}r")):
        p_elem.remove(r)
    # Build the replacement run
    r = etree.SubElement(p_elem, f"{{{W}}}r")
    if rpr_xml is not None:
        r.append(etree.fromstring(rpr_xml))
    t = etree.SubElement(r, f"{{{W}}}t")
    t.text = new_text
    # Preserve leading/trailing whitespace in <w:t>
    if new_text != new_text.strip():
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def update_cover(body) -> int:
    """Walk SDT cover paragraphs and apply COVER_REPLACEMENTS by full-text match."""
    sdt = body.find(f"{{{W}}}sdt")
    if sdt is None:
        return 0
    n = 0
    for p in sdt.iter(f"{{{W}}}p"):
        # Skip paragraphs containing drawings/anchors
        if list(p.iter(f"{{{W}}}drawing")) or list(p.iter(f"{{{W}}}object")):
            continue
        full = "".join(t.text or "" for t in p.iter(f"{{{W}}}t"))
        new = COVER_REPLACEMENTS.get(full)
        if new is not None and new != full:
            replace_paragraph_text(p, new)
            n += 1
    return n


def make_paragraph(style_id: str, text: str, page_break_before: bool) -> etree._Element:
    p = etree.Element(f"{{{W}}}p")
    pPr = etree.SubElement(p, f"{{{W}}}pPr")
    pStyle = etree.SubElement(pPr, f"{{{W}}}pStyle")
    pStyle.set(f"{{{W}}}val", style_id)
    if page_break_before:
        etree.SubElement(pPr, f"{{{W}}}pageBreakBefore")
    r = etree.SubElement(p, f"{{{W}}}r")
    t = etree.SubElement(r, f"{{{W}}}t")
    t.text = text
    return p


def make_normal_paragraph(text: str = "") -> etree._Element:
    p = etree.Element(f"{{{W}}}p")
    if text:
        r = etree.SubElement(p, f"{{{W}}}r")
        t = etree.SubElement(r, f"{{{W}}}t")
        t.text = text
    return p


def rebuild_body(body) -> None:
    """Drop everything between the SDT cover and the trailing <w:sectPr>; insert
    the NLP section skeleton.
    """
    children = list(body)
    sdt_idx = next(i for i, c in enumerate(children) if etree.QName(c).localname == "sdt")
    sectPr_idx = next(
        i for i, c in enumerate(children)
        if etree.QName(c).localname == "sectPr"
    )
    # Bookmarks immediately after SDT (bookmarkStart/End that wrap the body) get
    # preserved so the document structure stays valid.
    keep_after_sdt = []
    for c in children[sdt_idx + 1:sectPr_idx]:
        local = etree.QName(c).localname
        if local in {"bookmarkStart", "bookmarkEnd"}:
            keep_after_sdt.append(c)
    # Remove old body content
    for c in children[sdt_idx + 1:sectPr_idx]:
        body.remove(c)
    # Re-insert kept bookmarks immediately after SDT
    insert_at = sdt_idx + 1
    for bm in keep_after_sdt:
        body.insert(insert_at, bm)
        insert_at += 1
    # Insert section skeleton paragraphs
    for style, heading, page_break, placeholder in SECTION_SKELETON:
        body.insert(insert_at, make_paragraph(style, heading, page_break))
        insert_at += 1
        body.insert(insert_at, make_normal_paragraph(placeholder))
        insert_at += 1
        body.insert(insert_at, make_normal_paragraph(""))
        insert_at += 1


def patch(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_initial_template")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"backup written -> {backup.name}")

    with zipfile.ZipFile(path) as z:
        members = {name: z.read(name) for name in z.namelist()}

    doc_xml = members["word/document.xml"]
    root = etree.fromstring(doc_xml)
    body = root.find(f"{{{W}}}body")

    n_cover = update_cover(body)
    rebuild_body(body)

    members["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    # Rewrite the zip in place with the same compression
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    tmp.replace(path)
    print(f"patched {path.name}: cover lines updated = {n_cover}, body skeleton = {len(SECTION_SKELETON)} sections")


if __name__ == "__main__":
    patch(DOC)
