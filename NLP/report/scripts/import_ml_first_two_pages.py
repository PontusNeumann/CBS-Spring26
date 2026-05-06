"""One-off script: prepend the first two pages of the ML report onto the NLP
report. Nothing in the NLP body is replaced or deleted.

Pages 1 and 2 of `KAN-CDSCO2004U_..._Polymarket_Mispricing.docx` correspond to
the first 39 body children:
  - child 0: <w:sdt>            cover (page 1)
  - children 1..37: ToC heading, ToC entries, closing fldChar (page 2)
  - child 38: paragraph with the CBS logo drawing anchor (page 2 footer)

These are inserted at the very top of the NLP body. NLP's existing cover, ToC
and body sit immediately after, untouched. Image relationships referenced by
the copied content (`rId10`, `rId11`, `rId12`) are remapped to fresh rIds in
the NLP package, and the underlying media files are imported under new
filenames where needed. Bookmark IDs are offset to avoid clashes with the NLP
body.

The resulting docx will contain two cover blocks and two ToCs back-to-back; the
user prunes whichever they do not want by hand in Word.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
PR_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ML = Path(
    "/Users/pontusneumann/Documents/GitHub/CBS-Spring26/ML/report/"
    "KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx"
)
NLP = Path(__file__).resolve().parents[1] / (
    "KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Intent_Routing.docx"
)
ML_PAGES_END_INDEX = 39  # exclusive: copy children [0:39]
BOOKMARK_ID_OFFSET = 1000  # added to every w:id in copied content


def localname(el) -> str:
    return etree.QName(el).localname


def read_zip(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {name: z.read(name) for name in z.namelist()}


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    tmp.replace(path)


def parse_rels(xml: bytes) -> tuple[etree._Element, dict[str, tuple[str, str, str | None]]]:
    root = etree.fromstring(xml)
    out = {}
    for r in root.findall(f"{{{PR_NS}}}Relationship"):
        out[r.get("Id")] = (r.get("Type"), r.get("Target"), r.get("TargetMode"))
    return root, out


def next_rid(rels_root) -> int:
    """Return the next free rId number."""
    used = []
    for r in rels_root.findall(f"{{{PR_NS}}}Relationship"):
        m = re.match(r"rId(\d+)", r.get("Id") or "")
        if m:
            used.append(int(m.group(1)))
    return (max(used) + 1) if used else 1


def add_image_rel(rels_root, target: str) -> str:
    """Append an image relationship and return the new rId."""
    new = f"rId{next_rid(rels_root)}"
    rel = etree.SubElement(rels_root, f"{{{PR_NS}}}Relationship")
    rel.set("Id", new)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", target)
    return new


def remap_rids_in_xml(xml_bytes: bytes, mapping: dict[str, str]) -> bytes:
    """Run regex replace on rId references in serialized XML."""
    s = xml_bytes.decode("utf-8")
    for old, new in mapping.items():
        # Replace r:embed="rIdN", r:id="rIdN", r:link="rIdN" forms
        s = re.sub(
            rf'(r:(?:id|embed|link)=")({re.escape(old)})(")',
            rf"\g<1>{new}\g<3>",
            s,
        )
    return s.encode("utf-8")


def offset_bookmark_ids(xml_bytes: bytes, offset: int) -> bytes:
    """Offset all w:id values in bookmarkStart/bookmarkEnd to dodge clashes."""
    s = xml_bytes.decode("utf-8")

    def repl(m: re.Match) -> str:
        return f'w:id="{int(m.group(1)) + offset}"'

    # Only touch bookmarkStart and bookmarkEnd contexts
    s = re.sub(
        r'(<w:bookmark(?:Start|End)\b[^>]*?)w:id="(\d+)"',
        lambda m: m.group(1) + f'w:id="{int(m.group(2)) + offset}"',
        s,
    )
    return s.encode("utf-8")


def ensure_content_type(types_xml: bytes, ext: str, content_type: str) -> bytes:
    root = etree.fromstring(types_xml)
    for d in root.findall(f"{{{TYPES_NS}}}Default"):
        if (d.get("Extension") or "").lower() == ext.lower():
            return types_xml
    d = etree.SubElement(root, f"{{{TYPES_NS}}}Default")
    d.set("Extension", ext)
    d.set("ContentType", content_type)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def patch() -> None:
    if not NLP.exists():
        raise FileNotFoundError(NLP)
    if not ML.exists():
        raise FileNotFoundError(ML)

    backup = NLP.with_suffix(NLP.suffix + ".pre_ml_pages_import")
    if not backup.exists():
        shutil.copy2(NLP, backup)
        print(f"backup written -> {backup.name}")

    nlp_members = read_zip(NLP)
    ml_members = read_zip(ML)

    # 1. Parse both document.xml
    ml_doc = etree.fromstring(ml_members["word/document.xml"])
    ml_body = ml_doc.find(f"{{{W}}}body")

    nlp_doc = etree.fromstring(nlp_members["word/document.xml"])
    nlp_body = nlp_doc.find(f"{{{W}}}body")

    # 2. Pure prepend mode: nothing in the NLP body is dropped or replaced.
    print("prepend mode: NLP body content stays intact, ML pages are inserted at index 0")

    # 3. Parse both rels
    ml_rels_root, ml_rels = parse_rels(ml_members["word/_rels/document.xml.rels"])
    nlp_rels_root, nlp_rels = parse_rels(nlp_members["word/_rels/document.xml.rels"])

    # 4. Find rIds referenced by ML's first 39 children, then map to fresh NLP rIds
    rid_pat = re.compile(r'r:(?:id|embed|link)="(rId\d+)"')
    ml_first_pages_xml = b"".join(
        etree.tostring(c) for c in ml_body[:ML_PAGES_END_INDEX]
    )
    referenced_rids = sorted(
        set(rid_pat.findall(ml_first_pages_xml.decode("utf-8")))
    )
    print(f"ML first-pages rIds referenced: {referenced_rids}")

    rid_mapping: dict[str, str] = {}
    media_to_copy: list[tuple[str, str]] = []  # (src_path_in_ml, dst_path_in_nlp)

    for rid in referenced_rids:
        if rid not in ml_rels:
            print(f"  WARNING: rId {rid} not in ML rels — skipping")
            continue
        rel_type, target, mode = ml_rels[rid]
        if not target.startswith("media/"):
            # Non-media rels (hyperlinks, etc.) — only support image for this script
            print(f"  WARNING: non-media rel {rid} -> {target}; skipping")
            continue
        # Decide a dest filename in NLP: if NLP already has the same path with
        # identical bytes, reuse its rId; otherwise import under a new filename
        # to avoid overwriting NLP's existing media.
        ml_src = f"word/{target}"
        ml_bytes = ml_members.get(ml_src)
        if ml_bytes is None:
            print(f"  WARNING: media {ml_src} missing in ML zip")
            continue
        nlp_path = f"word/{target}"
        existing = nlp_members.get(nlp_path)
        if existing == ml_bytes:
            # Identical file already present — find the NLP rId pointing to it
            for nrid, (_, ntarget, _) in nlp_rels.items():
                if ntarget == target:
                    rid_mapping[rid] = nrid
                    print(f"  {rid} -> reuse {nrid} ({target}, identical content)")
                    break
            else:
                new_rid = add_image_rel(nlp_rels_root, target)
                rid_mapping[rid] = new_rid
                # nlp_rels stays in sync via add_image_rel side-effect on root
                print(f"  {rid} -> new {new_rid} (identical content, fresh rel)")
        elif existing is None:
            # File doesn't exist in NLP, copy under same path
            media_to_copy.append((ml_src, nlp_path))
            new_rid = add_image_rel(nlp_rels_root, target)
            rid_mapping[rid] = new_rid
            print(f"  {rid} -> new {new_rid} ({target}, imported)")
        else:
            # File path collides with different content; rename
            stem, ext = target.rsplit(".", 1)
            new_target = f"{stem}_from_ml.{ext}"
            new_path = f"word/{new_target}"
            media_to_copy.append((ml_src, new_path))
            new_rid = add_image_rel(nlp_rels_root, new_target)
            rid_mapping[rid] = new_rid
            print(f"  {rid} -> new {new_rid} (renamed to {new_target})")

    # 5. Remap rIds and offset bookmark ids in the ML chunk
    chunk = ml_first_pages_xml
    chunk = remap_rids_in_xml(chunk, rid_mapping)
    chunk = offset_bookmark_ids(chunk, BOOKMARK_ID_OFFSET)

    # 6. Replace NLP body children [0:drop_until] with the parsed ML chunks.
    #    Wrap chunk in a fake root so lxml can parse it.
    chunk_root = etree.fromstring(
        b"<root xmlns:w=\"" + W.encode() + b"\" "
        b"xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        b"xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        b"xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        b"xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        b"xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
        b"xmlns:pic=\"http://schemas.openxmlformats.org/drawingml/2006/picture\" "
        b"xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        b"xmlns:adec=\"http://schemas.microsoft.com/office/drawing/2017/decorative\" "
        b"xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        b"xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        b"xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        b"xmlns:w15=\"http://schemas.microsoft.com/office/word/2012/wordml\" "
        b"xmlns:w16cex=\"http://schemas.microsoft.com/office/word/2018/wordml/cex\" "
        b"xmlns:w16cid=\"http://schemas.microsoft.com/office/word/2016/wordml/cid\" "
        b"xmlns:w16=\"http://schemas.microsoft.com/office/word/2018/wordml\" "
        b"xmlns:w16se=\"http://schemas.microsoft.com/office/word/2015/wordml/symex\" "
        b"xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        b"xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        b"xmlns:cx=\"http://schemas.microsoft.com/office/drawing/2014/chartex\" "
        b"xmlns:cx1=\"http://schemas.microsoft.com/office/drawing/2015/9/8/chartex\" "
        b"xmlns:cx2=\"http://schemas.microsoft.com/office/drawing/2015/10/21/chartex\" "
        b"xmlns:cx3=\"http://schemas.microsoft.com/office/drawing/2016/5/9/chartex\" "
        b"xmlns:cx4=\"http://schemas.microsoft.com/office/drawing/2016/5/10/chartex\" "
        b"xmlns:cx5=\"http://schemas.microsoft.com/office/drawing/2016/5/11/chartex\" "
        b"xmlns:cx6=\"http://schemas.microsoft.com/office/drawing/2016/5/12/chartex\" "
        b"xmlns:cx7=\"http://schemas.microsoft.com/office/drawing/2016/5/13/chartex\" "
        b"xmlns:cx8=\"http://schemas.microsoft.com/office/drawing/2016/5/14/chartex\" "
        b"xmlns:aink=\"http://schemas.microsoft.com/office/drawing/2016/ink\" "
        b"xmlns:am3d=\"http://schemas.microsoft.com/office/drawing/2017/model3d\" "
        b"xmlns:oel=\"http://schemas.microsoft.com/office/2019/extlst\""
        b">" + chunk + b"</root>"
    )
    new_children = list(chunk_root)
    print(f"parsed {len(new_children)} elements from ML chunk")

    # Insert at the front, in order (no removal of existing NLP body content)
    for i, el in enumerate(new_children):
        nlp_body.insert(i, el)

    # 7. Write back
    nlp_members["word/document.xml"] = etree.tostring(
        nlp_doc, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    nlp_members["word/_rels/document.xml.rels"] = etree.tostring(
        nlp_rels_root, xml_declaration=True, encoding="UTF-8", standalone=True
    )

    # Copy media files
    for src, dst in media_to_copy:
        nlp_members[dst] = ml_members[src]
        print(f"  imported media: {src} -> {dst}")

    # Make sure [Content_Types].xml has Defaults for png/emf
    ct = nlp_members.get("[Content_Types].xml")
    if ct is not None:
        ct = ensure_content_type(ct, "png", "image/png")
        ct = ensure_content_type(ct, "emf", "image/x-emf")
        nlp_members["[Content_Types].xml"] = ct

    write_zip(NLP, nlp_members)
    print(f"patched {NLP.name}.")


if __name__ == "__main__":
    patch()
