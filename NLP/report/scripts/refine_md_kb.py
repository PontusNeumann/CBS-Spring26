#!/usr/bin/env python3
"""Refine an already-markdown OneStream KB into the md-only v3 input folder.

The script is intentionally deterministic and dependency-free. It does not try
to understand Maersk content. It standardises markdown structure, mirrors the
folder taxonomy, writes cleaned .md files, and emits QA files that Linus can
review before running the analysis notebooks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path


HEADING_RE = re.compile(r"^(#{1,6})([^#\s].*)$")
SPACED_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SETEXT_H1_RE = re.compile(r"^=+\s*$")
SETEXT_H2_RE = re.compile(r"^-+\s*$")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
WINDOWS_USER_RE = re.compile(r"\b[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass
class RefinedDoc:
    input_relative_path: str
    output_relative_path: str
    class_label: str
    title: str
    char_count: int
    word_count: int
    heading_count: int
    h1_count_before: int
    h1_count_after: int
    h2_count_after: int
    table_line_count: int
    code_fence_count: int
    email_count: int
    url_count: int
    windows_user_path_count: int
    content_sha256: str
    flags: str
    actions: str


def slugify(value: str, fallback: str = "untitled") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or fallback


def title_from_stem(stem: str) -> str:
    words = re.sub(r"[_-]+", " ", stem).strip()
    return words.title() if words else "Untitled"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def convert_setext_headings(lines: list[str]) -> tuple[list[str], list[str]]:
    if len(lines) < 2:
        return lines, []
    actions: list[str] = []
    converted: list[str] = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and SETEXT_H1_RE.match(lines[i + 1].strip()):
            converted.append(f"# {lines[i].strip()}")
            actions.append("converted_setext_h1")
            i += 2
        elif i + 1 < len(lines) and lines[i].strip() and SETEXT_H2_RE.match(lines[i + 1].strip()):
            converted.append(f"## {lines[i].strip()}")
            actions.append("converted_setext_h2")
            i += 2
        else:
            converted.append(lines[i])
            i += 1
    return converted, actions


def clean_markdown(text: str, fallback_title: str) -> tuple[str, dict[str, int | str], list[str], list[str]]:
    text = normalize_newlines(text)
    lines, actions = convert_setext_headings(text.split("\n"))
    cleaned: list[str] = []
    in_fence = False
    h1_seen = False
    h1_count_before = 0
    h1_count_after = 0

    for raw_line in lines:
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            in_fence = not in_fence
            cleaned.append(line)
            continue
        if not in_fence:
            match = HEADING_RE.match(line)
            if match:
                line = f"{match.group(1)} {match.group(2).strip()}"
                actions.append("normalized_heading_spacing")
            spaced = SPACED_HEADING_RE.match(line)
            if spaced and len(spaced.group(1)) == 1:
                h1_count_before += 1
                if h1_seen:
                    line = f"## {spaced.group(2).strip()}"
                    actions.append("demoted_extra_h1")
                else:
                    h1_seen = True
                    h1_count_after += 1
        cleaned.append(line)

    if not h1_seen:
        cleaned.insert(0, f"# {fallback_title}")
        cleaned.insert(1, "")
        h1_count_after = 1
        actions.append("inserted_missing_h1")

    compacted: list[str] = []
    blank_run = 0
    for line in cleaned:
        if line.strip():
            blank_run = 0
            compacted.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                compacted.append("")
            else:
                actions.append("trimmed_extra_blank_lines")

    output = "\n".join(compacted).strip() + "\n"
    h1_after = len(re.findall(r"^#\s+\S+", output, flags=re.MULTILINE))
    h2_after = len(re.findall(r"^##\s+\S+", output, flags=re.MULTILINE))
    heading_count = len(re.findall(r"^#{1,6}\s+\S+", output, flags=re.MULTILINE))
    title_match = re.search(r"^#\s+(.+)$", output, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else fallback_title

    table_line_count = len([line for line in output.splitlines() if line.strip().startswith("|")])
    code_fence_count = len(re.findall(r"^```", output, flags=re.MULTILINE))
    email_count = len(EMAIL_RE.findall(output))
    url_count = len(URL_RE.findall(output))
    windows_user_count = len(WINDOWS_USER_RE.findall(output))
    words = re.findall(r"[A-Za-z0-9_]+", output)

    flags: list[str] = []
    if len(words) < 50:
        flags.append("short_doc_under_50_words")
    if h1_count_before == 0:
        flags.append("missing_h1_before_refine")
    if h1_count_before > 1:
        flags.append("multiple_h1_before_refine")
    if h2_after == 0:
        flags.append("no_h2_after_refine")
    if code_fence_count % 2:
        flags.append("unbalanced_code_fence")
    if table_line_count:
        flags.append("contains_markdown_table")
    if email_count:
        flags.append("contains_email")
    if windows_user_count:
        flags.append("contains_windows_user_path")
    if heading_count <= 1 and len(words) > 200:
        flags.append("long_doc_without_subheadings")

    stats = {
        "title": title,
        "char_count": len(output),
        "word_count": len(words),
        "heading_count": heading_count,
        "h1_count_before": h1_count_before,
        "h1_count_after": h1_after,
        "h2_count_after": h2_after,
        "table_line_count": table_line_count,
        "code_fence_count": code_fence_count,
        "email_count": email_count,
        "url_count": url_count,
        "windows_user_path_count": windows_user_count,
    }
    return output, stats, sorted(set(flags)), sorted(set(actions))


def unique_output_path(output_root: Path, rel_parent: Path, stem: str) -> Path:
    candidate = output_root / rel_parent / f"{stem}.md"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = output_root / rel_parent / f"{stem}_{counter}.md"
        if not candidate.exists():
            return candidate
        counter += 1


def class_label_for(relative_path: Path) -> str:
    if len(relative_path.parts) > 1:
        return slugify(relative_path.parts[0], fallback="root")
    return "root"


def refine_file(input_root: Path, output_root: Path, path: Path) -> RefinedDoc:
    relative = path.relative_to(input_root)
    fallback_title = title_from_stem(path.stem)
    text = path.read_text(encoding="utf-8", errors="replace")
    cleaned, stats, flags, actions = clean_markdown(text, fallback_title)
    rel_parent = Path(*[slugify(part, fallback="folder") for part in relative.parent.parts])
    output_stem = slugify(path.stem, fallback="document")
    output_path = unique_output_path(output_root, rel_parent, output_stem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return RefinedDoc(
        input_relative_path=relative.as_posix(),
        output_relative_path=output_path.relative_to(output_root).as_posix(),
        class_label=class_label_for(relative),
        title=str(stats["title"]),
        char_count=int(stats["char_count"]),
        word_count=int(stats["word_count"]),
        heading_count=int(stats["heading_count"]),
        h1_count_before=int(stats["h1_count_before"]),
        h1_count_after=int(stats["h1_count_after"]),
        h2_count_after=int(stats["h2_count_after"]),
        table_line_count=int(stats["table_line_count"]),
        code_fence_count=int(stats["code_fence_count"]),
        email_count=int(stats["email_count"]),
        url_count=int(stats["url_count"]),
        windows_user_path_count=int(stats["windows_user_path_count"]),
        content_sha256=digest,
        flags=";".join(flags),
        actions=";".join(actions),
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[RefinedDoc], unsupported: list[Path], input_root: Path) -> list[dict[str, object]]:
    flag_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for row in rows:
        for flag in filter(None, row.flags.split(";")):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        for action in filter(None, row.actions.split(";")):
            action_counts[action] = action_counts.get(action, 0) + 1
    summary = [
        {"metric": "input_root", "value": str(input_root)},
        {"metric": "markdown_documents_refined", "value": len(rows)},
        {"metric": "unsupported_non_md_files", "value": len(unsupported)},
        {"metric": "total_words", "value": sum(row.word_count for row in rows)},
        {"metric": "total_flags", "value": sum(len(list(filter(None, row.flags.split(';')))) for row in rows)},
    ]
    for key, value in sorted(flag_counts.items()):
        summary.append({"metric": f"flag_{key}", "value": value})
    for key, value in sorted(action_counts.items()):
        summary.append({"metric": f"action_{key}", "value": value})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine an already-markdown KB into a clean md-only folder.")
    parser.add_argument("--input", required=True, type=Path, help="Source KB folder containing .md files.")
    parser.add_argument("--output", required=True, type=Path, help="Destination refined md-only KB folder.")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for manifest and QA CSVs. Defaults to <output>/_refinement_reports.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete the output folder before writing refined files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    report_dir = (args.report_dir or output_root / "_refinement_reports").expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_root}")
    if input_root == output_root:
        raise SystemExit("Input and output folders must be different.")
    if args.clear_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(input_root.rglob("*.md"))
    unsupported = sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() != ".md")
    rows = [refine_file(input_root, output_root, path) for path in md_files]

    manifest_fields = list(RefinedDoc.__dataclass_fields__.keys())
    write_csv(report_dir / "kb_refinement_manifest.csv", [row.__dict__ for row in rows], manifest_fields)
    flagged = [row.__dict__ for row in rows if row.flags]
    write_csv(report_dir / "kb_refinement_flags.csv", flagged, manifest_fields)
    write_csv(
        report_dir / "unsupported_files.csv",
        [{"input_relative_path": path.relative_to(input_root).as_posix(), "suffix": path.suffix} for path in unsupported],
        ["input_relative_path", "suffix"],
    )
    write_csv(report_dir / "kb_refinement_summary.csv", build_summary(rows, unsupported, input_root), ["metric", "value"])

    print(f"Refined markdown documents: {len(rows)}")
    print(f"Unsupported non-md files: {len(unsupported)}")
    print(f"Output folder: {output_root}")
    print(f"Report folder: {report_dir}")
    if flagged:
        print(f"Flagged documents for review: {len(flagged)}")


if __name__ == "__main__":
    main()
