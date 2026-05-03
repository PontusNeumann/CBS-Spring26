# DPD Report — Brief for Claude / Codex

Read this before touching any file in `report/`. It supersedes assumptions you may have from older sessions or scripts.

## What this folder is

Final-exam paper for *Data, Platforms and Digitalization* (KAN-CDSCO2401U) at CBS. Single-author, max 10 pages body, Word delivered as PDF. Case is FNZ.

## Source-of-truth files

| Purpose | File |
|---|---|
| Active report (the deliverable) | `KAN-CDSCO2401U_185912_DPD_Spring2026.docx` |
| Historical package/style fallback (do not overwrite current manual cover from it) | `backup/KAN-CDSCO2401U_185912_DPD_Spring2026_Backup_28.docx` |
| Historical pre-content template — do not rebuild from this | `backup/Old_template.docx` |
| Editorial / formatting rules | `Design.md` |
| Project decisions, theory choices, structure | `project_plan.md` |
| Body text working draft (markdown mirror, not the deliverable) | `paper_body_draft.md` |

The `.docx` is the deliverable. The markdown files are planning artifacts; do not treat them as the live report.

## Editing the .docx — hard rules

1. **Patch in place.** Modify only `word/document.xml` (and `word/styles.xml` for style-level fixes) inside the active `.docx`. Preserve every other package part: cover image, CBS logo, accent shape, fonts, media, headers, and footers.
2. **Never rebuild from a template.** `backup/Old_template.docx` and any older snapshot will silently drop later formatting and front-page edits.
3. **The first page is frozen and manual-only.** Do not automate cover metadata, confidentiality text, image/logo placement, or blank cover paragraphs. Some visually blank cover paragraphs carry drawing anchors and editing them can make images disappear.
4. **The final deliverable intentionally has no decorative CBS back page.** Do not recreate, repair, or restore a back-page block in future scripts or manual cleanup.
5. **Model new patchers on `scripts/patch_docx_round6.py`** (surgical XML edits with a timestamped backup, idempotent paragraph-text replacements, Reference Text list maintenance, frozen cover handling, and no back-page creation). Earlier patchers (`patch_docx_round3.py`, `patch_docx_round4.py`, `patch_docx_round5.py`) are frozen snapshots and now hard-disable themselves with a `sys.exit` guard at module load — they reflect older docx states and would regress current content if re-run. The truly old rebuild scripts (`build_docx.py`, `patch_docx.py`, `patch_docx_round2.py`) live in `scripts/archive/` and must never be run; `build_docx.py` in particular still points at a pre-content-draft template and is the historical source of every "wrong template" regression.
6. **Take a backup before any structural patch.** Copy the active `.docx` to `backup/` with a `YYYY-MM-DD_<short-tag>` suffix.

## Formatting rules that bite

These are the ones that have caused regressions. Full detail in `Design.md` §1.1–§1.2.

- Body `Normal` is **11 pt**. Cover, references, and reference-heading styles keep their template sizes.
- Every paragraph: `space-before=0`, `space-after=0`, both autospacing flags zeroed.
- Body `Normal` paragraphs (including blank separator rows) need explicit `w:line="240" w:lineRule="auto"` — the template default is 680 twips exact and will produce huge gaps if not overridden.
- Vertical separation = one blank `Normal` paragraph. Exceptions: heading→heading, heading→body, caption→table, figure→caption (no blank).
- `pageBreakBefore` only on **References** and **Appendix** headings. No body Heading 1 carries a page break.
- The `Heading 1`/`Heading 2`/`Heading 3` **style definitions** must also have `pageBreakBefore` unset, otherwise Word inherits a page break onto every heading even when the paragraph itself does not have one.

## Voice

Academic third person by default. First person only in clearly bounded reflexive passages about the author's own FNZ practitioner experience, and only when anonymized: no client names, internal check names, colleague names, incidents, screenshots, or non-public operating details.

## Self-check before reporting a docx change as done

- Opened the .docx after the patch and confirmed the cover and headers/footers are intact, with no decorative CBS back page reintroduced.
- No new direct page breaks beyond References and Appendix; no inherited page breaks on heading styles.
- Spacing on body paragraphs visually matches the snapshot.
- Character count for body still under the 10-page × 2,275-char-per-page budget.
