# CLAUDE.md — ML report brief for AI agents in `report/`

CBS Machine Learning and Deep Learning final exam (KAN-CDSCO2004U). Group of four. Oral exam based on a max-15-page written product. Topic: *Mispricing on Polymarkets — Detecting Probability Asymmetries in Iran Geopolitical Markets with Machine Learning*. Not confidential.

This file is the operating brief for both Claude and Codex. Read it before editing anything in `ML/report/`.

## Read these first

| File | What it covers |
|---|---|
| `paper_guidelines.md` | Course rules, formatting limits, required sections, LLM disclosure template, submission checklist. **Source of truth for what the paper must satisfy.** |
| `project_plan.md` | Pipeline, data, methods, leakage policy, repository layout, status. **Source of truth for what the project does.** |
| `Design.md` | Figure and table conventions: palette, sizing, theme, paragraph spacing rules, page break rules. **Binding for any docx edit.** |
| `submission/README.md` | How to run the pipeline end-to-end. |
| `submission/data/MISSING_DATA.md` | Per-feature missing-value handling. |
| `report_tools/README.md` | Live docx automation scripts (not part of the graded submission). |

## Live filenames

- Live report (the deliverable): `KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx`
- Mirror PDF: `KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.pdf`
- Single source of truth for data: `submission/data/consolidated_modeling_data.parquet` (1,371,180 rows × 87 cols)
- Pipeline scripts: `submission/scripts/01_data_prep.py` through `06_tuning_optuna.py`
- Docx automation: `report_tools/28_*.py`, `29_*.py`, `30_*.py`, `31_*.py`, `36_*.py`, plus `eda.py` and `build_eda_appendix.py`
- Backups for docx patches: `report_tools/backup/`
- Historical material (do not rebuild from): `archive/`

## Hard rules

1. **Pages 1 and 2 of the live docx are manual-only.** Do not automate the cover, the ToC page, anchored drawings, the TOC field, the `TOCHeading` paragraph, the Introduction `pageBreakBefore`, or page-1/2 `sectPr`/`pgMar`. Some visually blank cover paragraphs carry drawing anchors; editing them can make images disappear.
2. **Patch in place; never rebuild from a template or archived snapshot.** Modify only the active `.docx` (or `word/document.xml` and `word/styles.xml` inside it). Preserve every other package part: cover image, CBS logo, fonts, media, headers, footers. Older `.pre_*.docx` files in `archive/` and `report_tools/backup/` are rollback artefacts, not templates — rebuilding from them silently drops later formatting and front-page edits.
3. **Take a backup before any structural patch.** Scripts in `report_tools/` already write `.pre_*.docx` into `report_tools/backup/`; new patchers must do the same with a `YYYY-MM-DD_<short-tag>` suffix.
4. **Comment-aware edits.** If Word comments or review markup are present in the area being edited, prefer a direct package patch that rewrites only the relevant `word/document.xml` nodes and preserves `word/comments*.xml`. `python-docx` patchers may normalise paragraph XML around comment markers; only use them after checking that comments are outside the edited range.
5. **15-page body cap. ≤ 2,275 chars/page average. 3 cm top/bottom, 2 cm L/R margins on body pages (front page exempt). ≥ 11 pt body font.** Cover, references, and appendices do not count toward the 15 pages; ToC and in-body figures/tables do.
6. **US English spelling everywhere** (e.g. *behavior*, *modeling*, *normalization*, *color*, *analyze*). Applies to prose, captions, and any text written into the docx.
7. **Citation style is APA 7.** Géron (2022) is the primary textbook. The course catalogue lists six learning objectives that the report should make visible.
8. **LLM Usage Disclosure is mandatory and on its own page** (template in `paper_guidelines.md` §7).
9. **Project Python is the conda env `py312`** at `/Applications/anaconda3/envs/py312/bin/python3.12`. Do not use `/usr/bin/python3` (3.9.6) and do not `pip install --user`. Activate with `conda activate py312` for installs.

## Pipeline status (2026-05-07)

- Pipeline scripts `01`–`06` are in `submission/scripts/` and were last verified end-to-end against the bundled parquet.
- Headline figures already in `submission/report_assets/figures/main/`; EDA appendix in `submission/report_assets/figures/appendix/eda/`.
- `submission/outputs/` is currently empty (placeholder dirs only). **Alex is running the pipeline now**, so `outputs/data/`, `outputs/models/`, `outputs/metrics/`, `outputs/backtest/`, `outputs/tuning/` will populate from his machine and feed the writing pass.
- The `.docx` exists; results-section integration of newly regenerated figures and tables is the next active workstream.

## Who runs what

- **Alex** is the run owner for the modelling pipeline; outputs land on his machine and are then synced into `submission/outputs/` and (where cited) into `submission/report_assets/`.
- The rest of the group designs, code-reviews, and drafts the paper.
- Hand-off contract: scripts run end-to-end against `submission/data/consolidated_modeling_data.parquet` with paths and seeds set in `submission/scripts/config.py`. Do not invent data; do not change the consolidated parquet.

## Voice and style

Academic third person. No first person ("I", "we", "our") in report prose. No em dashes, en dashes, semicolons, or colons as sentence shortcuts in body prose; recast with clauses, commas, and explicit linking words. Lead with the answer, then explain. APA-7 in-text citations resolve to a single References list.

## Self-check before reporting a docx change as done

A docx edit passes only if every box is ticked. If any fails, flag it explicitly before reporting the edit as done.

- [ ] Pages 1 and 2 visually unchanged (cover image, CBS logo, anchored drawings, ToC field, Introduction `pageBreakBefore` all intact).
- [ ] A `.pre_*.docx` backup landed in `report_tools/backup/` before the patch.
- [ ] No new direct page breaks beyond **References** and **Appendix**; no inherited `pageBreakBefore` introduced on `Heading 1` / `Heading 2` / `Heading 3` style definitions.
- [ ] Every paragraph carries `space-before = 0` and `space-after = 0`; both autospacing flags zeroed; vertical separation uses blank `Normal` paragraphs only (per `Design.md`).
- [ ] Tables: `cantSplit` on every row; `keepNext` on every paragraph in every non-last row and on the caption paragraph above the table. Figures: `keepNext` on the paragraph containing the figure so it stays with its caption.
- [ ] US English spelling preserved across the edited area; no em dashes, en dashes, or sentence-shortcut semicolons/colons introduced into body prose.
- [ ] Body still under the 15-page × 2,275-char-per-page budget.
- [ ] Active `.docx` filename unchanged; no archive snapshot opened or saved over the live file.
