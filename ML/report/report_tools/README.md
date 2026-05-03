# report_tools

Live automation for editing the report `.docx`. These are NOT part of the submission bundle and are not graded — they only exist to make report iteration faster.

## What is in here

| Script | Job |
|---|---|
| `eda.py` | Regenerate the 19 EDA figures + tables. Reads `submission/data/consolidated_modeling_data.parquet`, writes to `submission/report_assets/figures/appendix/eda/`. |
| `build_eda_appendix.py` | Pack the regenerated EDA figures into `eda_appendix.docx` (and `.md` preview). |
| `28_finalise_report.py` | One-shot finalise pass over the .docx body (post-results sweep). |
| `29_pin_appendix_tables.py` | Pin appendix tables so they don't reflow on page-margin changes. |
| `30_us_spelling_and_table_fix.py` | Normalise to US spelling and fix table-cell formatting. |
| `31_fix_page_total.py` | Fix the page total field. |
| `36_remove_stray_logo.py` | Remove a stray logo that crept into one body page. |

## Path conventions

- `ROOT = Path(__file__).resolve().parents[1]` — i.e. `ML/report/`.
- `DOCX = ROOT / "KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx"` — the live report.
- Backups land in `report_tools/backup/` so they don't pollute the repo root or the archive.
- EDA inputs / outputs go through `submission/data/` and `submission/report_assets/figures/appendix/eda/` so the submission bundle stays the source of truth.

## What NOT to touch

The first two pages of the report (cover + ToC) are off-limits for any script: `sectPr`, `pgMar`, anchors, footers, ToC heading. Cover-template and section-margin scripts (originals 25, 26, 32, 33, 34, 35) are intentionally left in `archive/scripts/` and not promoted here.

## Run order

For a normal report-iteration loop:

```bash
# 1. regenerate EDA artefacts (only when data or EDA logic changes)
python report_tools/eda.py
python report_tools/build_eda_appendix.py

# 2. apply post-results docx fixes (run only the ones you actually need)
python report_tools/28_finalise_report.py
python report_tools/29_pin_appendix_tables.py
python report_tools/30_us_spelling_and_table_fix.py
python report_tools/31_fix_page_total.py
python report_tools/36_remove_stray_logo.py
```

Every script writes a `.pre_*.docx` backup into `report_tools/backup/` before patching, so you can always roll back.
