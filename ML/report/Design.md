# Design System

Visual, structural, and editorial conventions for the final exam paper (KAN-CDSCO2004U Machine Learning and Deep Learning). The deliverable is a Word document exported to PDF, max 15 pages of body, group-written. Figures are generated in Python scripts, saved as PNG to `submission/report_assets/figures/`, and inserted into the Word document for export to PDF. Tables are produced in Python and either exported as CSV for native Word formatting or reproduced directly in Word using the styling below.

This file inherits the **CBS Formal Requirements** (≤ 2,275 chars/page, margins, font size, page count) from `paper_guidelines.md`. Anything below either reaffirms a binding inherited rule or specifies how the docx implements it.

## Active document and editing rule

The active Word document is `KAN-CDSCO2004U_161989_160363_185912_160714_Polymarket_Mispricing.docx` in this folder. The mirror PDF (`KAN-CDSCO2004U_*.pdf`) is a regenerated export and not a source.

- **Patch in place.** Modify only the active `.docx` (or `word/document.xml` and `word/styles.xml` inside it). Preserve every other package part: cover image, CBS logo, anchored drawings, fonts, media, headers, footers.
- **Never rebuild from a template or archived snapshot.** Older `.pre_*.docx` files in `archive/` and `report_tools/backup/` are rollback artefacts, not templates. Rebuilding silently drops later formatting and front-page edits.
- **The first two pages are frozen and manual-only.** Cover page and ToC page are off-limits to every script. Do not automate cover metadata, image/logo placement, anchored drawings, the TOC field, the `TOCHeading` paragraph, the Introduction `pageBreakBefore`, or page-1/2 `sectPr`/`pgMar`. Some visually blank cover paragraphs carry drawing anchors and editing them can make images disappear.
- **Take a backup before any structural patch.** Scripts in `report_tools/` already write `.pre_*.docx` rollbacks into `report_tools/backup/`. New patchers must do the same with a `YYYY-MM-DD_<short-tag>` suffix.
- **Comment-aware edits.** If Word comments or review markup are present in the area being edited, prefer a direct package patch that rewrites only the relevant `word/document.xml` nodes and preserves `word/comments*.xml`. `python-docx` patchers may normalise paragraph XML around comment markers; use them only after checking that comments are outside the edited range or can safely be removed.

## Language

The report uses **US English spelling** throughout (e.g. *behavior*, *modeling*, *normalization*, *color*, *analyze*). Apply the same convention to all generated text, captions, code comments destined for the report, and any new prose written into the docx.

## Output Pipeline

- Figures are saved with `fig.savefig(path, dpi=300, bbox_inches="tight")`.
- Output format is PNG. File names are numbered and descriptive: `01_class_balance.png`, `04_correlation_heatmap.png`.
- Captions and numbering are applied in Word using the Figure and Table caption styles, not baked into the image.
- Figures sit on a white background so they blend with the Word page.

## Page and Sizing

Body pages: A4, **3 cm top/bottom margin and 2 cm left/right margin** per `paper_guidelines.md` (≥ minima). Effective body text width ≈ 17 cm (6.7 inches). The cover page is exempt — its margins and layout are manual-only and must not be touched by automation.

Figures are sized to `FIG_W = 6.3 in` (16 cm) for headroom, so they render without horizontal overflow even if the body margins are widened. This conservative width is intentional and should not be raised without re-checking every existing figure in the docx.

| Parameter | Value |
|---|---|
| `FIG_W` | 6.3 (inches, full body-text width with safety margin) |
| `FIG_W_HALF` | 3.1 (inches, half width for side-by-side placement) |
| `FIG_W_WIDE` | 7.5 to 8.5 (inches, rotated or landscape-style figures) |
| Display DPI | 140 |
| Save DPI | 300 |
| `constrained_layout` | `True` on all figures |

Multi-panel layouts use a single `plt.subplots(1, 2, ...)` or `plt.subplots(2, 2, ...)` call with a shared figure. Heatmaps scale height to preserve square cells.

## Colour Palette

| Token | Value | Usage |
|---|---|---|
| `C_MAP` | `sns.color_palette("rocket_r", as_cmap=True)` | Continuous colourmap for heatmaps |
| `PAL_10` | `sns.color_palette("rocket_r", 10)` | Discrete palette for line, bar, and scatter plots |
| `PAL_K` | `[PAL_10[i] for i in [1, 3, 6, 8, 9]]` | Cluster or category colours (up to five) |
| `INK` | `PAL_10[8]` | Scatter point fill and annotation text |
| `COL_DARK` | `"0.15"` | Heatmap annotation text |
| `COL_CORRECT` | `PAL_10[6]` | Correct or positive class |
| `COL_INCORRECT` | `PAL_10[2]` | Incorrect or negative class |

All plots draw from this palette. Ad-hoc hex colours are avoided.

## Theme

Set once at the top of every plotting script:

```python
sns.set_theme(style="white", context="paper")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})
```

- White figure and axes background
- Grid disabled globally
- Top and right spines removed:

```python
def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
```

## Figures

- No in-image titles when a Word caption will sit below. A short axis-level title is acceptable for multi-panel layouts to distinguish panels.
- Axis labels are lowercase and concise.
- Legends sit inside the plot area when space allows, `frameon=False`.
- Saved file is the final artefact. Captioning is done in Word with the `Figure` style.

Caption format in Word: *Figure N. Descriptive caption text.*

## Heatmaps

- Square cells, no gridlines (`linewidths=0.0`)
- Annotation text white, size 8 (or 7 for dense matrices)
- X-tick rotation 45 degrees, y-tick rotation 0, label size 9
- Colourbar: `pad=0.02`, `aspect=30`, label font size 10
- Correlation maps: `vmin=-1, vmax=1, center=0`
- Covariance maps: `vmin=0`

## Tables

Two paths depending on complexity:

1. **Native Word table** (preferred for small tables, up to roughly 10 rows). Export the dataframe to CSV via `df.to_csv(path, index=False)` and paste into Word using a clean grid table style. Apply centred numeric columns and 4 decimal places where relevant.

2. **Image table** (for dense or styled output). Render with pandas Styler and save via `dataframe_image` or a matplotlib-backed render. Use the same caption convention as figures.

Word table style:
- Grid lines: 0.5 pt, black
- Header row: bold, no fill
- Cell padding: 6 pt left/right, 2 pt top/bottom
- Numeric columns centred, four decimal places where appropriate

Caption format in Word: *Table N. Descriptive caption text.*

## Numbering

Figures and tables are numbered independently, both starting at 1, in order of appearance. Numbering and cross-references are managed in Word using the built-in caption and reference features so the PDF export renders them correctly.

## Paragraph Spacing and Page Breaks

These rules are enforced programmatically by `scripts/23_docx_spacing_and_breaks.py` — re-run after any manual edit that adds new headings or paragraphs.

**No automatic paragraph spacing.**
- Every paragraph in the document has `space-before = 0` and `space-after = 0`. This applies to body text, every heading level (Heading 1 through Heading 3), caption paragraphs, and reference-list items.
- Do not rely on Word's built-in "space before / after heading" formatting. The Word template often injects `spacing before = 18 pt` on Heading 1 / Heading 2 by default; these are explicitly zeroed out.

**Spacing uses explicit blank rows.**
- Separation between content elements is produced by inserting a blank paragraph in the `Normal` style with the document's default body font size.
- The blank-paragraph height is therefore exactly one line of body text. This gives visually consistent vertical rhythm across headings, body, captions, and tables, regardless of which heading style sits above or below.
- Do not use the Word "empty paragraph with 12 pt spacing" convention to create space. Always an explicit blank `Normal` paragraph.

**Default rule: one blank row between every two content elements** (heading, body paragraph, figure, table, caption).

**Exceptions to the default one-row rule:**

1. **Heading immediately followed by another heading** — no blank row between them. A Heading 1 directly followed by a Heading 2 (e.g., `5. Methodology` → `5.1. Dataset Description`), or a Heading 2 directly followed by a Heading 3, sits flush with no separator paragraph in between.
2. **Heading immediately followed by body text** — no blank row between them. The first body paragraph of a section starts directly under its heading. The blank-row gap is only required *before the next* body paragraph, heading, figure, or table.
3. **Table caption ("table rubric") immediately above its table** — no blank row between the caption and the table. The caption paragraph and the table itself are treated as a single block. A blank row still separates the table from whatever follows.
4. **Figure caption immediately below its figure** — no blank row between the figure and its caption. The figure paragraph and the caption paragraph are treated as a single block. A blank row still separates the caption from whatever follows.

**Keep-together rules (no splits across pages):**

- **Tables**: every row carries `cantSplit`; every paragraph in every non-last row carries `keepNext`; the paragraph immediately before each table carries `keepNext`. The table caption paragraph above the table also carries `keepNext` so the caption never lands on a different page than the table itself.
- **Figures**: the paragraph containing the figure (drawing) carries `keepNext` so it stays with its caption paragraph below it. The caption itself carries no further `keepNext` unless the next element is logically tied to it.

**Placeholder (rubric) headings are treated the same.**
- Any placeholder heading that still reads `[ … to be written ]` or similar inherits the same zero-spacing + blank-row rule. This ensures that when the rubric is filled in, the page layout does not shift.

**References and Appendix start on new pages.**
- The top-level Heading 1 paragraphs `References` and `Appendix` each have `pageBreakBefore` set on their paragraph properties. Word (and PDF export) render them at the top of a fresh page regardless of where the preceding section ends.
- This applies only to those two top-level headings. Intermediate Heading 1 sections (Introduction, Methodology, Results, Ethical Consideration, Discussion, Conclusion and Future Work) flow inline after the preceding section without a forced break.
- The Feature Inventory tables (Table A.6 and Table A.7) sit inside the Appendix, not as a standalone section — no page break or subheading precedes them.

## Table captions

All tables in the paper carry a single-line descriptive caption in the format `Table N. Descriptive caption text.` Where a caption needs additional clarification of columns or filter conventions (as on Table A.6), the descriptive text continues on the same line without a paragraph break; the entire caption is one `Normal`-styled paragraph.

## Self-check after a docx edit

A docx change passes only if every box is ticked. If any box fails, flag it explicitly before reporting the edit as done.

- [ ] Pages 1 and 2 visually unchanged (cover image, CBS logo, anchored drawings, ToC field, Introduction `pageBreakBefore` all intact).
- [ ] No new direct page breaks beyond **References** and **Appendix**; no inherited `pageBreakBefore` introduced on the `Heading 1` / `Heading 2` / `Heading 3` style definitions.
- [ ] Every paragraph carries `space-before = 0` and `space-after = 0`; both autospacing flags are zeroed; vertical separation is created by blank `Normal` paragraphs only.
- [ ] Tables: `cantSplit` on every row, `keepNext` on every paragraph in every non-last row, `keepNext` on the table's caption paragraph above it.
- [ ] Figures: `keepNext` on the paragraph containing the figure so it stays with its caption.
- [ ] US English spelling preserved across the edited area; no em dashes, en dashes, or sentence-shortcut semicolons/colons introduced into body prose.
- [ ] Body still under the 15-page × 2,275-char-per-page budget (ToC + body sections + in-body figures count; cover, references, appendix do not).
- [ ] Active `.docx` filename unchanged; no archive snapshot opened or saved over the live file.
- [ ] A `.pre_*.docx` backup landed in `report_tools/backup/` before the patch.
