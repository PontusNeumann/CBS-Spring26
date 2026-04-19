# Design System

Visual conventions for figures and tables in the final exam paper. Figures are generated in Python scripts, saved as PNG to `report/outputs/`, and inserted into the Word document for export to PDF. Tables are produced in Python and either exported as CSV for native Word formatting or reproduced directly in Word using the styling below.

## Output Pipeline

- Figures are saved with `fig.savefig(path, dpi=300, bbox_inches="tight")`.
- Output format is PNG. File names are numbered and descriptive: `01_class_balance.png`, `04_correlation_heatmap.png`.
- Captions and numbering are applied in Word using the Figure and Table caption styles, not baked into the image.
- Figures sit on a white background so they blend with the Word page.

## Page and Sizing

Target Word layout: A4, 2.5 cm margins, approximately 16 cm (6.3 inches) text width.

| Parameter | Value |
|---|---|
| `FIG_W` | 6.3 (inches, full page width) |
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
