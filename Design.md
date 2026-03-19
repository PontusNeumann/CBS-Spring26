# Design Context

Design system for `pn_ma1.ipynb`. All figures, tables, and heatmaps follow the conventions below.

## Colour Palette

| Token | Value | Usage |
|---|---|---|
| `C_MAP` | `sns.color_palette("rocket_r", as_cmap=True)` | Continuous colourmap for heatmaps |
| `PAL_10` | `sns.color_palette("rocket_r", 10)` | Discrete 10-stop palette for line plots, histograms, scatter points |
| `SECTOR_PALETTE` | `sns.color_palette("magma", n_colors=len(sectors))` | One colour per sector (line, box, stack plots) |
| `PAL_K` | `[PAL_10[i] for i in [1, 3, 6, 8, 9]]` | Cluster colours (supports up to k = 5) |
| `INK` | `PAL_10[8]` | Scatter point fill and correlation annotation text |
| `COL_DARK` | `"0.15"` | Heatmap annotation text colour |

Commonly used palette indices: `PAL_10[2]` (elbow plots), `PAL_10[4]` (secondary histogram, trend line), `PAL_10[7]` (primary line, silhouette plots), `PAL_10[8]` (scatter ink).

## Layout and Sizing

| Parameter | Value |
|---|---|
| `FIG_W` | 9 (inches) |
| `DPI` | 140 |
| Cluster/wide plots | `FIG_W * 1.6` width |
| `constrained_layout` | `True` on all standard figures |

Figures are created through `fig_ax(height)` which returns a `(fig, ax)` tuple at fixed width `FIG_W`.

Heatmaps use `heatmap_fig_axes(data)` which scales height to preserve square cells and adds a dedicated colourbar column (`cbar_ratio=0.06`, `wspace=0.10`).

## Theme

Set by `set_clean_theme()`:

- `sns.set_theme(style="white", context="notebook")`
- White figure and axes background
- Grid disabled globally
- Top and right spines removed (`clean_ax`)

## Figures

Standard single-panel heights by chart type:

| Type | Height |
|---|---|
| Line (total emissions) | 3.6 |
| Line (multi-sector) | 4.6 |
| Histogram (side-by-side) | 3.6 |
| Boxplot | 4.8 |
| Stacked area | 4.8 |
| Scatter matrix | `FIG_W` x `FIG_W` (square) |
| Heatmap | auto-scaled to square cells |

Styling rules:

- Line width: 2.0 to 2.6 (primary lines thicker)
- Histogram alpha: 0.45, no edge colour
- Boxplot alpha: 0.40, width 0.6, flier size 2.5
- Stacked area alpha: 0.75
- Scatter: size 18, alpha 0.55, no edge
- Legends: `frameon=False`, placed outside plot area (`bbox_to_anchor=(1.02, 1.0)`)
- X-axis ticks: sparse, every 5th value via `sparse_xticks`
- X-axis left margin removed so y-axis starts at x = 0

## Heatmaps

Drawn by `heatmap_fixed()`:

- Square cells, no gridlines (`linewidths=0.0`)
- Annotation text white, size 8 (or 7 for covariance)
- X-tick rotation 45 degrees, y-tick rotation 0, label size 9
- Colourbar: `pad=0.02`, `aspect=30`, label font size 10
- Correlation maps: `vmin=-1, vmax=1, center=0`
- Covariance maps: `vmin=0`, clipped at lower bound 0

## Tables

- Numbered sequentially starting at Table 11
- Displayed inline via `IPython.display.display`
- No custom table styling beyond default pandas HTML output

## Numbering and Captions

Figures and tables are auto-numbered by `print_figure(title)` and `print_table(title)`.

- Figures start at 1; tables start at 11
- Captions formatted by `caption_title()`:
  - Title case with minor words lowercase (and, of, with, for, by, to, in, on, at, from, over)
  - Abbreviations preserved: YoY, UK
  - Colons and hyphens replaced with spaces
  - Format: `"Figure N: Caption Text"`