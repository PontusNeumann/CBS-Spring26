# Design Context

Design system for all projects, assignments, and reports in this repository. All figures, tables, and heatmaps follow the conventions below.

## CBS Formal Requirements

Source: CBS Student Hub, *Formal requirements* (studentcbs.sharepoint.com). Apply to all written home assignments including projects, bachelor projects, and master theses.

### Standard page

| Rule | Value |
|---|---|
| Characters per page (incl. spaces) | Max **2,275** on average |
| Top and bottom margin | At least **3 cm** |
| Left and right margin | At least **2 cm** |
| Font size (body) | At least **11 pt** |
| Footnote font | May be smaller, must remain readable |
| Headlines | Must differ graphically from body text |
| Page numbers | Required; table of contents must list main section starts |

### What counts toward page and character limits

Counts toward **both** pages and characters:

- Abstract
- Introduction, delimitation, research question
- Method section
- Indications of individualisation in group papers
- Footnotes and endnotes
- Conclusion and perspective

Counts toward **pages only** (not characters):

- Table of contents

Counts toward **neither**:

- Front page
- Bibliography and list of sources
- Appendices

### Tables, figures, illustrations

- Not included in the character count.
- Do count as the **page space** they occupy. A figure covering 3/4 of a page consumes 3/4 of a page regardless of its character content.
- Inserting illustrations does not justify exceeding the page maximum. If a figure does not say more than the text it displaces, move it to an appendix or remove it.

### When a paper is too long

A paper exceeds limits if any of the following hold:

- Page count exceeds the course requirement.
- Margins or font size violate the rules above.
- Average characters per page exceeds 2,275 incl. spaces.

### Front page (required fields)

- Title (and subtitle if any)
- Type of paper (e.g. exam paper for course X)
- Student name(s) — only on papers with oral defence; omitted on anonymous Digital Exam submissions
- Student number(s)
- Programme name
- Date of submission
- Supervisor name (if any)
- Group number (if relevant)
- Number of characters and number of pages
- Confidentiality status (if confidential)

### AI use

- Allowed by default in compulsory courses and compulsory courses taken as electives (autumn 2025 onwards).
- **Not allowed** in written sit-in exams (treated as cheating even if all aids are permitted).
- **Not allowed** in elective courses until the rules update in 2026.
- Allowed for bachelor projects and master theses submitted after 1 February 2024.
- Individual lecturers may impose stricter rules — check the course description.
- Declare AI use where required.

### Plagiarism

Handle references per CBS Library guidance (Bibliographies and references). Reference resource: stopplagiat.nu.


## Notebook Structure

- All imports go in a single cell at the top of the notebook.
- Only import libraries that are actually used in the notebook.
- No scattered inline imports in later cells.
- Caption and styling helpers (`fig_caption`, `table_style`, `clean_ax`) are defined in a dedicated cell immediately after imports.

## Colour Palette

| Token | Value | Usage |
|---|---|---|
| `C_MAP` | `sns.color_palette("rocket_r", as_cmap=True)` | Continuous colourmap for heatmaps |
| `PAL_10` | `sns.color_palette("rocket_r", 10)` | Discrete 10-stop palette for line plots, bar charts, histograms, scatter points |
| `SECTOR_PALETTE` | `sns.color_palette("magma", n_colors=len(sectors))` | One colour per sector (line, box, stack plots) |
| `PAL_K` | `[PAL_10[i] for i in [1, 3, 6, 8, 9]]` | Cluster colours (supports up to k = 5) |
| `INK` | `PAL_10[8]` | Scatter point fill and correlation annotation text |
| `COL_DARK` | `"0.15"` | Heatmap annotation text colour |

Commonly used palette indices: `PAL_10[2]` (elbow plots), `PAL_10[8]` (scatter ink).

## Layout and Sizing

| Parameter | Value |
|---|---|
| `FIG_W` | 9 (inches) |
| `DPI` | 140 |
| Wide/multi-panel plots | `FIG_W * 1.4` to `FIG_W * 1.6` width |
| `constrained_layout` | `True` on all figures |

Figures are created with `plt.subplots(figsize=(...), constrained_layout=True)` directly. Multi-panel layouts (e.g. side-by-side loss and accuracy curves) use a single `plt.subplots(1, 2, ...)` call.

Heatmaps use `heatmap_fig_axes(data)` which scales height to preserve square cells and adds a dedicated colourbar column (`cbar_ratio=0.06`, `wspace=0.10`).

## Theme

Set once at the top of the notebook:

```python
sns.set_theme(style="white", context="notebook")
plt.rcParams.update({"figure.dpi": DPI})
```

- White figure and axes background
- Grid disabled globally
- Top and right spines removed by `clean_ax(ax)`:

```python
def clean_ax(ax):
    """Remove top and right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
```

## Figures

Captions are rendered below each figure via `fig_caption(fig, number, text)`:

```python
def fig_caption(fig, number, text):
    """Show a numbered caption as a separate HTML row below the figure."""
    plt.show()
    display(HTML(
        f'<p style="text-align:center; font-style:italic; font-size:10pt; '
        f'margin-top:2px; margin-bottom:16px;">'
        f'Figure {number}: {text}</p>'
    ))
```

- Called after all plotting is complete for a given figure
- Italic, centred, 10pt
- Format: `"Figure N: Descriptive caption text"`

## Heatmaps

Drawn by `heatmap_fixed()`:

- Square cells, no gridlines (`linewidths=0.0`)
- Annotation text white, size 8 (or 7 for covariance)
- X-tick rotation 45 degrees, y-tick rotation 0, label size 9
- Colourbar: `pad=0.02`, `aspect=30`, label font size 10
- Correlation maps: `vmin=-1, vmax=1, center=0`
- Covariance maps: `vmin=0`, clipped at lower bound 0

## Tables

Styled inline via `table_style(styler, number, text)`:

```python
def table_style(styler, number, text):
    """Apply consistent table styling with a numbered caption above."""
    return (styler
        .set_caption(f"Table {number}: {text}")
        .set_table_styles([
            {"selector": "caption", "props": [
                ("font-weight", "bold"), ("font-size", "14px"),
                ("margin-bottom", "8px"), ("font-style", "italic")]},
            {"selector": "th, td", "props": [
                ("border", "1px solid black"), ("padding", "6px 12px"),
                ("text-align", "center")]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"), ("margin", "0 auto")]},
        ])
    )
```

- Numbered sequentially starting at Table 1
- Bold italic caption above the table (14px)
- All cells: 1px solid black border, centred text, 6px/12px padding
- Table centred on page with `margin: 0 auto`
- Displayed via `IPython.display.display(styled)`
- Numeric columns formatted to 4 decimal places where appropriate: `.format({"Col": "{:.4f}"})`

## Numbering

Figures and tables are numbered independently, both starting at 1. Format: `"Figure N: ..."` and `"Table N: ..."`.