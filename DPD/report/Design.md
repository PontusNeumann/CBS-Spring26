# Design System — DPD Final Exam

Visual, structural, and editorial conventions for the *Data, Platforms and Digitalization* (KAN-CDSCO2401U) individual exam paper. The deliverable is a Word document exported to PDF, max 10 pages, individually written, graded only against the four Learning Objectives (LO1–LO4).

This file inherits the **CBS Formal Requirements** from `../../Design.md` (repo root) and the **Word output pipeline / paragraph spacing rules** from `../../ML/report/Design.md`. Anything below either reaffirms a binding inherited rule or overrides it for the DPD-specific context.

## 1. Inherited Rules — Binding

From `../../Design.md`:

- Max 2,275 characters per page on average, including spaces.
- Top and bottom margin at least 3 cm; left and right at least 2 cm.
- Body font at least 11 pt. Headlines graphically distinct from body text.
- Page numbers required.
- Tables, figures, and illustrations occupy page space but do not count toward characters. They do not justify exceeding the page maximum.
- Front page fields: title, type of paper, student number, programme name, date, supervisor (if any), character and page count, confidentiality status (if applicable).
- Plagiarism and reference handling per CBS Library guidance.

From `../../ML/report/Design.md`:

- Word output pipeline, paragraph-spacing rules, and keep-together rules (zero auto spacing, blank `Normal` paragraphs as separators, `cantSplit` and `keepNext` on tables and figure captions).
- Caption format in Word: *Figure N. Descriptive caption text.* and *Table N. Descriptive caption text.*

## 1.1. DPD-Specific Template Rules — Typography, Cover, and Back Page

The DPD docx is built from the CBS branded template at `report/KAN-CDSCO2401U_185912_DPD_Spring2026.docx`. The first page of that template (the front page wrapped in a `<w:sdt>` block) carries the CBS logo, the cover image, and the accent-colour background shape. **Those elements are not moved or modified by any build step.** The same applies to the back-page block.

**Active document base and editing rule**

- The active Word document is `KAN-CDSCO2401U_185912_DPD_Spring2026.docx` in this report folder.
- The binding fallback (front page, styles, package parts) is the latest backup snapshot: `backup/KAN-CDSCO2401U_185912_DPD_Spring2026_Backup_28.docx`. If a content patch or restore is needed, preserve or restore the front page from this snapshot.
- Do **not** rebuild the final document from `backup/Old_template.docx` (the historical pre-content template) or any older backup. That path silently drops later template and formatting choices.
- Content-only edits to the final Word file should patch the existing `.docx` in place and leave cover, header, footer, styles, media, and back-page parts untouched. The safest automated route is to replace only `word/document.xml` text content while preserving all other `.docx` package parts.
- The first page is not edited by automation. Front-page metadata and confidentiality wording are added manually unless the user explicitly asks otherwise.

**Body text**

- All body paragraphs in the built-in `Normal` style are **11 pt**.
- The body font follows the CBS template default serif face.

**Headings**

- `Heading 1` stays at the template-defined setting: **CBS NEW, 36 pt, bold, color `#4967AA`**.
- `Heading 2` stays at the template-defined setting: **CBS NEW, 14 pt, bold, color `#4967AA`**.
- `Reference Heading` stays at the template-defined setting: **CBS NEW, 36 pt, bold**.

**Template-defined styles excluded from the 11 pt rule**

- Front page styles remain at template sizes:
  - `Cover - Title`: **42 pt**
  - `Cover - Subtitle`: **14 pt**
  - `Cover - Text`: **9 pt / 11 pt**, per template
- Back page:
  - `Back - Text`: **CBS Serif, 9 pt**
- References:
  - `Reference Text`: **CBS Serif, 9.5 pt**

**Rule of thumb**

- Only paragraphs in the built-in `Normal` style are forced to **11 pt**.
- Paragraphs using CBS template-specific styles such as `Cover - *`, `Back - Text`, `Reference Text`, and `Reference Heading` keep their template-defined sizes.

**Removed / cleaned up**

- No manual table-of-contents listing is kept on the cover or in front matter.
- No ISSN line is kept on the back page.

## 1.2. DPD-Specific Word Output Rules — Spacing and Page Breaks

**Paragraph spacing — one row, applied programmatically.**

- Every paragraph in the document has `space-before = 0` and `space-after = 0` set on its `<w:spacing>` element, regardless of the underlying style's defaults.
- All `w:beforeAutospacing` and `w:afterAutospacing` flags are zeroed out so the explicit zero values actually apply.
- Normal body paragraphs and blank `Normal` separator rows carry explicit single-line spacing (`w:line="240"`, `w:lineRule="auto"`) so they do not inherit any large document-default line spacing from the Word template.
- Vertical separation between content elements is created by inserting a single blank paragraph in the `Normal` style with the document's default body font size. The blank paragraph is one body-text line tall — no more, no less.
- Default rule: **one blank `Normal` row between every two content elements** (heading, body paragraph, figure, table, caption).

**Exceptions to the one-row rule:**

1. **Heading immediately followed by another heading** (e.g. `5. Discussion` → `5.1. Transaction costs`) — no blank row between them. The lower-level heading sits flush under the higher-level one.
2. **Heading immediately followed by body text** — no blank row between them. The first body paragraph of a section starts directly under its heading.
3. **Table caption immediately above its table** — no blank row between the caption and the table. Caption + table form one block.
4. **Figure caption immediately below its figure** — no blank row between figure and caption. Figure + caption form one block.

**Keep-together rules (no splits across pages):**

- Tables: every row carries `cantSplit`; every paragraph in every non-last row carries `keepNext`; the paragraph immediately before each table (the caption) carries `keepNext` so the caption never lands on a different page than the table.
- Figures: the paragraph containing the figure carries `keepNext` so it stays with its caption paragraph below.

**Page-break-before — locked sections only.**

Only two top-level paragraphs carry `pageBreakBefore` on their paragraph properties:

1. The **References** heading (`Reference Heading` style).
2. The **Appendix** heading (`Heading 1` style).

No body Heading 1 (Introduction, Background, Method, Results, Discussion, Conclusion) carries `pageBreakBefore`. Body sections flow inline so the 10-page budget is not wasted on forced breaks. The `Heading 1`, `Heading 2`, and `Heading 3` style definitions themselves must also have `pageBreakBefore` unset; otherwise Word will show "Page break before" as inherited on every heading even when the individual paragraph does not contain a direct page-break flag.

The back-page block in the CBS template already carries its own `pageBreakBefore` and is left untouched.

## 2. DPD-Specific Page Discipline

| Rule | Value |
|---|---|
| Maximum length | **10 pages** (hard cap) |
| Problem introduction / case description | **≤ 1 page** |
| Theories applied in depth | **2 to 3** (more is shallow, one is insufficient) |
| Standalone theory-summary section | Not required — introduce theory at point of use |
| Structure | Open — choose the structure that best serves the argument |

Every page must earn its place against an LO. Cut anything that does not.

## 3. Single Grading Lens — The Four Learning Objectives

Every section and every paragraph is scored against:

1. **LO1 — Explain** key concepts (data, digital platforms, digitalization).
2. **LO2 — Illustrate** their influence on markets and firms via a real business case or sectoral analysis.
3. **LO3 — Apply** theoretical tools to economic, managerial, and societal impacts.
4. **LO4 — Evaluate** and propose innovative, justified solutions accounting for the interplay of economic, managerial, and societal factors.

A paragraph that serves none of these four does not belong in the paper.

## 4. Suggested Section Skeleton

Open structure, but a workable default:

1. Front page, if used by the CBS template.
2. Introduction — research question, purpose, scope, and theory selection, ≤ 1 page.
3. Background — FNZ and the wealth-platform shift, including public case evidence.
4. Method — case design, interview logic, and source limitations.
5. Results — interview extraction tables and core empirical findings.
6. Discussion — theory-driven interpretation, trade-offs, and solution implications.
7. Conclusion and limitations.
8. References (page break before; outside the 10-page body cap).
9. Appendix, if any (page break before; outside the 10-page body cap).

## 5. Writing Conventions

- **Language:** US English, academic tone, clear and concise.
- **Voice:** Use academic third person by default. First person is allowed only in clearly bounded reflexive passages about the author's own practitioner experience, and only when it improves transparency. Such passages must remain anonymized: no client data, named internal checks, colleague names, incidents, screenshots, or non-public operating details.
- **Punctuation:** No em dash and no en dash used as a sentence connector. Use a comma, colon, semicolon, or recast.
- **Sentence rhythm:** Lead with the answer, then explain. Short sentences with selective descriptive word choices.
- **Headings:** Sentence case. Heading 1 for top-level sections, Heading 2 for subsections, Heading 3 sparingly.
- **Spacing:** No blank spaces inside paragraphs to create rhythm — use full blank rows in `Normal` style only, per the inherited paragraph-spacing rules.
- **Concept handling:** Define a term precisely on first use, then pivot to application. The audience is the examiner, who already knows the literature.
- **No concept-dropping:** A term named is a term applied. If a concept appears only as a label, remove it.

## 6. How to Apply Theory (binding pattern)

For every theoretical move in the paper, follow this shape:

> "In light of \[theory], the \[mechanism / property] present in \[case] is X. This influences \[specific behavior or outcome] as Y. The implication for the firm is Z."

Theories are introduced at the point of use, only as much as needed to apply them. If a framework has multiple components and only some are used, state the scope explicitly:

> "The AI Factory framework comprises four elements; the analysis below uses only the data pipeline and the experimentation platform, because these bear directly on \[case]."

## 7. Citation Rules

Citation style must be **homogeneous** throughout. Choose one of:

- **APA 7** (CBS standard, default), or
- **Footnotes** (Chicago-style numeric).

### Citation required

- Any specific numerical claim or fact (revenues, market shares, growth rates, headcounts, dates).
- Every theoretical claim, cited to the originating article or book.

### Citation not required

- General statements without numbers (e.g. "AI is reshaping pharmaceutical R&D").

### Source hierarchy

1. Syllabus readings take priority. Cite the reading directly, never the lecture slide that references it.
2. Lecture slides may be cited only when the content is not covered in the syllabus readings.
3. External academic articles and credible industry sources are welcome for the empirical case.
4. Book citations: author, title (italicised), publisher, year.

The course backbone is **Iansiti, M. & Lakhani, K. R. (2020).** *Competing in the Age of AI.* Harvard Business Review Press. Supplement with the syllabus readings listed in `course_learning_objectives_and_exam_info.md`.

## 8. Figures and Tables

The DPD paper is text-driven. Use a figure or table only when it carries argumentative weight that prose cannot.

If used:

- Save figures as PNG at 300 dpi, `bbox_inches="tight"`, white background. File naming: `01_descriptive_name.png`.
- Insert into Word with the `Figure` caption style; format: *Figure N. Descriptive caption text.*
- Tables are native Word tables for anything up to roughly ten rows. Grid lines 0.5 pt black, header row bold, no fill, numeric columns centered.
- Captions: *Table N. Descriptive caption text.*
- Figures and tables are numbered independently, both starting at 1, in order of appearance.
- A figure or table that does not say more than the prose it displaces is moved to the appendix or removed.

**Table header color coding**

- Tables connected to Interview A use FNZ purple on the header row: `#6C1BEE`.
- Tables connected to Interview B use FNZ orange on the header row: `#F37340`.
- Tables connected to Interview C use FNZ yellow on the header row: `#F8D271`.
- The color coding applies only to the top row of each relevant table, including appendix tables.

**Lighter FNZ color variants for later use**

- Light purple: `#9154F2`.
- Light orange: `#F48052`.
- Light yellow: `#F9DA8A`.
- These are available for future accents, softer table headers, or figure elements where the stronger table-header colors are too saturated.

When the rare figure is generated in Python, follow the palette and theme rules from `../../ML/report/Design.md` (rocket_r palette, white background, `clean_ax`, no in-image titles).

## 9. Confidentiality

If the paper draws on confidential material (employer, NDA, internship), insert at the very top of the front page:

> *"This paper contains confidential information."*

Otherwise omit.

## 10. Self-Check Before Any Draft Is Returned

The draft passes only if every box is ticked:

- [ ] Real-life case, not hypothetical.
- [ ] Two to three theories applied in depth, not summarised.
- [ ] Problem introduction ≤ 1 page.
- [ ] Total length ≤ 10 pages, average ≤ 2,275 characters per page including spaces.
- [ ] Margins ≥ 3 cm top and bottom, ≥ 2 cm left and right; body font ≥ 11 pt.
- [ ] Each of LO1–LO4 demonstrably met.
- [ ] Economic, managerial, and societal dimensions all present and connected.
- [ ] Theory applied to case, not described in isolation.
- [ ] Every specific number carries a citation.
- [ ] Citation style homogeneous throughout (APA or footnotes, not both).
- [ ] Syllabus readings cited directly, not via lecture slides.
- [ ] Concrete, justified recommendation delivered.
- [ ] Scope of partially-used frameworks stated explicitly.
- [ ] First person appears only in bounded reflexive passages, with anonymization intact; no em dash; no concept-dropping.
- [ ] Confidentiality statement added if applicable.
- [ ] Front page carries title, type of paper, student number, programme, date, supervisor (if any), character count, page count.

If any box fails, flag it explicitly before the student reviews.
