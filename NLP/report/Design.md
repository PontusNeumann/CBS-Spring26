# Design System — NLP Final Exam

Visual, structural, and editorial conventions for the *Natural Language Processing and Text Analytics* (KAN-CDSCO1002U) final exam paper. The deliverable is a Word document exported to PDF, max 15 pages, written in a group of four students, handed in together with the analysis Jupyter notebook. The grade is based on an overall assessment of the written product and the individual oral performance, against the six learning objectives listed in section 3.

This file inherits the **CBS Formal Requirements** from `../../Design.md` (repo root) and the **Word output pipeline / paragraph spacing rules** from `../../ML/report/Design.md`. Anything below either reaffirms a binding inherited rule or overrides it for the NLP-specific context.

## 1. Inherited Rules — Binding

From `../../Design.md`:

- Max 2,275 characters per page on average, including spaces.
- Top and bottom margin at least 3 cm; left and right at least 2 cm.
- Body font at least 11 pt. Headlines graphically distinct from body text.
- Page numbers required.
- Tables, figures, and illustrations occupy page space but do not count toward characters. They do not justify exceeding the page maximum.
- Front page fields: title, type of paper, student numbers, programme name, date, supervisor (if any), character and page count, confidentiality status (if applicable).
- Plagiarism and reference handling per CBS Library guidance.

From `../../ML/report/Design.md`:

- Word output pipeline, paragraph-spacing rules, and keep-together rules (zero auto spacing, blank `Normal` paragraphs as separators, `cantSplit` and `keepNext` on tables and figure captions).
- Caption format in Word: *Figure N. Descriptive caption text.* and *Table N. Descriptive caption text.*

## 1.1. NLP-Specific Template Rules — Typography and Cover

The NLP docx is built from the CBS branded template inherited from the DPD report. The first page (the front page wrapped in a `<w:sdt>` block) carries the CBS logo, the cover image, and the accent-colour background shape. **Those elements are not moved or modified by any build step.** The decorative CBS back page has been intentionally removed and must not be recreated by automation.

**Active document base and editing rule**

- The active Word document is `KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Spring2026.docx` in this report folder.
- The pre-edit baseline template lives in `backup/` as `KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Intent_Routing.docx`. Do not overwrite the live cover from this snapshot unless the user explicitly asks for a one-off cover restore.
- Content-only edits to the live Word file should patch the existing `.docx` in place and leave cover, header, footer, styles, and media untouched. Replace only non-cover `word/document.xml` text content while preserving all other `.docx` package parts.
- The first page is frozen and manual-only after the initial template build. Front-page metadata, confidentiality wording, character count, page count, image and logo placement, and visually blank cover rows are not edited by automation. Some blank cover paragraphs carry drawing anchors.
- The final hand-in has no decorative CBS back page.

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
- References:
  - `Reference Text`: **CBS Serif, 9.5 pt**

**Rule of thumb**

- Only paragraphs in the built-in `Normal` style are forced to **11 pt**.
- Paragraphs using CBS template-specific styles such as `Cover - *`, `Reference Text`, and `Reference Heading` keep their template-defined sizes.

**Removed / cleaned up**

- No manual table-of-contents listing is kept on the cover or in front matter. A native Word ToC field sits on **page 2**, inserted once by `scripts/insert_toc_page.py` (`TOCHeading` style + `TOC \o "1-3" \h \z \u` field). The field is marked dirty so Word repopulates it on open; right-click → *Update Field* refreshes page numbers manually. The ToC counts toward pages but not toward characters.
- **Page 2 visual layout is manual-only and mirrors the ML report ToC.** The CBS logo is anchored in the lower-left corner of the ToC page by the user, in Word, and any other visual tweaks (spacing, leader dots, font tweaks) are also maintained by hand. Automation must not touch the TOCHeading paragraph, the TOC field paragraph, or any drawings anchored to that page once the user has laid it out. The inserter has an early-exit guard when a TOCHeading already exists; that guard must not be bypassed.
- No ISSN line or decorative CBS back page is kept in the final hand-in.

## 1.2. NLP-Specific Word Output Rules — Spacing and Page Breaks

**Paragraph spacing — one row, applied programmatically.**

- Every paragraph in the document has `space-before = 0` and `space-after = 0` on its `<w:spacing>` element, regardless of the underlying style's defaults.
- All `w:beforeAutospacing` and `w:afterAutospacing` flags are zeroed out so the explicit zero values actually apply.
- Body `Normal` paragraphs in the report body (Introduction through Conclusion) carry explicit 1.5 line spacing (`w:line="360"`, `w:lineRule="auto"`) for readability. This covers running prose and body-range table captions. Excluded from the 1.5 rule: headings, table cells, the references list, the front page, the appendix, and blank separator rows. Table cells stay at `w:line="276"` (1.15), references at `w:line="240"` (single), cover paragraphs untouched. Blank `Normal` separator rows remain one row tall and should not be used to create extra spacing.
- Vertical separation between content elements is created by inserting a single blank paragraph in the `Normal` style with the document's default body font size. The blank paragraph is one body-text line tall — no more, no less.
- Default rule: **one blank `Normal` row between every two content elements** (heading, body paragraph, figure, table, caption).

**Exceptions to the one-row rule:**

1. **Heading immediately followed by another heading** (e.g. `4. Method` → `4.1. Preprocessing`) — no blank row between them.
2. **Heading immediately followed by body text** — no blank row between them. The first body paragraph of a section starts directly under its heading.
3. **Table caption immediately above its table** — no blank row between the caption and the table. Caption + table form one block.
4. **Figure caption immediately below its figure** — no blank row between figure and caption. Figure + caption form one block.

**Keep-together rules (no splits across pages):**

- Tables: every row carries `cantSplit`; every paragraph in every non-last row carries `keepNext`; the paragraph immediately before each table (the caption) carries `keepNext` so the caption never lands on a different page than the table.
- Figures: the paragraph containing the figure carries `keepNext` so it stays with its caption paragraph below.

**Page-break-before — locked sections only.**

Four top-level paragraphs carry `pageBreakBefore` on their paragraph properties:

1. The **Table of contents** heading (`TOCHeading` style) — pushes the ToC onto page 2.
2. The **Introduction** heading (`Heading 1` style) — set manually in Word so the body starts on page 3, after the ToC. This flag is preserved by all automation; no script may strip it.
3. The **References** heading (`ReferenceHeading` style).
4. The **Appendix** heading (`Heading 1` style).

No other body Heading 1 (Background, Data, Method, Results, Discussion, Conclusion) carries `pageBreakBefore`. Body sections after Introduction flow inline so the 15-page budget is not wasted on forced breaks. The `Heading 1`, `Heading 2`, and `Heading 3` style definitions themselves should keep `pageBreakBefore` unset at the style level; the flag is applied per-paragraph only on the four sections listed above. If the style definition is ever changed to carry `pageBreakBefore`, every Heading 1 will inherit it and the page budget will collapse.

No page break is inserted for a decorative back page.

## 2. NLP-Specific Page Discipline

| Rule | Value |
|---|---|
| Maximum length | **15 pages** (hard cap) |
| Minimum group size | 2 |
| Maximum group size | 4 (this group is at the cap) |
| Problem introduction / case description | **≤ 1.5 pages** |
| Methods applied in depth | **3 to 5** distinct NLP techniques, each justified at point of use |
| Standalone theory chapter | Not required — introduce concepts and methods at point of use |
| Structure | IMRAD-style preferred (Introduction, Background, Data, Method, Results, Discussion, Conclusion) |
| Notebook hand-in | The `.ipynb` and a `.pdf` export accompany the paper; see section 9 |

Every page must earn its place against a learning objective. Cut anything that does not.

## 3. Single Grading Lens — The Six Learning Objectives

Source: KAN-CDSCO1002U course description, learning objectives section. Every section and every paragraph is scored against:

1. **LO1 — Characterize** the phenomena of text analytics and Natural Language Processing.
2. **LO2 — Summarize** different fundamental concepts, techniques, and methods of NLP.
3. **LO3 — Analyze and apply** different text analytics techniques to big or business datasets in organizational contexts.
4. **LO4 — Understand the linkages** between business intelligence and text analytics, and the potential benefits for organizations.
5. **LO5 — Summarize** the application areas, trends, and challenges in text analysis.
6. **LO6 — Demonstrate critical methodological awareness** of the choices made, with written skills to accepted academic standards.

A paragraph that serves none of these six does not belong in the paper.

## 4. Suggested Section Skeleton

IMRAD-style default, present in the live docx as inserted headings:

1. Front page (CBS template).
2. Table of contents (native Word, between cover and Introduction; counts toward pages, not characters).
3. **Introduction** — research question, business motivation, scope, roadmap. ≤ 1.5 pages.
4. **Background** — Maersk help-desk and OneStream chatbot context. Brief positioning against course-aligned NLP literature on intent classification, topic modelling, and embeddings.
5. **Data** — corpus description, key statistics, exploratory NLP findings, preprocessing decisions and rationale.
6. **Method** — pipeline overview tied to the live notebook: text preprocessing, LDA topic discovery, TF-IDF + logistic regression on LDA labels, word2vec semantic expansion, Dify export. Each method introduced at point of use with a short justification of hyperparameter choices.
7. **Results** — topic discovery, classifier metrics with appropriate cross-validation, semantic expansion outputs, routing recommendations.
8. **Discussion** — interpretation, business implications for Maersk's routing layer, methodological reflections and limitations (LO6).
9. **Conclusion** — contribution, recommendations, and future work.
10. **References** (page break before; outside the 15-page body cap).
11. **Appendix**, optional (page break before; outside the 15-page body cap).

The sub-sectioning of Method and Results should mirror the seven phase headings in `onestream_nlp_pipeline.ipynb` so the reader can move between the paper and the notebook without remapping concepts.

## 5. Writing Conventions

- **Language:** US English, academic tone, clear and concise.
- **Voice:** Academic third person by default. First person plural ("we") is allowed only in clearly bounded methodological reflections. No first person singular.
- **Punctuation:** No em dash and no en dash used as a sentence connector. Main body prose should also avoid semicolons and colons as sentence shortcuts. Use clauses, commas, and explicit linking words instead. APA reference punctuation is exempt.
- **Sentence rhythm:** Lead with the answer, then explain. Short sentences with selective descriptive word choices.
- **Headings:** Sentence case. Heading 1 for top-level sections, Heading 2 for subsections, Heading 3 sparingly.
- **Spacing:** No blank spaces inside paragraphs to create rhythm — use full blank rows in `Normal` style only, per the inherited paragraph-spacing rules.
- **Concept handling:** Define a term precisely on first use, then pivot to application. The audience is the examiner, who already knows the literature.
- **No concept-dropping:** A method named is a method applied. If a technique appears only as a label, remove it.
- **Code and library names:** Inline-code styling for library and function names (e.g. `gensim`, `LdaModel`, `TfidfVectorizer`).

## 6. How to Apply Methods (binding pattern)

For every methodological move in the paper, follow this shape:

> "Method M is introduced because \[data property or research question Y]. Applied to \[dataset / subset], it yields \[result]. The implication for \[business question / routing decision] is Z."

Methods are introduced at the point of use, only as much as needed to apply them. If a technique has multiple variants and only one is used, state the scope explicitly:

> "Latent Dirichlet Allocation has multiple inference variants; the analysis below uses online variational Bayes via `gensim.models.LdaModel`, because the corpus is too large for collapsed Gibbs sampling within the project's compute budget."

Every hyperparameter choice receives at least one short sentence of justification (number of topics, vocabulary size, n-gram range, embedding dimension, etc.).

## 7. Citation Rules

Citation style must be **homogeneous** throughout. Default: **APA 7** (CBS standard).

### Citation required

- Any specific numerical claim or fact (corpus size, headline metrics from prior work, dataset reference numbers).
- Every methodological claim, cited to the originating paper or course textbook chapter.

### Citation not required

- General statements without numbers (e.g. "topic modelling has become a standard tool for unsupervised text analysis").

### Source hierarchy

1. **Course backbone:** Jurafsky, D. & Martin, J. H. (2014). *Speech and Language Processing* (2nd ed.). Pearson, and Manning, C. D. & Schütze, H. (2003). *Foundations of Statistical Natural Language Processing*. MIT Press. Cite these directly when invoking foundational NLP concepts.
2. Syllabus readings beyond the textbooks. Cite the reading directly, never the lecture slide that references it.
3. Lecture slides may be cited only when the content is not covered in the syllabus readings.
4. External academic articles for empirical comparisons and method extensions.
5. Industry sources and tool documentation are welcome for the Maersk case context and library specifics.
6. Book citations: author, title (italicised), publisher, year.

## 8. Figures and Tables

The NLP paper supports its argument with visualisations from the analysis pipeline. Use a figure or table when it carries argumentative weight that prose cannot.

If used:

- Figures generated in the notebook follow the repo-wide conventions in `../../Design.md` (`PAL_10` rocket palette, `FIG_W = 9 in`, `DPI = 140`, `clean_ax`).
- Save figures destined for the Word document as PNG at 300 dpi (and PDF for vector reuse), `bbox_inches="tight"`, white background. File naming: `figNN_descriptive_name.{pdf,png}` in `assets/figures/`.
- Insert into Word with the `Figure` caption style; format: *Figure N. Descriptive caption text. Source: ...*
- Tables in the body are native Word tables for anything up to roughly ten rows. Long tables move to the appendix.
- Captions: *Table N. Descriptive caption text.*
- Figures and tables are numbered independently, both starting at 1, in order of appearance.
- A figure or table that does not say more than the prose it displaces is moved to the appendix or removed.

**Visual palette**

The NLP project does not use a brand palette tied to a specific case company. Use the repository default `rocket_r` palette from `../../Design.md`:

- `PAL_10 = sns.color_palette("rocket_r", 10)` — discrete points, bars, lines.
- `C_MAP = sns.color_palette("rocket_r", as_cmap=True)` — heatmaps and continuous colourmaps.
- Greys (`#333333`, `#6c757d`, `#e9ecef`, header grey `#D9D9D9`) for non-emphasis text, gridlines, table header shading, and alternating rows.

**Tables**

- Table body type is **9.5 pt**, 1.15 line spacing. This 9.5 pt rule applies only to tables. Body `Normal` prose stays **11 pt**.
- Table body text is left-aligned. Numeric columns are centered.
- Header row uses grey background `#D9D9D9` and bold text. Header text is left-aligned by default; headers for numeric columns are centered with the numbers.
- Column widths are flexible and optimized per table for body-page space rather than equal thirds.
- Width rule of thumb: headers should fit on one line where possible, and the longest body cell should not wrap more than two lines.
- Borders: 0.5 pt mid-grey on body cells.
- Alternating body rows may use white and `#e9ecef` (light grey) for readability on long tables. No alternating shading is needed for tables of four rows or fewer.

**Python-generated figures**

- White background, serif font matching the body (Times New Roman / Times), 9–10 pt for in-figure text.
- **No in-image title.** The figure title and any source line live in the Word caption, not inside the PNG/PDF. This avoids double titling when the figure sits next to a Word caption.
- **No in-image source / method caption.** Source attribution belongs in the Word caption.
- Legend outside the data area when feasible; never overlapping data.
- Single-column body width is roughly 6.4 inches — size figures to that width so they do not need resizing in Word.

## 9. Notebook Hand-in

The `.ipynb` file is part of the deliverable. The paper must be reproducible from the notebook alone, with no hidden state.

- The live notebook is `onestream_nlp_pipeline.ipynb` at the top of `report/`. Older versions stay in `notebooks/`.
- All imports go in a single cell at the top of the notebook (per repo `Design.md`). Only import libraries that are actually used. No scattered inline imports.
- Course-aligned libraries only: `scikit-learn`, `nltk`, `spaCy`, `gensim`. Other libraries require explicit justification in the Method section.
- Random seeds are set once at the top of the notebook so every reported metric is reproducible. State the seed value in the Method section.
- Cells are executed top-to-bottom in order, with output preserved for the hand-in. Re-run before submission so cell numbers are sequential.
- Every figure and table that appears in the paper must trace to a specific notebook cell, named or numbered consistently between paper and notebook.
- Hand-in bundle: the live `.ipynb`, a `.pdf` export of the executed notebook, and the live `.docx`. The PDF export is generated last so it reflects the final cell outputs.
- No raw confidential data ships with the notebook — see section 10.

## 10. Confidentiality

The Maersk help-desk corpus may contain customer-identifiable text and operational details. The default assumption is that the data is confidential.

- Insert at the very top of the front page (already present in the template):

> *"This paper contains confidential information."*

- The CBS confidentiality agreement signed by the four students lives in `admin/cbs_confidentiality_agreement_2021.pdf`.
- No raw ticket text is reproduced verbatim in the paper body. Examples used to illustrate preprocessing or topics are paraphrased or aggressively redacted.
- Customer names, employee names, internal system identifiers, and free-form contact details are removed from any quoted snippet.
- The notebook hand-in does not include the underlying ticket dump. A small synthetic or scrambled sample may be included for reproducibility, with a `data/README.md` explaining the substitution.

If, before submission, the team confirms with Maersk that the data may be released openly, this section can be relaxed and the cover statement removed.

## 11. Group Coordination

Four-author logistics that protect the writing flow:

- One section owner per Heading 1, rotated across the four students. Owners draft and revise; the others review.
- Cover-page authorship list lives in the `<w:sdt>` block on page 1 and is updated only by manual edit in Word, never by automation.
- A short *Indications of individualisation* paragraph at the end of the Method section lists which member led which subsection. CBS counts this paragraph toward both pages and characters.
- The oral exam is 20 minutes per student. The 5-minute opening presentation is shared, but each student must be ready to answer questions on every section, including those they did not draft.

**Data access and notebook execution.** Only **Linus** has access to the raw Maersk OneStream knowledge base and the help-desk ticket export, because the data is confidential and lives on Maersk infrastructure he is employed to operate. All notebooks, scripts, and pipelines in this report folder are designed, parameter-cleaned, and code-reviewed by the rest of the group, then handed to Linus for execution against the live data. The hand-off contract is: every notebook runs end-to-end on a fresh kernel with only `TICKETS_CSV` and `KB_FOLDER` to fill in, no other path edits. Outputs (figures, metrics, exports) flow back to the group for incorporation into the paper. Reproducibility for the examiner is preserved by shipping a synthetic or aggressively redacted sample alongside the notebook (see section 10).

## 12. Self-Check Before Any Draft Is Returned

The draft passes only if every box is ticked:

- [ ] Real corpus, not toy. Maersk OneStream tickets used in the analysis.
- [ ] Three to five NLP methods applied in depth, not summarised.
- [ ] Problem introduction ≤ 1.5 pages.
- [ ] Total length ≤ 15 pages, average ≤ 2,275 characters per page including spaces.
- [ ] Margins ≥ 3 cm top and bottom, ≥ 2 cm left and right; body font ≥ 11 pt.
- [ ] Each of LO1–LO6 demonstrably met.
- [ ] Concept defined precisely on first use; method then applied to data.
- [ ] Every specific number carries a citation.
- [ ] Citation style homogeneous throughout (APA 7).
- [ ] Course textbooks cited directly where relevant; lecture slides only when not covered in syllabus readings.
- [ ] Hyperparameter choices justified at the point each method is introduced.
- [ ] Figures and tables earn their page space; weak ones moved to appendix or removed.
- [ ] Notebook reproduces every metric, figure, and table in the paper from a clean kernel.
- [ ] Random seeds set; cell numbers sequential in the executed notebook hand-in.
- [ ] No raw confidential ticket text in the paper or the notebook hand-in; quoted snippets paraphrased or redacted.
- [ ] Confidentiality statement present on the front page.
- [ ] First person appears only in bounded methodological reflections; no em dash, en dash, semicolon, or colon used as a main-body sentence shortcut; no concept-dropping.
- [ ] Front page carries title, type of paper, all four student numbers, programme, date, instructor, character count, page count.
- [ ] Indications of individualisation paragraph present at the end of the Method section.

If any box fails, flag it explicitly before the team reviews.
