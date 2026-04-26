# Design System — DPD Final Exam

Visual, structural, and editorial conventions for the *Data, Platforms and Digitalization* (KAN-CDSCO2401U) individual exam paper. The deliverable is a Word document exported to PDF, max 10 pages, individually written, graded only against the four Learning Objectives (LO1–LO4).

This file inherits the **CBS Formal Requirements** from `../../Design.md` (repo root) and the **Word output pipeline / paragraph spacing rules** from `../../ML/report/Design.md`. Anything below either reaffirms a binding inherited rule or overrides it for the DPD-specific context.

## 1. Inherited Rules — Binding

From `../../Design.md`:

- Max 2,275 characters per page on average, including spaces.
- Top and bottom margin at least 3 cm; left and right at least 2 cm.
- Body font at least 11 pt. Headlines graphically distinct from body text.
- Page numbers required. Table of contents lists main section starts.
- Tables, figures, and illustrations occupy page space but do not count toward characters. They do not justify exceeding the page maximum.
- Front page fields: title, type of paper, student number, programme name, date, supervisor (if any), character and page count, confidentiality status (if applicable).
- Plagiarism and reference handling per CBS Library guidance.

From `../../ML/report/Design.md`:

- Word output pipeline, paragraph-spacing rules, and keep-together rules (zero auto spacing, blank `Normal` paragraphs as separators, `cantSplit` and `keepNext` on tables and figure captions, `pageBreakBefore` on `References` and `Appendix`).
- Caption format in Word: *Figure N. Descriptive caption text.* and *Table N. Descriptive caption text.*

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

1. Front page (excluded from page and character counts).
2. Table of contents (counts toward pages, not characters).
3. Introduction and problem statement — ≤ 1 page.
4. Brief scope note — which parts of which frameworks will be used and why.
5. Integrated analysis — theory and case woven together, covering economic, managerial, and societal dimensions.
6. Recommendations — concrete, justified, addressing trade-offs.
7. Conclusion, reflection, and limitations.
8. References (page break before; counts toward neither pages nor characters).
9. Appendix, if any (page break before; counts toward neither).

## 5. Writing Conventions

- **Language:** UK English, academic tone, clear and concise.
- **Voice:** No first person — avoid *I*, *we*, *our*. Use the passive or recast in third person.
- **Punctuation:** No em dash and no en dash used as a sentence connector. Use a comma, colon, semicolon, or recast.
- **Sentence rhythm:** Lead with the answer, then explain. Short sentences with selective descriptive word choices.
- **Headings:** Sentence case. Heading 1 for top-level sections, Heading 2 for subsections, Heading 3 sparingly.
- **Spacing:** No blank spaces inside paragraphs to create rhythm — use full blank rows in `Normal` style only, per the inherited paragraph-spacing rules.
- **Concept handling:** Define a term precisely on first use, then pivot to application. The audience is the examiner, who already knows the literature.
- **No concept-dropping:** A term named is a term applied. If a concept appears only as a label, remove it.

## 6. How to Apply Theory (binding pattern)

For every theoretical move in the paper, follow this shape:

> "In light of \[theory], the \[mechanism / property] present in \[case] is X. This influences \[specific behaviour or outcome] as Y. The implication for the firm is Z."

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
- Tables are native Word tables for anything up to roughly ten rows. Grid lines 0.5 pt black, header row bold, no fill, numeric columns centred.
- Captions: *Table N. Descriptive caption text.*
- Figures and tables are numbered independently, both starting at 1, in order of appearance.
- A figure or table that does not say more than the prose it displaces is moved to the appendix or removed.

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
- [ ] No first person, no em dash, no concept-dropping.
- [ ] Confidentiality statement added if applicable.
- [ ] Front page carries title, type of paper, student number, programme, date, supervisor (if any), character count, page count.

If any box fails, flag it explicitly before the student reviews.
