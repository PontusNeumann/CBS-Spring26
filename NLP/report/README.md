# NLP report — folder guide

CBS course KAN-CDSCO1002U, NLP and Text Analytics. Group project: improving intent
routing in the Maersk OneStream help-desk chatbot using a refined knowledge base.

## Where to look first

- **Working in the docx?** Read `Design.md`.
- **Working on the pipeline / notebook?** Read `onestream_nlp_pipeline.md`.
- **Working on paper structure or scope?** Read `planning/paper_scope_and_narrative.md`.
- **AI agent entry point?** Read `CLAUDE.md`.

## Data access and execution

Maersk OneStream KB and ticket data are confidential and live only on **Linus's**
corporate machine. Linus is the only group member with data access and the only one
who runs notebooks against live data. Notebooks parametrise paths so Linus edits
`TICKETS_CSV` and `KB_FOLDER` only. The rest of the group designs, code-reviews, and
drafts the paper.

## Live working files (top-level)

| File | Purpose |
|---|---|
| `KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Spring2026.docx` | Live paper on the CBS template. Page 1 (cover) and page 2 (ToC) are manual-only; body starts on page 3. |
| `KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Intent_Routing.pdf` | Static PDF export of the live docx. Filename predates the rename and is regenerated on next export. |
| `onestream_architecture_comparison.ipynb` | Live analysis notebook. Currently implements the v2 plan, pending refactor to v3 per `onestream_nlp_pipeline.md`. |
| `Design.md` | Visual, structural, and editorial conventions. Binding for any docx edit. |
| `onestream_nlp_pipeline.md` | Pipeline phase plan (v3, KB-driven). Authoritative for what the notebook should do. |
| `CLAUDE.md` | Short pointer doc for AI agents working in this folder. |
| `AGENTS.md` | One-line entry point that defers to `CLAUDE.md`. |

## Subfolders

| Folder | Contents |
|---|---|
| `assets/architecture/` | System architecture spec (`cleaned_architecture.docx` / `.pdf`). |
| `assets/figures/` | Figures rendered for the paper. |
| `literature/` | SLP3 chapter PDFs, Blei (2012), and supplementary readings. |
| `notebooks/` | Superseded analysis notebooks (`onestream_poc.ipynb`, `onestream_pipeline_v1.ipynb`). |
| `planning/` | `paper_scope_and_narrative.md` (current scope), `kasper_call_takeaways.md` (practitioner cross-check), Kasper call source docx, NLP exam plan, early chatbot blueprint. |
| `admin/` | CBS confidentiality agreement and other admin paperwork. |
| `archive/` | Original CBS template and other historical artefacts. |
| `backup/` | Pre-edit snapshots of the live docx, written by the build scripts. |
| `dify_exports/` | Output destination for Dify configuration JSON (populated by notebook on run). |
| `scripts/` | Docx build and patch scripts: `build_initial_nlp_paper.py`, `insert_toc_page.py`, `drop_nlp_old_cover_and_toc.py`, `import_ml_first_two_pages.py`. |

## Pipeline phase plan (current target)

Authoritative source: `onestream_nlp_pipeline.md`. Current phases:

0. Configuration
1. Loading and preprocessing
2. Taxonomy derivation from the KB
3. Selecting K
4. Classifier feature-representation benchmark
5. Keywords per class
6. Architecture comparison (5 architectures including pure RAG and classifier-plus-RAG hybrid)
7. Knowledge-base consolidation
8. Dify export

A future Phase 9, before-and-after spot-check on the deployed Maersk LLM interface,
is specified in `planning/paper_scope_and_narrative.md`.
