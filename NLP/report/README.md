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
| `onestream_pipeline_v3.ipynb` | Live analysis notebook (Codex build). Implements the eight-phase v3 plan from `onestream_nlp_pipeline.md`. |
| `onestream_spot_check.ipynb` | Stand-alone semi-manual notebook for the Maersk LLM before-and-after experiment (Codex build). |
| `Design.md` | Visual, structural, and editorial conventions. Binding for any docx edit. |
| `onestream_nlp_pipeline.md` | Pipeline phase plan (v3, KB-driven). Authoritative for what the notebooks should do. |
| `CLAUDE.md` | Short pointer doc for AI agents working in this folder. |
| `AGENTS.md` | One-line entry point that defers to `CLAUDE.md`. |

## Subfolders

| Folder | Contents |
|---|---|
| `assets/architecture/` | System architecture spec (`cleaned_architecture.docx` / `.pdf`). |
| `assets/figures/` | Figures rendered for the paper. |
| `literature/` | SLP3 chapter PDFs, Blei (2012), and supplementary readings. |
| `notebooks/` | Superseded notebooks: `onestream_architecture_comparison_v2.ipynb` (the pre-refactor v2 build), `onestream_poc.ipynb`, `onestream_pipeline_v1.ipynb`. |
| `planning/` | `paper_scope_and_narrative.md` (current scope), `kasper_call_takeaways.md` (practitioner cross-check), `codex_handoff_notes.md` (Codex's notebook hand-off summary). |
| `claude_implementation/` | Parallel Claude build of the v3 pipeline, fully isolated. Contains its own notebooks, builder script, `dify_exports/`, and `figures/`. Run independently of the Codex notebooks for comparison. |
| `admin/` | CBS confidentiality agreement and other admin paperwork. |
| `archive/` | Original CBS template and other historical artefacts. |
| `backup/` | Pre-edit snapshots of the live docx and superseded planning artefacts (`nlp_exam_plan.docx`, `nlp_paper_chatbot_blueprint.pdf`). |
| `dify_exports/` | Codex notebook outputs: phase CSVs, `dify_config.json`, `runtime_metadata.json`, spot-check trace CSVs. |
| `scripts/` | Docx build and patch scripts plus the v3 KB refiner: `build_initial_nlp_paper.py`, `insert_toc_page.py`, `drop_nlp_old_cover_and_toc.py`, `import_ml_first_two_pages.py`, `refine_md_kb.py`. |

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
