# NLP report — folder guide

CBS course KAN-CDSCO1002U, NLP and Text Analytics. Group project: improving intent
routing in a production help-desk chatbot using Maersk OneStream tickets.

## Live working files (top-level)

| File | Purpose |
|---|---|
| `KAN-CDSCO1002U_161989_160363_185912_160714_OneStream_Intent_Routing.docx` | Live paper on the CBS template. Cover (page 1) is manual-only; ToC (page 2) is a Word field that auto-populates on open; body sections start on page 3. |
| `nlp_paper.docx` | Earlier 29 KB draft kept for reference until its content is migrated into the templated docx. |
| `onestream_nlp_pipeline.ipynb` | Final analysis notebook (7 phases, unsupervised-first). |
| `onestream_nlp_pipeline.pdf` | Static export of the notebook for review. |
| `Design.md` | Visual, structural, and editorial conventions. Read before editing the docx. |

## Subfolders

| Folder | Contents |
|---|---|
| `assets/architecture/` | System architecture spec (`cleaned_architecture.docx` / `.pdf`). |
| `assets/figures/` | Figures rendered for the paper. |
| `assets/sources/` | Cited PDFs and source material. |
| `notebooks/` | Superseded analysis (`onestream_poc.ipynb`, `onestream_pipeline_v1.ipynb`). |
| `planning/` | Exam plan, early chatbot blueprint. |
| `admin/` | CBS confidentiality agreement and other admin paperwork. |
| `archive/` | Original CBS template and other historical artefacts. |
| `backup/` | Pre-edit snapshots of the live docx, written by the build scripts. |
| `scripts/` | Re-runnable docx patchers (`build_initial_nlp_paper.py`, `insert_toc_page.py`). |

## Pipeline phases (in `onestream_nlp_pipeline.ipynb`)

1. **0A** Installation and imports
2. **0B** Data loading
3. **1** Preprocessing
4. **2** Exploratory NLP
5. **3** LDA topic discovery (+ checkpoint to label topics)
6. **4** Keyword extraction (TF-IDF + logistic regression on LDA labels)
7. **5** Semantic keyword expansion (word2vec)
8. **6** Dify export and routing recommendations
