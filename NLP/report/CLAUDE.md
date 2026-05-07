# CLAUDE.md — fast localisation for AI agents in `report/`

CBS NLP final exam project on Maersk OneStream help-desk routing. Group of four. Oral
exam based on a max-15-page written product. Confidential data.

## Read these first

| File | What it says |
|---|---|
| `Design.md` | Formatting, page rules, citation style, group coordination, confidentiality. **Binding for any docx edit.** |
| `onestream_nlp_pipeline.md` | Pipeline phase plan (v3). What the notebook should do. |
| `planning/paper_scope_and_narrative.md` | Three-act paper plan, Maersk LLM spot-check methodology, scope. **The current source of truth for what the paper is about.** |
| `planning/kasper_call_takeaways.md` | Practitioner notes on RAG (chunking, embeddings, k, re-ranking) cross-checked with SLP3. |
| `README.md` | Folder map. |

## Hard rules

- **Pages 1 and 2 of the live docx are manual-only.** Never touch the cover, the ToC, anchored drawings, the TOCHeading paragraph, the TOC field, or the Introduction `pageBreakBefore`.
- **Data lives only on Linus's machine.** Notebooks parametrise paths via `TICKETS_CSV` and `KB_FOLDER`. Do not invent ticket text or KB content.
- **No em dashes, no en dashes, no semicolons or colons as sentence shortcuts in body prose.** Per `Design.md` section 5.
- **15-page hard cap. Problem introduction ≤ 1.5 pages. 3 to 5 NLP methods applied in depth.**
- Citation style is **APA 7**. Course backbone is **SLP3** (Jurafsky & Martin) and Manning & Schütze (2003).

## Live filenames

- Live docx: `KAN-CDSCO1002U_161989_160363_185912_160714_NLP_Spring2026.docx`
- Live pipeline notebook: `onestream_pipeline_v3.ipynb` (Codex build)
- Live spot-check notebook: `onestream_spot_check.ipynb` (Codex build)
- Parallel Claude build: `claude_implementation/claude_pipeline_v3.ipynb` and `claude_spot_check.ipynb`, fully isolated
- KB refiner script: `scripts/refine_md_kb.py`
- Backup snapshots: `backup/`

## Who runs what

- **Linus** runs every notebook against live data on his Maersk machine.
- The rest of the group designs, code-reviews, and drafts the paper.
- Hand-off contract: notebooks run end-to-end with only path placeholders to fill in.
