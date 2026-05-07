# Kasper Call, RAG Practitioner Takeaways (Cross-Checked Against Literature)

**Source.** Internal call with a practitioner (Kasper) on building RAG systems for enterprise customer support, dated 2026-05-06. The original transcript docx (`planning/Kasper call on some ideas.docx`) was retired after distillation into this file; recoverable from git history at commit `c19c2b0` if a verbatim quote is ever needed.
**Purpose.** Distil the operationally useful guidance from the call and anchor each point to course or primary literature so the report can cite it.
**Reader.** Project group writing the NLP final exam paper on Maersk customer support automation.

---

## 1. Reference RAG architecture (Kasper's pipeline)

1. Documents are chunked, embedded, and stored in a vector database.
2. The query is embedded with the **same** embedding model used for the documents.
3. k-nearest-neighbour retrieval returns candidate chunks.
4. A re-ranker reorders the candidates, and a small cutoff (top-3 in Kasper's default) is applied.
5. The top chunks plus the query are passed to an LLM.

**Literature check.** SLP3 Ch. 7 §7.7 forward-references RAG to SLP3 Ch. 11, which is not in our literature folder, so the textbook does not anchor this architecture in the chapters available. The report should cite **Lewis et al. (2020)** for the RAG pattern itself, already in `onestream_nlp_pipeline.md` Phase 6 references.

## 2. Choosing the retrieval cutoff k

**Kasper.** No academically rigorous procedure, trial-and-error per knowledge base. Knowledge bases with many near-duplicate documents (e.g. 50 stock reports sharing a template, only numbers and tickers differ) force a higher k because everything is roughly equally relevant.

**Literature check.** SLP3 Ch. 7 and Ch. 8 use "top-k" for token-level decoding, not retrieval cutoff. SLP3 Ch. 11 (not in the folder) is the place where retrieval-side k is discussed.

**Implication for the project.** Sweep k on a held-out ticket sample and report the curve. Do not pick k by intuition.

## 3. Re-ranking after first-pass retrieval

**Kasper.** A re-ranker reorders the kNN output, and a small top-N (e.g. 3) is fed to the LLM. First-pass kNN is not always sorted by true relevance.

**Literature check.** Not covered in SLP3 chapters 5, 7, 8, or 9 in the folder. Cite **Pradeep et al. (2023)** for listwise re-ranking, already a Phase 6E reference. This positions Phase 6E of the v3 pipeline as a course-aligned design choice, not only an industry pattern.

## 4. Chunking quality dominates retrieval quality

**Kasper.** When the classifier was removed and pure RAG was run, retrieval found the *correct* document, but answers were poor because chunks were cut off. Example: the heading "S4 entities" was chunked alone, without the body description below it, because the description did not repeat the keyword.

**Literature check.** SLP3 chapters 5, 7, 8, 9 in the folder do not discuss chunking strategy explicitly.

**Implication for the project.** Chunking strategy is a first-class methodological choice and must be reported with the same rigour as classifier choice. Heading-only chunks must be merged with the body that follows.

**Tool flagged by Kasper.** **Docling** (IBM Research, 2024, Auer et al., arXiv:2408.09869). Open-source document parser that preserves layout, reading order, tables, and formulas during PDF / DOCX conversion to structured text. Addresses the exact failure mode Kasper described. Spelling is **Docling**, one *k*.

## 5. Embedding model is a key quality lever

**Kasper.** The embeddings model determines how well chunk content is represented. He uses **OpenAI text-embedding-3-small** or **text-embedding-3-large**.

**Literature check.** SLP3 Ch. 5 covers static embeddings (Word2Vec, GloVe, fastText), and §5.10 states the chapter is about *static* embeddings. Contextual embeddings (BERT) are introduced in SLP3 Ch. 8 §8.1. Sentence-level embeddings used in retrieval (Sentence-BERT, OpenAI embedding API) are not covered in the chapters available.

**Tool facts (OpenAI announcement, 25 January 2024).**

| Model | Native dim | MTEB avg | MIRACL | Price (USD per 1M tokens) |
|---|---|---|---|---|
| `text-embedding-3-small` | 1536 | 62.3% | n/a | 0.02 |
| `text-embedding-3-large` | 3072 | 64.6% | 54.9% | 0.13 |
| `text-embedding-ada-002` (legacy) | 1536 | 61.0% | 31.4% | 0.10 |

Both v3 models support Matryoshka-style dimension truncation via the `dimensions` API parameter, so a 256-dim cut of `-3-small` still beats full-size ada-002 on MTEB.

For the report, cite **Reimers and Gurevych (2019)** for sentence embeddings as a methodological foundation, and cite the **OpenAI announcement** for the deployed model.

## 6. Drop the classifier if retrieval already finds the right chunk

**Kasper.** If a no-classifier RAG flow retrieves the correct chunks already, the classifier adds little. The bottleneck observed at Maersk was chunking and embedding quality, not routing.

**Literature check.** SLP3 does not present a classifier-versus-retrieval taxonomy in the chapters available. The classical chatbot taxonomy (rule-based, IR-based, frame-based) lives in SLP3 Ch. 11 / Ch. 24 depending on edition, neither of which is in the folder.

**Implication for the project.** This is precisely the comparison Phase 6 of the pipeline already runs: Arch A (flat classifier), Arch D (no-classifier RAG), Arch E (classifier plus re-ranker). Kasper's first-hand experience supports the *empirical* framing of Phase 6, do not assume a classifier helps. Report retrieval@k for all three on the held-out ticket sample.

## 7. Contextual / hierarchical embeddings

**Kasper.** Mentioned "Cohere" contextual or hierarchical embeddings as a more context-aware alternative. He had not personally tested them but his CTO had.

**Clarification.** The technique is most likely **Anthropic's Contextual Retrieval** (19 September 2024), not a Cohere release. The confusion is reasonable because Anthropic's reference pipeline uses Cohere's `rerank-english-v3.0` for the re-ranking stage. The technique prepends a 50 to 100 token LLM-generated summary situating each chunk in its parent document *before* embedding and BM25 indexing.

**Headline claim (Anthropic, internal benchmarks, top-20 retrieval failure rate).**

- Contextual embeddings alone, 35% reduction (5.7% to 3.7%).
- Contextual embeddings + contextual BM25, 49% reduction (to 2.9%).
- Plus re-ranking, 67% reduction (to 1.9%).

**Literature check.** SLP3 Ch. 5 line 47 and Ch. 8 §8.1 cover the static-versus-contextual distinction (Word2Vec context-free vs transformer contextual). Hierarchical or document-structure-aware embeddings are not discussed in the available chapters.

**Status.** Vendor blog post, not peer-reviewed. Citable for industry context, not as a methodological foundation.

## 8. Vector database / nearest-neighbour search

**Kasper.** Different vector databases, different sizes. Query and document embeddings must come from the same model.

**Literature check.** SLP3 chapters 5, 7, 8, 9 in the folder do not cover approximate nearest-neighbour search, FAISS, or vector databases. SLP3 Ch. 5 references Manning, Raghavan, Schütze (2008) *Introduction to Information Retrieval* for retrieval-side detail. For the report, cite **Johnson, Douze, Jégou (2019)**, "Billion-scale similarity search with GPUs", only if FAISS is the deployed index.

---

## What this changes for the pipeline

| Pipeline phase | Kasper's input | Action |
|---|---|---|
| Phase 1, Loading | Chunking quality is first-class | Add a chunking-strategy sub-step, pilot Docling for the Maersk KB ingestion |
| Phase 4, Classifier benchmark | Use OpenAI text-embedding-3 as a retrieval embedding alongside Sentence-BERT | Add a 4G row: `text-embedding-3-small` + Logistic Regression |
| Phase 6, Architecture comparison | Drop-the-classifier hypothesis confirmed empirically by Kasper's testing | Frame Arch A vs Arch D vs Arch E as the central empirical question of the report |
| Phase 6E, Re-ranker | Top-3 cutoff after k=10 retrieval is Kasper's default | Adopt this as a tested baseline, then sweep around it |
| New, Contextual Retrieval | Optional ablation | If time allows, add an Arch F ablation: Anthropic-style contextual chunks plus Arch D |

---

## Citations to add or confirm in the report

| Reference | Stage | Status |
|---|---|---|
| Lewis et al. (2020), RAG | Phase 6D | Already cited |
| Pradeep et al. (2023), RankZephyr | Phase 6E | Already cited |
| Reimers and Gurevych (2019), Sentence-BERT | Phase 4F, Phase 6 | Already cited |
| Auer et al. (2024), Docling Technical Report (arXiv:2408.09869) | Phase 1 ingestion | New, candidate |
| OpenAI (2024), New embedding models and API updates | Phase 4G, Phase 6 | New, candidate |
| Anthropic (2024), Introducing Contextual Retrieval | Phase 6 ablation | New, optional, vendor source |
| Johnson, Douze, Jégou (2019), Billion-scale similarity search | Phase 6 retrieval | New, only if FAISS is used |

---

## Open questions raised by the call

1. Is the Maersk KB redundant in Kasper's "stock report" sense (many near-template documents)? If yes, Phase 6 needs a higher k floor.
2. What is the largest document in the Maersk KB? Kasper noted some Maersk documents are 2,000+ pages, chunking strategy matters more on long documents.
3. Does the Dify deployment expose an embedding-model swap? Without that, the embedding-model benchmark in Phase 4 is hypothetical for production.
4. Is there budget for OpenAI embedding API calls at the corpus scale? At 0.02 USD per million tokens for `text-embedding-3-small`, even a 1M-token KB embedding costs 0.02 USD, so this is not a cost barrier.

---

## Primary sources

- Auer, C., Antognini, D., Lysak, M., et al. (2024). *Docling Technical Report*. arXiv:2408.09869. https://arxiv.org/abs/2408.09869
- Anthropic (2024, September 19). *Introducing Contextual Retrieval*. https://www.anthropic.com/news/contextual-retrieval
- OpenAI (2024, January 25). *New embedding models and API updates*. https://openai.com/index/new-embedding-models-and-api-updates/
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.
- Pradeep, R., et al. (2023). RankZephyr: Effective and robust zero-shot listwise reranking. arXiv:2312.02724.
- Reimers, N., and Gurevych, I. (2019). Sentence-BERT. *EMNLP 2019*.
- Johnson, J., Douze, M., and Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*.
