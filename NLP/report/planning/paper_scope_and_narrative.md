# Paper Scope and Narrative

**Course.** KAN-CDSCO1002U, NLP and Text Analytics, CBS BADS.
**Deliverable.** Group oral exam product, max 15 pages, 4 authors, Maersk OneStream case.
**Status.** Working scope as of 2026-05-07. Supersedes the v2 unsupervised-first framing in `nlp_paper_chatbot_blueprint.pdf`.
**Reader.** Project group writing the NLP final exam paper.

This document defines what the paper argues, what evidence it presents, and how the work is divided between the methodologically active group and Linus, who holds Maersk data access and runs all live experiments.

---

## 1. The contribution in one paragraph

The Maersk OneStream help-desk chatbot routes user tickets to a knowledge base. An earlier classifier-on-tickets pipeline produced poor routing because the taxonomy and the chunking did not reflect what the KB can actually answer. This paper restructures the knowledge base, benchmarks five NLP approaches against the refined KB, and tests whether retrieval-augmented generation can replace classifier-based routing entirely. Statistical retrieval results from the bench are paired with before-and-after spot-checks on the production Maersk LLM interface, holding the model constant and varying only the knowledge base. The headline finding the paper aims to defend: **knowledge-base quality, not model sophistication, is the dominant lever for help-desk answer quality.** The contribution generalises beyond Maersk. When the KB is well structured, simpler retrieval beats heavier classification, and KB curation is a higher-leverage intervention than model selection.

---

## 2. The three-act structure

### Act 1, Why the original approach failed (1.5 to 2 pages)

A short, honest account of the v1/v2 pipeline and its diagnosis. Three structural problems:

1. Taxonomy inferred from user complaints (tickets) rather than from what the KB can answer. Routing produced classes with no real KB sections behind them.
2. Knowledge-base documents were unstructured, with chunking that left heading-only fragments cut off from their explanatory body (Kasper's "S4 entities" example).
3. Multi-class membership ignored. Genuinely cross-cutting documents were forced into one bucket.

Anchors LO6 (critical methodological reflection) and motivates the pivot.

### Act 2, Model bench on the refined knowledge base (5 to 6 pages, methodological core)

Six approaches benchmarked on the same refined KB with the same evaluation harness.

| ID | Approach | Course anchor |
|---|---|---|
| A | Naïve Bayes baseline on KB-derived classes | SLP3 Appendix B, Lecture 4 |
| B | TF-IDF + Logistic Regression | SLP3 Ch. 5, Lecture 5 |
| C | Word2Vec averaged + LR | SLP3 Ch. 6, Lecture 8 |
| D | Sentence-BERT + LR | SLP3 Ch. 8, Lecture 8 / 10 |
| E | Pure RAG, dense retrieval over chunked KB | Lewis et al. (2020), Lecture 11 |
| F | Classifier + RAG hybrid | Lecture 4 / 5 + Lecture 11 |

**Why all six.** Curriculum coverage for the oral defence (Lectures 4, 5, 8, 10, 11). Honest comparison so the eventual recommendation rests on numbers, not preference.

**Reported metrics.** Macro-F1 on KB held-out, retrieval@1, retrieval@3, retrieval@10 on a held-out ticket sample, with 95% bootstrap confidence intervals. Per-class F1 reported separately so class imbalance is visible.

**Output.** A single comparison table and a single retrieval@k figure. The pipeline notebook sits behind these results as a methodological appendix, not as the centrepiece of the paper.

### Act 3, Does the refined KB make the deployed Maersk LLM better? (3 to 4 pages, headline contribution)

The experiment that makes the paper's claim concrete and visible.

**Design.** Hold the deployed Maersk LLM constant. Vary only the knowledge base it points at. Measure the change.

Three streams of evidence:

1. **Statistical retrieval comparison from Act 2.** Retrieval@k for the two best architectures on the held-out ticket sample, computed against the old KB and against the refined KB.
2. **Production-interface spot-checks (the new piece).** A fixed set of 15 to 20 representative tickets is run through the deployed Maersk chatbot twice, once with the old KB and once with the refined KB. For each run, the trace records:
   - the document the LLM retrieved,
   - the chunk it surfaced,
   - the final answer or the no-answer state,
   - a short qualitative grade (correct, partial, wrong, no-answer).

   Side-by-side presentation in the paper. Linus has already piloted this protocol on a single sample question, where the LLM found the correct file but failed to extract the answer because the relevant chunk did not contain the keyword. That single example becomes the motivating vignette for the experiment.
3. **Our own RAG vs the best classifier-based architecture on the refined KB.** Independent corroboration that the effect visible in the Maersk interface also shows up in a clean re-implementation. This isolates whether the gain comes from the KB alone or from the LLM-plus-KB combination.

**Why three streams.** Statistics tell the examiner the effect exists. Traces show *why* it matters operationally. The independent re-implementation shows the effect is not an artefact of Maersk's specific deployment.

The combination is the paper's claim: refined KB drives the improvement, and the improvement is reproducible outside Maersk's stack.

---

## 3. Spot-check methodology (Act 3 detail)

### Query set

15 to 20 tickets sampled from the held-out evaluation set, stratified across the KB-derived classes so every class is represented. Sampling protocol logged in the notebook so the set is reproducible. A short rationale is recorded for each ticket explaining why it was chosen (clarity, ambiguity, multi-document, edge case).

### Run protocol per ticket

Each ticket is submitted to the Maersk chatbot interface in two configurations:

1. **Before**, KB pointer set to the original unrefined corpus.
2. **After**, KB pointer set to the refined md-only chunked corpus.

For each run, the following is captured:

- Ticket ID and ticket text (paraphrased in the paper for confidentiality).
- Retrieved document name.
- Retrieved chunk text or the chunk excerpt the LLM cites.
- LLM final response or no-answer state.
- Latency, if the interface exposes it.
- A two-axis grade: factual correctness (correct, partial, wrong, none) and operational usefulness (resolved, partially resolved, escalation needed).

Grading is done by Linus with cross-review by one other group member. Any disagreement is logged.

### Reporting in the paper

A compact table with one row per ticket and two columns (before, after) showing the grade pair, plus aggregate counts: how many tickets moved from no-answer to answered, from wrong to correct, from partial to correct, and the rare regressions. Two or three illustrative traces are reproduced in full as figures, including the "S4 entities" style chunk-cut-off case Linus already documented.

The aggregate counts pair with the Act 2 retrieval@k numbers to give the examiner both a statistical and a qualitative picture.

### Confidentiality

Ticket text is paraphrased before any reproduction in the paper. Customer names, employee names, internal identifiers, and free-form contact details are removed. The cover statement on the report flags the dataset as confidential. See `Design.md` section 10.

---

## 4. Scope discipline, what is in and what is out

| In scope | Out of scope |
|---|---|
| KB refinement (md-only, heading-aware chunking) | Building a new LLM or fine-tuning Maersk's |
| Six-way model bench on the refined KB | Pre-training or large-scale fine-tuning |
| Retrieval@k statistical evaluation with bootstrap CIs | A/B testing in production |
| Before-and-after spot-checks via the deployed Maersk interface | Quantitative latency or cost benchmarking unless the interface exposes it cleanly |
| Independent RAG re-implementation for corroboration | A full alternative chatbot product |
| Critical reflection on chunking, embedding model, and routing trade-offs (LO6) | A standalone literature review chapter (concepts introduced at point of use, per `Design.md`) |
| Recommendations for Maersk routing and KB curation | Recommendations on Maersk vendor or model contracts |

If a section does not serve at least one of the six learning objectives in `Design.md` section 3, it is cut.

---

## 5. Pipeline status and what each member runs

The pipeline notebook (`onestream_architecture_comparison.ipynb`) currently implements v2 (taxonomy from tickets, no chunking, fake RAG) and is **not ready** to send to Linus. The targeted refactor:

1. Phase 1, add documented chunking with a quality sanity table.
2. Phases 2 to 4, swap the training corpus from tickets to KB documents with KB-derived labels.
3. Phase 6, replace the doc-level cosine "RAG" with a real chunk-level retrieval, top-k, optional re-ranker, LLM call.
4. Phase 9, new, the spot-check harness and result table.

A separate **stand-alone spot-check notebook** for Phase 9 can be drafted first so Linus can run the before-and-after experiment on his machine while the rest of the pipeline is being refactored. This parallelises the data-access bottleneck.

**Hand-off contract.** Linus edits only path placeholders (`TICKETS_CSV`, `KB_FOLDER_OLD`, `KB_FOLDER_NEW`) and an optional API key, then runs end-to-end. Outputs (figures, metrics, exports, spot-check trace tables) flow back to the rest of the group for paper incorporation.

---

## 6. Citations to carry into the report

Anchored in the course backbone where possible, with primary literature where SLP3 does not cover the topic.

| Reference | Used in | Source |
|---|---|---|
| SLP3 Ch. 2 | Phase 1 preprocessing, Act 1 diagnosis | Course textbook |
| SLP3 Ch. 4, Appendix B | Approaches A, B (classifier baselines) | Course textbook |
| SLP3 Ch. 5 | Approach C (word vectors), TF-IDF keywords | Course textbook |
| SLP3 Ch. 8 | Approach D (Sentence-BERT), contextual vs static distinction | Course textbook |
| SLP3 Ch. 11 (to be sourced) | Approach E (RAG), Approach F (hybrid), Act 3 framing | Course textbook (chapter not yet in `literature/`) |
| Lewis et al. (2020), RAG | Approach E framing | NeurIPS 2020 |
| Reimers and Gurevych (2019), Sentence-BERT | Approach D | EMNLP 2019 |
| Pradeep et al. (2023), RankZephyr | Re-ranker stage | arXiv |
| Auer et al. (2024), Docling | Phase 1 ingestion if used | arXiv:2408.09869 |
| OpenAI (2024), text-embedding-3 | Approach D variant if API access available | Vendor announcement |
| Anthropic (2024), Contextual Retrieval | Optional ablation in Approach E | Vendor blog |
| Johnson, Douze, Jégou (2019), FAISS | Vector index, only if FAISS is the deployed backend | IEEE TBD |
| Manning (2022), Daedalus | Introduction framing | Course reading |

The Kasper takeaways in `planning/kasper_call_takeaways.md` distil practitioner input on each of these and flag where SLP3 does not cover a claim.

---

## 7. Threats to validity

**KB selection bias.** A refined KB authored by the group (or by a Maersk content owner) reflects what the curator thought users would ask, not what users actually ask. Mitigation: the held-out ticket sample tests coverage. Tickets that no architecture can route in either KB are flagged as candidate KB gaps rather than counted as model failures.

**Spot-check sample size.** 15 to 20 tickets is small. Mitigation: stratify the sample, report per-class outcomes, anchor the qualitative evidence to the larger statistical evaluation in Act 2 rather than letting it stand alone.

**Production-interface confound.** The Maersk LLM may have caching, system prompts, or guard-rails that are not visible in the trace. Mitigation: run before-and-after queries in interleaved order, document interface settings, and re-run a subsample on the independent re-implementation in stream 3 of Act 3.

**Embedding-model confound in retrieval gold.** If retrieval gold is defined by the same embedding model used for retrieval, the evaluation is circular. Mitigation: gold is defined at the KB folder level (any document in the correct folder is a hit), per `onestream_nlp_pipeline.md` Phase 6.

**Cherry-picked vignettes.** The "S4 entities" chunk-cut-off case is illustrative. To avoid bias, illustrative traces in the paper are sampled blind from a pre-registered shortlist, not selected after grading.

---

## 8. Open questions to resolve before the bench is final

1. Is the Maersk KB redundant in Kasper's "stock report" sense (many near-template documents)? If yes, retrieval k must be higher.
2. What is the largest document in the refined KB? Chunking strategy needs sizing.
3. Does the Maersk interface expose retrieval traces in a copyable form, or does the team need screenshots? Trace fidelity affects what evidence Act 3 can present.
4. Is OpenAI embedding API available on Linus's corporate machine? If not, Approach D variants are limited to local Sentence-BERT.
5. Final query set for the spot-check: 15, 20, or larger? Decision logged before any run.

---

## 9. What this document is not

This file is not the paper. It is a planning artefact that helps the group converge before drafting. The cover, ToC, and front-page rules in `Design.md` remain authoritative for the deliverable. Any change to scope is reflected here first, then propagated to `onestream_nlp_pipeline.md` and the docx.
