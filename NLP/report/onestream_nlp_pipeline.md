# OneStream Help-Desk NLP Pipeline — v3 (KB-Driven)

**Course.** KAN-CDSCO1002U — Natural Language Processing and Text Analytics, CBS BADS.
**Companion notebooks.** `report/onestream_pipeline_v3.ipynb` (live Codex build) and `report/onestream_spot_check.ipynb` (Maersk LLM before-and-after). A parallel Claude build sits in `report/claude_implementation/`.
**Companion notes.** `planning/kasper_call_takeaways.md` — practitioner takeaways from the Kasper call cross-checked against SLP3 and primary literature. Covers chunking, embedding-model choice, k-selection, and the no-classifier-RAG argument that motivates Phase 6. Read before any change to Phases 1, 4, or 6.
**Paper scope.** `planning/paper_scope_and_narrative.md` — three-act paper plan, model bench, before-and-after Maersk LLM spot-check methodology, hand-off contract with Linus. Read before refactoring any phase, since some current phases are slated to move from headline to appendix or to be replaced.
**Data access.** The raw Maersk KB and ticket file are confidential and accessible only to **Linus** (Maersk-employed group member). Every executable phase below is designed by the group, then run by Linus on his corporate machine against the real data. Path placeholders (`TICKETS_CSV`, `KB_FOLDER`) are the only fields Linus edits before a run. Outputs return to the group for paper incorporation.
**Status.** Plan document. Updates from v2 are flagged in each section.

---

## What changed from v2

v2 was *unsupervised-first*: LDA discovered topics from ticket text, those pseudo-labels trained a classifier, and KB documents were assigned to classes by argmax cosine similarity. Three structural problems surfaced during testing:

1. The taxonomy was inferred from user complaints, not from what the KB can answer. Routing produced classes with no real KB sections behind them.
2. Change-request tickets were dropped at preprocessing as non-FAQ-answerable. They are a real intent the bot must serve.
3. Each KB document was assigned to one class. Documents that genuinely span topics were patched in afterwards as "bridge documents" rather than treated as multi-class members from the start.

v3 reframes the problem: **the KB defines the taxonomy. Tickets are routing queries.** This matches the production scenario and aligns the design with supervised text classification (SLP3 Ch. 4 and Ch. 5) rather than unsupervised topic discovery on the wrong artefact.

---

## Phase 0 — Configuration

Two additions to v2 config.

```python
# KB structure flag
KB_HIERARCHY = 'folder'       # 'folder' | 'top_heading' | 'cluster'

# Metadata KB is tagged, not excluded
KB_METADATA_PATTERNS = [r'metadata']

# ID and entity normalisation table — empty by default;
# fill with domain rules confirmed by content owners.
ID_NORMALISATION_RULES = {
    r'\bTAX[-_ ]?\d+\b':  'tax_code',
    r'\bCCY[-_ ]?\d+\b':  'currency_code',
    r'\bENT[-_ ]?\d+\b':  'entity_code',
    r'\bWF[-_ ]?\d+\b':   'workflow_step',
}

# Special non-KB classes
SPECIAL_CLASSES = {
    'access_change_request': {
        'patterns': [
            r'\bchange[\s_]?request\b',
            r'\bCR[-\s]?\d+\b',
            r'\baccess\s+(to|for)\b',
            r'\brole\s+access\b',
            r'\brequest(ing)?\s+access\b',
        ],
        'redirect_url': 'https://maersk-access-portal.example/...',
        'kb_documents': []
    }
}
```

**Reference.** SLP3 Ch. 2 — Basic Text Processing.

---

## Phase 1 — Loading and preprocessing

Three behavioural changes from v2.

1. **Change-request tickets are not dropped.** A ticket matching `SPECIAL_CLASSES['access_change_request']['patterns']` gets `intent = 'access_change_request'` and is held out of topic modelling. It still flows to evaluation as a labelled sample for the access-redirect class.
2. **Metadata KB is loaded and tagged**, not excluded. The metadata file is dropped from the topic-model corpus only. It is preserved in `kb_documents` with `is_metadata=True` and exported to the runtime metadata lookup in Phase 8.
3. **ID and entity normalisation runs before tokenisation**, replacing systematic codes (`TAX-NNN`, `ENT-NNN`) with category tokens (`tax_code`, `entity_code`). Optional. Skip if codes are not systematic.

```python
def normalise_ids(text):
    for pat, repl in ID_NORMALISATION_RULES.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    return text
```

**References.** SLP3 Ch. 2 — preprocessing pipeline. Manning & Schütze (2003) ch. 4 retained for statistical foundations of token co-occurrence.

---

## Phase 2 — Taxonomy derivation from the KB

v2 ran LDA on `ticket_tokens`. v3 derives classes from the KB. Three options, ordered by quality.

| Option | Method | When to use |
|---|---|---|
| 2A | KB folder hierarchy. Top-level folder name becomes the class label. | When the KB already has a curated taxonomy. This is the preferred path. |
| 2B | Markdown top-heading parsing. Cluster the H1s of every KB document. | When the folder structure is flat. |
| 2C | LDA / BERTopic / KMeans+SBERT on `kb_tokens`. | When neither folder nor heading structure is reliable. |

The k-sweep + coherence + classifier-F1 logic from v2 Phase 3 is retained but its corpus is now the KB, not tickets.

The `access_change_request` class is added to the discovered set. It carries no KB documents.

**References.** SLP3 Ch. 9 — chatbot design and intent inventory. Blei (2012) — LDA primer for option 2C. Grootendorst (2022) — BERTopic.

---

## Phase 3 — Selecting K

Joint coherence + classifier macro-F1 sweep, unchanged in spirit from v2.

Change in v3: F1 is measured by training a classifier on **KB documents** with their derived class labels and validating on a held-out KB slice. K is the largest value where coherence and KB-side macro-F1 both stay near their maxima.

**Reference.** SLP3 Ch. 4 — multinomial classification setup.

---

## Phase 4 — Classifier feature-representation benchmark

Six representations on the same labels, same 5-fold stratified cross-validation, same Logistic Regression head where applicable.

| ID | Representation | Citation |
|---|---|---|
| 4A | Bag-of-Words + Multinomial Naïve Bayes | SLP3 Appendix B |
| 4B | TF-IDF unigrams + Logistic Regression | SLP3 Ch. 5 |
| 4C | TF-IDF unigrams + bigrams + Logistic Regression | SLP3 Ch. 5 |
| 4D | TF-IDF + Linear SVM | Lab 07 alternative head |
| 4E | Word2Vec averaged + Logistic Regression | Mikolov et al. (2013); SLP3 Ch. 6 |
| 4F | Sentence-BERT + Logistic Regression | Reimers & Gurevych (2019); SLP3 Ch. 7 and Ch. 8 |

Training corpus is now KB documents tagged with their class. Tickets become the **routing test set** consumed by Phase 6.

**Selection rule.** Highest macro-F1 wins. When a TF-IDF baseline is within statistical noise of a transformer representation, deploy the cheaper one.

---

## Phase 5 — Keywords per class N

Top-N TF-IDF features per class, extracted from the winning Phase 4 classifier coefficients. N selected at the elbow of the precision-vs-N curve on held-out tickets.

Output is the keyword list pasted into Dify classifier-node descriptions.

**Reference.** SLP3 Ch. 5 — TF-IDF weighting.

---

## Phase 6 — Architecture comparison

Five candidate routing architectures, scored on **retrieval@k** against held-out tickets.

| Arch | Description |
|---|---|
| A | Flat single-label. Classifier picks one class → that class's KB. |
| B | Two-stage hierarchical. Stage 1 splits informational / navigation / escalation; Stage 2 picks domain class only for informational. |
| C | Multi-label fan-out. Soft probabilities; classes above threshold all contribute candidates; merged by query-document cosine. |
| D | No-classifier RAG. Rank all KB documents by query-document cosine. |
| E | Classifier + cross-encoder reranker. Arch A wins routing; cross-encoder reranks the chosen KB. |

**Two structural changes from v2.**

1. KB-document membership is now multi-class:
   ```python
   kb_doc_class_matrix = sim_kb_to_class >= MEMBERSHIP_THRESHOLD   # (n_docs, n_classes) bool
   docs_by_class = {c: np.where(kb_doc_class_matrix[:, c])[0].tolist()
                    for c in range(n_classes)}
   ```
   A document participates in every class it serves. Arch C becomes a real comparator instead of a near-clone of A.

2. The retrieval gold for a held-out ticket is "any document in the correct KB folder", not "argmax cosine to any document". This removes the embedding-leak threat that v2 acknowledged in its threats-to-validity section.

The `access_change_request` class never enters retrieval. Tickets routed to it are scored on classifier accuracy alone; the bot emits the redirect URL instead of a KB result.

**95% bootstrap confidence intervals on R@3 are reported** so the architecture recommendation rests on statistical significance, not a point estimate.

**References.** SLP3 Ch. 9 — retrieval-vs-generative chatbot design. Lewis et al. (2020) — RAG. Pradeep et al. (2023) — listwise reranking.

---

## Phase 7 — Knowledge-base consolidation

Lighter than v2 because the KB count is now defined upfront by the taxonomy.

Sanity checks:

- **Class-pair Jaccard** on the multi-membership matrix. Surfaces classes whose document sets overlap heavily — candidates to fold.
- **Orphan documents** — KB documents no class claims at the membership threshold. Routed to a content-team review queue, not to retrieval.

---

## Phase 8 — Dify export

Two artefacts.

1. **`dify_exports/dify_config.json`** — classifier configuration:
   - `metadata` block (K, N, embedding model, classifier, recommended architecture).
   - One entry per class: `class_id`, `keywords`, `kb_documents` (multi-membership list).
   - `access_change_request` class with `redirect_url` and empty `kb_documents`.
   - `orphan_documents` list.
   - `evaluation` block: coherence, macro-F1, R@3 with bootstrap CI.
2. **`dify_exports/runtime_metadata.json`** — runtime lookup table built from the metadata KB (excluded from training, retained for ticket-specific bot responses).

---

## Citations carried into the report

Primary references, all from the course syllabus.

| Reading | Use in report |
|---|---|
| SLP3 Ch. 2 | Phase 1 preprocessing |
| SLP3 Ch. 4 | Phase 4 classifier head, K-selection |
| SLP3 Appendix B | Phase 4A Naïve Bayes baseline |
| SLP3 Ch. 5 | Phase 4B–F vector representations, Phase 5 TF-IDF keywords |
| SLP3 Ch. 7 and Ch. 8 | Phase 4F Sentence-BERT, Phase 6E cross-encoder reranker |
| SLP3 Ch. 9 | Phase 6 chatbot architecture comparison framing |
| Blei (2012) — *Communications of the ACM* | Phase 2C LDA fallback |
| Grootendorst (2022) — BERTopic | Phase 2C BERTopic option |
| Mikolov et al. (2013) — Word2Vec | Phase 4E |
| Reimers and Gurevych (2019) — Sentence-BERT | Phase 4F, Phase 6 retrieval |
| Lewis et al. (2020) — RAG | Phase 6D |
| Pradeep et al. (2023) — RankZephyr | Phase 6E reranker |
| Manning (2022) — *Daedalus* | Report introduction framing |

PDFs of all readings live in `report/literature/`.

---

## Threats to validity

- **KB taxonomy bias.** A taxonomy authored by content owners reflects what they thought users would ask, not what users actually ask. Mitigation: reserve a held-out ticket sample as a coverage test. Tickets that no class can route are flagged as candidate gaps in the KB.
- **Class imbalance from KB folder sizes.** Folders with few documents become small classes with noisy centroids. Mitigation: report per-class F1 and minimum-class-size threshold; merge under-populated classes with their nearest neighbour by Jaccard.
- **Embedding-model confounding in retrieval gold.** Mitigated in v3 by defining gold at the folder level rather than the per-document argmax.
