# Paper guidelines: KAN-CDSCO2004U Machine Learning and Deep Learning

Single source of truth for what the report must satisfy. Consolidated from the three guideline PDFs in `archive/guidelines/` plus the formatting rules confirmed by the team.

## Course

| Item | Value |
|---|---|
| Course code | KAN-CDSCO2004U |
| Title | Machine Learning and Deep Learning |
| Programme | MSc Business Administration and Data Science |
| ECTS | 7.5 |
| Course coordinator | Somnath Mazumdar (DIGI) |
| Primary textbook | Géron (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 3rd ed., O'Reilly, ISBN 978-1098125974 |

## Examination

| Item | Value |
|---|---|
| Form | Oral exam based on written group product |
| Group size | 2–4 students |
| Written product size | **Max. 15 pages** |
| Citation style | **APA, 7th Edition** |
| Oral duration | 20 min/student (incl. examiner discussion + grade) |
| Grading | 7-point scale |
| Examiners | Internal + second internal |
| Exam period | Summer |
| Re-exam | Same form; same or revised project |

## Formatting (must satisfy)

| Rule | Value |
|---|---|
| Avg characters per page | **≤ 2,275** |
| Margins (top, bottom) | **≥ 3 cm** |
| Margins (left, right) | **≥ 2 cm** |
| Minimum font size | **≥ 11 pt** |
| Page count | **15 pages body max** |

What counts toward the 15-page limit: ToC, all body sections, in-body figures and tables.
What does NOT count: cover page, references, appendices.

Figures count as page space. Squeezing a half-page figure costs ~1,138 characters of text.

## Required report structure

Order matters. Section 4 (LLM Disclosure) is highlighted as **MUST FOLLOW** in the extended guidelines and is on its own page.

1. **Title**
2. **Author(s)**
3. **Abstract** — 200 words
4. **Keywords** — at least 5
5. **Introduction**
   - Motivation
   - Research Question(s)
   - Related Work (use Google Scholar; cite peer-reviewed work on similar RQs)
6. **Conceptual Framework**
   - Concepts of relevant data analytics methods and techniques
   - Problem statement with problem modelling (if relevant)
7. **Contribution & LLM Usage Disclosure** — *dedicated page, MUST FOLLOW*
   - Table with columns: `ID | Component/Module | Contributor(s) | LLM Tool (Model) | Type of LLM Usage | % LLM-Generated Code | Validation/Modification | Remarks`
   - Type examples: code generation, debugging, optimization, documentation assistance
   - `% LLM-Generated Code` must reflect actual dependency, not rough guesses
   - `Validation/Modification` must describe human oversight (testing, rewriting, benchmarking)
8. **Methodology**
   - Dataset Description
   - EDA (Data Analysis Process with diagram if any)
   - Data Pre-Processing (methods/tools)
   - Data Filtering, Transformation and Combination
   - Data Analytics: Modeling, Methods and Tools
   - **Evaluation Metrics**
   - **Model complexity analysis (running time vs baseline)**
9. **Results**
   - **Key Findings** — what was discovered, with causal explanation, assumptions, conditions
   - **Actionable Insights** — interpretations or predictions; what decisions or changes follow
   - **Practical Outcomes** — how results apply in real scenarios; recommendations / use cases
10. **Ethical Consideration** — top-level section in the extended guidelines
11. **Discussion**
    - Answers to the Research Question(s)
    - Implications for Research / Learning Reflections
    - Limitations of the dataset/work
12. **Conclusion & Future Work**
13. **References (APA 7th Edition)**
14. **Appendices** *(optional)*

## Dataset rules (the project satisfies these; documented for completeness)

- Custom dataset of own choice (not a coursework dataset, unless strong reason).
- Suitable to answer all the questions.
- Minimum a few thousand rows, a good number of columns, not too many missing/NA values.

The team's `consolidated_modeling_data.parquet` exceeds these floors: 1,371,180 rows × 87 columns, ~50.3% positive class, missing-data policy documented in `submission/data/MISSING_DATA.md`.

## CBS GenAI rules (full text in `archive/guidelines/03_CBS_GenAI_Guidelines.pdf`)

| Use case | Declaration required? | Where |
|---|---|---|
| Proofreading (spelling, grammar, punctuation, formatting) | No | — |
| Search engine | No | — |
| Idea generation / conceptualisation (structure, elaborating concepts) | **Yes** | Methodology section (or Introduction if no Methodology) |
| Generating text / images / output for a written product | **Yes — with specific reference** | Inline citation per CBS Library APA-7 guide |
| Sparring partner for oral exam prep | No | — |
| Tangible product for oral exam (slides) | **Yes** | Declared orally or in writing within the product |
| Generating answers during oral exam | **Not allowed** | — |

Hard constraints:
- Never input personal, confidential, or proprietary data into GenAI tools.
- Prefer CBS-provided tools (Copilot via CBS) for GDPR compliance.
- Assessors must be able to distinguish at all times between the student's own contribution and GenAI output. Failure to declare = exam cheating, reportable to CBS Legal.

## Learning objectives the report should evidence

The course catalogue lists six learning objectives. The report should make each of them visible:

1. Understand fundamental challenges of machine learning models (selection, complexity).
2. Detect strengths and weaknesses of machine learning models.
3. Design and implement machine learning models and deep learning techniques for realistic applications.
4. Summarize application areas, trends, and challenges in machine learning.
5. Exhibit deeper knowledge and understanding of covered topics.
6. Reflect on critical awareness of methodological choices with written skills to accepted academic standards.

## Course topics that may be examined

Coverage signals from the lecture plan. The report does not need to use every method, but the oral exam may probe any of these.

- Data pre-processing and exploratory data analysis
- Loss functions, bias-variance trade-off, cross-validation schemes
- Unsupervised: K-Means, DBSCAN, Gaussian Mixture Models
- Supervised: KNN, linear / logistic regression, SVM
- Regularization: L1 / L2 / Elastic Net
- Decision Trees, ensembles (Random Forest, Gradient Boosting)
- Dimensionality reduction: Principal Component Analysis
- Performance metrics: ROC, PR, AUC, F1
- Outlier detection: Isolation Forest
- Class imbalance: SMOTE, ADASYN
- Gradient descent, backpropagation, early stopping, activation functions
- Hyperparameter optimization: grid search, random search
- Neural networks: MLP, dropout, batch normalisation
- RNN: LSTM, GRU, Bi-LSTM
- CNN: VGG, ResNet, MobileNet, YOLO
- Autoencoders, transformer / attention introduction
- Adversarial attacks
- Federated learning, explainable AI
- Ethics of ML and DL, AI alignment

## Submission checklist

- [ ] Body ≤ 15 pages (ToC + sections + in-body figures), avg ≤ 2,275 chars/page
- [ ] Margins ≥ 3 cm top/bottom, ≥ 2 cm left/right; font ≥ 11 pt
- [ ] Abstract = 200 words
- [ ] ≥ 5 keywords
- [ ] All 14 sections present and in order (cover, abstract, keywords, intro, conceptual framework, **LLM disclosure page**, methodology, results, ethics, discussion, conclusion, references, appendix)
- [ ] LLM disclosure table filled with real %s and validation notes
- [ ] APA 7th in-text citations and reference list
- [ ] Dataset description references 1.37M rows × 87 cols, ~50.3% positive class, missing-data policy
- [ ] Evaluation metrics named (AUC, F1, calibration); complexity vs baseline reported
- [ ] Ethics section present
- [ ] All in-body figures legible at the print size and cited in text
- [ ] References list = APA 7th, all in-text citations resolve
- [ ] `submission/` bundle accompanies the .docx: scripts, data, report_assets

## Source documents

Originals in `archive/guidelines/`:

- `01_Project_Guidelines_original.pdf` — original 6-slide deck
- `02_Project_Guidelines_extended.pdf` — **MUST FOLLOW** version, adds LLM disclosure + 200-word abstract + Evaluation Metrics + Ethical Consideration as top-level
- `03_CBS_GenAI_Guidelines.pdf` — CBS-wide GenAI rules (5 pages)
- `learning_objectives_ml.pdf` — official course catalogue page (KAN-CDSCO2004U)
- `lecture_plan_ml.pdf` — 16-lecture plan + literature list
