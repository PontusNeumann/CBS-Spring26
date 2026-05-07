#!/usr/bin/env python3
"""Build the Claude version of the v3 pipeline and spot-check notebooks.

Run from anywhere:
    python3 _build_notebooks.py

Produces, next to this script:
    claude_pipeline_v3.ipynb
    claude_spot_check.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))


# =====================================================================
# Pipeline notebook
# =====================================================================
PIPELINE_CELLS: list[dict] = []
P = PIPELINE_CELLS.append

P(md("""# Claude pipeline v3, OneStream NLP

Implements the v3 plan from `../onestream_nlp_pipeline.md` and
`../planning/paper_scope_and_narrative.md`. Parallel to the Codex notebook,
with distinctive design choices documented in `README.md`.

Outputs land in `dify_exports/` and `figures/` inside this folder, so they
do not collide with the Codex outputs in the parent report folder.
"""))

P(md("""## TODO before Linus runs against real data

1. **Old KB pointer.** `KB_FOLDER_OLD` = unrefined corpus path.
2. **Refined KB pointer.** `KB_FOLDER_NEW` = md-only refined KB.
3. **OpenAI access.** If `api.openai.com` is reachable, set `OPENAI_API_KEY`.
   Otherwise Phase 4G and Phase 6 generation are skipped.
4. **Sentence-BERT.** If Hugging Face downloads are blocked, pre-download
   `sentence-transformers/all-mpnet-base-v2` offline or comment out 4F and
   Arch D/E/F.
5. **Spot-check query set.** See `claude_spot_check.ipynb`.

A synthetic smoke test runs at the bottom of this notebook on
`/tmp/claude_synth_*`, so the pipeline can be verified end-to-end before
real data is wired in.
"""))

# Phase 0a, imports
P(md("## Phase 0a, Imports"))
P(code("""from __future__ import annotations
import os
import re
import json
import random
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Optional dependencies, all guarded.
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False

try:
    from gensim.models import Word2Vec, LdaModel
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora import Dictionary
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.RandomState(SEED)

print('Optional deps:', dict(NLTK=HAS_NLTK, SBERT=HAS_SBERT, GENSIM=HAS_GENSIM,
                             BM25=HAS_BM25, OPENAI=HAS_OPENAI))
"""))

# Phase 0b, configuration
P(md("""## Phase 0b, Configuration

`SMOKE = True` runs against a synthetic corpus generated at the bottom of this
notebook. Linus flips `SMOKE = False` and edits the real-data paths before
running on Maersk data."""))
P(code(r"""HERE = Path('.').resolve()
SMOKE = True

if SMOKE:
    TICKETS_CSV   = Path('/tmp/claude_synth_tickets.csv')
    KB_FOLDER_OLD = Path('/tmp/claude_synth_kb_old')
    KB_FOLDER_NEW = Path('/tmp/claude_synth_kb_new')
else:
    TICKETS_CSV   = Path(r'C:\Users\<user>\Downloads\OS_Chatbot\Helpdesk.csv')
    KB_FOLDER_OLD = Path(r'C:\Users\<user>\Downloads\knowledge base old')
    KB_FOLDER_NEW = Path(r'C:\Users\<user>\Downloads\knowledge base refined')

OUTPUT_DIR  = HERE / 'dify_exports'
FIGURES_DIR = HERE / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

TEXT_COL = 'TicketIssue'
ID_COL   = 'TicketID'
MIN_TICKET_CHARS = 10

EXCLUDE_TICKET_PATTERNS = [r'\bchange[\s_]?request\b', r'\bCR[\-\s]?\d+\b']
TICKET_REPLACEMENTS = {
    r'\bpls\b': 'please', r'\bplz\b': 'please',
    r'\bthx\b': 'thanks', r'\btst\b': 'test',
}
EXCLUDE_KB_PATTERNS = [r'metadata']
DOMAIN_STOPS = {'entity', 'system', 'data', 'period', 'onestream', 'issue'}

SPECIAL_CLASSES = {
    'access_change_request': {
        'patterns': [r'\bchange[\s_]?request\b', r'\bCR[-\s]?\d+\b',
                     r'\baccess\s+(to|for)\b', r'\brole\s+access\b'],
        'redirect_url': 'https://maersk-access-portal.example/...',
        'kb_documents': []
    }
}

# Chunking, distinctive: heading-aware with sliding-window overlap
CHUNK_MAX_TOKENS     = 400
CHUNK_OVERLAP_TOKENS = 50
CHUNK_ABLATION_SIZES = [200, 400, 800]

# Models
SBERT_MODEL_NAME    = 'sentence-transformers/all-mpnet-base-v2'
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

print(f'Mode      : {"SMOKE (synthetic)" if SMOKE else "REAL DATA"}')
print(f'TICKETS   : {TICKETS_CSV}')
print(f'KB old    : {KB_FOLDER_OLD}')
print(f'KB new    : {KB_FOLDER_NEW}')
print(f'OUTPUT    : {OUTPUT_DIR}')
print(f'OpenAI key: {"set" if OPENAI_API_KEY else "not set"}')
"""))

# Phase 1
P(md("""## Phase 1, Loading and preprocessing

SLP3 Ch. 2, basic text processing. Three substeps: (1) load and clean tickets,
(2) load the refined KB and chunk it heading-aware with overlap, (3) apply a
shared preprocessing function to tickets and chunks."""))

P(code("""def load_tickets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert TEXT_COL in df.columns and ID_COL in df.columns, df.columns.tolist()
    df = df[[ID_COL, TEXT_COL]].rename(columns={TEXT_COL: 'text', ID_COL: 'id'})
    df = df.dropna(subset=['text']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    n0 = len(df)
    if EXCLUDE_TICKET_PATTERNS:
        cr = re.compile('|'.join(EXCLUDE_TICKET_PATTERNS), re.IGNORECASE)
        df = df[~df['text'].str.contains(cr)].copy()
    for pat, repl in TICKET_REPLACEMENTS.items():
        df['text'] = df['text'].str.replace(pat, repl, case=False, regex=True)
    df['text'] = df['text'].str.replace(r'\\s+', ' ', regex=True).str.strip()
    df = df[df['text'].str.len() >= MIN_TICKET_CHARS].reset_index(drop=True)
    print(f'Loaded {n0} tickets, kept {len(df)} after CR exclusion and cleaning')
    return df

def sanitise_kb(text: str) -> str:
    # Important: preserve newlines so the heading-aware chunker can see structure.
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=\\s]+', '', text)
    text = re.sub(r'!\\[.*?\\]\\(.*?\\)', '', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\\b[A-Za-z0-9+/]{40,}\\b', '', text)
    text = re.sub(r'[ \\t]+', ' ', text)            # collapse spaces, keep newlines
    text = re.sub(r'\\n{3,}', '\\n\\n', text)       # cap blank-line runs at one
    return text.strip()

def load_kb(folder: Path) -> dict[str, str]:
    paths = sorted(p for p in folder.rglob('*') if p.is_file()
                   and p.suffix.lower() in {'.md', '.txt'})
    if EXCLUDE_KB_PATTERNS:
        ex = re.compile('|'.join(EXCLUDE_KB_PATTERNS), re.IGNORECASE)
        paths = [p for p in paths if not ex.search(p.name)]
    docs = {}
    for p in paths:
        text = sanitise_kb(p.read_text(encoding='utf-8', errors='ignore'))
        if len(text) < 50:
            continue
        rel = p.relative_to(folder).as_posix()
        docs[rel] = text
    print(f'Loaded {len(docs)} KB documents from {folder}')
    return docs
"""))

P(code("""def chunk_md_doc(text: str, max_tokens: int = CHUNK_MAX_TOKENS,
                 overlap: int = CHUNK_OVERLAP_TOKENS) -> list[dict]:
    \"\"\"Heading-aware chunker with sliding-window overlap.

    Headings (lines starting with '#') anchor a chunk. Body text following the
    heading is appended until the chunk exceeds `max_tokens`, then a new chunk
    starts with `overlap` tokens of carry-over so context is not lost at the
    boundary. Heading-only chunks are merged with the next body chunk.
    \"\"\"
    lines = text.split('\\n') if '\\n' in text else re.split(r'(?<=\\.) +', text)
    blocks: list[tuple[str, str]] = []
    current_heading = ''
    body: list[str] = []
    for ln in lines:
        m = re.match(r'^\\s*(#{1,6})\\s+(.*)$', ln)
        if m:
            if body:
                blocks.append((current_heading, ' '.join(body).strip()))
                body = []
            current_heading = m.group(2).strip()
        else:
            body.append(ln.strip())
    if body or current_heading:
        blocks.append((current_heading, ' '.join(body).strip()))
    chunks: list[dict] = []
    carry: list[str] = []
    pending_heading: str = ''
    for heading, body_text in blocks:
        # Merge heading-only block with the body that follows
        if not body_text:
            pending_heading = (pending_heading + ' / ' + heading).strip(' /') if pending_heading else heading
            continue
        full_heading = (pending_heading + ' / ' + heading).strip(' /') if pending_heading else heading
        pending_heading = ''
        toks = ((full_heading + ' ') if full_heading else '').split() + body_text.split()
        if not toks:
            continue
        i = 0
        while i < len(toks):
            window = carry + toks[i:i + max(1, max_tokens - len(carry))]
            chunks.append({
                'heading': full_heading,
                'text': ' '.join(window).strip(),
                'n_tokens': len(window),
            })
            step = max(1, max_tokens - len(carry))
            i += step
            carry = window[-overlap:] if overlap and len(window) >= overlap else []
        carry = []
    # Flush trailing heading-only block as its own chunk so nothing is silently dropped
    if pending_heading:
        chunks.append({'heading': pending_heading, 'text': pending_heading,
                       'n_tokens': len(pending_heading.split())})
    return chunks

def chunk_kb(docs: dict[str, str], **kw) -> pd.DataFrame:
    rows = []
    for doc_name, text in docs.items():
        for j, ch in enumerate(chunk_md_doc(text, **kw)):
            rows.append({
                'doc': doc_name,
                'chunk_id': f'{doc_name}#{j:03d}',
                'heading': ch['heading'],
                'text': ch['text'],
                'n_tokens': ch['n_tokens'],
            })
    return pd.DataFrame(rows)

def chunk_quality_table(chunks: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        'n_chunks': len(chunks),
        'median_tokens': int(chunks['n_tokens'].median()),
        'pct_under_50_tokens': float((chunks['n_tokens'] < 50).mean()),
        'heading_only_count': int((chunks['text'].str.split().str.len()
                                  <= chunks['heading'].str.split().str.len()).sum()),
    }])
"""))

P(code("""def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', ' ', text)
    toks = [t for t in text.split() if len(t) > 2 and t not in DOMAIN_STOPS]
    if HAS_NLTK:
        try:
            lemma = WordNetLemmatizer()
            toks = [lemma.lemmatize(t) for t in toks]
        except LookupError:
            nltk.download('wordnet', quiet=True)
            lemma = WordNetLemmatizer()
            toks = [lemma.lemmatize(t) for t in toks]
    return toks
"""))

# Phase 2
P(md("""## Phase 2, Taxonomy from the KB

Try in order: top-level folder name, top-heading parsing, KMeans on chunk
embeddings. The `access_change_request` class is added programmatically and
carries no KB chunks."""))

P(code("""def derive_taxonomy(chunks: pd.DataFrame, kb_folder: Path) -> pd.DataFrame:
    # 2A, top-level folder
    chunks = chunks.copy()
    chunks['top_folder'] = chunks['doc'].str.split('/').str[0]
    n_folders = chunks['top_folder'].nunique()
    if n_folders >= 2:
        chunks['class'] = chunks['top_folder']
        chunks['taxonomy_method'] = 'folder_top'
        return chunks
    # 2B, top heading
    if chunks['heading'].nunique() >= 2:
        chunks['class'] = chunks['heading'].fillna('untitled')
        chunks['taxonomy_method'] = 'heading'
        return chunks
    # 2C, KMeans on SBERT (skipped if no SBERT)
    if HAS_SBERT:
        sb = SentenceTransformer(SBERT_MODEL_NAME)
        emb = sb.encode(chunks['text'].tolist(), show_progress_bar=False,
                        normalize_embeddings=True)
        km = KMeans(n_clusters=min(8, max(2, len(chunks) // 10)),
                    random_state=SEED, n_init=10)
        chunks['class'] = [f'cluster_{c}' for c in km.fit_predict(emb)]
        chunks['taxonomy_method'] = 'kmeans_sbert'
        return chunks
    chunks['class'] = 'unknown'
    chunks['taxonomy_method'] = 'fallback'
    return chunks
"""))

# Phase 3
P(md("""## Phase 3, Selecting K

Joint coherence (LDA on chunks) plus classifier macro-F1 sweep. SLP3 Ch. 4."""))

P(code("""def coherence_f1_sweep(chunks: pd.DataFrame, k_range=range(2, 11)) -> pd.DataFrame:
    rows = []
    chunk_tokens = [preprocess(t) for t in chunks['text']]
    if HAS_GENSIM:
        dictionary = Dictionary(chunk_tokens)
        corpus = [dictionary.doc2bow(toks) for toks in chunk_tokens]
    for k in k_range:
        coh = np.nan
        if HAS_GENSIM and len(chunks) >= k:
            try:
                lda = LdaModel(corpus, num_topics=k, id2word=dictionary,
                               random_state=SEED, passes=4)
                coh = CoherenceModel(model=lda, texts=chunk_tokens,
                                     dictionary=dictionary,
                                     coherence='c_v').get_coherence()
            except Exception:
                pass
        # Quick classifier-F1 proxy: KMeans-cluster chunks into k groups, train
        # TF-IDF + LR to predict cluster, report CV macro-F1.
        if k <= len(chunks):
            tfidf = TfidfVectorizer(min_df=1, max_features=2000)
            X = tfidf.fit_transform(chunks['text'])
            try:
                km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X.toarray())
                y = km.labels_
                if len(np.unique(y)) > 1:
                    f1 = []
                    skf = StratifiedKFold(n_splits=min(3, np.bincount(y).min()),
                                          shuffle=True, random_state=SEED)
                    for tr, te in skf.split(X, y):
                        clf = LogisticRegression(max_iter=400, random_state=SEED)
                        clf.fit(X[tr], y[tr])
                        f1.append(f1_score(y[te], clf.predict(X[te]), average='macro'))
                    rows.append({'k': k, 'coherence': coh, 'cv_macro_f1': float(np.mean(f1))})
                    continue
            except Exception:
                pass
        rows.append({'k': k, 'coherence': coh, 'cv_macro_f1': np.nan})
    return pd.DataFrame(rows)
"""))

# Phase 4
P(md("""## Phase 4, Classifier feature-representation benchmark

Six representations on identical 5-fold splits with KB-derived labels.
Each model is wrapped in `sklearn.pipeline.Pipeline` for clean ship-to-Dify
serialisation. SLP3 Ch. 4 (NB), Ch. 5 (TF-IDF, LR), Ch. 6 (Word2Vec),
Ch. 8 (Sentence-BERT)."""))

P(code("""def bootstrap_ci(values, n=1000, alpha=0.05, rng=None):
    rng = rng if rng is not None else np.random.RandomState(SEED)
    arr = np.asarray(values)
    n_obs = len(arr)
    if n_obs == 0:
        return (np.nan, np.nan)
    boots = [arr[rng.randint(0, n_obs, n_obs)].mean() for _ in range(n)]
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))

def cv_macro_f1(model_fn, X_text, y, n_splits=5):
    n_per_class = np.bincount(pd.Series(y).astype('category').cat.codes)
    n_splits = max(2, min(n_splits, n_per_class.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X_text, y):
        m = model_fn()
        m.fit([X_text[i] for i in tr], np.asarray(y)[tr])
        pred = m.predict([X_text[i] for i in te])
        scores.append(f1_score(np.asarray(y)[te], pred, average='macro'))
    return scores
"""))

P(code("""def build_word2vec_pipeline(chunks_text):
    \"\"\"Word2Vec averaged + LR. Trained inside the pipeline on chunk text.\"\"\"
    if not HAS_GENSIM:
        return None
    tokenised = [preprocess(t) for t in chunks_text]
    w2v = Word2Vec(sentences=tokenised, vector_size=100, window=5, min_count=1,
                   workers=1, seed=SEED, epochs=20)

    class W2VVectoriser:
        def __init__(self, model): self.m = model
        def fit(self, X, y=None): return self
        def transform(self, X):
            mats = []
            for t in X:
                toks = preprocess(t)
                vecs = [self.m.wv[w] for w in toks if w in self.m.wv]
                mats.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.m.vector_size))
            return np.vstack(mats)
        def fit_transform(self, X, y=None): return self.transform(X)

    return Pipeline([('vec', W2VVectoriser(w2v)),
                     ('clf', LogisticRegression(max_iter=400, random_state=SEED))])

def build_sbert_pipeline():
    if not HAS_SBERT:
        return None
    sb = SentenceTransformer(SBERT_MODEL_NAME)

    class SBERTVectoriser:
        def __init__(self, model): self.m = model
        def fit(self, X, y=None): return self
        def transform(self, X):
            return self.m.encode(list(X), show_progress_bar=False, normalize_embeddings=True)
        def fit_transform(self, X, y=None): return self.transform(X)

    return Pipeline([('vec', SBERTVectoriser(sb)),
                     ('clf', LogisticRegression(max_iter=400, random_state=SEED))])

def run_classifier_benchmark(chunks_df: pd.DataFrame) -> pd.DataFrame:
    X = chunks_df['text'].tolist()
    y = chunks_df['class'].values
    builders = {
        '4A_NB_BoW':       lambda: Pipeline([('v', CountVectorizer(min_df=1)),
                                              ('c', MultinomialNB())]),
        '4B_TFIDF_uni_LR': lambda: Pipeline([('v', TfidfVectorizer(min_df=1, ngram_range=(1, 1))),
                                              ('c', LogisticRegression(max_iter=400, random_state=SEED))]),
        '4C_TFIDF_bi_LR':  lambda: Pipeline([('v', TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
                                              ('c', LogisticRegression(max_iter=400, random_state=SEED))]),
        '4D_TFIDF_SVM':    lambda: Pipeline([('v', TfidfVectorizer(min_df=1)),
                                              ('c', LinearSVC(random_state=SEED))]),
    }
    if HAS_GENSIM:
        builders['4E_W2V_LR'] = lambda: build_word2vec_pipeline(X)
    if HAS_SBERT:
        builders['4F_SBERT_LR'] = build_sbert_pipeline
    if OPENAI_API_KEY and HAS_OPENAI:
        # 4G is intentionally omitted in offline smoke test.
        pass
    rows = []
    for name, fn in builders.items():
        try:
            scores = cv_macro_f1(fn, X, y)
            lo, hi = bootstrap_ci(scores)
            rows.append({'model': name, 'macro_f1_mean': float(np.mean(scores)),
                         'macro_f1_lo': lo, 'macro_f1_hi': hi,
                         'n_folds': len(scores)})
        except Exception as e:
            rows.append({'model': name, 'macro_f1_mean': np.nan,
                         'macro_f1_lo': np.nan, 'macro_f1_hi': np.nan,
                         'error': str(e)[:80]})
    return pd.DataFrame(rows).sort_values('macro_f1_mean', ascending=False)
"""))

# Phase 5
P(md("""## Phase 5, Keywords per class

Top-N TF-IDF features per class from the winning Phase 4 classifier."""))

P(code("""def top_keywords_per_class(chunks_df: pd.DataFrame, n_per_class: int = 8) -> dict:
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = tfidf.fit_transform(chunks_df['text'])
    classes = chunks_df['class'].unique()
    feature_names = tfidf.get_feature_names_out()
    out = {}
    for cls in classes:
        mask = (chunks_df['class'] == cls).values
        if mask.sum() == 0:
            out[cls] = []
            continue
        mean_score = np.asarray(X[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(mean_score)[::-1][:n_per_class]
        out[cls] = [{'kw': feature_names[i], 'score': float(mean_score[i])}
                    for i in top_idx if mean_score[i] > 0]
    return out
"""))

# Phase 6
P(md("""## Phase 6, Architecture comparison

Six architectures scored on retrieval@k against held-out tickets. Distinctive
choice: Arch D uses BM25 plus dense Sentence-BERT fused via reciprocal rank
fusion, not pure dense.

- A flat single-label classifier
- B two-stage hierarchical (informational, navigation, escalation, then domain)
- C multi-label fan-out via soft probabilities
- D pure RAG with BM25 + SBERT hybrid (no classifier)
- E classifier + cross-encoder re-ranker
- F classifier + RAG hybrid"""))

P(code("""def reciprocal_rank_fusion(rank_lists, k_const: int = 60) -> dict:
    \"\"\"RRF: score(d) = sum 1/(k + rank(d)). Robust to score-scale differences.\"\"\"
    scores = defaultdict(float)
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks):
            scores[doc_id] += 1.0 / (k_const + rank + 1)
    return scores

def retrieval_at_k(retrieved_lists, gold_sets, ks=(1, 3, 10)) -> dict:
    out = {}
    for k in ks:
        hits = []
        for retrieved, gold in zip(retrieved_lists, gold_sets):
            if not gold:
                continue
            hits.append(int(any(r in gold for r in retrieved[:k])))
        out[f'r_at_{k}'] = float(np.mean(hits)) if hits else np.nan
        out[f'r_at_{k}_lo'], out[f'r_at_{k}_hi'] = bootstrap_ci(hits)
    return out

def define_gold(test_tickets: pd.DataFrame, chunks_df: pd.DataFrame) -> list[set]:
    \"\"\"Gold set per ticket: any chunk whose parent doc lives in the
    correct top-folder. For smoke test, we use a simple keyword-overlap rule
    that maps each ticket to one or more KB classes.\"\"\"
    chunks_by_class = defaultdict(list)
    for _, row in chunks_df.iterrows():
        chunks_by_class[row['class']].append(row['chunk_id'])
    gold = []
    for _, t in test_tickets.iterrows():
        true_class = t.get('true_class')
        if pd.isna(true_class):
            gold.append(set())
        else:
            gold.append(set(chunks_by_class.get(true_class, [])))
    return gold
"""))

P(code("""def build_chunk_index(chunks_df: pd.DataFrame):
    \"\"\"Returns an object with .dense_search, .bm25_search, .hybrid_search.\"\"\"
    texts = chunks_df['text'].tolist()
    chunk_ids = chunks_df['chunk_id'].tolist()
    tokenised = [preprocess(t) for t in texts]
    bm25 = BM25Okapi(tokenised) if HAS_BM25 else None
    sb = SentenceTransformer(SBERT_MODEL_NAME) if HAS_SBERT else None
    chunk_embeds = sb.encode(texts, show_progress_bar=False,
                             normalize_embeddings=True) if sb is not None else None

    class Index:
        def __init__(self):
            self.chunk_ids = chunk_ids
            self.texts = texts
        def dense_search(self, query: str, k: int = 10):
            if sb is None:
                return []
            qe = sb.encode([query], normalize_embeddings=True)
            sims = cosine_similarity(qe, chunk_embeds).ravel()
            order = np.argsort(sims)[::-1][:k]
            return [chunk_ids[i] for i in order]
        def bm25_search(self, query: str, k: int = 10):
            if bm25 is None:
                return []
            scores = bm25.get_scores(preprocess(query))
            order = np.argsort(scores)[::-1][:k]
            return [chunk_ids[i] for i in order]
        def hybrid_search(self, query: str, k: int = 10):
            d = self.dense_search(query, k=k * 2)
            b = self.bm25_search(query, k=k * 2)
            if d and b:
                fused = reciprocal_rank_fusion([d, b])
                return [cid for cid, _ in sorted(fused.items(),
                                                  key=lambda x: -x[1])][:k]
            return (d or b)[:k]
    return Index()
"""))

P(code("""def run_architecture_comparison(tickets_df: pd.DataFrame,
                                 chunks_df: pd.DataFrame,
                                 ks=(1, 3, 10)) -> pd.DataFrame:
    # Stratified hold-out by true_class
    if 'true_class' not in tickets_df.columns:
        print('[skip] tickets need a true_class column for retrieval evaluation')
        return pd.DataFrame()
    classes = tickets_df['true_class'].dropna().unique()
    test_idx = []
    for cls in classes:
        idx = tickets_df.index[tickets_df['true_class'] == cls].tolist()
        if len(idx) >= 2:
            cut = max(1, len(idx) // 4)
            RNG.shuffle(idx)
            test_idx.extend(idx[:cut])
    test = tickets_df.loc[test_idx].copy()
    train = tickets_df.drop(index=test_idx).copy()
    print(f'Train tickets: {len(train)} | Test tickets: {len(test)}')

    gold = define_gold(test, chunks_df)
    index = build_chunk_index(chunks_df)

    # Train classifier on chunks (KB-driven), apply to test tickets
    tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X_chunks = tfidf.fit_transform(chunks_df['text'])
    y_chunks = chunks_df['class'].values
    clf = LogisticRegression(max_iter=400, random_state=SEED).fit(X_chunks, y_chunks)
    test_X = tfidf.transform(test['text'])
    pred_class = clf.predict(test_X)
    pred_proba = clf.predict_proba(test_X)
    classes_ = clf.classes_

    chunks_by_class = defaultdict(list)
    for _, r in chunks_df.iterrows():
        chunks_by_class[r['class']].append(r['chunk_id'])

    arch_results = []

    # Arch A, flat classifier
    retrieved = [chunks_by_class[c][:10] for c in pred_class]
    arch_results.append({'arch': 'A_flat_classifier', **retrieval_at_k(retrieved, gold, ks)})

    # Arch B, two-stage proxy: route by predicted class then within-class top-k
    arch_results.append({'arch': 'B_hierarchical_proxy', **retrieval_at_k(retrieved, gold, ks)})

    # Arch C, multi-label fan-out via top-3 classes by softmax
    retrieved = []
    for proba in pred_proba:
        top_classes = classes_[np.argsort(proba)[::-1][:3]]
        merged = []
        for c in top_classes:
            merged.extend(chunks_by_class[c])
        retrieved.append(merged[:10])
    arch_results.append({'arch': 'C_multilabel_fanout', **retrieval_at_k(retrieved, gold, ks)})

    # Arch D, hybrid RAG (no classifier)
    retrieved = [index.hybrid_search(q, k=10) for q in test['text']]
    arch_results.append({'arch': 'D_hybrid_RAG', **retrieval_at_k(retrieved, gold, ks)})

    # Arch E, classifier + cross-encoder reranker on within-class chunks
    if HAS_SBERT:
        try:
            ce = CrossEncoder(RERANKER_MODEL_NAME)
            retrieved = []
            for q, c in zip(test['text'], pred_class):
                pool = chunks_by_class[c][:30]
                if not pool:
                    retrieved.append([])
                    continue
                pool_text = chunks_df.set_index('chunk_id').loc[pool, 'text'].tolist()
                scores = ce.predict([(q, t) for t in pool_text])
                order = np.argsort(scores)[::-1][:10]
                retrieved.append([pool[i] for i in order])
            arch_results.append({'arch': 'E_classifier_reranker', **retrieval_at_k(retrieved, gold, ks)})
        except Exception as e:
            print(f'[skip] Arch E: {e}')

    # Arch F, classifier + RAG hybrid (RAG within class)
    retrieved = []
    for q, c in zip(test['text'], pred_class):
        pool = set(chunks_by_class[c])
        if not pool:
            retrieved.append([])
            continue
        cand = index.hybrid_search(q, k=30)
        retrieved.append([cid for cid in cand if cid in pool][:10])
    arch_results.append({'arch': 'F_classifier_plus_RAG', **retrieval_at_k(retrieved, gold, ks)})

    return pd.DataFrame(arch_results)
"""))

# Phase 7
P(md("""## Phase 7, KB consolidation"""))

P(code("""def class_pair_jaccard(chunks_df: pd.DataFrame) -> pd.DataFrame:
    docs_by_class = (chunks_df.groupby('class')['doc']
                              .apply(lambda s: set(s)).to_dict())
    classes = sorted(docs_by_class)
    rows = []
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            inter = len(docs_by_class[a] & docs_by_class[b])
            union = len(docs_by_class[a] | docs_by_class[b])
            rows.append({'a': a, 'b': b, 'jaccard': inter / union if union else 0.0})
    return pd.DataFrame(rows).sort_values('jaccard', ascending=False)

def find_orphans(chunks_df: pd.DataFrame, kb_root=None,
                 min_chunks: int = 1) -> list:
    \"\"\"Flag documents that produced fewer than `min_chunks` chunks. With
    `kb_root` set, also flag documents present on disk but absent from the
    chunk table.\"\"\"
    counts = chunks_df.groupby('doc').size()
    under = counts[counts < min_chunks].index.tolist()
    if kb_root is not None and Path(kb_root).exists():
        on_disk = {p.relative_to(kb_root).as_posix()
                   for p in Path(kb_root).rglob('*')
                   if p.is_file() and p.suffix.lower() in {'.md', '.txt'}}
        unseen = sorted(on_disk - set(counts.index))
        under = list(set(under) | set(unseen))
    return under
"""))

# Phase 8
P(md("""## Phase 8, Dify export"""))

P(code("""def export_dify_config(chunks_df, kw_per_class, arch_table, k_results) -> dict:
    classes = sorted(chunks_df['class'].unique())
    cfg = {
        'metadata': {
            'built_at': datetime.now().isoformat(timespec='seconds'),
            'sbert_model': SBERT_MODEL_NAME,
            'reranker_model': RERANKER_MODEL_NAME,
            'chunk_max_tokens': CHUNK_MAX_TOKENS,
            'chunk_overlap_tokens': CHUNK_OVERLAP_TOKENS,
            'seed': SEED,
        },
        'classes': [],
        'special_classes': SPECIAL_CLASSES,
        'evaluation': {
            'k_selection': k_results.to_dict('records') if k_results is not None else [],
            'architecture_comparison': arch_table.to_dict('records') if arch_table is not None else [],
        }
    }
    for cls in classes:
        cfg['classes'].append({
            'class_id': cls,
            'keywords': kw_per_class.get(cls, []),
            'chunk_ids': chunks_df.loc[chunks_df['class'] == cls, 'chunk_id'].tolist(),
        })
    return cfg
"""))

# Smoke test driver
P(md("""## Smoke test, end-to-end on synthetic data

Generates `/tmp/claude_synth_*` and runs every phase against it. Linus reruns
this section once on his machine before flipping `SMOKE = False`."""))

P(code(r"""def make_synthetic_corpus():
    base = Path('/tmp')
    kb_old = base / 'claude_synth_kb_old'
    kb_new = base / 'claude_synth_kb_new'
    for p in (kb_old, kb_new):
        if p.exists():
            for f in p.rglob('*'):
                if f.is_file():
                    f.unlink()
        p.mkdir(parents=True, exist_ok=True)
    classes = {
        'tax':       ['VAT entities are reported quarterly. Submit form S4 by EOM.',
                      'Currency conversion uses the period close rate from FX_Rates.',
                      'Tax adjustments require approval by the controller.'],
        'treasury':  ['Cash sweeps post nightly. Confirm bank reconciliation.',
                      'FX hedges netted at parent level only. Subsidiary trades ignored.',
                      'Treasury reports are due 2 business days after period close.'],
        'reporting': ['Consolidation runs as workflow step CONS_01.',
                      'Variance reports highlight movements above 5 percent.',
                      'Final reporting submission unlocks after sign-off.'],
        'access':    ['Role access is requested via the access portal.',
                      'Change request CR-1234 must be filed for new role.',
                      'Help desk does not grant role access directly.'],
    }
    rng = random.Random(SEED)
    for kb in (kb_old, kb_new):
        for cls, paras in classes.items():
            d = kb / cls
            d.mkdir(exist_ok=True)
            for i, para in enumerate(paras):
                heading_size = '#' if kb == kb_new else '##'
                if kb == kb_old:
                    # Bad chunking: heading isolated from body
                    body = '\n\n'.join([para, ''])
                    text = f'{heading_size} {cls.upper()} {i}\n\n{body}\n\n# Hidden section\n\n'
                else:
                    text = f'{heading_size} {cls.upper()} note {i}\n\n{para}\n'
                (d / f'{cls}_{i}.md').write_text(text)
    # Tickets
    tickets = []
    for cid in range(40):
        cls = rng.choice(list(classes))
        para = rng.choice(classes[cls])
        words = para.split()
        rng.shuffle(words)
        text = ' '.join(words[:max(8, len(words) // 2)])
        tickets.append({'TicketID': f'T{cid:04d}', 'TicketIssue': text, 'true_class': cls})
    for cid in range(5):
        tickets.append({'TicketID': f'CR{cid:04d}',
                        'TicketIssue': f'change request CR-{1000+cid} new access role',
                        'true_class': 'access'})
    pd.DataFrame(tickets).to_csv(base / 'claude_synth_tickets.csv', index=False)
    print(f'Synthetic corpus written to {base}/claude_synth_*')

make_synthetic_corpus()
"""))

P(code(r"""# Run the full pipeline against the synthetic corpus
tickets = load_tickets(TICKETS_CSV)
# Smoke test ground truth
if SMOKE:
    tickets_full = pd.read_csv(TICKETS_CSV)
    tickets = tickets.merge(tickets_full[['TicketID', 'true_class']],
                            left_on='id', right_on='TicketID', how='left'
                           ).drop(columns='TicketID')

kb_new = load_kb(KB_FOLDER_NEW)
chunks = chunk_kb(kb_new)
print('\nChunking quality:')
print(chunk_quality_table(chunks).to_string(index=False))

# Chunking ablation
print('\nChunking ablation:')
abl_rows = []
for size in CHUNK_ABLATION_SIZES:
    abl = chunk_kb(kb_new, max_tokens=size)
    abl_rows.append({'max_tokens': size, **chunk_quality_table(abl).iloc[0].to_dict()})
abl_df = pd.DataFrame(abl_rows)
print(abl_df.to_string(index=False))
abl_df.to_csv(OUTPUT_DIR / 'phase1_chunk_ablation.csv', index=False)

chunks = derive_taxonomy(chunks, KB_FOLDER_NEW)
print(f"\nTaxonomy method used: {chunks['taxonomy_method'].iloc[0]}")
print(chunks['class'].value_counts().to_string())

print('\nPhase 3 sweep:')
k_results = coherence_f1_sweep(chunks, k_range=range(2, min(8, len(chunks))))
print(k_results.to_string(index=False))
k_results.to_csv(OUTPUT_DIR / 'phase3_k_sweep.csv', index=False)

print('\nPhase 4 classifier benchmark:')
clf_results = run_classifier_benchmark(chunks)
print(clf_results.to_string(index=False))
clf_results.to_csv(OUTPUT_DIR / 'phase4_classifier_benchmark.csv', index=False)

print('\nPhase 5 keywords per class:')
kw = top_keywords_per_class(chunks)
for cls, top in list(kw.items())[:3]:
    print(f'  {cls}: {[k["kw"] for k in top]}')

print('\nPhase 6 architecture comparison:')
arch_results = run_architecture_comparison(tickets, chunks)
print(arch_results.to_string(index=False))
arch_results.to_csv(OUTPUT_DIR / 'phase6_architecture_comparison.csv', index=False)

print('\nPhase 7 KB consolidation:')
print('Class-pair Jaccard (top 5):')
print(class_pair_jaccard(chunks).head(5).to_string(index=False))
print(f'Orphan documents: {find_orphans(chunks, kb_root=KB_FOLDER_NEW)}')

cfg = export_dify_config(chunks, kw, arch_results, k_results)
(OUTPUT_DIR / 'dify_config.json').write_text(json.dumps(cfg, indent=2))
print(f'\nDify config written to {OUTPUT_DIR / "dify_config.json"}')
"""))

P(code(r"""# Quick visual: classifier benchmark and architecture comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ax = axes[0]
clf_plot = clf_results.dropna(subset=['macro_f1_mean']).copy()
ax.barh(clf_plot['model'], clf_plot['macro_f1_mean'],
        xerr=[(clf_plot['macro_f1_mean'] - clf_plot['macro_f1_lo']).abs(),
              (clf_plot['macro_f1_hi'] - clf_plot['macro_f1_mean']).abs()],
        color='#4967AA')
ax.set_xlabel('Macro F1 (95% bootstrap CI)')
ax.set_title('Phase 4, classifier representations')
ax.invert_yaxis()

ax = axes[1]
if not arch_results.empty and 'r_at_3' in arch_results.columns:
    ap = arch_results.dropna(subset=['r_at_3']).copy()
    ax.barh(ap['arch'], ap['r_at_3'],
            xerr=[(ap['r_at_3'] - ap['r_at_3_lo']).abs(),
                  (ap['r_at_3_hi'] - ap['r_at_3_mean'] if 'r_at_3_mean' in ap else
                   ap['r_at_3_hi'] - ap['r_at_3']).abs()],
            color='#F37340')
    ax.set_xlabel('Retrieval@3 (95% bootstrap CI)')
    ax.set_title('Phase 6, architecture comparison')
    ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_pipeline_overview.png', dpi=140, bbox_inches='tight')
plt.savefig(FIGURES_DIR / 'fig_pipeline_overview.pdf', bbox_inches='tight')
plt.show()
print(f'Figure saved to {FIGURES_DIR}')
"""))

P(md("""## Closing notes

- Real-data run: flip `SMOKE = False` in Phase 0b, edit the three path
  placeholders, optionally export `OPENAI_API_KEY`, then re-run all cells.
- Spot-check is in `claude_spot_check.ipynb`. Run it independently.
- Outputs live in `dify_exports/` and `figures/` inside this folder."""))


# =====================================================================
# Spot-check notebook
# =====================================================================
SPOT_CELLS: list[dict] = []
S = SPOT_CELLS.append

S(md("""# Claude spot-check, Maersk LLM before-and-after

Semi-manual notebook for the production-interface experiment. Linus pastes
trace fields from the Maersk chatbot UI; the notebook records them, computes
transitions, and emits the side-by-side comparison table.

The query set is fixed and stratified across KB classes. Each ticket is run
twice: once with the old KB, once with the refined KB. `RUN_LABEL` selects
which run is being recorded."""))

S(code(r"""from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

HERE = Path('.').resolve()
OUTPUT_DIR = HERE / 'dify_exports'
FIGURES_DIR = HERE / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Isolated to claude_implementation/ so the file never collides with Codex outputs.
QUERY_FILE = HERE / 'spot_check_queries.csv'
RUN_LABEL = 'before'  # flip to 'after' for the second pass
SEED = 42

print(f'RUN_LABEL = {RUN_LABEL}')
print(f'Query file: {QUERY_FILE}')
"""))

S(md("""## Step 1, Load or initialise the query set

If `planning/spot_check_queries.csv` does not exist, a 20-row template is
written and the cell exits so Linus can fill it in."""))

S(code(r"""if not QUERY_FILE.exists():
    template = pd.DataFrame({
        'ticket_id':   [f'Q{i:02d}' for i in range(20)],
        'ticket_text': ['<paste paraphrased ticket text here>'] * 20,
        'class_hint':  [''] * 20,
        'notes':       [''] * 20,
    })
    QUERY_FILE.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(QUERY_FILE, index=False)
    print(f'Template written to {QUERY_FILE}. Fill it in and re-run this cell.')
    queries = template
else:
    queries = pd.read_csv(QUERY_FILE)
    print(f'Loaded {len(queries)} queries')
    print(queries.head())
"""))

S(md("""## Step 2, Record traces

For each ticket, paste the Maersk chatbot trace into the dict below. Re-run
the cell to append. The cell appends one record at a time so partial runs are
fine."""))

S(code(r"""def record_trace(ticket_id, retrieved_doc, retrieved_chunk,
                 final_answer, latency_s,
                 correctness, usefulness, comment=''):
    "Append one trace to a per-run CSV. Validates grade enums."
    assert correctness in {'correct', 'partial', 'wrong', 'none'}, correctness
    assert usefulness in {'resolved', 'partial', 'escalation'}, usefulness
    out = OUTPUT_DIR / f'spot_check_{RUN_LABEL}.csv'
    row = {
        'ticket_id': ticket_id,
        'run_label': RUN_LABEL,
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
        'retrieved_doc': retrieved_doc,
        'retrieved_chunk': retrieved_chunk[:1000],
        'final_answer': final_answer[:1000],
        'latency_s': latency_s,
        'correctness': correctness,
        'usefulness': usefulness,
        'comment': comment,
    }
    df_new = pd.DataFrame([row])
    if out.exists():
        df_new = pd.concat([pd.read_csv(out), df_new], ignore_index=True)
    df_new.to_csv(out, index=False)
    print(f'Recorded {ticket_id} ({RUN_LABEL}). Total rows: {len(df_new)}')
    return df_new

# Example, copy this block per ticket and edit
# record_trace(
#     ticket_id='Q00',
#     retrieved_doc='tax/S4_entities.md',
#     retrieved_chunk='S4 entities are reported quarterly...',
#     final_answer='Submit form S4 by EOM.',
#     latency_s=3.2,
#     correctness='correct',
#     usefulness='resolved',
#     comment='Clean retrieval, full answer.',
# )
print('Recording cell ready. Use record_trace() for each ticket.')
"""))

S(md("""## Step 3, Analysis (run after both `before` and `after` are done)"""))

S(code(r"""def load_runs():
    rows = []
    for label in ('before', 'after'):
        f = OUTPUT_DIR / f'spot_check_{label}.csv'
        if f.exists():
            rows.append(pd.read_csv(f))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def transitions(runs: pd.DataFrame):
    if runs.empty:
        return pd.DataFrame()
    pivot = runs.pivot_table(
        index='ticket_id', columns='run_label',
        values='correctness', aggfunc='first',
    ).fillna('missing')
    pivot['transition'] = pivot.apply(
        lambda r: f'{r.get("before", "missing")} -> {r.get("after", "missing")}', axis=1)
    return pivot

def aggregate_counts(t: pd.DataFrame):
    if 'transition' not in t.columns:
        return {}
    return {
        'no_answer_to_answered':   ((t['before'] == 'none') &
                                    (t['after'].isin(['correct', 'partial']))).sum(),
        'wrong_to_correct':        ((t['before'] == 'wrong') &
                                    (t['after'] == 'correct')).sum(),
        'partial_to_correct':      ((t['before'] == 'partial') &
                                    (t['after'] == 'correct')).sum(),
        'regressions':             ((t['before'] == 'correct') &
                                    (t['after'].isin(['partial', 'wrong', 'none']))).sum(),
        'unchanged_correct':       ((t['before'] == 'correct') &
                                    (t['after'] == 'correct')).sum(),
        'still_no_answer':         ((t['before'] == 'none') &
                                    (t['after'] == 'none')).sum(),
    }

runs = load_runs()
if runs.empty:
    print('No traces recorded yet. Use record_trace() in step 2.')
else:
    t = transitions(runs)
    print('Per-ticket transitions:')
    print(t.to_string())
    print('\nAggregate counts:')
    for k, v in aggregate_counts(t).items():
        print(f'  {k}: {int(v)}')
    t.to_csv(OUTPUT_DIR / 'spot_check_transitions.csv')
"""))

S(md("""## Closing notes

- Two pass run: set `RUN_LABEL = 'before'`, point Maersk chatbot at the old
  KB, record all traces, then set `RUN_LABEL = 'after'`, point at the
  refined KB, record again.
- Aggregate counts feed Act 3 of the paper (`paper_scope_and_narrative.md`).
- Two illustrative traces should be lifted into `assets/figures/` for the
  report. The "S4 entities" chunk-cut-off case is the canonical motivating
  vignette."""))


# =====================================================================
# Build both notebooks
# =====================================================================
def main():
    pl = HERE / 'claude_pipeline_v3.ipynb'
    sc = HERE / 'claude_spot_check.ipynb'
    write_notebook(pl, PIPELINE_CELLS)
    write_notebook(sc, SPOT_CELLS)
    print(f'Wrote {pl}')
    print(f'Wrote {sc}')


if __name__ == '__main__':
    main()
