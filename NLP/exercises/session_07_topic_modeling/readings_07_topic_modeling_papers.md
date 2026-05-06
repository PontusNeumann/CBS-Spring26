# Session 7 — Topic Modeling Readings

## BERTopic: Neural topic modeling with a class-based TF-IDF procedure

**Author:** Maarten Grootendorst

**Abstract.** Topic models can be useful tools to discover latent topics in collections of documents. Recent studies have shown the feasibility of approach topic modeling as a clustering task. We present BERTopic, a topic model that extends this process by extracting coherent topic representation through the development of a class-based variation of TF-IDF. More specifically, BERTopic generates document embedding with pre-trained transformer-based language models, clusters these embeddings, and finally, generates topic representations with the class-based TF-IDF procedure. BERTopic generates coherent topics and remains competitive across a variety of benchmarks involving classical models and those that follow the more recent clustering approach of topic modeling.

**Comments.** BERTopic has a Python implementation.

**Subjects.** Computation and Language (cs.CL)

**Cite as.** arXiv:2203.05794 [cs.CL] (or arXiv:2203.05794v1 [cs.CL] for this version)

**DOI.** https://doi.org/10.48550/arXiv.2203.05794

**Submission history.** From: Maarten Grootendorst. v1: Fri, 11 Mar 2022 08:35:15 UTC (384 KB)

---

## Top2Vec: Distributed Representations of Topics

**Author:** Dimo Angelov

**Abstract.** Topic modeling is used for discovering latent semantic structure, usually referred to as topics, in a large collection of documents. The most widely used methods are Latent Dirichlet Allocation and Probabilistic Latent Semantic Analysis. Despite their popularity they have several weaknesses. In order to achieve optimal results they often require the number of topics to be known, custom stop-word lists, stemming, and lemmatization. Additionally these methods rely on bag-of-words representation of documents which ignore the ordering and semantics of words. Distributed representations of documents and words have gained popularity due to their ability to capture semantics of words and documents. We present top2vec, which leverages joint document and word semantic embedding to find topic vectors. This model does not require stop-word lists, stemming or lemmatization, and it automatically finds the number of topics. The resulting topic vectors are jointly embedded with the document and word vectors with distance between them representing semantic similarity. Our experiments demonstrate that top2vec finds topics which are significantly more informative and representative of the corpus trained on than probabilistic generative models.

**Comments.** Implementation available online.

**Subjects.** Computation and Language (cs.CL); Machine Learning (cs.LG); Machine Learning (stat.ML)

**Cite as.** arXiv:2008.09470 [cs.CL] (or arXiv:2008.09470v1 [cs.CL] for this version)

**DOI.** https://doi.org/10.48550/arXiv.2008.09470

**Submission history.** From: Dimo Angelov. v1: Wed, 19 Aug 2020 20:58:27 UTC (25,110 KB)
