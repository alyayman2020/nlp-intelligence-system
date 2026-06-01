# PLAN.md — Design Document

NLP Intelligence System · LAB1, Individual NLP Course, Week 1.

This document records the *why* behind the engineering decisions. The README is the front door; the RUNBOOK is the command reference; this is the rationale.

---

## 1. Problem Framing

Build a single NLP system that does two production tasks — **sentiment classification** and **document search** — over two deliberately contrasting datasets, packaged as a tracked, versioned, deployable software project rather than a notebook.

The contrast between datasets is the pedagogical core: identical code, identical algorithms, different behavior. Every design choice below is made so that *one* implementation serves *both* domains and the differences fall out of configuration and data, not forked code.

---

## 2. Architecture Overview

```
raw CSV ──► make_dataset ──► processed parquet ──► train (3 vectorizers) ──► best pipeline.joblib
                                     │                                              │
                                     └────────► build_index ──► search_index.pkl ◄──┘ (predicted sentiments)
                                                                     │
                                                          FastAPI (predict + search)
                                                                     │
                                                              Docker container
```

Tracking and versioning wrap the whole thing: **MLflow** logs every training run; **DVC** versions raw data and pins the pipeline stages.

---

## 3. Component Decisions

### 3.1 One configurable preprocessor (`TextPreprocessor`)

The single most important design choice. Instead of `clean_reviews()` and `clean_tweets()`, there is one class driven by a `PreprocessConfig` dataclass. Factory methods (`for_reviews()`, `for_tweets()`) return domain-tuned instances.

* **Reviews**: strip HTML (`<br>` is rampant in Amazon data), lemmatize, no repeat-collapsing.
* **Tweets**: strip `@mentions` and URLs, keep hashtag *text* (`#deal → deal`), collapse character runs (`soooo → soo`).
* **Shared and non-negotiable**: negation words (`not`, `no`, `never`, …) survive stopword removal. Dropping them inverts sentiment ("not good" → "good"), the single most common preprocessing bug in sentiment work.

It implements `fit`/`transform`/`fit_transform` so it drops into an sklearn `Pipeline` and pickles with the trained model — the API loads one artifact, no preprocessing drift between train and serve.

### 3.2 Vectorizers (BoW, TF-IDF, BM25)

BoW and TF-IDF are thin factories over scikit-learn. **BM25 is implemented from scratch** as a `BaseEstimator`/`TransformerMixin`:

* Fits an internal `CountVectorizer`, learns IDF and average document length at `fit`.
* At `transform`, applies Okapi BM25 term weighting: `tf · (k1+1) / (tf + k1·(1−b + b·dl/avgdl))`, scales columns by IDF, L2-normalizes rows.
* `k1=1.5`, `b=0.75` (standard Okapi defaults; surfaced in `params.yaml`).

**Why bother when TF-IDF exists?** TF-IDF's term-frequency term grows without bound; BM25 *saturates* it (the `k1` term) and *normalizes by length* (the `b` term). On long Amazon reviews where a word appears 20×, BM25 prevents that single document from dominating. The hypothesis to test empirically: BM25 ≥ TF-IDF > BoW, with the largest gap on Amazon (long docs) and a smaller gap on Sentiment140 (short docs, length normalization matters less).

### 3.3 Classifier

Logistic Regression (`class_weight="balanced"`, `C=1.0`). Deliberately simple: the lab's variable is the *vectorizer*, so the classifier is held constant to make the comparison clean. LogReg is also fast over 1.6M tweets and gives calibrated-enough probabilities for the API's confidence score.

### 3.4 Word embeddings

Word2Vec (gensim) trained on the corpus tokens, projected to 2-D via t-SNE (PCA fallback). The deliverable is the *visualization plus the articulation*: show that `excellent`, `great`, `delicious` cluster, which one-hot/TF-IDF vectors (orthogonal by construction) can never represent. This is qualitative evidence for the "distributional meaning" objective, not a classifier input.

### 3.5 Search engine

`rank-bm25` (`BM25Okapi`) for the inverted-index scoring — battle-tested, and reimplementing the index structure adds no learning value beyond the vectorizer BM25 above. Tokenization reuses the shared `TextPreprocessor`, so the search vocabulary matches the classifier's.

**Sentiment filtering** is post-scoring: score all docs, take a generous candidate pool (default 1000), filter to the requested class, truncate to `top_k`. This preserves BM25 relevance ranking *within* the chosen sentiment. Each document's sentiment is the classifier's *prediction* (search returns what the model thinks), not the ground-truth label.

### 3.6 API

FastAPI with a lifespan handler that loads all classifiers and indexes once at startup into module-level registries. Requests are served from warm memory. Missing artifacts degrade gracefully — the service still boots and `/health` reports what's available, so a partially-trained system is still inspectable.

---

## 4. Data Handling

* **Label mapping** per `DatasetSpec`: Amazon 1–2★→neg, 4–5★→pos, **3★ dropped** (genuine neutral, no clean binary signal). Sentiment140 0→neg, 4→pos.
* **Dedup** exact-duplicate texts (both datasets contain many).
* **Stratified split** 80/10/10 train/val/test, seeded.
* **Output** parquet (columnar, fast, typed) under `data/processed/<dataset>/`.
* `--sample-frac` flag for fast dev iterations without touching the full 1.6M rows.

---

## 5. Analysis the Lab Expects (filled during runs)

1. **Vectorizer comparison table** — val/test F1 + ROC-AUC for BoW/TF-IDF/BM25 on each dataset (auto-logged to MLflow; mirrored in `models/<dataset>/metrics.json`).
2. **Cross-dataset contrast** — vocabulary sizes, OOV behavior, where BM25's length normalization helps most.
3. **Embedding neighborhoods** — nearest-neighbor probes for a fixed word set on each corpus.
4. **Error analysis** — negation handling, sarcasm on tweets, the dropped-neutral effect on Amazon.

---

## 6. Reproducibility

* **DVC** (`dvc.yaml`): `prepare → train → index` stages with declared deps/outs; `dvc repro` skips unchanged stages; `params.yaml` enables `dvc exp run --set-param`.
* **MLflow**: params, metrics, and the dataset/vectorizer tags for every run; local file store by default, swappable via `MLFLOW_TRACKING_URI`.
* **Seeds** centralized in `config.SEED`, applied in every entry point.
* **Pinned deps** in `pyproject.toml`; CI runs on 3.10 and 3.11.

---

## 7. Acceptance Gates

| Gate | Threshold |
|---|---|
| `make smoke` runs E2E (sample) | passes, API serves |
| Test suite | 17/17 green |
| Lint | ruff + black clean |
| Each dataset: 3 vectorizers trained + logged | MLflow shows 3 runs/dataset |
| Best model persisted + loadable by API | `/health` lists it |
| Search returns relevant, sentiment-filterable hits | manual + `test_search_and_api.py` |
| Docker image builds and `/health` returns 200 | CI `docker` job |

Sanity expectation on real data (not a hard gate): TF-IDF/BM25 val F1 ≥ BoW; absolute F1 ≥ ~0.85 (Amazon) / ~0.78 (Sentiment140) with this simple LogReg setup. Treat large deviations as a signal to audit preprocessing.

---

## 8. Deliberate Non-Goals

* No deep transformer fine-tuning — out of scope for Week 1; the embeddings section motivates *why* one would move there next.
* No hyperparameter search on the classifier — it's the controlled variable.
* No distributed serving — single-container FastAPI is the target; horizontal scaling is a deployment concern, not a lab one.

---

## 9. Future Work

* Swap LogReg → fine-tuned DistilBERT, reusing the same data + API contract.
* Approximate-NN index (FAISS) over sentence embeddings as a semantic-search arm alongside BM25.
* Cross-dataset transfer: train on Amazon, evaluate on Sentiment140, quantify domain shift.
