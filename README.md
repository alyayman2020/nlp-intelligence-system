# NLP Intelligence System

> Production-grade sentiment analysis and document search over noisy, real-world text — a reusable NLP pipeline that ships a sentiment classifier **and** a BM25 search engine as a live, Dockerized FastAPI service.

[![CI](https://github.com/alyayman2020/nlp-intelligence-system/actions/workflows/ci.yml/badge.svg)](https://github.com/alyayman2020/nlp-intelligence-system/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Stack:** Python 3.10+ · scikit-learn · NLTK · rank-bm25 · gensim · MLflow · DVC · FastAPI · Docker · uv

---

## Overview

This system takes raw, messy text and turns it into two production capabilities behind a single API:

1. **Sentiment classification** — predict positive/negative sentiment with calibrated confidence.
2. **Semantic-aware search** — BM25 retrieval over hundreds of thousands of documents, with optional filtering by predicted sentiment.

It is built as a complete software project, not a notebook: data is versioned with **DVC**, every experiment is tracked in **MLflow**, the best model is persisted as a portable artifact, and the whole thing is served from a **Dockerized FastAPI** container with live, health-checked endpoints.

The design centers on one principle — **one configurable pipeline, many domains**. The same code runs over two deliberately contrasting datasets (long, well-formed product reviews and short, extremely noisy tweets); only the *configuration* changes, never the code. This makes the system a reusable foundation rather than a one-off model.

Design rationale lives in **[`PLAN.md`](PLAN.md)**. Step-by-step operational commands live in **[`RUNBOOK.md`](RUNBOOK.md)**.

> **Just want to see it run?** `make smoke` (macOS/Linux) trains and indexes a 2% sample end to end in under a minute. On Windows, run the three `uv run` smoke commands in the RUNBOOK §1.

---

## Highlights

* **One configurable preprocessor** (`TextPreprocessor`) adapts to domains via config — reviews strip HTML, tweets strip `@mentions` and collapse `loooove → loove`, and negation words are always protected from stopword removal. No copy-pasted pipelines.
* **Three vectorizers, one interface**: Bag-of-Words, TF-IDF, and a **custom Okapi BM25 transformer** built as a proper scikit-learn `TransformerMixin` (length-normalized, term-frequency saturated, L2-normalized output).
* **Empirical model selection**: the training stage sweeps all three vectorizers per dataset, logs every run to MLflow, and persists the best by validation F1 — so model choice is evidence-driven, not assumed.
* **Word embeddings**: Word2Vec trained on-corpus plus a 2-D projection (t-SNE/PCA) that surfaces semantic structure classical sparse vectors cannot represent.
* **Search engine**: BM25 retrieval over 500K+ documents with **sentiment filtering** (e.g. `"battery"` restricted to negative reviews only).
* **Production API**: Dockerized FastAPI exposing `/predict`, `/search`, and `/health`; artifacts are loaded once at startup and served from warm memory; runs as a non-root user with a container healthcheck.
* **Reproducible by construction**: DVC pipeline (`dvc repro`), MLflow tracking, a committed `uv.lock`, CI on GitHub Actions, and a full test suite.

---

## Architecture

```
raw CSV ──► make_dataset ──► processed parquet ──► train (3 vectorizers) ──► best pipeline.joblib
                                   │                                                │
                                   └────────► build_index ──► search_index.pkl ◄────┘ (predicted sentiments)
                                                                   │
                                                        FastAPI (predict + search)
                                                                   │
                                                            Docker container
```

MLflow tracks every training run; DVC versions the raw data and pins each pipeline stage. The trained pipeline bundles preprocessing + vectorizer + classifier into a single artifact, so there is no train/serve skew — the API loads exactly what was trained.

---

## Quickstart

This project uses **[uv](https://docs.astral.sh/uv/)** for environment and dependency management. With uv you don't need to manually activate the venv — `uv run <cmd>` always uses the project environment.

### 1. Install uv (one time)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal, then verify: `uv --version`.

### 2. Clone & install

```bash
git clone https://github.com/alyayman2020/nlp-intelligence-system.git
cd nlp-intelligence-system

uv venv --python 3.11        # create .venv with Python 3.11
uv pip install -e ".[dev]"   # runtime + dev deps (editable)
uv run python -m nlp_system.pipeline.nltk_setup   # download NLTK corpora
```

> A committed **`uv.lock`** pins exact versions for reproducible installs. To install the locked set exactly, run `uv sync --extra dev` (creates the venv and installs from the lockfile in one step).

> **Activation is optional with uv.** If you prefer an activated shell:
> - **Windows (PowerShell):** `.venv\Scripts\activate`
> - **macOS / Linux:** `source .venv/bin/activate`
>
> Otherwise just prefix commands with `uv run` as shown below.

### 3. Configure (optional)

```bash
cp .env.example .env   # PowerShell: copy .env.example .env
# Defaults work out of the box. Set KAGGLE_USERNAME / KAGGLE_KEY for downloads,
# or place your kaggle.json (see RUNBOOK §0).
```

### 4. Run the pipeline

```bash
uv run nlp-download                              # pull both datasets -> data/raw
uv run nlp-data                                  # clean, label-map, stratified split -> parquet
uv run nlp-train                                 # BoW/TF-IDF/BM25 per dataset, logged to MLflow
uv run nlp-index --dataset amazon_fine_food      # build BM25 search index
uv run uvicorn nlp_system.api.main:app --reload  # FastAPI on http://localhost:8000 (docs at /docs)
```

On macOS/Linux you can also use the `make` shortcuts (`make data`, `make train`, …); they call the same commands.

### 5. Hit the API

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"the battery dies within an hour, very disappointed\",\"dataset\":\"amazon_fine_food\"}"

curl -s -X POST localhost:8000/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"battery life\",\"top_k\":5,\"sentiment\":0,\"dataset\":\"amazon_fine_food\"}"
```

> **Windows tip:** PowerShell aliases `curl` to `Invoke-WebRequest` and mangles inline JSON. Use the interactive docs at http://localhost:8000/docs, or `Invoke-RestMethod` (see RUNBOOK §8).

---

## Deployment (Docker)

The service ships as a container that loads trained artifacts at startup. Train the models first (steps above), then:

```bash
# Build the image (installs the package + pre-downloads NLTK corpora, non-root user)
docker build -f docker/Dockerfile -t nlp-intelligence-system:latest .

# Run it, mounting your trained models into the container
docker run --rm -p 8000:8000 -v ${PWD}/models:/app/models nlp-intelligence-system:latest
```

`docker ps` will report the container as `(healthy)` once the `/health` healthcheck passes. The image deliberately does **not** bake in models or data — artifacts are mounted at runtime, matching how real services load versioned artifacts from storage.

**Full stack (API + MLflow tracking server):**

```bash
docker compose -f docker/docker-compose.yml up --build
```

API on `:8000`, MLflow UI on `:5000`. The same image runs on any container host (AWS App Runner / ECS, Google Cloud Run, Azure Container Apps, etc.).

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service status + which models/indexes are loaded |
| `POST` | `/predict` | Classify one text → `{label, sentiment, confidence}` |
| `POST` | `/search` | BM25 retrieval with optional `sentiment` filter |
| `GET` | `/docs` | Interactive OpenAPI (Swagger) UI |

**Example `/predict` response:**

```json
{ "label": 0, "sentiment": "negative", "confidence": 0.966, "dataset": "amazon_fine_food" }
```

---

## The Two Datasets

The system is validated on two intentionally contrasting corpora to prove the pipeline generalizes across domains:

| | **Amazon Fine Food Reviews** | **Sentiment140** |
|---|---|---|
| Unit | Long product reviews | Short tweets (≤140 chars) |
| Noise | HTML tags, run-ons | `@mentions`, URLs, `#tags`, `sooo gooood` |
| Label source | Star score (1–2 neg, 4–5 pos, 3 dropped) | 0 = neg, 4 = pos |
| Encoding | utf-8 | latin-1 |
| Preprocessor | `TextPreprocessor.for_reviews()` | `TextPreprocessor.for_tweets()` |

Same code path, two configs. The behavioral differences this surfaces — vocabulary size, BM25's length normalization mattering more on long reviews, embeddings being noisier on tweets — are explored in the notebooks and `PLAN.md §5`.

---

## Repository Structure

```
nlp-intelligence-system/
├── PLAN.md                      # Design doc: decisions, trade-offs, acceptance gates
├── RUNBOOK.md                   # Exact CLI commands, troubleshooting
├── README.md                    # This file
├── pyproject.toml               # Deps + console scripts + tooling config
├── uv.lock                      # Pinned, reproducible dependency set
├── Makefile                     # Pipeline shortcuts (wrap uv run)
├── dvc.yaml / params.yaml       # DVC pipeline + surfaced hyperparameters
├── .env.example                 # Copy -> .env
│
├── src/nlp_system/
│   ├── config.py                # Paths, seeds, DatasetSpec (Amazon + Sentiment140)
│   ├── utils.py                 # logger, set_seed, timer
│   ├── pipeline/
│   │   ├── preprocess.py        # TextPreprocessor — the configurable core
│   │   └── nltk_setup.py        # Idempotent corpora download
│   ├── data/
│   │   ├── download.py          # kagglehub -> data/raw
│   │   └── make_dataset.py      # clean, label-map, dedup, stratified split
│   ├── features/
│   │   ├── vectorizers.py       # BoW / TF-IDF factory + BM25Vectorizer
│   │   └── embeddings.py        # Word2Vec train + t-SNE/PCA plot
│   ├── models/
│   │   └── train.py             # 3-vectorizer sweep, MLflow, best-model persist
│   ├── search/
│   │   ├── engine.py            # SearchEngine (BM25 + sentiment filter)
│   │   └── build_index.py       # Build + serialize the index
│   └── api/
│       ├── main.py              # FastAPI app (lifespan-loaded artifacts)
│       └── schemas.py           # Pydantic request/response models
│
├── tests/                       # Unit + smoke tests (pytest)
├── notebooks/                   # EDA + vectorizer comparison + embeddings
├── docker/                      # Dockerfile + docker-compose (API + MLflow)
├── .github/workflows/ci.yml     # Lint + test (3.10/3.11) + docker build
├── data/{raw,interim,processed} # gitignored; DVC-tracked
├── models/                      # trained pipelines + search indexes (gitignored)
└── reports/figures/             # embedding plots, comparison charts
```

---

## Capabilities → Where They Live

| Capability | Implementation |
|---|---|
| Reusable, configurable preprocessing | `pipeline/preprocess.py` |
| Vectorization (BoW / TF-IDF / BM25) + comparison | `features/vectorizers.py` + `models/train.py` |
| Word embeddings & distributional analysis | `features/embeddings.py` + `notebooks/03_embeddings.ipynb` |
| Data + experiment versioning | `dvc.yaml` + MLflow in `models/train.py` |
| BM25 search with sentiment filtering | `search/engine.py` |
| Production API & container | `api/` + `docker/` |

---

## Development

```bash
uv run pytest                  # test suite
uv run ruff check src tests    # lint
uv run black --check src tests # format check
# macOS/Linux shortcuts: make test / make lint / make format
```

MLflow UI: `uv run mlflow ui` then open http://localhost:5000 (or `docker compose -f docker/docker-compose.yml up mlflow`).

---

## License

MIT — see [`LICENSE`](LICENSE).

---

**Author:** Aly Ayman
