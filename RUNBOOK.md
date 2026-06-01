# RUNBOOK.md — Operations Reference

Exact, copy-pasteable commands for every stage, using **uv**. Assumes you're in the repo root.

> **No activation needed.** Every command below uses `uv run`, which automatically uses the project's `.venv`. If you'd rather activate the shell once:
> **Windows (PowerShell):** `.venv\Scripts\activate` · **macOS/Linux:** `source .venv/bin/activate` — then drop the `uv run` prefix.

> **PowerShell note:** PowerShell separates commands with `;`, **not** `&&`. And `curl` is an alias for `Invoke-WebRequest` — use `curl.exe` for the API examples, or just use http://localhost:8000/docs.

---

## 0. One-time setup

Install uv (skip if already installed — check with `uv --version`):
```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment and install the project:
```bash
uv venv --python 3.11           # creates .venv
uv pip install -e ".[dev]"      # runtime + dev deps, editable
uv run python -m nlp_system.pipeline.nltk_setup   # NLTK corpora

# config (optional; defaults work)
cp .env.example .env            # PowerShell: copy .env.example .env
```

**Kaggle credentials** (only for downloading data): set environment variables —
```powershell
# PowerShell (current session)
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_key"
```
```bash
# macOS / Linux
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```
— or place `kaggle.json` at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) / `~/.kaggle/kaggle.json` (macOS/Linux, then `chmod 600`).

---

## 1. Fast path — prove it works in <1 min

```bash
uv run python -m nlp_system.data.make_dataset --dataset sentiment140 --sample-frac 0.02
uv run python -m nlp_system.models.train --dataset sentiment140
uv run python -m nlp_system.search.build_index --dataset sentiment140
uv run uvicorn nlp_system.api.main:app --reload    # http://localhost:8000/docs
```
(macOS/Linux shortcut: `make smoke` then `make serve`.)

In another terminal:
```bash
curl.exe -s localhost:8000/health
curl.exe -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d "{\"text\":\"this is awful and broke immediately\",\"dataset\":\"sentiment140\"}"
```

---

## 2. Full pipeline

### 2.1 Download

```bash
uv run nlp-download                              # both datasets
uv run nlp-download --dataset sentiment140       # just one
```
Raw CSVs land in `data/raw/{amazon_fine_food,sentiment140}.csv`.

### 2.2 Prepare (clean + split)

```bash
uv run nlp-data                                  # both
uv run nlp-data --dataset amazon_fine_food       # one
uv run nlp-data --sample-frac 0.05               # 5% sample for fast dev
```
Writes `data/processed/<dataset>/{train,val,test}.parquet`.

### 2.3 Train

```bash
uv run nlp-train                                 # both datasets, all 3 vectorizers
uv run nlp-train --dataset sentiment140

# custom vectorizer subset:
uv run nlp-train --dataset amazon_fine_food --methods tfidf bm25 --max-features 80000
```
Outputs: `models/<dataset>/model.joblib` (best by val F1) + `metrics.json`. Every run logged to MLflow.

### 2.4 Build search index

```bash
uv run nlp-index --dataset amazon_fine_food
uv run nlp-index --dataset sentiment140 --max-docs 200000
uv run nlp-index --dataset amazon_fine_food --true-labels   # ground truth instead of predictions
```
Outputs: `models/search_index_<dataset>.pkl`.

### 2.5 Serve

```bash
uv run uvicorn nlp_system.api.main:app --reload                       # dev (auto-reload)
# production-style:
uv run uvicorn nlp_system.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

> On macOS/Linux the `make` shortcuts also work: `make download`, `make data DATASET=sentiment140`, `make train`, `make index`, `make serve`. They wrap the same `uv run` commands.

---

## 3. MLflow

```bash
uv run mlflow ui                    # http://localhost:5000
# or via compose:
docker compose -f docker/docker-compose.yml up mlflow
```
Compare runs by filtering on `params.dataset` and `params.vectorizer`; sort by `metrics.val_f1`.

---

## 4. DVC

```bash
make dvc-init                       # dvc init + dvc add data/raw
dvc repro                           # run the prepare -> train -> index DAG
dvc repro train@amazon_fine_food    # a single stage
dvc metrics show                    # tabulate models/*/metrics.json
dvc exp run --set-param bm25.k1=2.0 # experiment with a hyperparameter
dvc exp show
```
To push data/models to remote storage (e.g. S3): set a remote with `dvc remote add` then `dvc push`.

---

## 5. Docker

```bash
make docker-build                   # builds nlp-intelligence-system:latest
make docker-run                     # runs API, mounts ./models read-only

# full stack (API + MLflow):
docker compose -f docker/docker-compose.yml up --build
```
The image pre-downloads NLTK corpora, runs as non-root, and has a `/health` healthcheck.

> **Note:** train + index *before* `docker-run` — the container mounts `./models`. An image with no models still boots; `/health` will show empty lists.

---

## 6. Quality gates

```bash
uv run pytest                                       # 17 tests
uv run ruff check src tests                         # lint
uv run black --check src tests                      # format check
uv run pytest --cov=nlp_system --cov-report=term-missing
```
(macOS/Linux: `make test`, `make lint`, `make format`.)

---

## 7. Embeddings (notebook or REPL)

Launch a REPL or Jupyter inside the env:
```bash
uv run python            # REPL
uv run jupyter notebook  # notebooks (add jupyter first: uv pip install jupyter)
```
```python
from nlp_system.data.make_dataset import load_split
from nlp_system.pipeline.preprocess import TextPreprocessor
from nlp_system.features.embeddings import train_word2vec, most_similar_report, plot_embeddings

df = load_split("amazon_fine_food", "train")
pre = TextPreprocessor.for_reviews()
docs = [pre.tokens(t) for t in df["text"]]

model = train_word2vec(docs, vector_size=100, epochs=5)
print(most_similar_report(model, ["great", "awful", "delicious"]))
plot_embeddings(model, n_words=200, method="tsne")   # -> reports/figures/embeddings_tsne.png
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `The token '&&' is not a valid statement separator` | You're in PowerShell — use `;` instead of `&&`, run commands one per line |
| `source : The term 'source' is not recognized` | PowerShell activation is `.venv\Scripts\activate` (or just use `uv run`) |
| `uv : command not found` | Reopen the terminal after installing uv, or re-run the install one-liner |
| `No virtual environment found` | Run `uv venv --python 3.11` first |
| `curl` behaves oddly on Windows | PowerShell aliases `curl`→`Invoke-WebRequest`; use `curl.exe` or the `/docs` UI |
| `LookupError: Resource stopwords not found` | `uv run python -m nlp_system.pipeline.nltk_setup` |
| `FileNotFoundError: ...amazon_fine_food.csv` | run `uv run nlp-download` first, or check `DATA_DIR` in `.env` |
| `kagglehub` 403 / auth error | set `KAGGLE_USERNAME`/`KAGGLE_KEY` or place `kaggle.json` (see §0) |
| `Only N usable rows ... after cleaning/dedup` | wrong file/encoding, or you sampled too hard — check `data/raw/*.csv` is the real Kaggle file |
| `/predict` returns 404 | model not trained for that dataset — `uv run nlp-train --dataset ...` |
| `/search` returns 404 | index not built — `uv run nlp-index --dataset ...` |
| API slow on first request | first call warms the NLTK lemmatizer cache; later calls are fast |
| Docker `/health` shows empty lists | mount trained models: `-v ${PWD}/models:/app/models` |

---

## 9. Console scripts (installed by `uv pip install -e .`)

```bash
uv run nlp-download --dataset all
uv run nlp-data     --dataset amazon_fine_food
uv run nlp-train    --dataset sentiment140
uv run nlp-index    --dataset amazon_fine_food
```
Equivalent to the `uv run python -m nlp_system.*` invocations above. (If the env is activated, drop the `uv run` prefix.)
