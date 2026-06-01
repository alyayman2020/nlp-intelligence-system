.PHONY: help venv install install-dev nltk download data train index serve all \
        test lint format smoke docker-build docker-run dvc-init clean

# All Python commands run through uv (no manual venv activation needed).
RUN ?= uv run
DATASET ?= all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the uv virtual environment (.venv) with Python 3.11
	uv venv --python 3.11

install:  ## Install runtime deps (editable) via uv
	uv pip install -e .

install-dev:  ## Install runtime + dev deps via uv
	uv pip install -e ".[dev]"

nltk:  ## Download required NLTK corpora
	$(RUN) python -m nlp_system.pipeline.nltk_setup

download:  ## Download raw Kaggle datasets -> data/raw
	$(RUN) python -m nlp_system.data.download --dataset $(DATASET)

data:  ## Clean + split -> data/processed  (use SAMPLE=0.05 for fast runs)
	$(RUN) python -m nlp_system.data.make_dataset --dataset $(DATASET) $(if $(SAMPLE),--sample-frac $(SAMPLE),)

train:  ## Train BoW/TF-IDF/BM25 models, log to MLflow, save best
	$(RUN) python -m nlp_system.models.train --dataset $(DATASET)

index:  ## Build BM25 search index (DATASET must be a single dataset)
	$(RUN) python -m nlp_system.search.build_index --dataset $(if $(filter all,$(DATASET)),amazon_fine_food,$(DATASET))

serve:  ## Run the FastAPI service locally
	$(RUN) uvicorn nlp_system.api.main:app --host 0.0.0.0 --port 8000 --reload

all: nltk data train index  ## Full pipeline (assumes data already downloaded)

test:  ## Run the test suite
	$(RUN) pytest

smoke:  ## Fast end-to-end on a 2% sample of one dataset
	$(RUN) python -m nlp_system.data.make_dataset --dataset sentiment140 --sample-frac 0.02
	$(RUN) python -m nlp_system.models.train --dataset sentiment140
	$(RUN) python -m nlp_system.search.build_index --dataset sentiment140

lint:  ## Lint (ruff + black --check)
	$(RUN) ruff check src tests
	$(RUN) black --check src tests

format:  ## Auto-format (isort + black + ruff --fix)
	$(RUN) isort src tests
	$(RUN) black src tests
	$(RUN) ruff check --fix src tests

dvc-init:  ## Initialize DVC and track raw data
	$(RUN) dvc init
	$(RUN) dvc add data/raw

docker-build:  ## Build the API image
	docker build -f docker/Dockerfile -t nlp-intelligence-system:latest .

docker-run:  ## Run the API container (mounts ./models)
	docker run --rm -p 8000:8000 -v $(PWD)/models:/app/models nlp-intelligence-system:latest

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.egg-info build dist
