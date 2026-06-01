"""FastAPI application: live sentiment + search endpoints.

Artifacts (one classifier and one search index per dataset) are loaded once at
startup via the lifespan handler and held in module state, so requests are
served from warm memory. Missing artifacts degrade gracefully: the service
still boots and reports what is available at ``/health``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from nlp_system import __version__, config
from nlp_system.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
)
from nlp_system.pipeline.nltk_setup import ensure_nltk
from nlp_system.search.engine import SearchEngine
from nlp_system.utils import get_logger

logger = get_logger("nlp_system.api")

# Warm in-memory artifact registries.
_MODELS: dict[str, object] = {}
_INDEXES: dict[str, SearchEngine] = {}

_LABELS = {0: "negative", 1: "positive"}


def _load_artifacts() -> None:
    ensure_nltk()
    for name in config.DATASETS:
        model_path = config.MODELS_DIR / name / "model.joblib"
        if model_path.exists():
            _MODELS[name] = joblib.load(model_path)
            logger.info("Loaded classifier: %s", name)
        index_path = config.MODELS_DIR / f"search_index_{name}.pkl"
        if index_path.exists():
            _INDEXES[name] = SearchEngine.load(Path(index_path))
            logger.info("Loaded search index: %s", name)
    if not _MODELS:
        logger.warning("No classifiers loaded — run training first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NLP Intelligence System v%s", __version__)
    _load_artifacts()
    yield
    _MODELS.clear()
    _INDEXES.clear()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="NLP Intelligence System",
    description="Sentiment classification + BM25 search over noisy text.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=sorted(_MODELS),
        indexes_loaded=sorted(_INDEXES),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pipe = _MODELS.get(req.dataset)
    if pipe is None:
        raise HTTPException(
            404, f"No model for dataset '{req.dataset}'. Loaded: {sorted(_MODELS)}"
        )
    proba = float(pipe.predict_proba([req.text])[0][1])
    label = int(proba >= 0.5)
    return PredictResponse(
        label=label,
        sentiment=_LABELS[label],
        confidence=proba if label == 1 else 1.0 - proba,
        dataset=req.dataset,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    engine = _INDEXES.get(req.dataset)
    if engine is None:
        raise HTTPException(
            404, f"No index for dataset '{req.dataset}'. Loaded: {sorted(_INDEXES)}"
        )
    hits = engine.search(req.query, top_k=req.top_k, sentiment=req.sentiment)
    return SearchResponse(
        query=req.query,
        count=len(hits),
        results=[
            SearchHitOut(
                doc_id=h.doc_id, score=h.score, text=h.text, sentiment=h.sentiment
            )
            for h in hits
        ],
    )


@app.get("/")
def root():
    return {"service": "nlp-intelligence-system", "version": __version__, "docs": "/docs"}
