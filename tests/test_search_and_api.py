"""Tests for the BM25 search engine and a FastAPI smoke test."""

import pytest

from nlp_system.pipeline.preprocess import PreprocessConfig, TextPreprocessor
from nlp_system.search.engine import SearchEngine

DOCS = [
    "the battery life on this phone is amazing and lasts all day",
    "terrible battery, dies within an hour, very disappointed",
    "great camera quality and sharp display",
    "the screen cracked after one drop, poor build quality",
    "love this product, fast shipping and excellent packaging",
]
SENTS = [1, 0, 1, 0, 1]


@pytest.fixture
def engine():
    pre = TextPreprocessor(PreprocessConfig(lemmatize=False))
    return SearchEngine(pre).build(DOCS, SENTS)


def test_search_returns_relevant(engine):
    hits = engine.search("battery", top_k=5)
    assert hits, "expected at least one battery hit"
    assert all("battery" in h.text for h in hits[:2])


def test_sentiment_filter(engine):
    neg = engine.search("battery", top_k=5, sentiment=0)
    assert all(h.sentiment == 0 for h in neg)


def test_empty_query(engine):
    assert engine.search("", top_k=5) == []


def test_save_load_roundtrip(engine, tmp_path):
    p = tmp_path / "idx.pkl"
    engine.save(p)
    loaded = SearchEngine.load(p)
    assert len(loaded.texts) == len(DOCS)
    assert loaded.search("camera", top_k=3)


def test_api_health():
    from fastapi.testclient import TestClient

    from nlp_system.api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
