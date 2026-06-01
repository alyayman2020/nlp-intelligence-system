"""BM25-based retrieval over 500K+ documents with sentiment filtering.

Uses ``rank_bm25`` for the inverted-index scoring (battle-tested Okapi BM25)
and the shared :class:`TextPreprocessor` for query/document tokenization so the
search vocabulary matches the classifier's. Each document carries a predicted
sentiment label so queries can be filtered ("show me only negative reviews
mentioning 'battery'").

Index build is the expensive step; it is serialized to disk and memory-mapped
back at API startup.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nlp_system import config
from nlp_system.pipeline.preprocess import TextPreprocessor
from nlp_system.utils import get_logger, timer

logger = get_logger(__name__)


@dataclass
class SearchHit:
    doc_id: int
    score: float
    text: str
    sentiment: int


class SearchEngine:
    """In-memory BM25 index with optional sentiment filtering."""

    def __init__(self, preprocessor: TextPreprocessor | None = None):
        self.preprocessor = preprocessor or TextPreprocessor.for_reviews()
        self._bm25 = None
        self.texts: list[str] = []
        self.sentiments: np.ndarray | None = None

    # ----- build / persist ------------------------------------------------- #
    def build(self, texts: list[str], sentiments: list[int]) -> SearchEngine:
        from rank_bm25 import BM25Okapi

        if len(texts) != len(sentiments):
            raise ValueError("texts and sentiments must be the same length")

        with timer(f"tokenize {len(texts)} docs", logger):
            tokenized = [self.preprocessor.tokens(t) for t in texts]
        with timer("build BM25 index", logger):
            self._bm25 = BM25Okapi(tokenized)
        self.texts = list(texts)
        self.sentiments = np.asarray(sentiments, dtype=np.int8)
        logger.info("Indexed %d documents", len(self.texts))
        return self

    def save(self, path: Path | None = None) -> Path:
        path = path or (config.MODELS_DIR / "search_index.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "texts": self.texts,
                    "sentiments": self.sentiments,
                    "config": self.preprocessor.config,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Saved search index -> %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> SearchEngine:
        path = path or (config.MODELS_DIR / "search_index.pkl")
        with open(path, "rb") as f:
            blob = pickle.load(f)
        engine = cls(TextPreprocessor(blob["config"]))
        engine._bm25 = blob["bm25"]
        engine.texts = blob["texts"]
        engine.sentiments = blob["sentiments"]
        logger.info("Loaded search index (%d docs) from %s", len(engine.texts), path)
        return engine

    # ----- query ----------------------------------------------------------- #
    def search(
        self,
        query: str,
        top_k: int = 10,
        sentiment: int | None = None,
        candidate_pool: int = 1000,
    ) -> list[SearchHit]:
        """Return top-k hits, optionally filtered to a sentiment class.

        We score all docs with BM25, take a generous candidate pool, then apply
        the sentiment filter and truncate to ``top_k``. Filtering post-scoring
        keeps relevance ranking intact within the chosen class.
        """
        if self._bm25 is None:
            raise RuntimeError("Index not built/loaded.")
        q_tokens = self.preprocessor.tokens(query)
        if not q_tokens:
            return []

        scores = self._bm25.get_scores(q_tokens)
        pool = min(candidate_pool, len(scores))
        order = np.argpartition(scores, -pool)[-pool:]
        order = order[np.argsort(scores[order])[::-1]]

        hits: list[SearchHit] = []
        for idx in order:
            if scores[idx] <= 0:
                continue
            sent = int(self.sentiments[idx])
            if sentiment is not None and sent != sentiment:
                continue
            hits.append(
                SearchHit(
                    doc_id=int(idx),
                    score=float(scores[idx]),
                    text=self.texts[idx],
                    sentiment=sent,
                )
            )
            if len(hits) >= top_k:
                break
        return hits
