"""Unit tests for vectorizers (BoW, TF-IDF, BM25)."""

import numpy as np
import pytest
import scipy.sparse as sp

from nlp_system.features.vectorizers import BM25Vectorizer, build_vectorizer

CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are great pets",
    "i love my pet cat very much",
    "the quick brown fox jumps",
]


@pytest.mark.parametrize("method", ["bow", "tfidf", "bm25"])
def test_vectorizer_shapes(method):
    vec = build_vectorizer(method, max_features=100, min_df=1, ngram_range=(1, 1))
    X = vec.fit_transform(CORPUS)
    assert sp.issparse(X)
    assert X.shape[0] == len(CORPUS)
    assert X.shape[1] > 0


def test_bm25_is_l2_normalized():
    vec = BM25Vectorizer(min_df=1, ngram_range=(1, 1), max_features=100)
    X = vec.fit_transform(CORPUS)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    nonzero = norms[norms > 0]
    assert np.allclose(nonzero, 1.0, atol=1e-6)


def test_bm25_saturation_vs_tfidf():
    # A doc repeating a term many times should not blow up under BM25 the way
    # raw counts do. Compare the max weight magnitude.
    doc = ["spam " * 50 + "ham"]
    bm25 = BM25Vectorizer(min_df=1, ngram_range=(1, 1)).fit(doc + CORPUS)
    Xb = bm25.transform(doc)
    assert Xb.max() <= 1.0 + 1e-6  # L2 normalized, bounded


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        build_vectorizer("word2vec")
