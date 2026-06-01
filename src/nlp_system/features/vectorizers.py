"""Vectorizers: BoW, TF-IDF, and a custom BM25 transformer.

BoW and TF-IDF are thin factories over scikit-learn. BM25 is implemented as a
proper sklearn transformer (Okapi BM25 weighting applied on top of a count
matrix) so it slots into pipelines and the search engine alike.

Why BM25 for vectorization? TF-IDF grows linearly with term frequency; BM25
saturates it (the `k1` term) and normalizes by document length (the `b` term).
For sentiment this often helps on long, verbose reviews where a word repeated
20 times should not dominate.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_vectorizer(method: str, **kwargs):
    """Factory returning a fitted-on-call vectorizer for the given method.

    Parameters
    ----------
    method : {"bow", "tfidf", "bm25"}
    kwargs : forwarded to the underlying vectorizer (e.g. ``max_features``,
        ``ngram_range``, ``min_df``).
    """
    method = method.lower()
    defaults = dict(max_features=50_000, ngram_range=(1, 2), min_df=2)
    defaults.update(kwargs)
    if method == "bow":
        return CountVectorizer(**defaults)
    if method == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **defaults)
    if method == "bm25":
        return BM25Vectorizer(**defaults)
    raise ValueError(f"Unknown method '{method}'. Use bow|tfidf|bm25.")


class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """Okapi BM25 as an sklearn transformer.

    Internally fits a :class:`CountVectorizer`, learns IDF and average document
    length at ``fit`` time, then applies BM25 term weighting at ``transform``.
    The output is L2-normalized so it behaves well for linear classifiers and
    cosine similarity in the search engine.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        max_features: int | None = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
    ):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df

    def fit(self, X, y=None):
        self._count = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
        )
        counts = self._count.fit_transform(X)
        n_docs, _ = counts.shape
        df = np.asarray((counts > 0).sum(axis=0)).ravel()
        # BM25 idf with the +1 smoothing that keeps weights non-negative.
        self.idf_ = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        self.avgdl_ = float(doc_len.mean()) if n_docs else 0.0
        self.vocabulary_ = self._count.vocabulary_
        return self

    def transform(self, X):
        counts = self._count.transform(X).tocsr().astype(np.float64)
        doc_len = np.asarray(counts.sum(axis=1)).ravel()
        # Length normalization factor per row: k1 * (1 - b + b * dl/avgdl)
        denom_norm = self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl_ + 1e-9)))
        # Saturate term frequencies row-by-row, then scale columns by idf.
        data, _indices, indptr = counts.data, counts.indices, counts.indptr
        out = counts.copy()
        for row in range(counts.shape[0]):
            start, end = indptr[row], indptr[row + 1]
            tf = data[start:end]
            out.data[start:end] = tf * (self.k1 + 1.0) / (tf + denom_norm[row])
        out = out.multiply(self.idf_)  # broadcast idf across columns
        out = sp.csr_matrix(out)
        # L2 normalize rows.
        norms = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        inv = sp.diags(1.0 / norms)
        return inv @ out

    def get_feature_names_out(self, input_features=None):
        return self._count.get_feature_names_out(input_features)
