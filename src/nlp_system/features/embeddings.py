"""Word embeddings: train Word2Vec and visualize the distributional space.

This module backs the "understand distributional meaning" objective. It trains
a small Word2Vec model on the corpus and produces a 2-D projection so we can
*see* that classical one-hot/TF-IDF vectors cannot capture similarity
("excellent" near "great"), while embeddings can.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nlp_system import config
from nlp_system.utils import get_logger, timer

logger = get_logger(__name__)


def train_word2vec(
    tokenized_docs: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    workers: int = 4,
):
    """Train a gensim Word2Vec model on pre-tokenized documents."""
    from gensim.models import Word2Vec

    with timer("train Word2Vec", logger):
        model = Word2Vec(
            sentences=tokenized_docs,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            seed=config.SEED,
        )
    logger.info("Vocabulary size: %d", len(model.wv))
    return model


def most_similar_report(model, words: list[str], topn: int = 8) -> dict:
    """Return nearest neighbours for a list of probe words (skips OOV)."""
    report = {}
    for w in words:
        if w in model.wv:
            report[w] = model.wv.most_similar(w, topn=topn)
    return report


def plot_embeddings(
    model,
    words: list[str] | None = None,
    n_words: int = 200,
    method: str = "tsne",
    out_path: Path | None = None,
):
    """Project embeddings to 2-D and save a scatter plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if words is None:
        words = list(model.wv.index_to_key[:n_words])
    vecs = np.array([model.wv[w] for w in words])

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, len(words) - 1))
        coords = TSNE(
            n_components=2, random_state=config.SEED, perplexity=perplexity
        ).fit_transform(vecs)
    else:
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=config.SEED).fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6)
    for i, w in enumerate(words):
        ax.annotate(w, (coords[i, 0], coords[i, 1]), fontsize=8, alpha=0.75)
    ax.set_title(f"Word embedding space ({method.upper()})")
    fig.tight_layout()

    out_path = out_path or (config.FIGURES_DIR / f"embeddings_{method}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved embedding plot -> %s", out_path)
    return out_path
