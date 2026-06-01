"""Build and persist the BM25 search index.

Pulls the corpus from the processed splits (train+val+test combined), assigns
each document a sentiment label using the trained classifier where the true
label is not the point (search returns predicted sentiment), and serializes the
index for the API.
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd

from nlp_system import config
from nlp_system.data.make_dataset import load_split
from nlp_system.pipeline.nltk_setup import ensure_nltk
from nlp_system.pipeline.preprocess import TextPreprocessor
from nlp_system.search.engine import SearchEngine
from nlp_system.utils import get_logger, timer

logger = get_logger(__name__)


def build_index(dataset: str, use_predicted: bool = True, max_docs: int | None = None):
    ensure_nltk()
    frames = [load_split(dataset, s) for s in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["text"])
    if max_docs:
        df = df.head(max_docs)
    logger.info("Corpus for index: %d docs", len(df))

    if use_predicted:
        model_path = config.MODELS_DIR / dataset / "model.joblib"
        if model_path.exists():
            pipe = joblib.load(model_path)
            with timer("predict sentiments for corpus", logger):
                sentiments = pipe.predict(df["text"].tolist()).astype(int).tolist()
        else:
            logger.warning("No trained model found; falling back to true labels.")
            sentiments = df["label"].astype(int).tolist()
    else:
        sentiments = df["label"].astype(int).tolist()

    pre = (
        TextPreprocessor.for_tweets()
        if dataset == "sentiment140"
        else TextPreprocessor.for_reviews()
    )
    engine = SearchEngine(pre).build(df["text"].tolist(), sentiments)
    out = config.MODELS_DIR / f"search_index_{dataset}.pkl"
    engine.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 search index.")
    parser.add_argument(
        "--dataset", choices=list(config.DATASETS), default="amazon_fine_food"
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--true-labels",
        action="store_true",
        help="Use ground-truth instead of predicted.",
    )
    args = parser.parse_args()
    build_index(args.dataset, use_predicted=not args.true_labels, max_docs=args.max_docs)


if __name__ == "__main__":
    main()
