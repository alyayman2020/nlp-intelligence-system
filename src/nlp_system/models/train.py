"""Train sentiment classifiers and track every run with MLflow.

The training entry point sweeps the vectorization methods (BoW, TF-IDF, BM25)
for a given dataset, logs metrics/params/artifacts to MLflow, and persists the
best pipeline (preprocessor + vectorizer + classifier) as a single pickle that
the FastAPI service loads at startup.

This is where the lab's core empirical claim gets demonstrated: run the *same*
classifier over three vectorizers and let the numbers show when TF-IDF/BM25 beat
raw counts — and on which dataset the gap is largest.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from nlp_system import config
from nlp_system.data.make_dataset import load_split
from nlp_system.features.vectorizers import build_vectorizer
from nlp_system.pipeline.nltk_setup import ensure_nltk
from nlp_system.pipeline.preprocess import TextPreprocessor
from nlp_system.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


@dataclass
class RunMetrics:
    accuracy: float
    f1: float
    roc_auc: float


def _preprocessor_for(dataset: str) -> TextPreprocessor:
    return (
        TextPreprocessor.for_tweets()
        if dataset == "sentiment140"
        else TextPreprocessor.for_reviews()
    )


def build_pipeline(dataset: str, method: str, max_features: int) -> Pipeline:
    """Assemble preprocess -> vectorizer -> logistic regression."""
    return Pipeline(
        steps=[
            ("preprocess", _preprocessor_for(dataset)),
            ("vectorizer", build_vectorizer(method, max_features=max_features)),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=1.0,
                    solver="liblinear",
                    random_state=config.SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _evaluate(pipe: Pipeline, X, y) -> RunMetrics:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return RunMetrics(
        accuracy=float(accuracy_score(y, pred)),
        f1=float(f1_score(y, pred)),
        roc_auc=float(roc_auc_score(y, proba)),
    )


def train_dataset(
    dataset: str,
    methods: list[str],
    max_features: int = 50_000,
) -> dict:
    """Train one model per vectorizer method; log to MLflow; keep the best."""
    import mlflow

    set_seed(config.SEED)
    ensure_nltk()
    config.ensure_dirs()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    train_df = load_split(dataset, "train")
    val_df = load_split(dataset, "val")
    test_df = load_split(dataset, "test")
    logger.info(
        "Loaded %s: train=%d val=%d test=%d",
        dataset,
        len(train_df),
        len(val_df),
        len(test_df),
    )

    results: dict[str, dict] = {}
    best_method, best_f1, best_pipe = None, -1.0, None

    for method in methods:
        run_name = f"{dataset}-{method}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "dataset": dataset,
                    "vectorizer": method,
                    "classifier": "logreg",
                    "max_features": max_features,
                    "n_train": len(train_df),
                }
            )
            pipe = build_pipeline(dataset, method, max_features)
            with timer(f"fit {run_name}", logger):
                pipe.fit(train_df["text"].tolist(), train_df["label"].values)

            val_m = _evaluate(pipe, val_df["text"].tolist(), val_df["label"].values)
            test_m = _evaluate(pipe, test_df["text"].tolist(), test_df["label"].values)

            mlflow.log_metrics(
                {
                    "val_accuracy": val_m.accuracy,
                    "val_f1": val_m.f1,
                    "val_roc_auc": val_m.roc_auc,
                    "test_accuracy": test_m.accuracy,
                    "test_f1": test_m.f1,
                    "test_roc_auc": test_m.roc_auc,
                }
            )
            logger.info(
                "%s | val_f1=%.4f val_auc=%.4f | test_f1=%.4f test_auc=%.4f",
                run_name,
                val_m.f1,
                val_m.roc_auc,
                test_m.f1,
                test_m.roc_auc,
            )
            results[method] = {"val": asdict(val_m), "test": asdict(test_m)}

            if val_m.f1 > best_f1:
                best_f1, best_method, best_pipe = val_m.f1, method, pipe

    # Persist the best pipeline for this dataset.
    model_dir = config.MODELS_DIR / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(best_pipe, model_path)

    meta = {
        "dataset": dataset,
        "best_method": best_method,
        "best_val_f1": best_f1,
        "results": results,
        "model_path": str(model_path),
    }
    (model_dir / "metrics.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        "Best for %s: %s (val_f1=%.4f) -> %s", dataset, best_method, best_f1, model_path
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sentiment models.")
    parser.add_argument("--dataset", choices=[*config.DATASETS, "all"], default="all")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bow", "tfidf", "bm25"],
        choices=["bow", "tfidf", "bm25"],
    )
    parser.add_argument("--max-features", type=int, default=50_000)
    args = parser.parse_args()

    names = list(config.DATASETS) if args.dataset == "all" else [args.dataset]
    summary = {}
    for name in names:
        summary[name] = train_dataset(name, args.methods, args.max_features)

    print("\n=== Training summary ===")
    print(json.dumps({k: v["results"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
