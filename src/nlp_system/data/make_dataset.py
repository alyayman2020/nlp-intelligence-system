"""Load raw CSVs, map labels to binary sentiment, split, and persist parquet.

This is the ``make data`` stage. It is dataset-agnostic: it reads the
:class:`~nlp_system.config.DatasetSpec` and applies the right columns, encoding,
and label mapping. Output is a clean, deduplicated, stratified train/val/test
split written to ``data/processed/<dataset>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from nlp_system import config
from nlp_system.utils import get_logger, set_seed, timer

logger = get_logger(__name__)


def _read_raw(spec: config.DatasetSpec, raw_path: Path) -> pd.DataFrame:
    """Read a raw CSV with dataset-specific quirks handled."""
    if spec.name == "sentiment140":
        # Sentiment140 ships headerless with a fixed 6-column schema.
        cols = ["target", "ids", "date", "flag", "user", "text"]
        df = pd.read_csv(raw_path, encoding=spec.encoding, names=cols, header=None)
    else:
        df = pd.read_csv(raw_path, encoding=spec.encoding)
    return df


def prepare(
    dataset: str,
    sample_frac: float | None = None,
    test_size: float = 0.10,
    val_size: float = 0.10,
) -> dict[str, Path]:
    """Clean and split one dataset; return paths to the written parquet files."""
    set_seed(config.SEED)
    config.ensure_dirs()
    spec = config.get_dataset(dataset)

    raw_path = config.RAW_DIR / f"{spec.name}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python -m nlp_system.data.download "
            f"--dataset {spec.name}` first."
        )

    with timer(f"load raw {spec.name}", logger):
        df = _read_raw(spec, raw_path)
    logger.info("Raw rows: %d", len(df))

    # Keep only text + label, rename to canonical names.
    df = df[[spec.text_col, spec.label_col]].rename(
        columns={spec.text_col: "text", spec.label_col: "raw_label"}
    )

    # Map to binary sentiment; rows outside the map (e.g. 3-star neutral) drop.
    df["label"] = df["raw_label"].map(spec.label_map)
    before = len(df)
    df = df.dropna(subset=["label", "text"])
    df = df[df["text"].astype(str).str.strip().astype(bool)]
    df["label"] = df["label"].astype(int)
    logger.info("Dropped %d rows (neutral/empty/unmapped)", before - len(df))

    # Deduplicate identical texts (common in both datasets).
    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    logger.info("Dropped %d exact-duplicate texts", before - len(df))

    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=config.SEED).reset_index(drop=True)
        logger.info("Sampled to %d rows (frac=%.3f)", len(df), sample_frac)

    df = df[["text", "label"]]
    logger.info(
        "Class balance: %s", df["label"].value_counts(normalize=True).round(3).to_dict()
    )

    # Guard: stratified splitting needs at least 2 samples per class, and a
    # tiny corpus almost always means a data problem upstream (bad path, wrong
    # encoding, over-aggressive dedup) rather than something to silently accept.
    n_total = len(df)
    min_required = 20
    if n_total < min_required:
        raise ValueError(
            f"Only {n_total} usable rows for '{spec.name}' after cleaning/dedup. "
            f"Expected thousands+. Check that data/raw/{spec.name}.csv is the real "
            f"Kaggle file (Amazon ~568K rows, Sentiment140 1.6M rows) and that the "
            f"encoding/columns match the DatasetSpec. If you intentionally sampled, "
            f"raise --sample-frac."
        )

    # Stratified train / val / test.
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=config.SEED
    )
    rel_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=rel_val,
        stratify=train_df["label"],
        random_state=config.SEED,
    )

    out_dir = config.PROCESSED_DIR / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = out_dir / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(p, index=False)
        paths[name] = p
        logger.info("Wrote %s: %d rows -> %s", name, len(part), p)

    return paths


def load_split(dataset: str, split: str) -> pd.DataFrame:
    """Load a previously written split."""
    spec = config.get_dataset(dataset)
    p = config.PROCESSED_DIR / spec.name / f"{split}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run the data stage first.")
    return pd.read_parquet(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed splits.")
    parser.add_argument("--dataset", choices=[*config.DATASETS, "all"], default="all")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional row fraction for fast dev runs (e.g. 0.05).",
    )
    args = parser.parse_args()

    names = list(config.DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        prepare(name, sample_frac=args.sample_frac)


if __name__ == "__main__":
    main()
