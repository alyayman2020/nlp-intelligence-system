"""Download raw datasets from Kaggle into ``data/raw/``.

Uses ``kagglehub`` (the handles the user supplied). Kaggle credentials must be
available, either via ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables.

Run::

    python -m nlp_system.data.download              # both datasets
    python -m nlp_system.data.download --dataset sentiment140
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from nlp_system import config
from nlp_system.utils import get_logger

logger = get_logger(__name__)


def download_one(spec: config.DatasetSpec) -> Path:
    """Download a dataset and copy its primary CSV into ``data/raw/``."""
    import kagglehub

    config.ensure_dirs()
    logger.info("Downloading %s (%s) …", spec.name, spec.kaggle_handle)
    src_dir = Path(kagglehub.dataset_download(spec.kaggle_handle))
    logger.info("kagglehub cache: %s", src_dir)

    matches = list(src_dir.rglob(spec.raw_filename))
    if not matches:
        available = [p.name for p in src_dir.rglob("*.csv")]
        raise FileNotFoundError(
            f"Expected '{spec.raw_filename}' in {src_dir}. Found CSVs: {available}"
        )
    dest = config.RAW_DIR / f"{spec.name}.csv"
    shutil.copy2(matches[0], dest)
    logger.info("Saved -> %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw Kaggle datasets.")
    parser.add_argument(
        "--dataset",
        choices=[*config.DATASETS, "all"],
        default="all",
        help="Which dataset to download.",
    )
    args = parser.parse_args()

    targets = (
        list(config.DATASETS.values())
        if args.dataset == "all"
        else [config.get_dataset(args.dataset)]
    )
    for spec in targets:
        download_one(spec)


if __name__ == "__main__":
    main()
