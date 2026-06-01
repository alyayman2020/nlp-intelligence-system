"""Central configuration.

All paths are derived from the project root and overridable via environment
variables (loaded from a ``.env`` file when present). Nothing here triggers
heavy imports, so this module is safe to import everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# MLflow — SQLite is the recommended default backend (file store deprecated
# as of MLflow 3.7 / Feb 2026). Override via MLFLOW_TRACKING_URI for a remote
# server, Postgres, etc.
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "nlp-intelligence-system")

# Reproducibility
SEED = int(os.getenv("SEED", "42"))


# --------------------------------------------------------------------------- #
# Dataset specifications
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of one input dataset.

    The two datasets are deliberately contrasting: long, well-formed product
    reviews (Amazon) vs. short, extremely noisy tweets (Sentiment140). The same
    pipeline runs on both; the *config* differs, not the code.
    """

    name: str
    kaggle_handle: str
    raw_filename: str
    text_col: str
    label_col: str
    # Maps a raw label value -> binary sentiment (0 = negative, 1 = positive).
    label_map: dict = field(default_factory=dict)
    encoding: str = "utf-8"
    # Domain hints consumed by the preprocessing pipeline.
    domain: str = "generic"


AMAZON = DatasetSpec(
    name="amazon_fine_food",
    kaggle_handle="snap/amazon-fine-food-reviews",
    raw_filename="Reviews.csv",
    text_col="Text",
    label_col="Score",
    # 1-2 stars -> negative, 4-5 -> positive, 3 dropped as neutral (see loader).
    label_map={1: 0, 2: 0, 4: 1, 5: 1},
    encoding="utf-8",
    domain="reviews",
)

SENTIMENT140 = DatasetSpec(
    name="sentiment140",
    kaggle_handle="kazanova/sentiment140",
    raw_filename="training.1600000.processed.noemoticon.csv",
    text_col="text",
    label_col="target",
    # Sentiment140 encodes 0 = negative, 4 = positive (no neutral in train).
    label_map={0: 0, 4: 1},
    encoding="latin-1",
    domain="tweets",
)

DATASETS: dict[str, DatasetSpec] = {
    AMAZON.name: AMAZON,
    SENTIMENT140.name: SENTIMENT140,
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Choices: {list(DATASETS)}")
    return DATASETS[name]


def ensure_dirs() -> None:
    """Create the directory tree if it does not yet exist."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
