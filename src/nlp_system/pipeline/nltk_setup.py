"""Download the NLTK corpora the pipeline needs (idempotent)."""

from __future__ import annotations

from nlp_system.utils import get_logger

logger = get_logger(__name__)

_RESOURCES = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk() -> None:
    """Ensure required NLTK data is present, downloading only what's missing."""
    import nltk

    for path, pkg in _RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info("Downloading NLTK resource: %s", pkg)
            nltk.download(pkg, quiet=True)


if __name__ == "__main__":
    ensure_nltk()
    logger.info("NLTK resources ready.")
