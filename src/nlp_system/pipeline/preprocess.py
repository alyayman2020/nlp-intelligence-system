"""A single, configurable preprocessing class for every text domain.

The lab's central engineering idea: *one* pipeline that adapts to different
domains via configuration rather than copy-pasted code. A tweet pipeline keeps
hashtags and strips @mentions; a reviews pipeline strips HTML. Both share the
same normalization, tokenization, and stemming/lemmatization machinery.

The class is deliberately sklearn-compatible (``fit``/``transform``) so it can
drop straight into a ``Pipeline`` and be pickled with a trained model.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

# Lazy NLTK imports happen inside methods so importing the module is cheap and
# does not require corpora to be present (the API only needs transform()).

# --------------------------------------------------------------------------- #
# Precompiled regexes (module-level: compiled once)
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")  # "loooove" -> "loove"
_TOKEN_RE = re.compile(r"\b\w[\w']*\b")

# A compact, dependency-free contraction map (covers the common cases that
# matter for sentiment without pulling a heavy library).
_CONTRACTIONS = {
    "won't": "will not",
    "can't": "can not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}


@dataclass
class PreprocessConfig:
    """Domain-adaptive switches for :class:`TextPreprocessor`.

    Sensible defaults target generic English text. Use the factory methods on
    :class:`TextPreprocessor` for the two lab datasets.
    """

    lowercase: bool = True
    strip_html: bool = False
    strip_urls: bool = True
    strip_mentions: bool = False
    keep_hashtag_text: bool = True
    expand_contractions: bool = True
    reduce_repeats: bool = False
    remove_non_alpha: bool = True
    remove_stopwords: bool = True
    # Mutually exclusive in practice; stem wins if both are True.
    stem: bool = False
    lemmatize: bool = True
    min_token_len: int = 2
    extra_stopwords: frozenset[str] = field(default_factory=frozenset)
    keep_negations: bool = True  # don't drop "not", "no", "never" as stopwords


# Negation words that are semantically critical for sentiment and must survive
# stopword removal regardless of the stoplist used.
_NEGATIONS = frozenset({"not", "no", "nor", "never", "none", "cannot"})


class TextPreprocessor:
    """Configurable, sklearn-style text cleaner.

    Example
    -------
    >>> pre = TextPreprocessor.for_reviews()
    >>> pre.transform_one("This <b>product</b> isn't GREAT!!!")
    'product not great'
    """

    def __init__(self, config: PreprocessConfig | None = None):
        self.config = config or PreprocessConfig()
        self._stopwords: frozenset[str] | None = None
        self._stemmer = None
        self._lemmatizer = None
        self._stem_cache: dict[str, str] = {}
        self._lemma_cache: dict[str, str] = {}

    # ----- factory helpers ------------------------------------------------- #
    @classmethod
    def for_reviews(cls) -> TextPreprocessor:
        """Tuned for long, HTML-laden product reviews (Amazon)."""
        return cls(
            PreprocessConfig(
                strip_html=True,
                strip_urls=True,
                strip_mentions=False,
                keep_hashtag_text=True,
                reduce_repeats=False,
                lemmatize=True,
                stem=False,
            )
        )

    @classmethod
    def for_tweets(cls) -> TextPreprocessor:
        """Tuned for short, noisy tweets (Sentiment140)."""
        return cls(
            PreprocessConfig(
                strip_html=False,
                strip_urls=True,
                strip_mentions=True,
                keep_hashtag_text=True,
                reduce_repeats=True,
                lemmatize=True,
                stem=False,
            )
        )

    # ----- lazy resources -------------------------------------------------- #
    @property
    def stopwords(self) -> frozenset[str]:
        if self._stopwords is None:
            try:
                from nltk.corpus import stopwords as nltk_sw

                base = set(nltk_sw.words("english"))
            except Exception:
                base = set(_FALLBACK_STOPWORDS)
            base |= set(self.config.extra_stopwords)
            if self.config.keep_negations:
                base -= set(_NEGATIONS)
            self._stopwords = frozenset(base)
        return self._stopwords

    def _get_stemmer(self):
        if self._stemmer is None:
            from nltk.stem import PorterStemmer

            self._stemmer = PorterStemmer()
        return self._stemmer

    def _get_lemmatizer(self):
        if self._lemmatizer is None:
            from nltk.stem import WordNetLemmatizer

            self._lemmatizer = WordNetLemmatizer()
        return self._lemmatizer

    # ----- core steps ------------------------------------------------------ #
    def normalize(self, text: str) -> str:
        cfg = self.config
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        text = html.unescape(text)
        if cfg.strip_html:
            text = _HTML_TAG_RE.sub(" ", text)
        if cfg.strip_urls:
            text = _URL_RE.sub(" ", text)
        if cfg.strip_mentions:
            text = _MENTION_RE.sub(" ", text)
        if cfg.keep_hashtag_text:
            text = _HASHTAG_RE.sub(r"\1", text)
        if cfg.lowercase:
            text = text.lower()
        if cfg.expand_contractions:
            for pat, repl in _CONTRACTIONS.items():
                text = text.replace(pat, repl)
        if cfg.reduce_repeats:
            text = _REPEAT_CHAR_RE.sub(r"\1\1", text)
        if cfg.remove_non_alpha:
            text = _NON_ALPHA_RE.sub(" ", text)
        return _MULTISPACE_RE.sub(" ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text)

    def _filter_and_reduce(self, tokens: list[str]) -> list[str]:
        cfg = self.config
        out: list[str] = []
        sw = self.stopwords if cfg.remove_stopwords else frozenset()
        stem_fn = self._stem_token if cfg.stem else None
        lemma_fn = self._lemma_token if (cfg.lemmatize and not cfg.stem) else None
        for tok in tokens:
            if len(tok) < cfg.min_token_len and tok not in _NEGATIONS:
                continue
            if tok in sw:
                continue
            if stem_fn is not None:
                tok = stem_fn(tok)
            elif lemma_fn is not None:
                tok = lemma_fn(tok)
            out.append(tok)
        return out

    def _stem_token(self, token: str) -> str:
        cached = self._stem_cache.get(token)
        if cached is None:
            cached = self._get_stemmer().stem(token)
            self._stem_cache[token] = cached
        return cached

    def _lemma_token(self, token: str) -> str:
        cached = self._lemma_cache.get(token)
        if cached is None:
            cached = self._get_lemmatizer().lemmatize(token)
            self._lemma_cache[token] = cached
        return cached

    # ----- public API ------------------------------------------------------ #
    def tokens(self, text: str) -> list[str]:
        """Full path: normalize -> tokenize -> filter/stem/lemmatize."""
        return self._filter_and_reduce(self.tokenize(self.normalize(text)))

    def transform_one(self, text: str) -> str:
        return " ".join(self.tokens(text))

    def transform(self, texts) -> list[str]:
        """Vectorized transform over an iterable of strings (sklearn-style)."""
        return [self.transform_one(t) for t in texts]

    def fit(self, X=None, y=None):  # noqa: D401 - sklearn compatibility
        """No-op: the preprocessor is stateless. Present for Pipeline use."""
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)


# Minimal fallback stoplist if NLTK corpora are unavailable. Negations excluded.
_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "these",
    "those",
    "i",
    "you",
    "they",
    "we",
    "she",
    "but",
    "or",
    "if",
    "then",
    "so",
    "than",
    "too",
    "very",
    "can",
    "just",
    "have",
    "had",
    "do",
    "does",
    "did",
    "am",
    "been",
    "being",
    "would",
    "could",
    "should",
    "there",
    "their",
    "them",
    "what",
    "which",
    "who",
    "when",
    "where",
    "while",
    "about",
    "into",
    "over",
    "after",
    "before",
    "again",
    "here",
}
