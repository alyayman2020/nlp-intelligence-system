"""Unit tests for the configurable preprocessor (no NLTK corpora required)."""

from nlp_system.pipeline.preprocess import PreprocessConfig, TextPreprocessor


def _no_external_resources():
    # Force the fallback stoplist and disable lemmatization so tests run
    # without downloading NLTK data.
    cfg = PreprocessConfig(lemmatize=False, stem=False)
    pre = TextPreprocessor(cfg)
    # Trigger the fallback path by simulating missing corpora is hard here;
    # instead rely on whatever is installed. Tests assert structural behavior.
    return pre


def test_strip_html_for_reviews():
    pre = TextPreprocessor.for_reviews()
    pre.config.lemmatize = False
    out = pre.normalize("This <b>product</b> is great")
    assert "<b>" not in out and "product" in out


def test_url_and_mention_removal_for_tweets():
    pre = TextPreprocessor.for_tweets()
    out = pre.normalize("@user check http://x.co amazing #Deal")
    assert "@user" not in out
    assert "http" not in out
    assert "deal" in out  # hashtag text kept, lowercased


def test_negation_survives_stopwords():
    pre = TextPreprocessor(PreprocessConfig(lemmatize=False))
    toks = pre.tokens("this is not good")
    assert "not" in toks


def test_repeat_reduction_for_tweets():
    pre = TextPreprocessor.for_tweets()
    out = pre.normalize("soooo goooood")
    assert "ooooo" not in out


def test_transform_returns_string_list():
    pre = TextPreprocessor.for_reviews()
    pre.config.lemmatize = False
    out = pre.transform(["Great product", "Terrible service"])
    assert isinstance(out, list) and all(isinstance(s, str) for s in out)


def test_empty_and_none_safe():
    pre = TextPreprocessor.for_reviews()
    assert pre.transform_one("") == ""
    assert pre.transform_one(None) == ""
