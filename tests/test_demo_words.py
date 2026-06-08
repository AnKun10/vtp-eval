# tests/test_demo_words.py
from vtp_eval.demo.sections.words import content_words


def test_content_words_keeps_content_drops_stopwords_and_image():
    out = content_words("Is there a snowboard in the image?")
    assert "snowboard" in out
    for w in ("Is", "there", "the", "image"):
        assert w not in out and w.lower() not in out


def test_content_words_preserves_case_and_dedups():
    assert content_words("Red car and red bus") == ["Red", "car", "bus"]


def test_content_words_empty_when_all_stopwords():
    assert content_words("is there a") == []
