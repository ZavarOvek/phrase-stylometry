"""Tests for corpus loading.

`load_corpus` takes the spaCy pipeline as a plain callable, so a stub that
returns a pre-built `Doc` replaces the 500 MB model entirely.
"""

from __future__ import annotations

import pytest
from spacy.tokens import Doc

from phrase_analysis.corpus import AuthorStats, load_corpus
from phrase_analysis.messages import Reporter

ANDRUKHOVYCH = "Андрухович_Рекреації"
ZABUZHKO = "Забужко_Казка_про_калинову_сопілку"


class FakeNlp:
    """Stand-in for a loaded spaCy pipeline: returns a fixed Doc, records input."""

    def __init__(self, doc):
        self.doc = doc
        self.texts: list[str] = []

    def __call__(self, text: str):
        self.texts.append(text)
        return self.doc


@pytest.fixture
def sentence(make_doc):
    """'Старий воїн швидко читав давню книгу' — 6 phrases across all three types."""
    return make_doc(
        words=["Старий", "воїн", "швидко", "читав", "давню", "книгу"],
        lemmas=["старий", "воїн", "швидко", "читати", "давній", "книга"],
        pos=["ADJ", "NOUN", "ADV", "VERB", "ADJ", "NOUN"],
        heads=[1, 3, 3, 3, 5, 3],
        deps=["amod", "nsubj", "advmod", "ROOT", "amod", "obj"],
    )


@pytest.fixture
def reporter():
    return Reporter("clean")


def write_samples(directory, texts: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for author, text in texts.items():
        (directory / f"{author}.txt").write_text(text, encoding="utf-8")


def test_missing_directory_raises(tmp_path, reporter):
    with pytest.raises(FileNotFoundError, match="Text directory not found"):
        load_corpus(tmp_path / "nope", FakeNlp(None), set(), 100, 3, reporter)


def test_directory_without_txt_files_raises(tmp_path, reporter):
    (tmp_path / "notes.md").write_text("нічого", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"No \.txt files found"):
        load_corpus(tmp_path, FakeNlp(None), set(), 100, 3, reporter)


def test_one_row_per_extracted_phrase(tmp_path, sentence, reporter, capsys):
    write_samples(tmp_path, {ANDRUKHOVYCH: "Старий воїн швидко читав давню книгу"})

    df, stats = load_corpus(tmp_path, FakeNlp(sentence), set(), 10_000, 3, reporter)
    capsys.readouterr()

    assert list(df.columns) == ["author", "type", "phrase"]
    assert set(df["author"]) == {ANDRUKHOVYCH}
    assert sorted(df.loc[df["type"] == "verbal", "phrase"]) == [
        "читати воїн",
        "читати книга",
        "читати швидко",
    ]
    assert stats[ANDRUKHOVYCH] == AuthorStats(
        words=6, verbal=3, nominal=2, adverbial=1, total=6
    )


def test_stop_words_and_min_length_are_applied(tmp_path, sentence, reporter, capsys):
    write_samples(tmp_path, {ANDRUKHOVYCH: "Старий воїн швидко читав давню книгу"})

    df, stats = load_corpus(tmp_path, FakeNlp(sentence), {"книга"}, 10_000, 3, reporter)
    capsys.readouterr()

    assert "читати книга" not in set(df["phrase"])
    assert "давній книга" not in set(df["phrase"])
    assert stats[ANDRUKHOVYCH].total == 4


def test_text_is_truncated_to_max_words(tmp_path, sentence, reporter, capsys):
    write_samples(tmp_path, {ANDRUKHOVYCH: "один два три чотири п'ять шість"})
    nlp = FakeNlp(sentence)

    _, stats = load_corpus(tmp_path, nlp, set(), 3, 3, reporter)
    capsys.readouterr()

    assert nlp.texts == ["один два три"]
    assert stats[ANDRUKHOVYCH].words == 3


def test_authors_are_read_in_filename_order(tmp_path, sentence, reporter, capsys):
    write_samples(tmp_path, {ZABUZHKO: "текст", ANDRUKHOVYCH: "текст"})

    df, stats = load_corpus(tmp_path, FakeNlp(sentence), set(), 100, 3, reporter)
    capsys.readouterr()

    assert list(stats) == sorted([ANDRUKHOVYCH, ZABUZHKO])
    assert list(dict.fromkeys(df["author"])) == sorted([ANDRUKHOVYCH, ZABUZHKO])


def test_empty_corpus_still_returns_the_expected_columns(tmp_path, vocab, reporter, capsys):
    write_samples(tmp_path, {ANDRUKHOVYCH: ""})

    df, stats = load_corpus(tmp_path, FakeNlp(Doc(vocab, words=[])), set(), 100, 3, reporter)
    capsys.readouterr()

    assert df.empty
    assert list(df.columns) == ["author", "type", "phrase"]
    assert stats[ANDRUKHOVYCH] == AuthorStats(words=0, verbal=0, nominal=0, adverbial=0, total=0)
