"""Tests for corpus loading.

`load_corpus` takes the spaCy pipeline as a plain callable, so a stub that
returns a pre-built `Doc` replaces the 500 MB model entirely.
"""

from __future__ import annotations

import pytest
from spacy.tokens import Doc

from phrase_analysis.corpus import AuthorStats, load_corpus, normalize_apostrophes
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
        "читати книга",
        "читати швидко",
    ]
    assert list(df.loc[df["type"] == "predicative", "phrase"]) == ["воїн читати"]
    assert stats[ANDRUKHOVYCH] == AuthorStats(
        words=6, verbal=2, nominal=2, adverbial=1, predicative=1, total=6
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
    assert stats[ANDRUKHOVYCH] == AuthorStats(
        words=0, verbal=0, nominal=0, adverbial=0, predicative=0, total=0
    )


# --- apostrophes ------------------------------------------------------------

VARIANT_APOSTROPHES = ["\u2019", "\u02bc", "\u2018", "`", "\u00b4", "\u2032", "\u02b9", "\uff07"]


@pytest.mark.parametrize("mark", VARIANT_APOSTROPHES)
def test_every_apostrophe_variant_becomes_ascii(mark):
    assert normalize_apostrophes(f"м{mark}ясо з{mark}їв") == "м'ясо з'їв"


def test_ascii_apostrophe_is_left_alone():
    assert normalize_apostrophes("м'ясо") == "м'ясо"


@pytest.mark.parametrize(
    "text",
    [
        "\u2018цитата\u2019",
        "він сказав: \u2018так\u2019.",
        "\u2019 \u02bc ` на самоті",
        "слово\u2019 наступне",
        "123\u201945",
    ],
)
def test_marks_that_are_not_between_two_letters_are_untouched(text):
    assert normalize_apostrophes(text) == text


def test_apostrophe_variant_is_counted_with_its_ascii_twin(tmp_path, vocab, reporter, capsys):
    """'м’ясо' and "м'ясо" must end up as one phrase, because the text reaches
    the pipeline already normalized."""

    def nlp(text: str):
        words = text.split()
        n = len(words)
        return Doc(
            vocab,
            words=words,
            lemmas=words,
            pos=["ADJ", "NOUN"] * (n // 2),
            heads=[i + 1 if i % 2 == 0 else i for i in range(n)],
            deps=["amod", "ROOT"] * (n // 2),
        )

    write_samples(tmp_path, {ANDRUKHOVYCH: "свіже м\u2019ясо свіже м\u02bcясо свіже м'ясо"})

    df, stats = load_corpus(tmp_path, nlp, set(), 100, 3, reporter)
    capsys.readouterr()

    assert list(df["phrase"]) == ["свіже м'ясо"] * 3
    assert stats[ANDRUKHOVYCH].nominal == 3


def test_text_reaches_nlp_normalized(tmp_path, sentence, reporter, capsys):
    write_samples(tmp_path, {ANDRUKHOVYCH: "зʼїв \u2018м\u2019ясо\u2019"})
    nlp = FakeNlp(sentence)

    load_corpus(tmp_path, nlp, set(), 100, 3, reporter)
    capsys.readouterr()

    assert nlp.texts == ["з'їв \u2018м'ясо\u2019"]


def test_stop_words_with_apostrophe_variants_are_normalized(tmp_path, make_doc, reporter, capsys):
    doc = make_doc(
        words=["Старий", "м\u2019ясо"],
        lemmas=["старий", "м'ясо"],
        pos=["ADJ", "NOUN"],
        heads=[1, 1],
        deps=["amod", "ROOT"],
    )
    write_samples(tmp_path, {ANDRUKHOVYCH: "текст"})

    df, _ = load_corpus(tmp_path, FakeNlp(doc), {"м\u2019ясо"}, 100, 3, reporter)
    capsys.readouterr()

    assert df.empty
