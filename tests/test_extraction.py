"""Tests for phrase extraction and filtering.

Every ``Doc`` here is hand-built from parallel lists (see ``conftest.py``), so
each test pins down exactly one branch of the dependency walk.
"""

from __future__ import annotations

import pytest
from spacy.tokens import Doc

from phrase_analysis.extraction import _VERB_OBJECT_DEPS, extract_phrases, filter_phrases

EMPTY: dict[str, list[str]] = {"verbal": [], "nominal": [], "adverbial": []}


def test_empty_doc_yields_three_empty_lists(vocab):
    """An empty document must return the full key set, not raise or return {}."""
    assert extract_phrases(Doc(vocab, words=[])) == EMPTY


def test_full_sentence_reference_case(make_doc):
    """Regression case for the whole walk: 'Старий воїн швидко читав давню книгу'."""
    doc = make_doc(
        words=["Старий", "воїн", "швидко", "читав", "давню", "книгу"],
        lemmas=["старий", "воїн", "швидко", "читати", "давній", "книга"],
        pos=["ADJ", "NOUN", "ADV", "VERB", "ADJ", "NOUN"],
        heads=[1, 3, 3, 3, 5, 3],
        deps=["amod", "nsubj", "advmod", "ROOT", "amod", "obj"],
    )
    assert extract_phrases(doc) == {
        "verbal": ["читати воїн", "читати швидко", "читати книга"],
        "nominal": ["старий воїн", "давній книга"],
        "adverbial": ["швидко читати"],
    }


# --- verbal -----------------------------------------------------------------


@pytest.mark.parametrize("dep", sorted(_VERB_OBJECT_DEPS))
def test_verb_takes_noun_child_on_object_like_deps(make_doc, dep):
    doc = make_doc(
        words=["читав", "книгу"],
        lemmas=["читати", "книга"],
        pos=["VERB", "NOUN"],
        heads=[0, 0],
        deps=["ROOT", dep],
    )
    assert extract_phrases(doc)["verbal"] == ["читати книга"]


@pytest.mark.parametrize("dep", ["appos", "conj", "flat", "advcl"])
def test_verb_ignores_noun_child_on_other_deps(make_doc, dep):
    doc = make_doc(
        words=["читав", "книгу"],
        lemmas=["читати", "книга"],
        pos=["VERB", "NOUN"],
        heads=[0, 0],
        deps=["ROOT", dep],
    )
    assert extract_phrases(doc) == EMPTY


def test_verb_takes_adverb_child_and_adverb_takes_its_head(make_doc):
    """One adverb modifier feeds both the verbal and the adverbial bucket."""
    doc = make_doc(
        words=["швидко", "читав"],
        lemmas=["швидко", "читати"],
        pos=["ADV", "VERB"],
        heads=[1, 1],
        deps=["advmod", "ROOT"],
    )
    assert extract_phrases(doc) == {
        "verbal": ["читати швидко"],
        "nominal": [],
        "adverbial": ["швидко читати"],
    }


def test_verb_hanging_off_a_noun_is_recorded(make_doc):
    """Participle-style constructions: the verb's *head* is a noun.

    This branch has no counterpart in the other buckets and is easy to drop
    during a refactor, hence an explicit test.
    """
    doc = make_doc(
        words=["книга", "написана"],
        lemmas=["книга", "написати"],
        pos=["NOUN", "VERB"],
        heads=[0, 0],
        deps=["ROOT", "acl"],
    )
    assert extract_phrases(doc)["verbal"] == ["написати книга"]


# --- nominal ----------------------------------------------------------------


def test_adjective_child_of_noun(make_doc):
    doc = make_doc(
        words=["давню", "книгу"],
        lemmas=["давній", "книга"],
        pos=["ADJ", "NOUN"],
        heads=[1, 1],
        deps=["amod", "ROOT"],
    )
    assert extract_phrases(doc)["nominal"] == ["давній книга"]


def test_noun_child_of_noun_keeps_head_first(make_doc):
    """Genitive-style modification is emitted head-first, unlike the ADJ case."""
    doc = make_doc(
        words=["голос", "серця"],
        lemmas=["голос", "серце"],
        pos=["NOUN", "NOUN"],
        heads=[0, 0],
        deps=["ROOT", "nmod"],
    )
    assert extract_phrases(doc)["nominal"] == ["голос серце"]


def test_noun_hanging_off_an_adjective(make_doc):
    doc = make_doc(
        words=["сповнений", "радості"],
        lemmas=["сповнений", "радість"],
        pos=["ADJ", "NOUN"],
        heads=[0, 0],
        deps=["ROOT", "obl"],
    )
    assert extract_phrases(doc)["nominal"] == ["сповнений радість"]


# --- adverbial --------------------------------------------------------------


@pytest.mark.parametrize(
    ("head_pos", "head_word", "head_lemma"),
    [
        ("VERB", "читав", "читати"),
        ("ADJ", "гарний", "гарний"),
        ("ADV", "швидко", "швидко"),
    ],
)
def test_adverb_attaches_to_verb_adjective_and_adverb(make_doc, head_pos, head_word, head_lemma):
    doc = make_doc(
        words=["дуже", head_word],
        lemmas=["дуже", head_lemma],
        pos=["ADV", head_pos],
        heads=[1, 1],
        deps=["advmod", "ROOT"],
    )
    assert f"дуже {head_lemma}" in extract_phrases(doc)["adverbial"]


def test_adverb_attached_to_a_noun_is_ignored(make_doc):
    doc = make_doc(
        words=["майже", "опівніч"],
        lemmas=["майже", "опівніч"],
        pos=["ADV", "NOUN"],
        heads=[1, 1],
        deps=["advmod", "ROOT"],
    )
    assert extract_phrases(doc) == EMPTY


def test_root_adverb_pairs_with_itself_and_is_dropped_by_the_filter(make_doc):
    """A ROOT adverb is its own head, so extraction emits a self-pair.

    That is by design cheap to produce and cheap to remove: ``filter_phrases``
    is what guarantees no self-pair reaches the corpus.
    """
    doc = make_doc(
        words=["дуже", "швидко"],
        lemmas=["дуже", "швидко"],
        pos=["ADV", "ADV"],
        heads=[1, 1],
        deps=["advmod", "ROOT"],
    )
    phrases = extract_phrases(doc)
    assert phrases["adverbial"] == ["дуже швидко", "швидко швидко"]
    assert filter_phrases(phrases, stop_words=frozenset())["adverbial"] == ["дуже швидко"]


# --- filter_phrases ---------------------------------------------------------


def test_filter_keeps_every_key_even_when_all_phrases_are_dropped():
    filtered = filter_phrases({"verbal": ["бути бути"], "nominal": [], "adverbial": []}, set())
    assert filtered == EMPTY


@pytest.mark.parametrize(
    "phrase",
    ["бути воїн", "воїн бути"],
    ids=["stop-word-first", "stop-word-second"],
)
def test_stop_words_are_checked_in_both_positions(phrase):
    phrases = {"verbal": [phrase, "читати книга"], "nominal": [], "adverbial": []}
    filtered = filter_phrases(phrases, stop_words=frozenset({"бути"}))
    assert filtered["verbal"] == ["читати книга"]


@pytest.mark.parametrize(
    "phrase",
    ["око бачити", "бачити око"],
    ids=["short-word-first", "short-word-second"],
)
def test_min_word_length_is_checked_in_both_positions(phrase):
    phrases = {"verbal": [phrase, "читати книга"], "nominal": [], "adverbial": []}
    filtered = filter_phrases(phrases, stop_words=set(), min_word_length=4)
    assert filtered["verbal"] == ["читати книга"]


def test_min_word_length_defaults_to_three():
    phrases = {"verbal": ["бачити око", "бачити ліс"], "nominal": [], "adverbial": []}
    assert filter_phrases(phrases, stop_words=set())["verbal"] == ["бачити око", "бачити ліс"]
    assert filter_phrases(phrases, stop_words=set(), min_word_length=4)["verbal"] == []


def test_filter_does_not_mutate_its_input():
    phrases = {"verbal": ["бути воїн"], "nominal": [], "adverbial": []}
    filter_phrases(phrases, stop_words=frozenset({"бути"}))
    assert phrases["verbal"] == ["бути воїн"]
