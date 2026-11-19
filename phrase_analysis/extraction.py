"""Extraction of two-word syntactic phrases from a spaCy-parsed document.

Four phrase types are collected, based on dependency relations:

- ``verbal``:      verb + object/oblique noun, verb + adverb modifier
- ``nominal``:     adjective + noun, noun + noun (genitive-style modification)
- ``adverbial``:   adverb + the verb/adjective/adverb it modifies
- ``predicative``: subject noun + verb predicate (``nsubj``)

Subject + predicate is kept apart from ``verbal`` on purpose. In Ukrainian
grammar it is the grammatical basis of a clause (граматична основа), not a
word combination (словосполучення); counting it with verb + object pairs,
as English-oriented phrase statistics often do, mixes two different units.

An ``advmod`` arc between a verb and an adverb lands in both ``verbal``
("verb adverb") and ``adverbial`` ("adverb verb"). That is deliberate: the
same link is a verbal combination and an adverbial one at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spacy.tokens import Doc

PhraseDict = dict[str, list[str]]

_VERB_OBJECT_DEPS = {"obj", "iobj", "obl", "nmod"}
_SUBJECT_DEPS = {"nsubj"}


def _add_pair(items: list[str], first: str, second: str) -> None:
    """Append ``"first second"`` unless a lemma is empty or contains whitespace.

    A phrase is stored as one space-joined string and split back into its two
    words later, so a lemma that is empty or has a space in it would make that
    round trip ambiguous (the model leaves lemmas empty for some tokens).
    """
    if first.split() == [first] and second.split() == [second]:
        items.append(f"{first} {second}")


def extract_phrases(doc: Doc) -> PhraseDict:
    """Walk the dependency tree and collect verbal, nominal, and adverbial phrases.

    Args:
        doc: A spaCy ``Doc`` that has been parsed with POS tags and dependencies.

    Returns:
        A dict with keys ``"verbal"``, ``"nominal"``, ``"adverbial"``,
        ``"predicative"``, each mapping to a list of ``"<lemma> <lemma>"``
        phrase strings. Predicative pairs are subject first ("петро читати").
    """
    phrases: PhraseDict = {"verbal": [], "nominal": [], "adverbial": [], "predicative": []}

    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.pos_ == "NOUN" and child.dep_ in _SUBJECT_DEPS:
                    _add_pair(phrases["predicative"], child.lemma_, token.lemma_)
                elif (child.pos_ == "NOUN" and child.dep_ in _VERB_OBJECT_DEPS) or (
                    child.pos_ == "ADV" and child.dep_ == "advmod"
                ):
                    _add_pair(phrases["verbal"], token.lemma_, child.lemma_)
            if token.head.pos_ == "NOUN":
                _add_pair(phrases["verbal"], token.lemma_, token.head.lemma_)

        elif token.pos_ == "NOUN":
            for child in token.children:
                if child.pos_ == "ADJ":
                    _add_pair(phrases["nominal"], child.lemma_, token.lemma_)
                elif child.pos_ == "NOUN":
                    _add_pair(phrases["nominal"], token.lemma_, child.lemma_)
            if token.head.pos_ == "ADJ":
                _add_pair(phrases["nominal"], token.head.lemma_, token.lemma_)

        elif token.pos_ == "ADV":
            if token.head.pos_ in {"VERB", "ADJ", "ADV"}:
                _add_pair(phrases["adverbial"], token.lemma_, token.head.lemma_)

    return phrases


def filter_phrases(
    phrases: PhraseDict,
    stop_words: frozenset[str] | set[str],
    min_word_length: int = 3,
) -> PhraseDict:
    """Drop malformed, self-repeating, stop-word, and too-short phrases.

    Args:
        phrases: Output of :func:`extract_phrases`.
        stop_words: Lemmas to exclude (e.g. auxiliary/light verbs, pronouns).
        min_word_length: Minimum character length required for *both* words.

    Returns:
        A new phrase dict with the same keys, filtered.
    """
    filtered: PhraseDict = {phrase_type: [] for phrase_type in phrases}

    for phrase_type, items in phrases.items():
        for phrase in items:
            words = phrase.split()
            if len(words) != 2:
                continue
            word_a, word_b = words
            if word_a == word_b:
                continue
            if word_a in stop_words or word_b in stop_words:
                continue
            if len(word_a) < min_word_length or len(word_b) < min_word_length:
                continue
            filtered[phrase_type].append(phrase)

    return filtered
