"""Loads per-author .txt files and turns them into a phrase corpus dataframe."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .extraction import extract_phrases, filter_phrases
from .messages import Reporter

if TYPE_CHECKING:
    from spacy.language import Language

# The apostrophe the Ukrainian pymorphy3 dictionary knows is ASCII U+0027; with
# U+2019 / U+02BC the word is unknown to it and its lemma comes out wrong
# (мʼясо -> "мʼяса"). Only a mark between two letters is rewritten, so
# ‘quotes’ around a word are left alone.
_LETTER = r"[^\W\d_]"
_INTRA_WORD_APOSTROPHE = re.compile(
    rf"(?<={_LETTER})[\u2019\u02bc\u2018`\u00b4\u2032\u02b9\uff07](?={_LETTER})"
)


def normalize_apostrophes(text: str) -> str:
    """Replace look-alike apostrophes between two letters with ASCII ``'``.

    Handles U+2019, U+02BC, U+2018, backtick, U+00B4, U+2032, U+02B9 and U+FF07.
    Marks that are not surrounded by letters on both sides (quotation marks,
    for instance) are kept as they are.
    """
    return _INTRA_WORD_APOSTROPHE.sub("'", text)


class EmptyCorpusError(ValueError):
    """No author yielded a single phrase, so there is nothing to analyze."""


@dataclass(slots=True)
class AuthorStats:
    """Per-author corpus and extraction counts."""

    words: int
    verbal: int
    nominal: int
    adverbial: int
    predicative: int
    total: int


def load_corpus(
    text_dir: Path,
    nlp: Language,
    stop_words: frozenset[str] | set[str],
    max_words_per_author: int,
    min_word_length: int,
    reporter: Reporter,
) -> tuple[pd.DataFrame, dict[str, AuthorStats]]:
    """Parse every ``.txt`` file in ``text_dir`` and extract its phrases.

    Args:
        text_dir: Directory with one ``<author>.txt`` file per author.
        nlp: A loaded spaCy pipeline with POS tagging and dependency parsing.
        stop_words: Lemmas to exclude from extracted phrases. Apostrophes in
            them are normalized the same way as in the texts.
        max_words_per_author: Truncate each author's text to this many words.
        min_word_length: Minimum character length for either word in a phrase.
        reporter: Console narrator (clean or gothic vibe).

    Returns:
        A tuple of:
            - a long dataframe with columns ``author``, ``type``, ``phrase``
            - a dict mapping author name to :class:`AuthorStats`

    Raises:
        FileNotFoundError: If ``text_dir`` doesn't exist or has no ``.txt`` files.
    """
    if not text_dir.exists():
        raise FileNotFoundError(f"Text directory not found: {text_dir}")

    txt_files = sorted(text_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {text_dir}")

    reporter.say("loading", text_dir=text_dir)
    stop_words = frozenset(normalize_apostrophes(word) for word in stop_words)

    records: list[dict[str, str]] = []
    author_stats: dict[str, AuthorStats] = {}

    for file_path in txt_files:
        author = file_path.stem
        reporter.say("processing_author", author=author)

        words = file_path.read_text(encoding="utf-8").split()[:max_words_per_author]
        text = normalize_apostrophes(" ".join(words))
        doc = nlp(text)

        phrases = filter_phrases(extract_phrases(doc), stop_words, min_word_length)

        author_stats[author] = AuthorStats(
            words=len(words),
            verbal=len(phrases["verbal"]),
            nominal=len(phrases["nominal"]),
            adverbial=len(phrases["adverbial"]),
            predicative=len(phrases["predicative"]),
            total=sum(len(items) for items in phrases.values()),
        )

        for phrase_type, items in phrases.items():
            for phrase in items:
                records.append({"author": author, "type": phrase_type, "phrase": phrase})

    df = pd.DataFrame(records, columns=["author", "type", "phrase"])
    reporter.say("loaded", count=len(df), n_authors=len(author_stats))
    return df, author_stats


def _empty_author_reason(stats: AuthorStats) -> str:
    if stats.words == 0:
        return "файл порожній (0 слів)"
    return (
        f"слів {stats.words}, але жодне словосполучення не пройшло відбір і фільтри: "
        "перевірте мову тексту, модель spaCy, стоп-слова та --min-word-length"
    )


def ensure_corpus_not_empty(
    df: pd.DataFrame, author_stats: dict[str, AuthorStats], reporter: Reporter
) -> None:
    """Fail clearly on an empty corpus; warn about single empty authors.

    Args:
        df: The phrase dataframe returned by :func:`load_corpus`.
        author_stats: The per-author counts returned by :func:`load_corpus`.
        reporter: Console narrator, used for the per-author warning.

    Raises:
        EmptyCorpusError: If no author has a single phrase. The message names
            every author and the likely cause (empty file vs. nothing survived
            extraction and filtering).
    """
    empty = [author for author, stats in author_stats.items() if stats.total == 0]
    if df.empty:
        lines = [f"  - {author}: {_empty_author_reason(author_stats[author])}" for author in empty]
        raise EmptyCorpusError(
            "Не знайдено жодного словосполучення в жодному з текстів:\n" + "\n".join(lines)
        )
    for author in empty:
        reporter.say(
            "empty_author", author=author, reason=_empty_author_reason(author_stats[author])
        )
