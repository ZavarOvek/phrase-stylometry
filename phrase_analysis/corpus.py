"""Loads per-author .txt files and turns them into a phrase corpus dataframe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .extraction import extract_phrases, filter_phrases
from .messages import Reporter

if TYPE_CHECKING:
    from spacy.language import Language


@dataclass(slots=True)
class AuthorStats:
    """Per-author corpus and extraction counts."""

    words: int
    verbal: int
    nominal: int
    adverbial: int
    total: int


def load_corpus(
    text_dir: Path,
    nlp: "Language",
    stop_words: frozenset[str] | set[str],
    max_words_per_author: int,
    min_word_length: int,
    reporter: Reporter,
) -> tuple[pd.DataFrame, dict[str, AuthorStats]]:
    """Parse every ``.txt`` file in ``text_dir`` and extract its phrases.

    Args:
        text_dir: Directory with one ``<author>.txt`` file per author.
        nlp: A loaded spaCy pipeline with POS tagging and dependency parsing.
        stop_words: Lemmas to exclude from extracted phrases.
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

    records: list[dict[str, str]] = []
    author_stats: dict[str, AuthorStats] = {}

    for file_path in txt_files:
        author = file_path.stem
        reporter.say("processing_author", author=author)

        words = file_path.read_text(encoding="utf-8").split()[:max_words_per_author]
        text = " ".join(words)
        doc = nlp(text)

        phrases = filter_phrases(extract_phrases(doc), stop_words, min_word_length)

        author_stats[author] = AuthorStats(
            words=len(words),
            verbal=len(phrases["verbal"]),
            nominal=len(phrases["nominal"]),
            adverbial=len(phrases["adverbial"]),
            total=sum(len(items) for items in phrases.values()),
        )

        for phrase_type, items in phrases.items():
            for phrase in items:
                records.append({"author": author, "type": phrase_type, "phrase": phrase})

    df = pd.DataFrame(records, columns=["author", "type", "phrase"])
    reporter.say("loaded", count=len(df), n_authors=len(author_stats))
    return df, author_stats
