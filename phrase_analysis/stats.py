"""Frequency, vocabulary-uniqueness, and Jaccard-similarity statistics.

Note on the Jaccard table: the original script only ever compared
``authors[0]`` against ``authors[1]``, silently ignoring everyone else and
crashing on a single-author corpus. :func:`build_jaccard_table` computes
every pairwise combination instead, so it scales to any number of authors.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import pandas as pd


def build_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each (author, type, phrase) combination occurs."""
    return (
        df.groupby(["author", "type", "phrase"])
        .size()
        .reset_index(name="count")
        .sort_values(["author", "type", "count"], ascending=[True, True, False])
    )


def build_uniqueness_table(
    freq_df: pd.DataFrame, authors: Iterable[str], phrase_types: Iterable[str]
) -> pd.DataFrame:
    """For each author/type, split their vocabulary into unique vs. shared-with-others."""
    rows = []
    for phrase_type in phrase_types:
        type_df = freq_df[freq_df["type"] == phrase_type]
        for author in authors:
            author_phrases = set(type_df.loc[type_df["author"] == author, "phrase"])
            other_phrases = set(type_df.loc[type_df["author"] != author, "phrase"])
            unique = len(author_phrases - other_phrases)
            shared = len(author_phrases & other_phrases)
            total = len(author_phrases)
            rows.append(
                {
                    "author": author,
                    "type": phrase_type,
                    "unique": unique,
                    "shared": shared,
                    "total": total,
                    "unique_pct": (unique / total * 100) if total else 0,
                }
            )
    return pd.DataFrame(rows)


def build_average_frequency_table(
    freq_df: pd.DataFrame, authors: Iterable[str], phrase_types: Iterable[str]
) -> pd.DataFrame:
    """Mean/median/max phrase frequency per author and phrase type."""
    rows = []
    for author in authors:
        for phrase_type in phrase_types:
            subset = freq_df[(freq_df["author"] == author) & (freq_df["type"] == phrase_type)]
            if len(subset) == 0:
                continue
            rows.append(
                {
                    "author": author,
                    "type": phrase_type,
                    "mean_freq": subset["count"].mean(),
                    "median_freq": subset["count"].median(),
                    "max_freq": subset["count"].max(),
                    "unique_phrases": len(subset),
                }
            )
    return pd.DataFrame(rows)


def build_jaccard_table(
    freq_df: pd.DataFrame, authors: list[str], phrase_types: Iterable[str]
) -> pd.DataFrame:
    """Pairwise Jaccard similarity of phrase vocabularies, per phrase type.

    Computes one row per (author pair, phrase type) for every combination of
    two authors — not just the first two — so it works for any author count.
    Returns an empty dataframe if fewer than two authors are present.
    """
    rows = []
    for author_a, author_b in combinations(authors, 2):
        for phrase_type in phrase_types:
            phrases_a = set(
                freq_df.loc[
                    (freq_df["author"] == author_a) & (freq_df["type"] == phrase_type), "phrase"
                ]
            )
            phrases_b = set(
                freq_df.loc[
                    (freq_df["author"] == author_b) & (freq_df["type"] == phrase_type), "phrase"
                ]
            )
            intersection = len(phrases_a & phrases_b)
            union = len(phrases_a | phrases_b)
            smaller = min(len(phrases_a), len(phrases_b))
            rows.append(
                {
                    "author_a": author_a,
                    "author_b": author_b,
                    "type": phrase_type,
                    "a_unique": len(phrases_a - phrases_b),
                    "b_unique": len(phrases_b - phrases_a),
                    "shared": intersection,
                    "jaccard_index": intersection / union if union else 0,
                    "overlap_pct": (intersection / smaller * 100) if smaller else 0,
                }
            )
    return pd.DataFrame(rows)
