"""Tests for the frequency, uniqueness, average and Jaccard tables."""

from __future__ import annotations

import pandas as pd
import pytest

from phrase_analysis.stats import (
    build_average_frequency_table,
    build_frequency_table,
    build_jaccard_table,
    build_uniqueness_table,
)

ANDRUKHOVYCH = "Андрухович"
ZABUZHKO = "Забужко"
KOKOTIUKHA = "Кокотюха"


def corpus(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Build the long (author, type, phrase) frame that `load_corpus` returns."""
    return pd.DataFrame(rows, columns=["author", "type", "phrase"])


def freq(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    """Build a frequency frame directly, skipping the groupby round-trip."""
    return pd.DataFrame(rows, columns=["author", "type", "phrase", "count"])


# --- build_frequency_table --------------------------------------------------


def test_frequency_table_counts_and_sort_order():
    df = corpus(
        [
            (ANDRUKHOVYCH, "verbal", "читати книга"),
            (ANDRUKHOVYCH, "verbal", "читати книга"),
            (ANDRUKHOVYCH, "verbal", "бачити світло"),
            (ANDRUKHOVYCH, "nominal", "давній книга"),
            (ZABUZHKO, "verbal", "читати книга"),
        ]
    )
    table = build_frequency_table(df)

    assert list(table.columns) == ["author", "type", "phrase", "count"]
    # author asc, type asc, count desc
    assert list(table.itertuples(index=False, name=None)) == [
        (ANDRUKHOVYCH, "nominal", "давній книга", 1),
        (ANDRUKHOVYCH, "verbal", "читати книга", 2),
        (ANDRUKHOVYCH, "verbal", "бачити світло", 1),
        (ZABUZHKO, "verbal", "читати книга", 1),
    ]


# --- build_uniqueness_table -------------------------------------------------


def test_uniqueness_with_disjoint_vocabularies():
    table = build_uniqueness_table(
        freq(
            [
                (ANDRUKHOVYCH, "verbal", "читати книга", 1),
                (ANDRUKHOVYCH, "verbal", "бачити світло", 1),
                (ZABUZHKO, "verbal", "писати лист", 1),
            ]
        ),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal"],
    ).set_index("author")

    assert table.loc[ANDRUKHOVYCH, ["unique", "shared", "total", "unique_pct"]].tolist() == [
        2,
        0,
        2,
        100.0,
    ]
    assert table.loc[ZABUZHKO, ["unique", "shared", "total", "unique_pct"]].tolist() == [
        1,
        0,
        1,
        100.0,
    ]


def test_uniqueness_with_fully_overlapping_vocabularies():
    table = build_uniqueness_table(
        freq(
            [
                (ANDRUKHOVYCH, "verbal", "читати книга", 5),
                (ZABUZHKO, "verbal", "читати книга", 1),
            ]
        ),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal"],
    )

    assert table["unique"].tolist() == [0, 0]
    assert table["shared"].tolist() == [1, 1]
    assert table["unique_pct"].tolist() == [0, 0]


def test_uniqueness_of_an_author_with_no_phrases_of_that_type():
    """`unique_pct` must not divide by zero for an empty vocabulary."""
    table = build_uniqueness_table(
        freq([(ANDRUKHOVYCH, "verbal", "читати книга", 1)]),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal"],
    ).set_index("author")

    assert table.loc[ZABUZHKO, ["unique", "shared", "total", "unique_pct"]].tolist() == [0, 0, 0, 0]


def test_uniqueness_covers_every_author_and_type():
    table = build_uniqueness_table(
        freq([(ANDRUKHOVYCH, "verbal", "читати книга", 1)]),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal", "nominal", "adverbial"],
    )
    assert len(table) == 6


# --- build_average_frequency_table ------------------------------------------


def test_average_frequency_mean_median_max():
    table = build_average_frequency_table(
        freq(
            [
                (ANDRUKHOVYCH, "verbal", "читати книга", 3),
                (ANDRUKHOVYCH, "verbal", "писати лист", 2),
                (ANDRUKHOVYCH, "verbal", "бачити світло", 1),
            ]
        ),
        authors=[ANDRUKHOVYCH],
        phrase_types=["verbal"],
    )

    assert len(table) == 1
    row = table.iloc[0]
    assert row["mean_freq"] == pytest.approx(2.0)
    assert row["median_freq"] == pytest.approx(2.0)
    assert row["max_freq"] == 3
    assert row["unique_phrases"] == 3


def test_average_frequency_skips_empty_author_type_pairs():
    table = build_average_frequency_table(
        freq([(ANDRUKHOVYCH, "verbal", "читати книга", 1)]),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal", "nominal"],
    )
    assert list(zip(table["author"], table["type"], strict=True)) == [(ANDRUKHOVYCH, "verbal")]


# --- build_jaccard_table ----------------------------------------------------


def test_jaccard_is_zero_for_disjoint_vocabularies():
    table = build_jaccard_table(
        freq(
            [
                (ANDRUKHOVYCH, "verbal", "читати книга", 1),
                (ZABUZHKO, "verbal", "писати лист", 1),
            ]
        ),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal"],
    )
    row = table.iloc[0]
    assert row["jaccard_index"] == pytest.approx(0.0)
    assert row["overlap_pct"] == pytest.approx(0.0)
    assert (row["a_unique"], row["b_unique"], row["shared"]) == (1, 1, 0)


def test_jaccard_is_one_for_identical_vocabularies():
    table = build_jaccard_table(
        freq(
            [
                (ANDRUKHOVYCH, "verbal", "читати книга", 4),
                (ZABUZHKO, "verbal", "читати книга", 1),
            ]
        ),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal"],
    )
    row = table.iloc[0]
    assert row["jaccard_index"] == pytest.approx(1.0)
    assert row["overlap_pct"] == pytest.approx(100.0)


def test_jaccard_partial_overlap():
    table = build_jaccard_table(
        freq(
            [
                (ANDRUKHOVYCH, "verbal", "читати книга", 1),
                (ANDRUKHOVYCH, "verbal", "бачити світло", 1),
                (ZABUZHKO, "verbal", "бачити світло", 1),
                (ZABUZHKO, "verbal", "писати лист", 1),
            ]
        ),
        authors=[ANDRUKHOVYCH, ZABUZHKO],
        phrase_types=["verbal"],
    )
    row = table.iloc[0]
    assert row["jaccard_index"] == pytest.approx(1 / 3)
    assert row["overlap_pct"] == pytest.approx(50.0)
    assert (row["a_unique"], row["b_unique"], row["shared"]) == (1, 1, 1)


def test_jaccard_covers_every_author_pair():
    """Regression guard: the original script only ever compared the first two authors."""
    authors = [ANDRUKHOVYCH, ZABUZHKO, KOKOTIUKHA]
    phrase_types = ["verbal", "nominal", "adverbial"]
    table = build_jaccard_table(
        freq([(author, "verbal", "читати книга", 1) for author in authors]),
        authors=authors,
        phrase_types=phrase_types,
    )

    assert len(table) == 9  # 3 pairs x 3 phrase types
    assert set(zip(table["author_a"], table["author_b"], strict=True)) == {
        (ANDRUKHOVYCH, ZABUZHKO),
        (ANDRUKHOVYCH, KOKOTIUKHA),
        (ZABUZHKO, KOKOTIUKHA),
    }


def test_jaccard_is_empty_for_a_single_author():
    table = build_jaccard_table(
        freq([(ANDRUKHOVYCH, "verbal", "читати книга", 1)]),
        authors=[ANDRUKHOVYCH],
        phrase_types=["verbal"],
    )
    assert table.empty
