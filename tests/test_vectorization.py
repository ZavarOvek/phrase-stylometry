"""Tests for Word2Vec phrase embeddings, aggregation and PCA projection.

Word2Vec is stochastic, so nothing here asserts concrete vector values — only
shapes, vocabulary membership and row/column bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phrase_analysis.vectorization import (
    add_phrase_vectors,
    aggregate_author_vectors,
    phrase_vector,
    project_pca,
    train_phrase_model,
)

VECTOR_SIZE = 8

PHRASES = [
    "читати книга",
    "читати газета",
    "писати лист",
    "писати книга",
    "бачити світло",
]
KNOWN_WORDS = {"читати", "книга", "газета", "писати", "лист", "бачити", "світло"}


@pytest.fixture(scope="module")
def model():
    """A deliberately tiny model — the tests only look at shapes."""
    return train_phrase_model(
        pd.Series(PHRASES),
        vector_size=VECTOR_SIZE,
        window=2,
        min_count=1,
        epochs=2,
        workers=1,
    )


def test_model_vocabulary_is_the_set_of_phrase_words(model):
    assert set(model.wv.index_to_key) == KNOWN_WORDS


def test_model_vectors_have_the_requested_size(model):
    assert model.wv["книга"].shape == (VECTOR_SIZE,)


def test_min_count_drops_rare_words():
    model = train_phrase_model(
        pd.Series(["читати книга", "читати газета"]),
        vector_size=VECTOR_SIZE,
        min_count=2,
        epochs=1,
        workers=1,
    )
    assert set(model.wv.index_to_key) == {"читати"}


def test_phrase_vector_averages_known_words(model):
    vector = phrase_vector("читати книга", model)
    assert vector.shape == (VECTOR_SIZE,)
    expected = np.mean([model.wv["читати"], model.wv["книга"]], axis=0)
    assert np.allclose(vector, expected)


def test_phrase_vector_ignores_unknown_words(model):
    """A half-known phrase falls back to the known word alone."""
    assert np.allclose(phrase_vector("читати невідоме", model), model.wv["читати"])


def test_phrase_vector_is_none_when_nothing_is_known(model):
    assert phrase_vector("цілком невідоме", model) is None


def test_add_phrase_vectors_drops_rows_without_a_vector(model):
    df = pd.DataFrame(
        {
            "author": ["Андрухович", "Забужко"],
            "type": ["verbal", "verbal"],
            "phrase": ["читати книга", "цілком невідоме"],
        }
    )
    out = add_phrase_vectors(df, model)

    assert out["phrase"].tolist() == ["читати книга"]
    assert out.iloc[0]["vector"].shape == (VECTOR_SIZE,)
    assert "vector" not in df.columns  # input untouched


def test_aggregate_author_vectors_yields_one_row_per_author_and_type(model):
    df = pd.DataFrame(
        {
            "author": ["Андрухович", "Андрухович", "Андрухович", "Забужко"],
            "type": ["verbal", "verbal", "nominal", "verbal"],
            "phrase": ["читати книга", "читати газета", "писати лист", "бачити світло"],
        }
    )
    agg = aggregate_author_vectors(add_phrase_vectors(df, model))

    assert list(agg.columns) == ["author", "type", "vector"]
    assert set(zip(agg["author"], agg["type"], strict=True)) == {
        ("Андрухович", "verbal"),
        ("Андрухович", "nominal"),
        ("Забужко", "verbal"),
    }
    assert all(vector.shape == (VECTOR_SIZE,) for vector in agg["vector"])


def test_project_pca_adds_two_coordinate_columns():
    rng = np.random.default_rng(0)
    agg = pd.DataFrame(
        {
            "author": ["Андрухович", "Андрухович", "Забужко", "Забужко"],
            "type": ["verbal", "nominal", "verbal", "nominal"],
            "vector": [rng.random(VECTOR_SIZE) for _ in range(4)],
        }
    )
    out, pca = project_pca(agg, n_components=2)

    assert len(out) == len(agg)
    assert list(out.columns) == ["author", "type", "vector", "x", "y"]
    assert pca.n_components_ == 2
    assert len(pca.explained_variance_ratio_) == 2
    assert out[["x", "y"]].notna().all().all()
    assert "x" not in agg.columns  # input untouched
