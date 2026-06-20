"""Word2Vec phrase embeddings, aggregation, and PCA projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


def train_phrase_model(
    phrases: pd.Series,
    vector_size: int = 100,
    window: int = 3,
    min_count: int = 1,
    epochs: int = 20,
    workers: int = 4,
) -> Word2Vec:
    """Train a Word2Vec model treating each phrase as a 2-token "sentence"."""
    sentences = [phrase.split() for phrase in phrases]
    return Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )


def phrase_vector(phrase: str, model: Word2Vec) -> np.ndarray | None:
    """Average the Word2Vec vectors of a phrase's words, or None if none are known."""
    words = phrase.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else None


def add_phrase_vectors(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    """Attach a `vector` column and drop rows where no vector could be built."""
    out = df.copy()
    out["vector"] = out["phrase"].apply(lambda phrase: phrase_vector(phrase, model))
    return out.dropna(subset=["vector"])


def aggregate_author_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Average phrase vectors per (author, type) into a single point each."""
    return (
        df.groupby(["author", "type"])["vector"]
        .apply(lambda vectors: np.mean(np.vstack(vectors), axis=0))
        .reset_index()
    )


def project_pca(agg_df: pd.DataFrame, n_components: int = 2) -> tuple[pd.DataFrame, PCA]:
    """Project aggregated author/type vectors into `n_components` dimensions."""
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(np.vstack(agg_df["vector"]))
    out = agg_df.copy()
    out["x"], out["y"] = coords[:, 0], coords[:, 1]
    return out, pca
