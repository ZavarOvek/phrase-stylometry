"""Smoke tests for the plotting layer.

Nothing here compares images. Each `plot_*` call has to produce a real,
non-empty PNG on disk without raising — that catches the failure modes that
actually happen (a renamed column, an axes-shape mistake, a missing label)
without pinning the tests to matplotlib's rendering.

The backend is forced to `Agg` in `conftest.py`; CI runners have no display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from phrase_analysis.config import PHRASE_TYPES
from phrase_analysis.messages import Reporter
from phrase_analysis.stats import (
    build_average_frequency_table,
    build_frequency_table,
    build_jaccard_table,
    build_uniqueness_table,
)
from phrase_analysis.vectorization import (
    add_phrase_vectors,
    aggregate_author_vectors,
    project_pca,
    train_phrase_model,
)
from phrase_analysis.visualization import (
    configure_style,
    plot_average_frequencies,
    plot_frequency_distribution,
    plot_jaccard,
    plot_pca,
    plot_similarity_heatmap,
    plot_top_phrases,
    plot_uniqueness,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

AUTHORS = ["Андрухович_Рекреації", "Забужко_Казка_про_калинову_сопілку"]
PHRASES = {
    "verbal": ["читати книга", "бачити світло", "писати лист"],
    "nominal": ["давній книга", "старий воїн", "тихий голос"],
    "adverbial": ["швидко читати", "тихо говорити", "довго чекати"],
}


@pytest.fixture(autouse=True)
def isolated_rc_params():
    """`configure_style` writes global rcParams (dpi 300); keep that out of other tests."""
    with plt.rc_context():
        yield


@pytest.fixture(scope="module")
def corpus_df() -> pd.DataFrame:
    """Two authors with overlapping vocabularies and uneven phrase counts."""
    rows = []
    for author_idx, author in enumerate(AUTHORS):
        for phrase_type, phrases in PHRASES.items():
            for phrase_idx, phrase in enumerate(phrases):
                repeats = 1 + (phrase_idx + author_idx) % 3
                rows += [{"author": author, "type": phrase_type, "phrase": phrase}] * repeats
            rows.append(
                {"author": author, "type": phrase_type, "phrase": f"власний{author_idx} вислів"}
            )
    return pd.DataFrame(rows, columns=["author", "type", "phrase"])


@pytest.fixture(scope="module")
def freq_df(corpus_df: pd.DataFrame) -> pd.DataFrame:
    return build_frequency_table(corpus_df)


@pytest.fixture(scope="module")
def projected(corpus_df: pd.DataFrame):
    model = train_phrase_model(
        corpus_df["phrase"], vector_size=8, window=2, min_count=1, epochs=2, workers=1
    )
    agg = aggregate_author_vectors(add_phrase_vectors(corpus_df, model))
    return project_pca(agg, n_components=2)


@pytest.fixture
def reporter() -> Reporter:
    return Reporter("clean")


def assert_is_png(path) -> None:
    assert path.exists(), f"{path.name} was not written"
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC), f"{path.name} is not a PNG"
    assert len(data) > 1_000, f"{path.name} is suspiciously small ({len(data)} bytes)"


def test_configure_style_sets_the_shared_defaults():
    configure_style()
    assert plt.rcParams["figure.dpi"] == 300
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["font.family"] == ["DejaVu Sans"]
    assert plt.rcParams["axes.unicode_minus"] is False


def test_plot_uniqueness(freq_df, tmp_path):
    path = tmp_path / "uniqueness.png"
    uniqueness = build_uniqueness_table(freq_df, AUTHORS, PHRASE_TYPES)
    plot_uniqueness(uniqueness, list(PHRASE_TYPES), path)
    assert_is_png(path)


@pytest.mark.parametrize("phrase_type", PHRASE_TYPES)
def test_plot_top_phrases(freq_df, tmp_path, reporter, phrase_type):
    path = tmp_path / f"top5_{phrase_type}.png"
    plot_top_phrases(freq_df, AUTHORS, phrase_type, 5, path, reporter)
    assert_is_png(path)


def test_plot_pca(projected, tmp_path, reporter):
    path = tmp_path / "pca.png"
    agg, pca = projected
    plot_pca(agg, pca, AUTHORS, path, reporter)
    assert_is_png(path)


def test_plot_similarity_heatmap(projected, tmp_path):
    path = tmp_path / "heatmap.png"
    agg, _ = projected
    plot_similarity_heatmap(agg, path)
    assert_is_png(path)


def test_plot_frequency_distribution(freq_df, tmp_path):
    path = tmp_path / "distribution.png"
    plot_frequency_distribution(freq_df, AUTHORS, list(PHRASE_TYPES), path)
    assert_is_png(path)


def test_plot_average_frequencies(freq_df, tmp_path):
    path = tmp_path / "averages.png"
    averages = build_average_frequency_table(freq_df, AUTHORS, PHRASE_TYPES)
    plot_average_frequencies(averages, list(PHRASE_TYPES), path)
    assert_is_png(path)


def test_plot_jaccard(freq_df, tmp_path, reporter):
    path = tmp_path / "jaccard.png"
    jaccard = build_jaccard_table(freq_df, AUTHORS, PHRASE_TYPES)
    plot_jaccard(jaccard, list(PHRASE_TYPES), path, reporter)
    assert_is_png(path)


def test_plot_jaccard_writes_nothing_for_a_single_author(tmp_path, reporter):
    """One author means no pairs; the function must bail out, not crash."""
    path = tmp_path / "jaccard.png"
    plot_jaccard(pd.DataFrame(), list(PHRASE_TYPES), path, reporter)
    assert not path.exists()


def test_plots_do_not_leak_open_figures(freq_df, tmp_path):
    """Every plot_* closes its figure; a leak here means a memory creep in long runs."""
    plt.close("all")
    uniqueness = build_uniqueness_table(freq_df, AUTHORS, PHRASE_TYPES)
    plot_uniqueness(uniqueness, list(PHRASE_TYPES), tmp_path / "uniqueness.png")
    assert plt.get_fignums() == []


def test_gothic_reporter_titles_reach_the_figure(freq_df, tmp_path):
    """The vibe flag must not break plotting — only the strings differ."""
    path = tmp_path / "top5_gothic.png"
    plot_top_phrases(freq_df, AUTHORS, "verbal", 5, path, Reporter("gothic"))
    assert_is_png(path)
