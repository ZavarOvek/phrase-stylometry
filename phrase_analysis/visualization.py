"""Matplotlib/Seaborn figures for the phrase analysis pipeline.

Each ``plot_*`` function takes the data it needs plus an output path and
saves a PNG. None of them call ``plt.show()`` — these are meant to run
headless (e.g. from a CLI), not in a notebook.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from .config import PHRASE_TYPE_LABELS
from .messages import Reporter

MARKERS = {"verbal": "o", "nominal": "s", "adverbial": "^"}


def configure_style() -> None:
    """Apply the shared figure style. Call once before plotting."""
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _short(author: str) -> str:
    """Shorten an author file-stem like 'Кокотюха_Аномальна_Зона' for axis labels."""
    return author.split("_")[0]


def plot_uniqueness(uniqueness_df: pd.DataFrame, phrase_types: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(phrase_types), figsize=(5 * len(phrase_types), 4))
    if len(phrase_types) == 1:
        axes = [axes]

    for ax, phrase_type in zip(axes, phrase_types):
        data = uniqueness_df[uniqueness_df["type"] == phrase_type]
        x = np.arange(len(data))
        width = 0.35
        ax.bar(x - width / 2, data["unique"], width, label="Унікальні", alpha=0.8)
        ax.bar(x + width / 2, data["shared"], width, label="Спільні", alpha=0.8)
        ax.set_xlabel("Автор")
        ax.set_ylabel("Кількість")
        ax.set_title(PHRASE_TYPE_LABELS[phrase_type])
        ax.set_xticks(x)
        ax.set_xticklabels([_short(a) for a in data["author"]], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_top_phrases(
    freq_df: pd.DataFrame,
    authors: list[str],
    phrase_type: str,
    top_n: int,
    output_path: Path,
    reporter: Reporter,
) -> None:
    fig, axes = plt.subplots(1, len(authors), figsize=(6 * len(authors), 6), sharey=True)
    if len(authors) == 1:
        axes = [axes]

    for ax, author in zip(axes, authors):
        top = (
            freq_df[(freq_df["type"] == phrase_type) & (freq_df["author"] == author)]
            .nlargest(top_n, "count")
            .sort_values("count", ascending=True)
        )
        colors = sns.color_palette("viridis", len(top))
        ax.barh(range(len(top)), top["count"], color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["phrase"], fontsize=9)
        ax.set_xlabel("Частота")
        ax.set_title(_short(author), fontsize=10, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        for i, value in enumerate(top["count"]):
            ax.text(value, i, f" {value}", va="center", fontsize=8)

    axes[0].set_ylabel("Словосполучення")
    fig.suptitle(
        reporter.text("top_phrases_title", n=top_n, type_label=PHRASE_TYPE_LABELS[phrase_type]),
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_pca(agg_df: pd.DataFrame, pca, authors: list[str], output_path: Path, reporter: Reporter) -> None:
    plt.figure(figsize=(10, 7))
    colors = sns.color_palette("Set2", len(authors))

    for idx, author in enumerate(authors):
        author_data = agg_df[agg_df["author"] == author]
        for phrase_type, marker in MARKERS.items():
            type_data = author_data[author_data["type"] == phrase_type]
            if type_data.empty:
                continue
            plt.scatter(
                type_data["x"],
                type_data["y"],
                marker=marker,
                s=300,
                color=colors[idx],
                label=f"{_short(author)} — {PHRASE_TYPE_LABELS[phrase_type]}",
                edgecolors="black",
                linewidths=1.5,
                alpha=0.7,
            )
            for _, row in type_data.iterrows():
                plt.annotate(
                    phrase_type[:3],
                    (row["x"], row["y"]),
                    fontsize=8,
                    ha="center",
                    va="center",
                    fontweight="bold",
                )

    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]:.1%} дисперсії)", fontsize=12)
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]:.1%} дисперсії)", fontsize=12)
    plt.title(reporter.text("pca_title"), fontsize=14, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_similarity_heatmap(agg_df: pd.DataFrame, output_path: Path) -> None:
    vectors_matrix = np.vstack(agg_df["vector"].values)
    similarity = cosine_similarity(vectors_matrix)
    labels = [f"{_short(row['author'])}\n{row['type'][:3]}" for _, row in agg_df.iterrows()]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Косинусна схожість"},
        square=True,
        linewidths=0.5,
    )
    plt.title("Схожість векторних представлень", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_frequency_distribution(
    freq_df: pd.DataFrame, authors: list[str], phrase_types: list[str], output_path: Path
) -> None:
    fig, axes = plt.subplots(len(authors), len(phrase_types), figsize=(5 * len(phrase_types), 5 * len(authors)))
    if len(authors) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, author in enumerate(authors):
        for col_idx, phrase_type in enumerate(phrase_types):
            ax = axes[row_idx, col_idx]
            subset = freq_df[(freq_df["author"] == author) & (freq_df["type"] == phrase_type)]
            counts = Counter(subset["count"])
            x = sorted(counts.keys())
            y = [counts[k] for k in x]
            ax.bar(x, y, color=sns.color_palette("muted")[col_idx], alpha=0.7, edgecolor="black")
            ax.set_xlabel("Частота словосполучення")
            ax.set_ylabel("Кількість унікальних фраз")
            ax.set_title(f"{_short(author)} — {PHRASE_TYPE_LABELS[phrase_type]}", fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_average_frequencies(avg_freq_df: pd.DataFrame, phrase_types: list[str], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(phrase_types), figsize=(5 * len(phrase_types), 5))
    if len(phrase_types) == 1:
        axes = [axes]

    for ax, phrase_type in zip(axes, phrase_types):
        data = avg_freq_df[avg_freq_df["type"] == phrase_type]
        x = np.arange(len(data))
        width = 0.25
        ax.bar(x - width, data["mean_freq"], width, label="Середня", alpha=0.8)
        ax.bar(x, data["median_freq"], width, label="Медіана", alpha=0.8)
        ax.bar(x + width, data["max_freq"], width, label="Максимум", alpha=0.8)
        ax.set_xlabel("Автор")
        ax.set_ylabel("Частота")
        ax.set_title(PHRASE_TYPE_LABELS[phrase_type])
        ax.set_xticks(x)
        ax.set_xticklabels([_short(a) for a in data["author"]], rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_jaccard(jaccard_df: pd.DataFrame, phrase_types: list[str], output_path: Path, reporter: Reporter) -> None:
    """Plot Jaccard index and unique/shared phrase counts for the first author pair.

    The underlying table now covers every author pair (see stats.build_jaccard_table);
    this figure summarizes one pair at a time so it stays readable. For more than
    two authors, slice ``jaccard_df`` before calling this if you want a different pair.
    """
    if jaccard_df.empty:
        return

    first_pair = jaccard_df[["author_a", "author_b"]].iloc[0]
    pair_df = jaccard_df[
        (jaccard_df["author_a"] == first_pair["author_a"])
        & (jaccard_df["author_b"] == first_pair["author_b"])
    ].set_index("type").reindex(phrase_types).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(pair_df))
    colors = sns.color_palette("viridis", len(pair_df))
    bars = ax1.bar(x, pair_df["jaccard_index"], color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Тип словосполучень")
    ax1.set_ylabel("Індекс Жаккара")
    ax1.set_title(reporter.text("jaccard_title"), fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([PHRASE_TYPE_LABELS[t] for t in phrase_types])
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, pair_df["jaccard_index"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.3f}", ha="center", va="bottom", fontweight="bold")

    x2 = np.arange(len(pair_df))
    width = 0.25
    ax2.bar(x2 - width, pair_df["a_unique"], width, label=_short(first_pair["author_a"]), alpha=0.8)
    ax2.bar(x2, pair_df["shared"], width, label="Спільні", alpha=0.8)
    ax2.bar(x2 + width, pair_df["b_unique"], width, label=_short(first_pair["author_b"]), alpha=0.8)
    ax2.set_xlabel("Тип словосполучень")
    ax2.set_ylabel("Кількість фраз")
    ax2.set_title("Розподіл унікальних та спільних фраз", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([PHRASE_TYPE_LABELS[t] for t in phrase_types])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
