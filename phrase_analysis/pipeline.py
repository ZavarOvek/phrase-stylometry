"""Orchestrates the full phrase-analysis pipeline end to end."""

from __future__ import annotations

import pandas as pd

from .config import PHRASE_TYPE_LABELS, PHRASE_TYPES, AnalysisConfig
from .corpus import load_corpus
from .messages import Reporter
from .stats import (
    build_average_frequency_table,
    build_frequency_table,
    build_jaccard_table,
    build_uniqueness_table,
)
from .vectorization import (
    add_phrase_vectors,
    aggregate_author_vectors,
    project_pca,
    train_phrase_model,
)
from .visualization import (
    configure_style,
    plot_average_frequencies,
    plot_frequency_distribution,
    plot_jaccard,
    plot_pca,
    plot_similarity_heatmap,
    plot_top_phrases,
    plot_uniqueness,
)


def run(config: AnalysisConfig) -> None:
    """Run the full pipeline: load -> extract -> analyze -> visualize -> save.

    Args:
        config: All run parameters (paths, model name, hyperparameters, vibe).
    """
    import spacy  # imported lazily so --help doesn't require spaCy to be importable yet

    reporter = Reporter(config.vibe)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    nlp = spacy.load(config.spacy_model)

    df, author_stats = load_corpus(
        text_dir=config.text_dir,
        nlp=nlp,
        stop_words=config.stop_words,
        max_words_per_author=config.max_words_per_author,
        min_word_length=config.min_word_length,
        reporter=reporter,
    )
    authors = sorted(author_stats.keys())

    reporter.say("stats_header")
    stats_df = pd.DataFrame({a: vars(s) for a, s in author_stats.items()}).T
    print(stats_df)
    stats_df.to_excel(config.output_dir / "statistics.xlsx")

    reporter.say("freq_analysis")
    freq_df = build_frequency_table(df)
    with pd.ExcelWriter(config.output_dir / "phrases_by_type.xlsx", engine="openpyxl") as writer:
        for phrase_type in PHRASE_TYPES:
            freq_df[freq_df["type"] == phrase_type].to_excel(
                writer, sheet_name=phrase_type, index=False
            )

    reporter.say("uniqueness_analysis")
    uniqueness_df = build_uniqueness_table(freq_df, authors, PHRASE_TYPES)
    plot_uniqueness(
        uniqueness_df, list(PHRASE_TYPES), config.output_dir / "uniqueness_analysis.png"
    )

    reporter.say("top_phrases", n=config.top_n_phrases)
    for phrase_type in PHRASE_TYPES:
        plot_top_phrases(
            freq_df,
            authors,
            phrase_type,
            config.top_n_phrases,
            config.output_dir / f"top{config.top_n_phrases}_{phrase_type}.png",
            reporter,
        )

    reporter.say("vectorizing")
    model = train_phrase_model(
        df["phrase"],
        vector_size=config.word2vec_vector_size,
        window=config.word2vec_window,
        min_count=config.word2vec_min_count,
        epochs=config.word2vec_epochs,
    )
    df = add_phrase_vectors(df, model)
    agg_df = aggregate_author_vectors(df)

    reporter.say("pca")
    agg_df, pca = project_pca(agg_df, n_components=config.pca_components)
    plot_pca(agg_df, pca, authors, config.output_dir / "pca_analysis.png", reporter)

    reporter.say("heatmap")
    plot_similarity_heatmap(agg_df, config.output_dir / "similarity_heatmap.png")

    reporter.say("freq_distribution")
    plot_frequency_distribution(
        freq_df, authors, list(PHRASE_TYPES), config.output_dir / "frequency_distribution.png"
    )

    reporter.say("extra_stats")
    avg_freq_df = build_average_frequency_table(freq_df, authors, PHRASE_TYPES)
    plot_average_frequencies(
        avg_freq_df, list(PHRASE_TYPES), config.output_dir / "average_frequencies.png"
    )
    avg_freq_df.to_excel(config.output_dir / "frequency_statistics.xlsx", index=False)

    reporter.say("jaccard")
    jaccard_df = build_jaccard_table(freq_df, authors, PHRASE_TYPES)
    if not jaccard_df.empty:
        plot_jaccard(
            jaccard_df, list(PHRASE_TYPES), config.output_dir / "jaccard_analysis.png", reporter
        )
        jaccard_df.to_excel(config.output_dir / "jaccard_statistics.xlsx", index=False)

    reporter.say("done_header", output_dir=config.output_dir)
    for name in [
        "statistics.xlsx",
        "phrases_by_type.xlsx",
        "uniqueness_analysis.png",
        f"top{config.top_n_phrases}_*.png",
        "pca_analysis.png",
        "similarity_heatmap.png",
        "frequency_distribution.png",
        "average_frequencies.png",
        "jaccard_analysis.png",
        "frequency_statistics.xlsx",
        "jaccard_statistics.xlsx",
    ]:
        print(f"  - {name}")

    reporter.say("summary_header")
    reporter.say("summary_total", total=len(df))
    reporter.say("summary_unique", unique=len(freq_df))
    if not jaccard_df.empty:
        reporter.say("jaccard_summary_header")
        for _, row in jaccard_df.iterrows():
            reporter.say(
                "jaccard_line",
                author_a=row["author_a"],
                author_b=row["author_b"],
                type_label=PHRASE_TYPE_LABELS[row["type"]],
                jaccard=row["jaccard_index"],
                overlap=row["overlap_pct"],
            )
