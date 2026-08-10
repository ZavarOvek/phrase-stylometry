#!/usr/bin/env python3
"""CLI entrypoint for the phrase analysis pipeline.

Examples:
    python main.py
    python main.py --text-dir Samples --output-dir results
    python main.py --vibe gothic --top-n 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

from phrase_analysis.config import AnalysisConfig
from phrase_analysis.pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare authors' syntactic phrase usage: frequency, "
        "vocabulary uniqueness, Word2Vec/PCA projection, and Jaccard similarity.",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        default=Path("Samples"),
        help="Directory with one <author>.txt file per author (default: Samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to write tables and figures to (default: results)",
    )
    parser.add_argument(
        "--spacy-model",
        default="uk_core_news_lg",
        help="spaCy pipeline to use; must support POS tagging and dependency parsing "
        "(default: uk_core_news_lg)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=10_000,
        help="Truncate each author's text to this many words (default: 10000)",
    )
    parser.add_argument(
        "--min-word-length",
        type=int,
        default=3,
        help="Minimum character length for either word in a phrase (default: 3)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top phrases per author/type to plot (default: 10)",
    )
    parser.add_argument(
        "--w2v-vector-size",
        type=int,
        default=100,
        help="Word2Vec embedding size (default: 100)",
    )
    parser.add_argument(
        "--w2v-window",
        type=int,
        default=3,
        help="Word2Vec context window (default: 3)",
    )
    parser.add_argument(
        "--w2v-epochs",
        type=int,
        default=20,
        help="Word2Vec training epochs (default: 20)",
    )
    parser.add_argument(
        "--vibe",
        choices=["clean", "gothic"],
        default="clean",
        help="Console narration style: 'clean' (professional) or 'gothic' "
        "(the original script's... distinctive flavor). Default: clean",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AnalysisConfig(
        text_dir=args.text_dir,
        output_dir=args.output_dir,
        spacy_model=args.spacy_model,
        max_words_per_author=args.max_words,
        min_word_length=args.min_word_length,
        top_n_phrases=args.top_n,
        word2vec_vector_size=args.w2v_vector_size,
        word2vec_window=args.w2v_window,
        word2vec_epochs=args.w2v_epochs,
        vibe=args.vibe,
    )
    run(config)


if __name__ == "__main__":
    main()
