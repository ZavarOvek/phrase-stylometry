"""Run configuration for the phrase analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {"бути", "мати", "робити", "ставати", "говорити", "той", "цей", "такий", "який", "весь"}
)

PHRASE_TYPES: tuple[str, ...] = ("verbal", "nominal", "adverbial")
PHRASE_TYPE_LABELS: dict[str, str] = {
    "verbal": "Дієслівні",
    "nominal": "Іменні",
    "adverbial": "Прислівникові",
}


@dataclass(slots=True)
class AnalysisConfig:
    """All tunable parameters for a single pipeline run.

    Attributes:
        text_dir: Directory containing one ``.txt`` file per author.
        output_dir: Directory where tables and figures are written.
        spacy_model: Name of the spaCy pipeline to load (must support POS + deps).
        max_words_per_author: Truncate each author's text to this many words,
            so corpus size stays comparable across authors and runtime stays bounded.
        stop_words: Lemmas excluded from extracted phrases.
        min_word_length: Minimum character length for either word in a phrase.
        word2vec_vector_size: Embedding dimensionality for the phrase Word2Vec model.
        word2vec_window: Word2Vec context window size.
        word2vec_min_count: Minimum token frequency for Word2Vec to keep a word.
        word2vec_epochs: Training epochs for Word2Vec.
        pca_components: Number of PCA components for the 2D/3D projection.
        top_n_phrases: How many top phrases per author/type to plot.
        vibe: Console narration style — "clean" (professional) or "gothic"
            (preserves the original script's... distinctive flavor).
    """

    text_dir: Path = Path("Samples")
    output_dir: Path = Path("results")
    spacy_model: str = "uk_core_news_lg"
    max_words_per_author: int = 10_000
    stop_words: frozenset[str] = field(default_factory=lambda: DEFAULT_STOP_WORDS)
    min_word_length: int = 3
    word2vec_vector_size: int = 100
    word2vec_window: int = 3
    word2vec_min_count: int = 1
    word2vec_epochs: int = 20
    pca_components: int = 2
    top_n_phrases: int = 10
    vibe: str = "clean"

    def __post_init__(self) -> None:
        self.text_dir = Path(self.text_dir)
        self.output_dir = Path(self.output_dir)
        if self.vibe not in {"clean", "gothic"}:
            raise ValueError(f"vibe must be 'clean' or 'gothic', got {self.vibe!r}")
