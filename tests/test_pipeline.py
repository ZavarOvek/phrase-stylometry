"""End-to-end run of ``pipeline.run`` with a stubbed spaCy model.

``spacy.load`` is replaced by a callable that returns hand-built ``Doc``
objects, so the whole pipeline (corpus -> tables -> Word2Vec -> PCA ->
figures -> Excel) runs offline, without ``uk_core_news_lg``. Until this test
existed nothing exercised ``run`` past argument parsing, which is how a
crash on the statistics table went unnoticed.
"""

from __future__ import annotations

import pytest
import spacy
from spacy.tokens import Doc

from phrase_analysis import pipeline
from phrase_analysis.config import AnalysisConfig

NOUNS = ["воїн", "книга", "ліс", "вітер", "голос", "море"]
VERBS = ["читати", "бачити", "чути", "писати"]
ADVERBS = ["швидко", "тихо", "довго"]
ADJECTIVES = ["старий", "давній", "тихий"]


def _doc(vocab, offset: int) -> Doc:
    """A few dozen 'ADJ NOUN ADV VERB NOUN' clauses, varied per author."""
    words, pos, heads, deps = [], [], [], []
    for i in range(30):
        base = len(words)
        k = i + offset
        words += [
            ADJECTIVES[k % 3],
            NOUNS[k % 6],
            ADVERBS[k % 3],
            VERBS[k % 4],
            NOUNS[(k * 5 + 1) % 6],
        ]
        pos += ["ADJ", "NOUN", "ADV", "VERB", "NOUN"]
        heads += [base + 1, base + 3, base + 3, base + 3, base + 3]
        deps += ["amod", "nsubj", "advmod", "ROOT", "obj"]
    return Doc(vocab, words=words, lemmas=list(words), pos=pos, heads=heads, deps=deps)


@pytest.fixture
def stub_spacy(monkeypatch, vocab):
    docs = iter([_doc(vocab, 0), _doc(vocab, 7)])
    monkeypatch.setattr(spacy, "load", lambda _name: lambda _text: next(docs))


def test_full_run_writes_every_output(tmp_path, stub_spacy, capsys):
    samples = tmp_path / "Samples"
    samples.mkdir()
    for author in ("Автор_Перший", "Автор_Другий"):
        (samples / f"{author}.txt").write_text("текст", encoding="utf-8")
    out = tmp_path / "results"

    pipeline.run(AnalysisConfig(text_dir=samples, output_dir=out, top_n_phrases=3))
    capsys.readouterr()

    expected = {
        "statistics.xlsx",
        "phrases_by_type.xlsx",
        "uniqueness_analysis.png",
        "pca_analysis.png",
        "similarity_heatmap.png",
        "frequency_distribution.png",
        "average_frequencies.png",
        "frequency_statistics.xlsx",
        "jaccard_analysis.png",
        "jaccard_statistics.xlsx",
    }
    written = {p.name for p in out.iterdir()}
    assert expected <= written
    assert any(name.startswith("top3_") for name in written)
