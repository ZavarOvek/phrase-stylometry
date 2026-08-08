"""Tests for AnalysisConfig normalisation and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from phrase_analysis.config import (
    DEFAULT_STOP_WORDS,
    PHRASE_TYPE_LABELS,
    PHRASE_TYPES,
    AnalysisConfig,
)


def test_defaults_are_paths():
    config = AnalysisConfig()
    assert config.text_dir == Path("Samples")
    assert config.output_dir == Path("results")


def test_string_directories_are_coerced_to_paths():
    config = AnalysisConfig(text_dir="Тексти", output_dir="вивід/2026")
    assert isinstance(config.text_dir, Path)
    assert isinstance(config.output_dir, Path)
    assert config.text_dir == Path("Тексти")
    assert config.output_dir == Path("вивід/2026")


def test_paths_are_left_alone(tmp_path):
    config = AnalysisConfig(text_dir=tmp_path)
    assert config.text_dir == tmp_path


@pytest.mark.parametrize("vibe", ["clean", "gothic"])
def test_valid_vibes_are_accepted(vibe):
    assert AnalysisConfig(vibe=vibe).vibe == vibe


@pytest.mark.parametrize("vibe", ["whatever", "Clean", "GOTHIC", ""])
def test_invalid_vibe_raises(vibe):
    with pytest.raises(ValueError, match="vibe must be 'clean' or 'gothic'"):
        AnalysisConfig(vibe=vibe)


def test_stop_words_default_to_the_shared_frozenset():
    assert AnalysisConfig().stop_words == DEFAULT_STOP_WORDS


def test_phrase_types_and_labels_agree():
    assert set(PHRASE_TYPES) == set(PHRASE_TYPE_LABELS)
