"""Regression guards for the two message sets.

`Reporter.say()` formats a template with keyword arguments. If the two vibes
ever drift — a missing key, or a placeholder present in one style and not the
other — the failure shows up as a `KeyError` in exactly one mode, at runtime,
in production. These tests are cheap; that bug is not.
"""

from __future__ import annotations

from string import Formatter

import pytest

from phrase_analysis.messages import CLEAN, GOTHIC, Reporter

EXPECTED_KEY_COUNT = 22

# Every placeholder used by either style, with a value of a usable type
# (`jaccard_line` applies `:.3f` / `:.1f` format specs).
SAMPLE_VALUES: dict[str, object] = {
    "author": "Андрухович",
    "author_a": "Андрухович",
    "author_b": "Забужко",
    "count": 128,
    "jaccard": 0.5,
    "n": 10,
    "n_authors": 2,
    "output_dir": "results",
    "overlap": 12.5,
    "text_dir": "Samples",
    "total": 512,
    "type_label": "Дієслівні",
    "unique": 64,
}


def placeholders(template: str) -> set[str]:
    """Field names referenced by a `str.format` template."""
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def test_both_vibes_define_the_same_keys():
    assert set(CLEAN) == set(GOTHIC)


def test_key_count_is_stable():
    """Not a law of nature — a nudge to update both dicts together."""
    assert len(CLEAN) == EXPECTED_KEY_COUNT
    assert len(GOTHIC) == EXPECTED_KEY_COUNT


@pytest.mark.parametrize("key", sorted(CLEAN))
def test_placeholders_match_across_vibes(key):
    assert placeholders(CLEAN[key]) == placeholders(GOTHIC[key])


def test_sample_values_cover_every_placeholder():
    """Keeps SAMPLE_VALUES honest if a new placeholder is introduced."""
    used = set()
    for messages in (CLEAN, GOTHIC):
        for template in messages.values():
            used |= placeholders(template)
    assert used <= set(SAMPLE_VALUES)


@pytest.mark.parametrize("vibe", ["clean", "gothic"])
@pytest.mark.parametrize("key", sorted(CLEAN))
def test_every_message_formats_without_error(vibe, key):
    assert Reporter(vibe).text(key, **SAMPLE_VALUES)


def test_gothic_vibe_selects_the_gothic_set():
    assert Reporter("gothic").text("stats_header") == GOTHIC["stats_header"]


@pytest.mark.parametrize("vibe", ["clean", "", "GOTHIC", "whatever"])
def test_anything_other_than_gothic_selects_clean(vibe):
    assert Reporter(vibe).text("stats_header") == CLEAN["stats_header"]


def test_default_vibe_is_clean():
    assert Reporter().text("stats_header") == CLEAN["stats_header"]


def test_say_prints_the_formatted_message(capsys):
    Reporter("clean").say("processing_author", author="Забужко")
    assert capsys.readouterr().out == CLEAN["processing_author"].format(author="Забужко") + "\n"


def test_text_returns_without_printing(capsys):
    result = Reporter("clean").text("summary_total", total=7)
    assert "7" in result
    assert capsys.readouterr().out == ""
