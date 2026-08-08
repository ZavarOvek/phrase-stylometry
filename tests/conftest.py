"""Shared fixtures.

The point of this suite is that it runs without ``uk_core_news_lg`` (~500 MB).
``extract_phrases`` only ever reads ``pos_``, ``dep_``, ``lemma_``, ``head`` and
``children`` off a token, and every one of those can be set by hand — so the
tests build :class:`spacy.tokens.Doc` objects from parallel lists instead of
parsing anything. A blank vocab is enough; no model is ever loaded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import matplotlib
import pytest
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

# Must be set before anything imports pyplot: CI runners have no display.
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def vocab() -> Vocab:
    """A blank Ukrainian vocab — `spacy.blank` downloads nothing."""
    return spacy.blank("uk").vocab


@pytest.fixture
def make_doc(vocab: Vocab) -> Callable[..., Doc]:
    """Return a helper building a fully annotated ``Doc`` from parallel lists."""

    def _make_doc(
        words: Sequence[str],
        lemmas: Sequence[str],
        pos: Sequence[str],
        heads: Sequence[int],
        deps: Sequence[str],
    ) -> Doc:
        return Doc(
            vocab,
            words=list(words),
            lemmas=list(lemmas),
            pos=list(pos),
            heads=list(heads),
            deps=list(deps),
        )

    return _make_doc
