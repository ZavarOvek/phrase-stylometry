"""CLI behaviour of ``main.py``: a known failure is a message and exit code 1."""

from __future__ import annotations

import sys

import pytest
import spacy
from spacy.tokens import Doc

import main


def test_empty_corpus_prints_message_and_exits_with_code_1(tmp_path, monkeypatch, vocab, capsys):
    samples = tmp_path / "Samples"
    samples.mkdir()
    (samples / "Автор.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(spacy, "load", lambda _name: lambda _text: Doc(vocab, words=[]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--text-dir", str(samples), "--output-dir", str(tmp_path / "results")],
    )

    with pytest.raises(SystemExit) as info:
        main.main()

    assert info.value.code == 1
    captured = capsys.readouterr()
    assert "Автор: файл порожній" in captured.err
    assert "Traceback" not in captured.err
