# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Fixed
- `python main.py` crashed right after loading the corpus: `pipeline.run`
  built the statistics table with `vars()` on `AuthorStats`, a
  `slots=True` dataclass without `__dict__`. Now uses `dataclasses.asdict`.
  Present since the initial commit; nothing ran `run` past `--help` before.
- A token with an empty lemma (or one containing whitespace) killed the whole
  run: phrases are stored as `"<lemma> <lemma>"` and `filter_phrases` split
  them back with `phrase.split()` into exactly two words, raising
  `ValueError`. `extract_phrases` no longer builds such phrases, and
  `filter_phrases` skips anything that doesn't split into two words.

### Added
- `EmptyCorpusError` (a `ValueError`): when `.txt` files exist but yield no
  phrases at all, the run stops right after loading with a message that lists
  each author and the likely cause (empty file vs. nothing survived
  extraction/filtering). `main.py` prints it to stderr and exits with code 1.
  This used to reach gensim and fail with "you must first build vocabulary".
  If only some authors have no phrases, a per-author warning is printed.
- Apostrophe normalization: `’`, `ʼ`, `‘`, `` ` ``, `´`, `′`, `ʹ`, `＇` between
  two letters become ASCII `'` before the text reaches spaCy (stop words get
  the same treatment). The pymorphy3 Ukrainian dictionary only knows the
  ASCII form, so мʼясо / зʼїв used to be lemmatized wrongly (`мʼяса`, `зʼй`)
  and counted apart from м'ясо / з'їв. Phrase counts for texts with typographic
  apostrophes will change.
- `tests/test_pipeline.py`: an end-to-end run of `pipeline.run` with a
  stubbed spaCy model, checking that every output file is written.
- A fourth phrase type, `predicative`: subject noun + verb predicate
  (`nsubj`), written subject first (`воїн читати`). These pairs used to be
  counted as `verbal`; in Ukrainian grammar subject + predicate is the
  grammatical basis of a clause, not a word combination. Changes the
  `verbal` counts, the Jaccard/uniqueness tables and every figure.
  `AuthorStats` gains a `predicative` field.
- Test suite (`pytest`) covering extraction, statistics, messages, config,
  vectorization, corpus loading, and smoke-level plotting. The tests build
  spaCy `Doc` objects from parallel lists, so they run offline and never
  download `uk_core_news_lg`.
- `requirements-dev.txt` and a `pytest.ini` (`pythonpath = .`, since the
  project is run from its root rather than installed).
- CI `test` job with a Python 3.10–3.12 matrix.
- `Testing` section in both READMEs.

### Changed
- Adopted `ruff format` as the project formatter; the whole tree was
  reformatted in a single dedicated commit (see `.git-blame-ignore-revs`).
- `E501` disabled in `ruff check`: line length is the formatter's job, and
  the formatter cannot split string literals anyway.
- CI lint job now also runs `ruff format --check .`.

## [0.1.1] - 2026-08-08

### Added
- GitHub Actions CI: `ruff` lint job plus an import/`--help` smoke job
  (no spaCy model download needed).
- Ruff configuration in `ruff.toml` (`RUF001-003` disabled: they flag
  Cyrillic letters as ambiguous look-alikes of Latin ones, which is a false
  positive in a project about Ukrainian text).
- Status badges in both READMEs.

### Changed
- Long lines wrapped and `zip(..., strict=True)` made explicit to satisfy
  ruff; no behaviour changes.

## [0.1.0] - 2026-06-20

### Added
- Initial release: stylometric comparison of authors via syntactic phrases.
- Verbal / nominal / adverbial phrase extraction from spaCy dependency parses.
- Frequency, uniqueness and Jaccard-index statistics across all author pairs.
- Word2Vec phrase embeddings with PCA projection and cosine-similarity heatmap.
- Figures: uniqueness, top-N phrases, PCA, similarity heatmap, frequency
  distribution, average frequencies, Jaccard.
- `AnalysisConfig` dataclass collecting every run parameter in one place.
- `--vibe clean|gothic` console narration modes.
