# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
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
