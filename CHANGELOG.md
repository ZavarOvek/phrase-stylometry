# Changelog

All notable changes to this project are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
