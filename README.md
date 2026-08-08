**English** | [Українська](README.uk.md)

[![CI](https://github.com/ZavarOvek/phrase-stylometry/actions/workflows/ci.yml/badge.svg)](https://github.com/ZavarOvek/phrase-stylometry/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# Phrase Stylometry — Syntactic Phrase Analysis of Authorial Style

A tool for stylometric comparison of authors based on syntactic phrases: it
takes several texts, extracts verbal, nominal and adverbial phrases (via
`spaCy` dependency parsing), computes phrase frequencies and vocabulary
uniqueness, builds `Word2Vec` vectors of phrases, projects them with `PCA`,
and calculates cosine similarity and the Jaccard index between authors.

## Installation

```bash
pip install -r requirements.txt
python -m spacy download uk_core_news_lg
```

`uk_core_news_lg` is a large model (~500 MB). For faster runs you can point
to a smaller one with `--spacy-model uk_core_news_sm` (lower POS/dependency
accuracy, but enough to verify the pipeline).

## Data layout

Put one `.txt` file per author into a directory (default: `Samples/`):

```
Samples/
├── Author1_Title.txt
└── Author2_Title.txt
```

The file name (without the extension) is used as the author identifier.

## Running

```bash
python main.py
```

With options:

```bash
python main.py --text-dir Samples --output-dir results --top-n 15
```

| Flag | Default | Description |
| --- | --- | --- |
| `--text-dir` | `Samples` | directory with the authors' `.txt` files |
| `--output-dir` | `results` | where to write tables and plots |
| `--spacy-model` | `uk_core_news_lg` | spaCy pipeline (POS + dependency parsing required) |
| `--max-words` | `10000` | truncate each author's text to N words |
| `--min-word-length` | `3` | min. word length within a phrase |
| `--top-n` | `10` | how many top phrases per author/type to plot |
| `--w2v-vector-size` / `--w2v-window` / `--w2v-epochs` | `100` / `3` / `20` | Word2Vec hyperparameters |
| `--vibe` | `clean` | console output style — `clean` or `gothic` (see below) |

## Output

The `results/` directory will contain:

- `statistics.xlsx` — overall statistics per author
- `phrases_by_type.xlsx` — phrase vocabularies by type (verbal/nominal/adverbial)
- `uniqueness_analysis.png` — unique vs. shared phrases
- `top{N}_*.png` — top-N phrases per author for each type
- `pca_analysis.png` — 2D projection of Word2Vec phrase vectors
- `similarity_heatmap.png` — cosine similarity between authors/types
- `frequency_distribution.png` — phrase frequency distribution
- `average_frequencies.png` — mean/median/max frequency
- `jaccard_analysis.png`, `jaccard_statistics.xlsx` — Jaccard index for every author pair

## Code structure

```
phrase_analysis/
├── config.py          # AnalysisConfig — every run parameter in one place
├── extraction.py      # phrase extraction from a spaCy Doc (syntactic dependencies)
├── corpus.py          # reading .txt files → phrase dataframe
├── stats.py           # frequencies, uniqueness, Jaccard index
├── vectorization.py   # Word2Vec, vector aggregation, PCA
├── visualization.py   # all plots (matplotlib/seaborn)
├── messages.py        # console narration in two styles (clean / gothic)
└── pipeline.py        # ties everything together into a single run
main.py                # CLI (argparse)
```

## About `--vibe gothic`

The original version of the script narrated its progress with comments like
*"СЛОВА-ПРИМАРИ ЯКІ ТРЕБА ВИГНАТИ"* ("GHOST-WORDS THAT MUST BE BANISHED") and
*"ЗВІР ПРОКИНУВСЯ"* ("THE BEAST HAS AWAKENED"). That's part of the project's
character, so instead of simply deleting it, it now lives as a separate
message set in `messages.py`, enabled with a flag:

```bash
python main.py --vibe gothic
```

The pipeline logic is identical in both modes — only what gets printed to the
console during the run changes.

## Known limitations of the original approach

- The Jaccard index is now computed for all author pairs (previously — only
  the first two), but the `jaccard_analysis.png` plot still shows a single
  pair for readability; the full table is in `jaccard_statistics.xlsx`.
- `Word2Vec` is trained on the phrase corpus itself (not on the full texts),
  which makes the vectors specific to a given run rather than transferable
  between projects — a deliberate simplicity trade-off, not a bug.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The suite runs offline and does **not** need `uk_core_news_lg`: it builds
spaCy `Doc` objects directly from parallel word/lemma/POS/head/dep lists, which
is all `extract_phrases` ever reads. Plots are checked at smoke level only —
that a non-empty PNG gets written — using the matplotlib `Agg` backend.
