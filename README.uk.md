[English](README.md) | **Українська**

[![CI](https://github.com/ZavarOvek/phrase-stylometry/actions/workflows/ci.yml/badge.svg)](https://github.com/ZavarOvek/phrase-stylometry/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

# Phrase Analysis — синтаксичний фразовий аналіз авторського стилю

Інструмент для стилеметричного порівняння авторів за синтаксичними
словосполученнями: бере кілька текстів, виділяє дієслівні, іменні та
прислівникові словосполучення (`spaCy`-парсинг → залежності), рахує частоти
та унікальність словника, будує `Word2Vec`-вектори фраз, проєктує їх через
`PCA`, рахує косинусну схожість і індекс Жаккара між авторами.

## Встановлення

```bash
pip install -r requirements.txt
python -m spacy download uk_core_news_lg
```

`uk_core_news_lg` — велика модель (~500 МБ). Для швидших прогонів можна
вказати меншу через `--spacy-model uk_core_news_sm` (менша точність POS/dep,
але достатньо для перевірки пайплайна).

## Структура даних

Покладіть по одному `.txt`-файлу на автора в директорію (за замовчуванням —
`Samples/`):

```
Samples/
├── Автор1_Назва_твору.txt
└── Автор2_Назва_твору.txt
```

Ім'я файлу (без розширення) використовується як ідентифікатор автора.

## Запуск

```bash
python main.py
```

З опціями:

```bash
python main.py --text-dir Samples --output-dir results --top-n 15
```

| Прапорець | За замовчуванням | Опис |
|---|---|---|
| `--text-dir` | `Samples` | директорія з `.txt`-файлами авторів |
| `--output-dir` | `results` | куди писати таблиці й графіки |
| `--spacy-model` | `uk_core_news_lg` | spaCy-пайплайн (потрібні POS + dependency parsing) |
| `--max-words` | `10000` | обрізати текст кожного автора до N слів |
| `--min-word-length` | `3` | мін. довжина слова у фразі |
| `--top-n` | `10` | скільки топ-фраз на автора/тип будувати |
| `--w2v-vector-size` / `--w2v-window` / `--w2v-epochs` | `100` / `3` / `20` | гіперпараметри Word2Vec |
| `--vibe` | `clean` | стиль консольного виводу — `clean` або `gothic` (див. нижче) |

## Результат

У `results/` з'являються:

- `statistics.xlsx` — загальна статистика по авторах
- `phrases_by_type.xlsx` — словники фраз по типах (verbal/nominal/adverbial)
- `uniqueness_analysis.png` — унікальні vs. спільні фрази
- `top{N}_*.png` — топ-N словосполучень на автора по кожному типу
- `pca_analysis.png` — 2D-проєкція Word2Vec-векторів фраз
- `similarity_heatmap.png` — косинусна схожість між авторами/типами
- `frequency_distribution.png` — розподіл частот словосполучень
- `average_frequencies.png` — середня/медіана/максимум частоти
- `jaccard_analysis.png`, `jaccard_statistics.xlsx` — індекс Жаккара по кожній парі авторів

## Структура коду

```
phrase_analysis/
├── config.py          # AnalysisConfig — усі параметри запуску в одному місці
├── extraction.py       # витяг словосполучень з spaCy Doc (синтаксичні залежності)
├── corpus.py            # читання .txt-файлів → дата-фрейм фраз
├── stats.py              # частоти, унікальність, індекс Жаккара
├── vectorization.py     # Word2Vec, агрегація векторів, PCA
├── visualization.py     # усі графіки (matplotlib/seaborn)
├── messages.py           # консольний наратив у двох стилях (clean / gothic)
└── pipeline.py           # зв'язує все докупи в один прогін
main.py                   # CLI (argparse)
```

## Про `--vibe gothic`

Оригінальна версія скрипту супроводжувала прогрес коментарями на кшталт
*"СЛОВА-ПРИМАРИ ЯКІ ТРЕБА ВИГНАТИ"* та *"ЗВІР ПРОКИНУВСЯ"*. Це частина
характеру проєкту, тож замість того, щоб просто стерти, я виніс це окремим
набором повідомлень у `messages.py` — вмикається прапорцем:

```bash
python main.py --vibe gothic
```

Логіка пайплайна в обох режимах ідентична — змінюється лише те, що
друкується в консоль під час прогону.

## Відомі обмеження вихідного підходу

- Індекс Жаккара тепер рахується по всіх парах авторів (раніше — лише по
  перших двох), але графік `jaccard_analysis.png` все одно показує одну пару
  для читабельності; повна таблиця — у `jaccard_statistics.xlsx`.
- `Word2Vec` тренується на самому корпусі фраз (а не на повних текстах), що
  робить вектори специфічними для цього прогону, а не перенесюваними між
  проєктами — це навмисний компроміс простоти, а не помилка.

## Тестування

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

Тести працюють офлайн і **не потребують** `uk_core_news_lg`: вони будують
spaCy-`Doc` напряму з паралельних списків слів / лем / POS / heads / deps —
це все, що читає `extract_phrases`. Графіки перевіряються лише на smoke-рівні
(що PNG створено й він не порожній), через бекенд matplotlib `Agg`.
