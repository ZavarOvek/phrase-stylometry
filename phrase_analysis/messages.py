"""Console narration for the pipeline, in two interchangeable styles.

The original script's progress messages were written in a deliberately
unhinged, "the words whisper to me in the dark" register. That's fun, but
not what you want a stranger reading your code to see first. Rather than
forking the whole codebase, the flavor lives here as data, and a single
``--vibe`` CLI flag switches between it and a plain professional version.
Same logic underneath either way — only the print() narration changes.
"""

from __future__ import annotations

CLEAN: dict[str, str] = {
    "loading": "Завантаження текстів з {text_dir}...",
    "processing_author": "  Обробка: {author}",
    "loaded": "Знайдено {count} словосполучень від {n_authors} авторів.",
    "stats_header": "Статистика по авторах:",
    "freq_analysis": "Частотний аналіз...",
    "uniqueness_analysis": "Аналіз унікальності словника...",
    "top_phrases": "Побудова графіків топ-{n} словосполучень...",
    "vectorizing": "Векторизація фраз (Word2Vec)...",
    "pca": "Проєкція PCA...",
    "heatmap": "Матриця косинусної схожості...",
    "freq_distribution": "Розподіл частот...",
    "extra_stats": "Додаткова статистика частот...",
    "jaccard": "Розрахунок індексу Жаккара...",
    "done_header": "Готово. Результати збережено в '{output_dir}':",
    "summary_header": "Резюме:",
    "summary_total": "  Усього словосполучень: {total}",
    "summary_unique": "  Унікальних фраз: {unique}",
    "jaccard_summary_header": "  Індекс Жаккара (схожість словників за парами авторів):",
    "jaccard_line": (
        "    {author_a} / {author_b} — {type_label}: "
        "{jaccard:.3f} ({overlap:.1f}% перекриття)"
    ),
    "pca_title": "Векторні відстані між типами словосполучень",
    "top_phrases_title": "Топ-{n} {type_label} словосполучень",
    "jaccard_title": "Схожість словників між авторами\n(індекс Жаккара)",
}

GOTHIC: dict[str, str] = {
    "loading": "📚 ЗАВАНТАЖУЄМО ТЕКСТИ З {text_dir}... (слова вишиковуються в ряди)",
    "processing_author": "  ОБРОБЛЯЄМО: {author} (ще одна жертва)",
    "loaded": "✅ ЗНАЙДЕНО {count} СЛОВОСПОЛУЧЕНЬ ВІД {n_authors} АВТОРІВ (вони серед нас)",
    "stats_header": "📊 СТАТИСТИКА ПО АВТОРАХ:",
    "freq_analysis": "🔢 ЧАСТОТНИЙ АНАЛІЗ... (слова мерехтять з різною частотою)",
    "uniqueness_analysis": "🔍 АНАЛІЗ УНІКАЛЬНОСТІ... (у кожного автора свій почерк)",
    "top_phrases": "📊 СТВОРЮЄМО ГРАФІКИ ТОП-{n}... (голоси стають гучнішими)",
    "vectorizing": "🧮 ВЕКТОРНИЙ АНАЛІЗ... (слова стають точками у просторі)",
    "pca": "📉 PCA ВІЗУАЛІЗАЦІЯ... (стискаємо виміри)",
    "heatmap": "🔥 ТЕПЛОВА КАРТА СХОЖОСТІ... (вимірюємо родинність душ)",
    "freq_distribution": "📈 РОЗПОДІЛ ЧАСТОТ... (ритми та патерни)",
    "extra_stats": "📊 ДОДАТКОВИЙ СТАТИСТИЧНИЙ АНАЛІЗ... (копаємо глибше)",
    "jaccard": "🔍 РОЗРАХУНОК ІНДЕКСУ ЖАККАРА... (спільні сни)",
    "done_header": "✅ ГОТОВО! РЕЗУЛЬТАТИ ЗБЕРЕЖЕНО В '{output_dir}':",
    "summary_header": "📊 РЕЗЮМЕ:",
    "summary_total": "  ВСЬОГО ЗНАЙДЕНО: {total} словосполучень (голосів з темряви)",
    "summary_unique": "  УНІКАЛЬНИХ ФРАЗ: {unique} (індивідуальних почерків)",
    "jaccard_summary_header": "  ІНДЕКСИ ЖАККАРА (родинність словників):",
    "jaccard_line": (
        "    {author_a} / {author_b} — {type_label}: "
        "{jaccard:.3f} ({overlap:.1f}% перекриття)"
    ),
    "pca_title": "ВЕКТОРНІ ВІДСТАНІ МІЖ ТИПАМИ СЛІВ",
    "top_phrases_title": "ТОП-{n} {type_label} ШЕПОТІВ",
    "jaccard_title": "СХОЖІСТЬ СЛОВНИКІВ МІЖ АВТОРАМИ\n(ІНДЕКС СПІЛЬНИХ СНІВ)",
}


class Reporter:
    """Thin wrapper that prints progress messages in the selected vibe."""

    def __init__(self, vibe: str = "clean") -> None:
        self._messages = GOTHIC if vibe == "gothic" else CLEAN

    def say(self, key: str, **kwargs: object) -> None:
        print(self._messages[key].format(**kwargs))

    def text(self, key: str, **kwargs: object) -> str:
        """Return a formatted message without printing it (for chart titles)."""
        return self._messages[key].format(**kwargs)
