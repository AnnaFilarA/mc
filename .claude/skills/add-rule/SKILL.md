---
name: add-rule
description: Добавить правило или синоним в Этап 1 AttrGuard (src/rules.py, config/rules/*.yaml) с обязательным span и тестами.
---
# add-rule

1. **Синоним значения** (самый дешёвый путь) — добавить в `config/rules/*.yaml`: `синоним: КаноническоеЗначение`. Каноническое значение обязано быть в `values` схемы категории.
2. **Регулярка** — в `src/rules.py`. Обязательно:
   - работает по **исходному** тексту источника (`prepared.texts[source]`), не по очищенному, чтобы span совпал;
   - возвращает `Candidate(attribute, raw_value, score, stage="rules", evidence=Evidence(text, start, end, source))`, где `texts[source][start:end] == text`;
   - score: словарь точно 0.95, регулярка 0.85, нечётко 0.6.
3. **Тест** — в `tests/test_rules.py`: минимум **3 позитивных** и **2 негативных** примера (негатив — похожий текст, где правило не должно сработать, например «microSD до 1 ТБ» для storage).
4. `python -m pytest -q`, затем скилл `run-eval`.
5. Нетривиальное решение (какое число считать ОЗУ и т.п.) — скилл `log-decision`.
