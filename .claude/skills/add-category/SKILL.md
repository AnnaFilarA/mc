---
name: add-category
description: Добавить новую категорию товаров в AttrGuard или утвердить черновик схемы (*.draft.yaml). Использовать, когда появилась новая категория или менеджер утвердил черновик.
---
# add-category

1. **Схема.** `config/categories/<code>.yaml` по образцу `smartphones.yaml`: `category`, `name_ru`, `keywords`, `attributes[]` (`code`, `name_ru`, `type: enum|number|text`, `required`, `unit`, `values`, `seller_keys`, `range`). Черновик от LLM (`<code>.draft.yaml`) — проверить и переименовать в `.yaml`.
2. **Словари.** Синонимы значений — `config/rules/` (бренды, цвета, специфичные синонимы категории). Только значения из `values` могут стать `proposed`.
3. **Регулярки.** Для `number`-атрибутов — шаблон «число + единица» с контекстом (подпись атрибута рядом). Каждое правило возвращает span (см. скилл `add-rule`).
4. **Фикстуры.** 2–3 карточки в `tests/fixtures/<prefix>NN.json` (формат: `card` + `expected`), включая один трудный случай.
5. **Тест.** `python -m pytest -q` — зелёный.
6. **Оценка.** Скилл `run-eval`, проверить, что метрики других категорий не упали.
7. **Журнал.** Скилл `log-decision`: какая категория, откуда атрибуты (EDA / LLM-черновик / кейс).
