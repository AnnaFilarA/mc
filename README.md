# AttrGuard — автоматическое заполнение характеристик товаров

Хакатон «Точка сборки», кейс М.Видео. Команда «Все из ничего».

Каскад «правила → лёгкий ML → LLM (по умолчанию выключена)»: текст карточки (`title`, `description`,
`seller_attributes`) → характеристики со значением, статусом, уверенностью и подтверждающим фрагментом текста.
Значение без фрагмента-подтверждения (`evidence`) никогда не получает статус `proposed`.

Документы: [INTENT.md](INTENT.md) — зачем и что, [SPEC.md](SPEC.md) — как, [DECISIONS.md](DECISIONS.md) — журнал решений,
[BACKLOG.md](BACKLOG.md) — идеи на потом.

> Черновик: Colab-инструкция и итоговые метрики дополнятся после прогона на датасете кейса.

## Установка

```bash
git clone <repo> attrguard && cd attrguard
pip install -r requirements.txt
python -m pytest -q
```

### Google Colab

```python
!git clone <repo> attrguard
%cd attrguard
!pip install -q -r requirements.txt
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p data/raw && cp /content/drive/MyDrive/<путь к датасету>/* data/raw/
```

## Запуск пайплайна

```bash
# пакетный режим (SPEC F11): .jsonl / .json / .csv / каталог фикстур
python -m src.pipeline --input <file|tests/fixtures> --output outputs/ [--config rules|rules_ml|full]
# → outputs/predictions.jsonl (ответ по контракту на карточку), outputs/predictions.csv (строка на атрибут)
```

CSV-вход: колонки `product_id, title, description, seller_attributes` (JSON-строка), `category_hint` (опц.).

## Метрики

```bash
python -m src.evaluate --pred outputs/predictions.jsonl --gold <gold.jsonl|tests/fixtures> --out outputs/metrics/ \
    [--compare rules=a.jsonl rules_ml=b.jsonl full=c.jsonl] [--weights weights.csv]
```

Файлы в `outputs/metrics/`: `per_attribute.csv`, `configs_comparison.csv`, `threshold_cost.csv` + `threshold_cost.png`, `summary.json`.
Процедура и порог «стало хуже» — скилл `.claude/skills/run-eval`.

## Веб-интерфейс и API

```bash
uvicorn src.api:app --port 8000
# демо из Colab: cloudflared tunnel --url http://localhost:8000
```

- `/` — выбор роли; `/manager` — кабинет менеджера (исключения, карточка, новые категории); `/seller` — кабинет продавца.
- API: `POST /v1/extract`, `POST /v1/extract/batch`, `POST /v1/feedback`, `GET /v1/queue`, `POST /v1/categories/{id}/approve`.
- Переменные окружения: `ATTRGUARD_DB` (SQLite, по умолчанию `data/attrguard.db`), `ATTRGUARD_FEEDBACK` (разметка, `data/feedback/`),
  `ATTRGUARD_DEMO` (демо-карточки: каталог json или .jsonl; по умолчанию `tests/fixtures`), `ATTRGUARD_CATEGORIES`.
- HTMX лежит в `web/static/`, интернет для интерфейса не нужен.
- Проверка сценариев SPEC 4.5 в браузере: `tests/e2e/ui_scenarios.py` (скриншоты — `outputs/screenshots/`).

## LLM (Этап 3)

По умолчанию **выключена** (`config/llm.yaml: enabled: false`) — пайплайн работает без неё. Включить:
`enabled: true`, `provider` (`openai_compatible` | `gigachat` | `yandexgpt`), `model`, `base_url`/`folder_id`, и
`export LLM_API_KEY=...`. Есть кэш ответов и лимит вызовов на прогон; значение без цитаты из текста отбрасывается,
статус LLM-значения — максимум `low_confidence`.

## Новая категория

Карточки неизвестной категории попадают в очередь; при накоплении 20 карточек
`python -m src.schema_bootstrap --queue data/new_category_queue.jsonl --category <code>` создаёт
`config/categories/<code>.draft.yaml`; менеджер утверждает его в `/manager/categories`.

## Состояние

| Фаза | Статус |
|---|---|
| 0. Каркас, контракт, 26 эталонных карточек | готово |
| 1. Ядро (правила), веб и API, оценка, LLM-интерфейс | готово, проверено на эталонных карточках |
| 1. EDA (поток A) | ждёт датасет в `data/raw/` |
| 2. Прогон на реальных данных, метрики | ждёт датасет |
| 3. ML, калибровка порогов, бизнес-метрики | после фазы 2 |
| 4. Финал, реальный LLM | после фазы 3; ключа LLM пока нет |

## Структура

| Путь | Что |
|---|---|
| `src/contracts.py` | контракт ответа (SPEC 3.3) |
| `src/prepare.py`, `rules.py`, `normalize.py`, `cascade.py` | Шаг 0, Этап 1, нормализация, статусы |
| `src/ml.py`, `src/llm.py` | Этапы 2 и 3 |
| `src/evaluate.py` | метрики → `outputs/metrics/` |
| `src/api.py`, `src/storage.py`, `web/` | API, SQLite, веб-интерфейс |
| `src/schema_bootstrap.py` | черновик схемы новой категории |
| `config/rules/*.yaml` | словари синонимов Этапа 1 |
| `config/categories/*.yaml` | схемы категорий |
| `config/thresholds.yaml`, `config/llm.yaml` | пороги, настройки LLM |
| `tests/fixtures/` | 26 эталонных карточек |
| `data/raw/` | датасет кейса (не в git) |
