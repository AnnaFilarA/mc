# AttrGuard — автоматическое заполнение характеристик товаров

Хакатон «Точка сборки», кейс М.Видео. Команда «Все из ничего».

Каскад «правила → лёгкий ML → LLM (по умолчанию выключена)»: текст карточки (`title`, `description`,
`seller_attributes`) → характеристики со значением, статусом, уверенностью и подтверждающим фрагментом текста.
Значение без фрагмента-подтверждения (`evidence`) никогда не получает статус `proposed`.

Документы: [INTENT.md](INTENT.md) — зачем и что, [SPEC.md](SPEC.md) — как, [DECISIONS.md](DECISIONS.md) — журнал решений,
[BACKLOG.md](BACKLOG.md) — идеи на потом.

> Черновик. Дополняется по мере готовности фаз.

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
python -m src.pipeline --input <file|tests/fixtures> --output outputs/
```

## Запуск веба

```bash
uvicorn src.api:app --port 8000
# демо из Colab: cloudflared tunnel --url http://localhost:8000
```

## Структура

| Путь | Что |
|---|---|
| `src/contracts.py` | контракт ответа (SPEC 3.3) |
| `src/prepare.py`, `rules.py`, `normalize.py`, `cascade.py` | Шаг 0, Этап 1, нормализация, статусы |
| `src/ml.py`, `src/llm.py` | Этапы 2 и 3 |
| `src/evaluate.py` | метрики → `outputs/metrics/` |
| `src/api.py`, `web/` | API и веб-интерфейс |
| `config/categories/*.yaml` | схемы категорий |
| `config/thresholds.yaml`, `config/llm.yaml` | пороги, настройки LLM |
| `tests/fixtures/` | 26 эталонных карточек |
| `data/raw/` | датасет кейса (не в git) |
