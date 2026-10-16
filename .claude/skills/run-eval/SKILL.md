---
name: run-eval
description: Прогнать пайплайн и метрики AttrGuard после любого изменения в rules/ml/llm/cascade и сравнить с прошлым запуском.
---
# run-eval

```bash
# 1. Тесты
python -m pytest -q
# 2. Предсказания (конфигурация: rules | rules_ml | full)
python -m src.pipeline --input <data/splits/val.jsonl | tests/fixtures> --output outputs/ --config rules
# 3. Метрики
python -m src.evaluate --pred outputs/predictions.jsonl --gold <тот же вход> --out outputs/metrics/
```

Обновляются: `outputs/predictions.jsonl`, `outputs/predictions.csv`, `outputs/metrics/per_attribute.csv`,
`configs_comparison.csv`, `threshold_cost.csv`.

**Сравнение с прошлым запуском:** перед прогоном скопировать `outputs/metrics/per_attribute.csv` в
`outputs/metrics/prev/`; после — сравнить F1 и precision(proposed) по атрибутам.

**Порог «стало хуже»:** падение F1 любого ключевого атрибута > 2 п.п., или precision(proposed) < 0.90,
или появилось хотя бы одно `proposed` без evidence → откатить изменение или доказать улучшение
по F1 с весом трафика (SPEC 8.1). Настройку делать только на validation, **никогда на test**.
