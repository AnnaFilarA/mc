"""Метрики (SPEC 2.3). Владелец — поток D.

Эталон (gold): jsonl {"product_id", "category", "attributes": {code: value|null}} или каталог фикстур.
Выход в outputs/metrics/: per_attribute.csv, configs_comparison.csv, threshold_cost.csv (+ png).
"""
from __future__ import annotations


def evaluate(predictions_path: str, gold_path: str, out_dir: str = "outputs/metrics") -> dict:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
