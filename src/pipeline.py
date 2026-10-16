"""Пакетный режим (SPEC F11): python -m src.pipeline --input X --output outputs/. Владелец — поток B.

Вход: .jsonl / .json (список карточек) / .csv / каталог фикстур tests/fixtures (берётся поле "card").
Выход: <output>/predictions.jsonl (ExtractResponse на строку) и <output>/predictions.csv (строка на атрибут).
"""
from __future__ import annotations

import argparse

from src.contracts import Card, ExtractResponse


def load_cards(path: str) -> list[Card]:
    raise NotImplementedError


def write_predictions(responses: list[ExtractResponse], out_dir: str) -> None:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AttrGuard: пакетное извлечение характеристик")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--config", default="rules", choices=["rules", "rules_ml", "full"])
    parser.parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
