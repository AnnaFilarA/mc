"""Черновик схемы новой категории (INTENT 4.2). Владелец — поток E.

N карточек из очереди «новая категория» → один вызов LLM → config/categories/<new>.draft.yaml.
Менеджер утверждает: правит и переименовывает в .yaml.
"""
from __future__ import annotations

from src.contracts import Card


def bootstrap_schema(cards: list[Card], category: str, llm=None, click_stats: dict | None = None) -> dict:
    raise NotImplementedError


def save_draft(schema: dict, category: str, config_dir: str = "config/categories") -> str:
    raise NotImplementedError
