"""Шаг 0: очистка текста, определение категории, загрузка схемы. Владелец — поток B.

Схемы категорий: config/categories/<category>.yaml (формат — см. smartphones.yaml).
Файлы *.draft.yaml не загружаются (черновики новых категорий, INTENT 4.2).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.contracts import Card, source_texts

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
UNKNOWN_CATEGORY = "unknown"   # категория не найдена → общие атрибуты из _common.yaml + очередь «новая категория»


@dataclass
class AttributeSpec:
    code: str
    name_ru: str
    type: str                          # "enum" | "number" | "text"
    required: bool = False
    unit: str | None = None
    values: list = field(default_factory=list)       # справочник допустимых значений
    seller_keys: list[str] = field(default_factory=list)
    range: list[float] | None = None   # [min, max] для number


@dataclass
class CategorySchema:
    category: str
    name_ru: str
    keywords: list[str]
    attributes: list[AttributeSpec]

    def attr(self, code: str) -> AttributeSpec | None:
        return next((a for a in self.attributes if a.code == code), None)


@dataclass
class PreparedCard:
    card: Card
    category: str
    category_confidence: float
    schema: CategorySchema
    texts: dict[str, str]              # source -> исходный текст (source_texts), evidence ссылается на него
    clean: dict[str, str]              # source -> очищенный текст (для классификации/ML; не для evidence)
    text_hash: str


def load_schemas(config_dir: Path | str = CONFIG_DIR / "categories") -> dict[str, CategorySchema]:
    """Загрузить все утверждённые схемы (*.yaml, кроме *.draft.yaml). Ключ — код категории; включает '_common'."""
    raise NotImplementedError


def clean_text(text: str) -> str:
    """Нормализация для поиска: lower, ё→е, схлопывание пробелов. Длина может меняться — не для evidence."""
    raise NotImplementedError


def text_hash(card: Card) -> str:
    """sha256 от title + description + seller_attributes (INTENT 4.1, идемпотентность)."""
    payload = json.dumps([card.title, card.description, card.seller_attributes], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def detect_category(card: Card, schemas: dict[str, CategorySchema]) -> tuple[str, float]:
    """category_hint, если он есть в схемах; иначе классификатор по title. Не уверены → (UNKNOWN_CATEGORY, conf)."""
    raise NotImplementedError


def prepare(card: Card, schemas: dict[str, CategorySchema]) -> PreparedCard:
    raise NotImplementedError


__all__ = ["AttributeSpec", "CategorySchema", "PreparedCard", "UNKNOWN_CATEGORY", "load_schemas",
           "clean_text", "text_hash", "detect_category", "prepare", "source_texts"]
