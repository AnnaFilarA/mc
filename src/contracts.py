"""Замороженный контракт AttrGuard (SPEC 3.3).

Менять этот файл может только оркестратор. Подагент, которому нужно изменение,
пишет запрос в DECISIONS.md и останавливается.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["title", "description", "seller_attributes"]
Stage = Literal["rules", "ml", "llm", "none"]
CardStatus = Literal["ready", "needs_manager", "needs_seller"]


class Status(str, Enum):
    unchanged = "unchanged"
    proposed = "proposed"
    low_confidence = "low_confidence"
    conflict = "conflict"
    missing = "missing"


class Card(BaseModel):
    """Входная карточка товара (SPEC F1)."""

    product_id: str
    title: str = ""
    description: str = ""
    # код атрибута (или сырое имя поля поставщика) -> значение в свободной форме
    seller_attributes: dict[str, str] = Field(default_factory=dict)
    category_hint: str | None = None


def seller_attributes_text(seller_attributes: dict[str, str]) -> str:
    """Сериализация seller_attributes для evidence: строки «ключ: значение», разделитель «\\n»,
    порядок — порядок ключей в dict. Позиции evidence с source="seller_attributes" — в этой строке."""
    return "\n".join(f"{k}: {v}" for k, v in seller_attributes.items())


def source_texts(card: Card) -> dict[str, str]:
    """Исходные тексты карточки по источникам. Evidence всегда ссылается на один из них (сырой, без очистки)."""
    return {
        "title": card.title,
        "description": card.description,
        "seller_attributes": seller_attributes_text(card.seller_attributes),
    }


def evidence_is_valid(card: Card, ev: "Evidence | None") -> bool:
    """Инвариант evidence: фрагмент реально есть в исходном тексте на указанной позиции."""
    if ev is None or not ev.text:
        return False
    src = source_texts(card).get(ev.source, "")
    return 0 <= ev.start < ev.end <= len(src) and src[ev.start:ev.end] == ev.text


class Evidence(BaseModel):
    """Фрагмент исходного текста. Инвариант: source_text[start:end] == text."""

    text: str
    start: int
    end: int
    source: Source


class Candidate(BaseModel):
    """Кандидат значения от любого этапа (rules / ml / llm) до нормализации и статусов."""

    attribute: str
    raw_value: str
    value: str | None = None  # канон после normalize; None — ещё не нормализован или не удалось
    unit: str | None = None
    score: float = 0.0
    stage: Stage = "rules"
    evidence: Evidence | None = None


class AttributeResult(BaseModel):
    attribute: str
    was: str | None
    value: str | None
    unit: str | None
    status: Status
    confidence: float
    stage: Stage
    evidence: Evidence | None
    question_to_seller: dict | None  # {"text": str, "options": [str]}


class ExtractResponse(BaseModel):
    product_id: str
    category: str
    category_confidence: float
    attributes: list[AttributeResult]
    card_status: CardStatus
    llm_calls: int
    elapsed_ms: int
    text_hash: str
