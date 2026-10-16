"""Нормализация к канону и проверка evidence (SPEC F4, F5). Владелец — поток B.

Канон: number — строка числа без единицы ("8", "6.67", 1 ТБ → "1024"), unit из схемы;
enum — значение из справочника схемы (values). Значение вне справочника не может быть proposed.
"""
from __future__ import annotations

from src.contracts import Candidate, Card, evidence_is_valid
from src.prepare import AttributeSpec


def normalize(candidate: Candidate, spec: AttributeSpec) -> Candidate:
    """Вернуть копию кандидата с value/unit в каноне; value=None, если привести нельзя."""
    raise NotImplementedError


def check_evidence(card: Card, candidate: Candidate) -> bool:
    """True, если у кандидата есть evidence и он реально есть в исходном тексте."""
    return evidence_is_valid(card, candidate.evidence)
