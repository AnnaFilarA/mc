"""Этап 1: правила и словари (SPEC 3.5). Владелец — поток B.

Каждое правило возвращает Candidate с evidence (span в исходном тексте источника).
Уверенность: точное совпадение со словарём 0.95; регулярка 0.85; нечёткое совпадение 0.6.
Словари синонимов — config/rules/*.yaml.
"""
from __future__ import annotations

from src.contracts import Candidate
from src.prepare import PreparedCard


def extract(prepared: PreparedCard) -> list[Candidate]:
    """Кандидаты для всех атрибутов схемы категории (несколько на атрибут допустимо)."""
    raise NotImplementedError
