"""Оркестрация этапов, статусы, вопрос продавцу (SPEC 3.4). Владелец — поток B, затем интеграция.

prepare → rules → [ml] → [llm]; после каждого этапа: normalize + check_evidence.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.contracts import AttributeResult, Card, ExtractResponse


@dataclass
class CascadeConfig:
    use_ml: bool = False
    use_llm: bool = False
    thresholds_path: str = "config/thresholds.yaml"
    categories_dir: str = "config/categories"


class Cascade:
    def __init__(self, config: CascadeConfig | None = None, ml=None, llm=None):
        self.config = config or CascadeConfig()
        self.ml = ml
        self.llm = llm

    def run(self, card: Card) -> ExtractResponse:
        raise NotImplementedError


_default: Cascade | None = None


def extract(card: Card) -> ExtractResponse:
    """Удобная точка входа: каскад с конфигурацией по умолчанию (только правила)."""
    global _default
    if _default is None:
        _default = Cascade()
    return _default.run(card)


__all__ = ["Cascade", "CascadeConfig", "extract", "AttributeResult"]
