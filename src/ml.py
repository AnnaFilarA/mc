"""Этап 2: лёгкие ML-модели (SPEC 3.5). Заглушка до фазы 3; владелец — поток F.

Вызывается только для атрибутов без уверенного ответа после Этапа 1 (SPEC F8).
"""
from __future__ import annotations

from src.contracts import Candidate
from src.prepare import PreparedCard


class MLStage:
    """Интерфейс Этапа 2."""

    def predict(self, prepared: PreparedCard, attributes: list[str], candidates: list[Candidate]) -> list[Candidate]:
        raise NotImplementedError


class NullML(MLStage):
    """Этап 2 выключен: ничего не добавляет (конфигурация «только правила»)."""

    def predict(self, prepared, attributes, candidates):
        return []


def train(train_path: str, val_path: str, out_dir: str = "models/") -> None:
    """Обучение на train, калибровка на validation. Test не трогаем."""
    raise NotImplementedError
