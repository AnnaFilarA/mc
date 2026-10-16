"""Этап 3: LLM (SPEC 3.5). По умолчанию выключен. Владелец — поток E.

Ключ только из переменной окружения LLM_API_KEY (SPEC N8). Значение без цитаты из текста отбрасывается.
"""
from __future__ import annotations

from src.contracts import Candidate
from src.prepare import PreparedCard


class LLMClient:
    """Интерфейс Этапа 3. calls — число реальных (не из кэша) вызовов за прогон."""

    calls: int = 0

    def resolve(self, prepared: PreparedCard, attributes: list[str], candidates: list[Candidate]) -> list[Candidate]:
        """Кандидаты stage="llm" только для переданных спорных атрибутов, каждый с проверенной evidence."""
        raise NotImplementedError

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Сырой вызов со строгой JSON-схемой ответа (нужен schema_bootstrap)."""
        raise NotImplementedError


class NullLLM(LLMClient):
    """Заглушка: ничего не вызывает, ничего не возвращает."""

    def resolve(self, prepared, attributes, candidates):
        return []

    def complete_json(self, prompt, schema):
        return {}


def load_llm(config_path: str = "config/llm.yaml") -> LLMClient:
    """enabled: false или нет LLM_API_KEY → NullLLM."""
    return NullLLM()
