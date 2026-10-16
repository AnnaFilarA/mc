"""Правило без исключений (INTENT 2.2): proposed/unchanged только с evidence, найденной в исходном тексте."""
import pytest

from src.contracts import Card, Status, evidence_is_valid
from tests.conftest import load_fixtures


def test_no_proposed_without_evidence():
    from src.cascade import extract
    for fx in load_fixtures():
        card = Card.model_validate(fx["card"])
        for r in extract(card).attributes:
            if r.status in (Status.proposed, Status.unchanged):
                assert evidence_is_valid(card, r.evidence), (card.product_id, r.attribute, r.evidence)
            if r.stage == "llm":
                assert r.evidence is not None and evidence_is_valid(card, r.evidence)
