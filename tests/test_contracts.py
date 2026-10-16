"""Контракт SPEC 3.3 и формат фикстур."""
from src.contracts import (AttributeResult, Card, Evidence, ExtractResponse, Status,
                           evidence_is_valid, source_texts)
from tests.conftest import load_fixtures


def test_status_values():
    assert {s.value for s in Status} == {"unchanged", "proposed", "low_confidence", "conflict", "missing"}


def test_extract_response_roundtrip():
    card = Card(product_id="x", title="Смартфон 8/256GB", seller_attributes={"Цвет": "Черный"})
    title = source_texts(card)["title"]
    ev = Evidence(text="8/256GB", start=title.index("8/256GB"), end=title.index("8/256GB") + 7, source="title")
    assert evidence_is_valid(card, ev)
    res = ExtractResponse(
        product_id="x", category="smartphones", category_confidence=1.0,
        attributes=[AttributeResult(attribute="ram", was=None, value="8", unit="GB", status=Status.proposed,
                                    confidence=0.9, stage="rules", evidence=ev, question_to_seller=None)],
        card_status="ready", llm_calls=0, elapsed_ms=1, text_hash="h")
    assert ExtractResponse.model_validate_json(res.model_dump_json()) == res


def test_seller_attributes_evidence():
    card = Card(product_id="x", seller_attributes={"Бренд": "Apple", "Цвет": "Черный"})
    text = source_texts(card)["seller_attributes"]
    assert text == "Бренд: Apple\nЦвет: Черный"
    s = text.index("Черный")
    assert evidence_is_valid(card, Evidence(text="Черный", start=s, end=s + 6, source="seller_attributes"))
    assert not evidence_is_valid(card, Evidence(text="Белый", start=s, end=s + 5, source="seller_attributes"))
    assert not evidence_is_valid(card, None)


def test_fixtures_well_formed():
    fx = load_fixtures()
    assert 20 <= len(fx) <= 30
    statuses = {s.value for s in Status}
    for f in fx:
        Card.model_validate(f["card"])
        for code, exp in f["expected"]["attributes"].items():
            st = exp.get("status")
            for s in (st if isinstance(st, list) else [st] if st else []):
                assert s in statuses, (f["card"]["product_id"], code, s)
