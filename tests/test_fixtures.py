"""Каскад на эталонных карточках (SPEC N6)."""
import pytest

from src.contracts import Card, Status
from tests.conftest import load_fixtures

FIXTURES = load_fixtures()
IDS = [f["card"]["product_id"] for f in FIXTURES]


def _run(card_dict):
    from src.cascade import extract
    return extract(Card.model_validate(card_dict))


@pytest.mark.parametrize("fx", FIXTURES, ids=IDS)
def test_fixture(fx):
    resp = _run(fx["card"])
    exp = fx["expected"]
    assert resp.category == exp["category"]
    if "card_status" in exp:
        assert resp.card_status == exp["card_status"]
    got = {a.attribute: a for a in resp.attributes}
    for code, e in exp["attributes"].items():
        assert code in got, f"нет атрибута {code}"
        r = got[code]
        if "status" in e:
            allowed = e["status"] if isinstance(e["status"], list) else [e["status"]]
            assert r.status.value in allowed, f"{code}: статус {r.status.value}, ожидали {allowed}"
        if "value" in e:
            assert r.value == e["value"], f"{code}: {r.value!r} != {e['value']!r}"
        if "unit" in e:
            assert r.unit == e["unit"], f"{code}: unit {r.unit!r} != {e['unit']!r}"
        if r.status in (Status.conflict, Status.missing):
            q = r.question_to_seller
            assert q and q.get("text") and isinstance(q.get("options"), list), f"{code}: нет вопроса продавцу"
