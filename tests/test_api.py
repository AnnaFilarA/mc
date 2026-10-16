"""Тесты API и веба (поток C): extract, batch, идемпотентность, feedback, сценарии SPEC 4.5, страницы.

Все тесты работают на временной БД и временных каталогах; data/ и config/ не трогаются.
Тесты не зависят от того, подключено ли ядро: проверяют поведение API и инварианты контракта.
"""
import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import Ctx, create_app
from src.contracts import Card, Status, evidence_is_valid
from tests.conftest import ROOT, load_fixtures

HX = {"HX-Request": "true"}
BLANK = Card(product_id="zz1", title="Смартфон ZZ", description="", seller_attributes={})


@pytest.fixture
def env(tmp_path):
    cats = tmp_path / "categories"
    shutil.copytree(ROOT / "config" / "categories", cats)
    app = create_app(db_path=tmp_path / "t.db", feedback_path=tmp_path / "feedback" / "feedback.jsonl",
                     categories_dir=cats)
    with TestClient(app) as client:
        client.cats = cats
        client.fb_path = tmp_path / "feedback" / "feedback.jsonl"
        client.ctx = app.state.ctx
        client.ctx.storage.save_card(BLANK, owner="seller")
        yield client


def test_all_pages_ok(env):
    pid = load_fixtures()[0]["card"]["product_id"]
    for url in ["/", "/manager", f"/manager/card/{pid}", "/manager/card/zz1", "/manager/categories",
                "/seller", "/seller/new", "/seller/questions", "/health", "/v1/queue", "/static/style.css"]:
        r = env.get(url)
        assert r.status_code == 200, url
    assert env.get("/manager/card/nope").status_code == 404


def test_role_cookie_and_switcher(env):
    r = env.get("/role/manager", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/manager"
    assert env.cookies.get("role") == "manager"
    assert "/role/seller" in env.get("/manager").text and "/role/manager" in env.get("/seller").text
    assert env.get("/role/boss", follow_redirects=False).status_code == 422


def test_extract_contract_and_evidence(env):
    for fx in load_fixtures():
        r = env.post("/v1/extract", json=fx["card"])
        assert r.status_code == 200, fx["card"]["product_id"]
        data = r.json()
        card = Card.model_validate(fx["card"])
        assert data["product_id"] == card.product_id and data["text_hash"]
        assert data["category"] == fx["expected"]["category"]
        for a in data["attributes"]:
            if a["status"] in (Status.proposed.value, Status.unchanged.value):
                from src.contracts import Evidence
                assert evidence_is_valid(card, Evidence.model_validate(a["evidence"])), (card.product_id, a["attribute"])
            if a["status"] in ("conflict", "missing"):
                assert a["question_to_seller"] and a["question_to_seller"]["text"]


def test_extract_arbitrary_card(env):
    r = env.post("/v1/extract", json={"product_id": "x1", "title": "Смартфон Foo", "seller_attributes": {"Цвет": "Синий"}})
    assert r.status_code == 200
    data = r.json()
    assert data["category"] == "smartphones"
    codes = {a["attribute"] for a in data["attributes"]}
    assert {"brand", "ram", "storage"} <= codes
    assert env.post("/v1/extract", json={"title": "без id"}).status_code == 422


def test_batch(env):
    cards = [fx["card"] for fx in load_fixtures()[:3]]
    r = env.post("/v1/extract/batch", json=cards)
    assert r.status_code == 200
    assert [x["product_id"] for x in r.json()] == [c["product_id"] for c in cards]
    assert env.post("/v1/extract/batch", json=[]).json() == []


def test_idempotent_cache(env, monkeypatch):
    calls = []
    real = Ctx.run_core

    def counting(self, card):
        calls.append(card.product_id)
        return real(self, card)

    monkeypatch.setattr(Ctx, "run_core", counting)
    card = {"product_id": "idem1", "title": "Смартфон Bar", "description": "текст"}
    a = env.post("/v1/extract", json=card).json()
    b = env.post("/v1/extract", json=card).json()
    assert a == b and calls == ["idem1"]
    env.post("/v1/extract/batch", json=[card])
    assert calls == ["idem1"]
    env.post("/v1/extract", json={**card, "description": "другой текст"})  # хэш изменился → пересчёт
    assert calls == ["idem1", "idem1"]
    env.post("/v1/extract", json={**card, "product_id": "idem2"})  # другой product_id → пересчёт
    assert len(calls) == 3


def test_feedback_validation_and_storage(env):
    base = {"product_id": "zz1", "attribute": "brand", "action": "fix", "value": "Samsung", "actor": "manager"}
    r = env.post("/v1/feedback", json=base)
    assert r.status_code == 200 and r.json()["ok"]
    assert env.post("/v1/feedback", json={**base, "product_id": "nope"}).status_code == 404
    assert env.post("/v1/feedback", json={**base, "attribute": "nope"}).status_code == 404
    assert env.post("/v1/feedback", json={**base, "action": "fix", "value": ""}).status_code == 422
    assert env.post("/v1/feedback", json={**base, "action": "dance"}).status_code == 422
    assert env.post("/v1/feedback", json={**base, "action": "seller_answer"}).status_code == 422  # actor=manager
    assert env.post("/v1/feedback", json={**base, "action": "ask_seller", "attribute": "ram"}).status_code == 200
    rows = env.ctx.storage.feedback_for("zz1")
    assert [(x["action"], x["attribute"]) for x in rows] == [("fix", "brand"), ("ask_seller", "ram")]
    lines = [json.loads(l) for l in env.fb_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["value"] == "Samsung" and lines[0]["product_id"] == "zz1"
    assert {"ts", "actor", "action", "attribute", "was"} <= set(lines[0])


def test_feedback_form_encoded_htmx_returns_partial(env):
    r = env.post("/v1/feedback", headers=HX, data={
        "product_id": "zz1", "attribute": "brand", "action": "fix", "value": "Apple", "actor": "manager", "view": "manager"})
    assert r.status_code == 200 and 'id="card-panel"' in r.text and "Apple" in r.text


def _problem_fields(client, pid):
    html = client.get(f"/manager/card/{pid}").text
    return re.findall(r'<tr id="row-(\w+)" class="row-problem">', html)


def _exception_ids(client):
    return re.findall(r'<tr data-pid="([^"]+)">', client.get("/manager").text.split('id="exceptions"')[-1].split("</table>")[0]) \
        if 'id="exceptions"' in client.get("/manager").text else []


def _pick_value(client, pid, code):
    v = client.ctx.view(pid)
    spec = next(a.spec for a in v.attrs if a.code == code)
    return spec.values[0] if spec.values else "1"


def test_scenario_manager_resolves_exception(env):
    """SPEC 4.5, сценарий 2: менеджер открывает спорную карточку → принимает/исправляет → карточка уходит из исключений."""
    assert "zz1" in _exception_ids(env)
    fields = _problem_fields(env, "zz1")
    assert fields, "у пустой карточки должны быть проблемные поля"
    for code in fields:
        v = env.ctx.view("zz1")
        av = next(a for a in v.attrs if a.code == code)
        if av.base.value and code == fields[0]:
            r = env.post("/v1/feedback", json={"product_id": "zz1", "attribute": code, "action": "accept", "actor": "manager"})
        else:
            r = env.post("/v1/feedback", json={"product_id": "zz1", "attribute": code, "action": "fix",
                                               "value": _pick_value(env, "zz1", code), "actor": "manager"})
        assert r.status_code == 200
    assert _problem_fields(env, "zz1") == []
    assert env.ctx.view("zz1").card_status == "ready"
    assert "zz1" not in _exception_ids(env)
    panel = env.get("/manager/card/zz1").text
    assert "решено: менеджер" in panel


def test_accept_without_value_rejected(env):
    v = env.ctx.view("zz1")
    empty = next(a for a in v.attrs if a.base.value is None)
    r = env.post("/v1/feedback", json={"product_id": "zz1", "attribute": empty.code, "action": "accept", "actor": "manager"})
    assert r.status_code == 422


def test_scenario_seller_creates_card_and_answers(env):
    """SPEC 4.5, сценарий 1: продавец вводит карточку → предзаполнение и вопросы → отвечает → менеджер видит «ответ получен»."""
    assert "Название" in env.get("/seller/new").text
    form = {"product_id": "new-t1", "title": "Смартфон Foo X1", "description": "", "characteristics": "Цвет: Синий"}
    preview = env.post("/seller/check", data=form, headers=HX)
    assert preview.status_code == 200
    assert "Попадёт в" in preview.text and "из" in preview.text and 'name="f_ram"' in preview.text
    assert 'value="Синий"' in preview.text or "Синий" in preview.text  # предзаполнение из характеристик продавца
    assert "Введите название" in env.post("/seller/check", data={"title": ""}, headers=HX).text

    saved = env.post("/seller/save", data={**form, "f_marker": "1", "f_brand": "Samsung"}, headers=HX)
    assert saved.status_code == 200 and saved.headers["HX-Redirect"] == "/seller?saved=new-t1"
    assert "new-t1" in env.get("/seller").text
    assert env.ctx.storage.get_card("new-t1")[1] == "seller"
    # значение, которое продавец указал вручную, записано как разметка
    assert any(f["attribute"] == "brand" and f["actor"] == "seller" for f in env.ctx.storage.feedback_for("new-t1"))

    qpage = env.get("/seller/questions").text
    codes = re.findall(r'id="q-new-t1-(\w+)"', qpage)
    assert codes, "продавец должен увидеть вопросы по карточке"
    assert "нужно уточнить" in env.get("/seller").text
    for code in codes:
        r = env.post("/v1/feedback", headers=HX, data={
            "product_id": "new-t1", "attribute": code, "action": "seller_answer", "actor": "seller",
            "view": "seller_q", "value": _pick_value(env, "new-t1", code)})
        assert r.status_code == 200 and 'id="q-new-t1"' in r.text
    v = env.ctx.view("new-t1")
    assert not v.questions and v.answered
    manager_page = env.get("/manager").text
    assert "Ответ получен" in manager_page and "Смартфон Foo X1" in manager_page


def test_manager_sees_request_states(env):
    r = env.post("/v1/feedback", json={"product_id": "zz1", "attribute": "ram", "action": "ask_seller", "actor": "manager"})
    assert r.status_code == 200
    page = env.get("/manager").text
    assert "отправлен" in page
    env.get("/seller/questions")  # продавец открыл вопросы
    assert "прочитан" in env.get("/manager").text
    env.post("/v1/feedback", json={"product_id": "zz1", "attribute": "ram", "action": "seller_answer",
                                   "value": "8", "actor": "seller"})
    assert "Ответ получен" in env.get("/manager").text


def test_queue_and_category_approval(env):
    q = env.get("/v1/queue").json()
    assert any(x["product_id"] == "u01" for x in q) and all(x["category"] == "unknown" for x in q)
    assert "u01" in env.get("/manager/categories").text

    assert env.post("/v1/categories/robot_vacuums/approve").status_code == 404  # нет черновика
    assert env.post("/v1/categories/..%2Fx/approve").status_code == 404
    draft = env.cats / "robot_vacuums.draft.yaml"
    draft.write_text("category: robot_vacuums\nname_ru: Роботы-пылесосы\nattributes:\n  - code: brand\n    name_ru: Бренд\n    type: enum\n    required: true\n    values: [Dreame]\n", encoding="utf-8")
    page = env.get("/manager/categories").text
    assert "robot_vacuums" in page and "Утвердить" in page and "Править" in page

    bad = env.post("/manager/categories/robot_vacuums/draft", data={"yaml": "a: [unclosed"}, follow_redirects=False)
    assert bad.status_code == 303 and "error=" in bad.headers["location"]
    good = env.post("/manager/categories/robot_vacuums/draft",
                    data={"yaml": draft.read_text(encoding="utf-8").replace("Dreame", "Dreame, Roborock")}, follow_redirects=False)
    assert good.status_code == 303 and "saved=" in good.headers["location"]
    assert "Roborock" in draft.read_text(encoding="utf-8")

    r = env.post("/v1/categories/robot_vacuums/approve")
    assert r.status_code == 200 and r.json()["ok"]
    assert not draft.exists() and (env.cats / "robot_vacuums.yaml").exists()
    assert env.post("/v1/categories/robot_vacuums/approve").status_code == 404


def test_demo_cards_from_env_path(tmp_path):
    demo = tmp_path / "demo.jsonl"
    demo.write_text(json.dumps({"product_id": "d1", "title": "Смартфон Demo"}, ensure_ascii=False) + "\n", encoding="utf-8")
    app = create_app(db_path=tmp_path / "d.db", feedback_path=tmp_path / "fb.jsonl", demo=demo)
    with TestClient(app) as c:
        assert [x[0].product_id for x in c.app.state.ctx.storage.list_cards()] == ["d1"]
        assert c.get("/manager/card/d1").status_code == 200


def test_fallback_never_proposes_without_evidence(tmp_path):
    ctx = Ctx(db_path=tmp_path / "f.db", feedback_path=tmp_path / "f.jsonl")
    for fx in load_fixtures():
        card = Card.model_validate(fx["card"])
        for r in ctx.fallback_extract(card).attributes:
            if r.status in (Status.proposed, Status.unchanged):
                assert evidence_is_valid(card, r.evidence)
