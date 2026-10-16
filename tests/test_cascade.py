"""Каскад: статусы (SPEC 3.4), вопросы продавцу, Этапы 2 и 3, пороги."""
import yaml

from src.cascade import Cascade, CascadeConfig, extract
from src.contracts import Candidate, Card, Evidence, Status, evidence_is_valid
from src.llm import LLMClient
from src.ml import MLStage


def card(title="", description="", seller=None, hint=None, pid="t"):
    return Card(product_id=pid, title=title, description=description, seller_attributes=seller or {}, category_hint=hint)


def attrs(resp):
    return {a.attribute: a for a in resp.attributes}


# ---------------------------------------------------------------- статусы

def test_proposed_has_evidence_and_canonical_value():
    c = card("Смартфон Samsung Galaxy A55 8/256GB синий", "Экран 6,6 дюйма")
    a = attrs(extract(c))
    assert a["ram"].status == Status.proposed and a["ram"].value == "8" and a["ram"].unit == "GB"
    assert evidence_is_valid(c, a["ram"].evidence)
    assert a["screen_size"].value == "6.6" and a["screen_size"].unit == "inch"


def test_unchanged_when_seller_value_is_already_canonical():
    a = attrs(extract(card("Apple iPhone 15 128 ГБ черный", seller={"Бренд": "Apple", "Цвет": "черный"}, hint="smartphones")))
    assert a["brand"].status == Status.unchanged and a["brand"].was == "Apple"
    assert a["color"].status == Status.unchanged   # регистр не важен


def test_seller_value_in_other_form_is_proposed():
    a = attrs(extract(card("Смартфон POCO X6 Pro белый", seller={"Оперативная память": "8 Гб", "Встроенная память": "256 Гб"})))
    assert a["ram"].status == Status.proposed and a["ram"].value == "8" and a["ram"].was == "8 Гб"
    assert a["storage"].status == Status.proposed


def test_unchanged_with_unit_in_was():
    a = attrs(extract(card("Смартфон Samsung 8/256GB", seller={"ОЗУ": "8 GB"})))
    assert a["ram"].status == Status.unchanged


def test_conflict_between_seller_and_description():
    c = card("Смартфон Xiaomi Redmi Note 13 6/128 Гб", "Корпус зелёного цвета.", seller={"Цвет": "Черный"})
    r = attrs(extract(c))["color"]
    assert r.status == Status.conflict and r.value in ("Черный", "Зеленый") and r.evidence is not None
    assert set(r.question_to_seller["options"]) == {"Черный", "Зеленый"}
    assert "Цвет" in r.question_to_seller["text"]


def test_conflict_title_vs_seller_number():
    r = attrs(extract(card("Ноутбук Acer Aspire 7 16GB/512GB i5", seller={"Оперативная память": "8 ГБ"})))["ram"]
    assert r.status == Status.conflict
    assert set(r.question_to_seller["options"]) == {"8 GB", "16 GB"}


def test_missing_has_question_with_schema_options():
    r = attrs(extract(card("Смартфон Honor X8b", "Отличный смартфон.", hint="smartphones")))
    assert r["ram"].status == Status.missing and r["ram"].value is None and r["ram"].stage == "none"
    assert r["ram"].evidence is None and r["ram"].confidence == 0.0
    q = r["ram"].question_to_seller
    assert q["text"] and "8 GB" in q["options"]
    assert r["color"].question_to_seller["options"][0] == "Черный"


def test_unparseable_seller_value_is_missing_with_hint():
    r = attrs(extract(card("Смартфон Honor X8b", seller={"Цвет": "цвет настроения"}, hint="smartphones")))["color"]
    assert r.status == Status.missing and r.was == "цвет настроения" and "цвет настроения" in r.question_to_seller["text"]


def test_low_confidence_has_value_and_evidence():
    c = card("Смартфон Honor X8b 8 ГБ", hint="smartphones")   # одиночное «8 ГБ» без подписи — неоднозначно
    r = attrs(extract(c))["ram"]
    assert r.status == Status.low_confidence and r.value == "8" and r.confidence < 0.8
    assert evidence_is_valid(c, r.evidence) and r.question_to_seller is None


def test_card_status_by_required_attributes():
    ready = extract(card("Смартфон Samsung Galaxy A55 8/256GB синий", "Экран 6,6 дюйма Android 14"))
    assert ready.card_status == "ready"
    seller = extract(card("Смартфон Honor X8b"))
    assert seller.card_status == "needs_seller"
    manager = extract(card("Смартфон Samsung Galaxy A55 синий 8 ГБ", "Экран 6,6 дюйма; 256 ГБ"))
    assert manager.card_status == "needs_manager"
    # необязательное поле (os) не делает карточку «needs_*»
    assert attrs(ready)["os"].status == Status.proposed
    no_os = extract(card("Смартфон Samsung Galaxy A55 8/256GB синий", "Экран 6,6 дюйма"))
    assert attrs(no_os)["os"].status == Status.missing and no_os.card_status == "ready"


def test_unknown_category_uses_common_schema():
    r = extract(card("Робот-пылесос Dreame D10 Plus белый", "Влажная уборка"))
    assert r.category == "unknown"
    assert {a.attribute for a in r.attributes} == {"brand", "color"}
    assert attrs(r)["color"].value == "Белый" and attrs(r)["brand"].status == Status.missing
    assert r.card_status == "needs_seller"


def test_response_metadata():
    c = card("Смартфон Samsung Galaxy A55 8/256GB")
    r = extract(c)
    from src.prepare import text_hash
    assert r.product_id == "t" and r.text_hash == text_hash(c) and r.llm_calls == 0 and r.elapsed_ms >= 0
    assert r.category == "smartphones" and 0 < r.category_confidence <= 1


# ---------------------------------------------------------------- пороги

def test_threshold_override_per_attribute(tmp_path):
    thr = tmp_path / "thr.yaml"
    thr.write_text(yaml.safe_dump({"default": {"high": 0.8, "low": 0.5}, "attributes": {"ram": {"high": 0.97}}}))
    c = card("Смартфон Samsung Galaxy A55 8/256GB синий")
    base = attrs(Cascade().run(c))
    strict = attrs(Cascade(CascadeConfig(thresholds_path=str(thr))).run(c))
    assert base["ram"].status == Status.proposed
    assert strict["ram"].status == Status.low_confidence and strict["storage"].status == Status.proposed


# ---------------------------------------------------------------- Этап 2

class FakeML(MLStage):
    def __init__(self, value="12", score=0.9, bad_evidence=False):
        self.calls, self.seen = [], None
        self.value, self.score, self.bad = value, score, bad_evidence

    def predict(self, prepared, attributes, candidates):
        self.seen = list(attributes)
        text = prepared.texts["title"]
        ev = Evidence(text="НЕТ В ТЕКСТЕ", start=0, end=12, source="title") if self.bad else \
            Evidence(text=text[:5], start=0, end=5, source="title")
        return [Candidate(attribute="ram", raw_value=self.value, score=self.score, stage="ml", evidence=ev),
                Candidate(attribute="brand", raw_value="Apple", score=0.99, stage="ml", evidence=ev)]


def test_ml_called_only_for_unresolved_attributes():
    c = card("Смартфон Honor X8b", hint="smartphones")
    ml = FakeML()
    r = Cascade(CascadeConfig(use_ml=True), ml=ml).run(c)
    assert "brand" not in ml.seen and "ram" in ml.seen   # brand решён правилами → ML не трогает
    a = attrs(r)
    assert a["ram"].status == Status.proposed and a["ram"].stage == "ml" and a["ram"].value == "12"
    assert a["brand"].stage == "rules"
    # выключен use_ml → ML не вызывается
    ml2 = FakeML()
    Cascade(CascadeConfig(use_ml=False), ml=ml2).run(c)
    assert ml2.seen is None


def test_ml_candidate_without_valid_evidence_is_dropped():
    c = card("Смартфон Honor X8b", hint="smartphones")
    r = Cascade(CascadeConfig(use_ml=True), ml=FakeML(bad_evidence=True)).run(c)
    assert attrs(r)["ram"].status == Status.missing


def test_ml_value_outside_schema_is_dropped():
    r = Cascade(CascadeConfig(use_ml=True), ml=FakeML(value="10")).run(card("Смартфон Honor X8b", hint="smartphones"))
    assert attrs(r)["ram"].status == Status.missing


# ---------------------------------------------------------------- Этап 3

class FakeLLM(LLMClient):
    def __init__(self, value="Зеленый", score=0.95):
        self.calls, self.seen, self.value, self.score = 0, None, value, score

    def resolve(self, prepared, attributes, candidates):
        self.seen = list(attributes)
        self.calls += 1
        t = prepared.texts["description"]
        i = t.index("зелёного")
        ev = Evidence(text=t[i:i + 8], start=i, end=i + 8, source="description")
        return [Candidate(attribute="color", raw_value=self.value, score=self.score, stage="llm", evidence=ev)]


CONFLICT = dict(title="Смартфон Xiaomi Redmi Note 13 6/128 Гб", description="Корпус зелёного цвета.",
                seller={"Цвет": "Черный"}, hint="smartphones")


def test_llm_only_for_disputed_and_capped_at_low_confidence():
    c = card(**CONFLICT)
    llm = FakeLLM()
    r = Cascade(CascadeConfig(use_llm=True), llm=llm).run(c)
    assert llm.seen == ["color"]                         # только спорное поле
    a = attrs(r)["color"]
    assert a.status == Status.low_confidence and a.stage == "llm" and a.value == "Зеленый"
    assert a.confidence < 0.8 and evidence_is_valid(c, a.evidence)
    assert r.llm_calls == 1
    assert attrs(r)["ram"].stage == "rules"


def test_llm_off_by_default_and_without_client():
    c = card(**CONFLICT)
    assert attrs(Cascade().run(c))["color"].status == Status.conflict
    assert attrs(Cascade(CascadeConfig(use_llm=True), llm=None).run(c))["color"].status == Status.conflict
    llm = FakeLLM()
    Cascade(CascadeConfig(use_llm=False), llm=llm).run(c)
    assert llm.seen is None


def test_llm_not_called_when_everything_is_confident():
    llm = FakeLLM()
    r = Cascade(CascadeConfig(use_llm=True), llm=llm).run(card("Смартфон Samsung Galaxy A55 8/256GB синий", "Экран 6,6 дюйма"))
    assert llm.seen is None and r.llm_calls == 0


def test_llm_value_outside_schema_is_dropped():
    r = Cascade(CascadeConfig(use_llm=True), llm=FakeLLM(value="Оранжевый")).run(card(**CONFLICT))
    assert attrs(r)["color"].status == Status.conflict
