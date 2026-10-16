"""Тесты Этапа 3 (src/llm.py): без сети, транспорт — мок."""
import json

import pytest

from src.contracts import Candidate, Card, Evidence, evidence_is_valid, source_texts
from src.llm import HttpLLM, NullLLM, find_quote, load_llm
from src.prepare import AttributeSpec, CategorySchema, PreparedCard, text_hash

CARD = Card(
    product_id="p1",
    title="Смартфон Samsung Galaxy S23 8/256GB черный",
    description="Мощный флагман. Оперативная память: 8 ГБ, встроенная память 256 ГБ. Экран 6.1 дюйма. "
                "Цвет корпуса — Черный фантом.",
    seller_attributes={"Цвет": "чёрный", "ОЗУ": "8"},
)

SCHEMA = CategorySchema(
    category="smartphones", name_ru="Смартфоны", keywords=["смартфон"],
    attributes=[
        AttributeSpec("brand", "Бренд", "enum", True, None, ["Apple", "Samsung", "Xiaomi"]),
        AttributeSpec("color", "Цвет", "enum", True, None, ["Черный", "Белый", "Серый"]),
        AttributeSpec("ram", "Оперативная память", "number", True, "GB", [2, 4, 8, 12, 16]),
        AttributeSpec("screen_size", "Диагональ", "number", True, "inch", [], range=[4.0, 8.0]),
    ],
)


def make_prepared(card: Card = CARD) -> PreparedCard:
    return PreparedCard(card=card, category="smartphones", category_confidence=0.9, schema=SCHEMA,
                        texts=source_texts(card), clean={}, text_hash=text_hash(card))


def cfg(tmp_path, **kw) -> dict:
    base = {"provider": "openai_compatible", "model": "test-model", "base_url": "http://llm.local/v1",
            "max_calls_per_run": 10, "context_chars": 30, "cache_path": str(tmp_path / "cache.jsonl")}
    base.update(kw)
    return base


class FakeTransport:
    """Мок HTTP: отдаёт заранее заданные тексты ответа модели (OpenAI-формат) и пишет запросы."""

    def __init__(self, *contents):
        self.contents = list(contents)
        self.requests: list[tuple[str, dict, dict]] = []

    def __call__(self, url, headers, payload, timeout):
        self.requests.append((url, headers, payload))
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {"choices": [{"message": {"content": content}}]}


def answer(*items) -> dict:
    return {"attributes": [dict(zip(("attribute", "value", "quote", "source"), it)) for it in items]}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "test-key")


# ───────── NullLLM / load_llm ─────────

def test_null_llm_returns_nothing():
    llm = NullLLM()
    assert llm.resolve(make_prepared(), ["ram"], []) == []
    assert llm.complete_json("x", {}) == {}
    assert llm.calls == 0


def test_load_llm_default_config_is_null():
    assert isinstance(load_llm("config/llm.yaml"), NullLLM)


def test_load_llm_without_key_is_null(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY")
    p = tmp_path / "llm.yaml"
    p.write_text("enabled: true\nprovider: openai_compatible\nmodel: m\nbase_url: http://x/v1\n", encoding="utf-8")
    assert isinstance(load_llm(str(p)), NullLLM)


def test_load_llm_enabled_with_key_is_http(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("enabled: true\nprovider: openai_compatible\nmodel: m\nbase_url: http://x/v1\n", encoding="utf-8")
    assert isinstance(load_llm(str(p)), HttpLLM)


def test_load_llm_bad_config_is_null(tmp_path):
    p = tmp_path / "llm.yaml"
    p.write_text("enabled: true\nprovider: nope\nmodel: m\n", encoding="utf-8")
    assert isinstance(load_llm(str(p)), NullLLM)


def test_http_llm_requires_env_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(RuntimeError):
        HttpLLM(cfg(tmp_path))


# ───────── resolve ─────────

def test_resolve_valid_answer_gives_candidate_with_valid_evidence(tmp_path):
    t = FakeTransport(answer(("ram", "8", "Оперативная память: 8 ГБ", "description"),
                             ("color", "черный", "Черный фантом", "description")))
    llm = HttpLLM(cfg(tmp_path), transport=t)
    res = llm.resolve(make_prepared(), ["ram", "color"], [])
    by = {c.attribute: c for c in res}
    assert set(by) == {"ram", "color"}
    assert llm.calls == 1
    for c in res:
        assert c.stage == "llm" and c.score <= 0.7
        assert evidence_is_valid(CARD, c.evidence)
    assert by["ram"].value == "8" and by["ram"].unit == "GB"
    assert by["color"].value == "Черный"           # канон из справочника
    # ключ передаётся только в заголовке, адрес собран из base_url
    url, headers, payload = t.requests[0]
    assert url == "http://llm.local/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "test-model"


def test_resolve_case_insensitive_quote(tmp_path):
    t = FakeTransport(answer(("ram", "8", "оперативная память: 8 гб", "description")))
    res = HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["ram"], [])
    assert len(res) == 1
    assert res[0].evidence.text == "Оперативная память: 8 ГБ"   # текст из исходника, регистр исходный
    assert evidence_is_valid(CARD, res[0].evidence)


def test_resolve_drops_invented_quote(tmp_path):
    t = FakeTransport(answer(("ram", "8", "Оперативная память 12 ГБ DDR5", "description")))
    assert HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["ram"], []) == []


def test_resolve_quote_in_wrong_source_is_dropped(tmp_path):
    t = FakeTransport(answer(("ram", "8", "Оперативная память: 8 ГБ", "title")))
    assert HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["ram"], []) == []


def test_resolve_drops_enum_outside_dictionary(tmp_path):
    t = FakeTransport(answer(("brand", "Nokia", "Samsung Galaxy", "title"),
                             ("color", "Фантомный", "Черный фантом", "description")))
    assert HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["brand", "color"], []) == []


def test_resolve_drops_number_outside_values_or_range(tmp_path):
    t = FakeTransport(answer(("ram", "256", "256 ГБ", "description"),
                             ("screen_size", "61", "6.1 дюйма", "description")))
    assert HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["ram", "screen_size"], []) == []


def test_resolve_ignores_attributes_not_requested(tmp_path):
    t = FakeTransport(answer(("brand", "Samsung", "Samsung", "title")))
    assert HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["ram"], []) == []


def test_resolve_invalid_json_gives_empty_and_single_retry(tmp_path):
    t = FakeTransport("это не JSON {")
    llm = HttpLLM(cfg(tmp_path), transport=t)
    assert llm.resolve(make_prepared(), ["ram"], []) == []
    assert llm.calls == 2           # один повтор, не больше
    assert len(t.requests) == 2


def test_resolve_retry_recovers(tmp_path):
    t = FakeTransport("мусор", answer(("ram", "8", "8 ГБ", "description")))
    res = HttpLLM(cfg(tmp_path), transport=t).resolve(make_prepared(), ["ram"], [])
    assert len(res) == 1


def test_resolve_wrong_shape_and_code_fence(tmp_path):
    llm = HttpLLM(cfg(tmp_path), transport=FakeTransport({"attributes": "нет"}))
    assert llm.resolve(make_prepared(), ["ram"], []) == []
    fenced = "```json\n" + json.dumps(answer(("ram", "8", "8 ГБ", "description"))) + "\n```"
    llm2 = HttpLLM(cfg(tmp_path, cache_path=None), transport=FakeTransport(fenced))
    assert len(llm2.resolve(make_prepared(), ["ram"], [])) == 1


def test_resolve_transport_error_gives_empty(tmp_path):
    def boom(*a, **k):
        raise OSError("network down")

    llm = HttpLLM(cfg(tmp_path), transport=boom)
    assert llm.resolve(make_prepared(), ["ram"], []) == []


def test_resolve_no_known_attributes_makes_no_call(tmp_path):
    t = FakeTransport(answer())
    llm = HttpLLM(cfg(tmp_path), transport=t)
    assert llm.resolve(make_prepared(), [], []) == []
    assert llm.resolve(make_prepared(), ["unknown_attr"], []) == []
    assert llm.calls == 0 and not t.requests


# ───────── кэш и лимит ─────────

def test_cache_second_call_does_not_increase_calls(tmp_path):
    t = FakeTransport(answer(("ram", "8", "8 ГБ", "description")))
    llm = HttpLLM(cfg(tmp_path), transport=t)
    r1 = llm.resolve(make_prepared(), ["ram"], [])
    r2 = llm.resolve(make_prepared(), ["ram"], [])
    assert llm.calls == 1 and len(t.requests) == 1
    assert r1 == r2
    # кэш на диске: новый клиент тоже не ходит в сеть
    llm2 = HttpLLM(cfg(tmp_path), transport=FakeTransport(answer()))
    assert llm2.resolve(make_prepared(), ["ram"], []) == r1
    assert llm2.calls == 0


def test_call_limit(tmp_path):
    t = FakeTransport(answer(("ram", "8", "8 ГБ", "description")))
    llm = HttpLLM(cfg(tmp_path, max_calls_per_run=1), transport=t)
    assert len(llm.resolve(make_prepared(), ["ram"], [])) == 1
    other = Card(product_id="p2", title="Смартфон Apple 16GB", description="ОЗУ 16 ГБ")
    assert llm.resolve(make_prepared(other), ["ram"], []) == []   # лимит исчерпан → без вызова
    assert llm.calls == 1 and len(t.requests) == 1


# ───────── промпт ─────────

def test_prompt_contains_only_disputed_attrs_values_and_context_window(tmp_path):
    long_desc = "Вводный текст без характеристик. " * 30 + "Оперативная память: 8 ГБ. " + "Прочее. " * 60
    card = Card(product_id="p3", title="Смартфон Samsung", description=long_desc)
    start = long_desc.index("8 ГБ")
    cand = Candidate(attribute="ram", raw_value="8", score=0.5, stage="rules",
                     evidence=Evidence(text="8 ГБ", start=start, end=start + 4, source="description"))
    llm = HttpLLM(cfg(tmp_path, context_chars=30), transport=FakeTransport(answer()))
    prompt = llm.build_prompt(make_prepared(card), ["ram"], [cand])
    assert "ram" in prompt and "допустимые значения: 2, 4, 8, 12, 16" in prompt
    assert "brand" not in prompt and "Samsung, " not in prompt     # чужие атрибуты/справочники не попадают
    assert "8 ГБ" in prompt
    assert len(prompt) < len(long_desc) // 2                        # только окно, не весь текст
    assert "[title]" not in prompt                                  # окна только вокруг evidence


def test_prompt_without_candidates_uses_truncated_head(tmp_path):
    card = Card(product_id="p4", title="Смартфон", description="Х" * 5000)
    llm = HttpLLM(cfg(tmp_path), transport=FakeTransport(answer()))
    prompt = llm.build_prompt(make_prepared(card), ["ram"], [])
    assert "[title] Смартфон" in prompt and len(prompt) < 1500


def test_find_quote_helper():
    texts = source_texts(CARD)
    assert find_quote(texts, "Samsung Galaxy", "title") == ("title", 9, 23)
    assert find_quote(texts, "нет такого", None) is None
    assert find_quote(texts, "", "title") is None


# ───────── провайдеры ─────────

def test_yandexgpt_adapter(tmp_path):
    seen = {}

    def tr(url, headers, payload, timeout):
        seen.update(url=url, headers=headers, payload=payload)
        return {"result": {"alternatives": [{"message": {"text": json.dumps(answer(("ram", "8", "8 ГБ", "description")))}}]}}

    llm = HttpLLM(cfg(tmp_path, provider="yandexgpt", model="yandexgpt-lite", folder_id="f1", base_url=None),
                  transport=tr)
    assert len(llm.resolve(make_prepared(), ["ram"], [])) == 1
    assert seen["payload"]["modelUri"] == "gpt://f1/yandexgpt-lite/latest"
    assert seen["headers"]["Authorization"] == "Api-Key test-key"


def test_complete_json_uses_bootstrap_model(tmp_path):
    t = FakeTransport({"ok": 1})
    llm = HttpLLM(cfg(tmp_path, schema_bootstrap={"model": "big-model"}), transport=t)
    assert llm.complete_json("привет", {"type": "object"}) == {"ok": 1}
    assert t.requests[0][2]["model"] == "big-model"
