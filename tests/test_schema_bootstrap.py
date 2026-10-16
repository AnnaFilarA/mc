"""Тесты черновика схемы новой категории (src/schema_bootstrap.py): без сети."""
import pytest
import yaml

from src.contracts import Card
from src.llm import NullLLM
from src.schema_bootstrap import (NewCategoryQueue, bootstrap_schema, heuristic_schema, main, save_draft,
                                  validate_schema)


def tablets(n: int) -> list[Card]:
    brands = ["Apple", "Samsung", "Xiaomi"]
    return [Card(product_id=f"t{i}", title=f"Планшет {brands[i % 3]} Tab {i} 128GB серый",
                 description="Планшет с большим экраном и мощным процессором.") for i in range(n)]


class MockLLM:
    calls = 0

    def __init__(self, response):
        self.response = response
        self.prompts = []

    def complete_json(self, prompt, schema):
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


GOOD = {
    "category": "tablets", "name_ru": "Планшеты", "keywords": ["планшет", "tablet"],
    "attributes": [
        {"code": "brand", "name_ru": "Бренд", "type": "enum", "required": True, "values": ["Apple", "Samsung"],
         "seller_keys": ["бренд"]},
        {"code": "storage", "name_ru": "Встроенная память", "type": "number", "required": True, "unit": "GB",
         "values": ["64", "128", "256"]},
        {"code": "Screen Size", "name_ru": "Диагональ", "type": "number", "required": False, "unit": "inch"},
        {"code": "broken", "name_ru": "", "type": "enum"},                       # без названия — отбросить
        {"code": "empty_enum", "name_ru": "Без справочника", "type": "enum", "required": False, "values": []},
    ],
}


def test_too_few_cards_raises():
    with pytest.raises(ValueError):
        bootstrap_schema(tablets(5), "tablets", MockLLM(GOOD))


def test_mock_llm_draft_saved(tmp_path):
    llm = MockLLM(GOOD)
    click = {"планшет samsung": 120, "планшет для рисования": 40}
    schema = bootstrap_schema(tablets(20), "tablets", llm, click_stats=click)
    assert llm.calls == 1
    assert "планшет samsung" in llm.prompts[0] and "Tab 3" in llm.prompts[0]
    assert schema["draft_source"] == "llm" and schema["category"] == "tablets"
    codes = [a["code"] for a in schema["attributes"]]
    assert codes == ["brand", "storage", "screen_size", "empty_enum"]
    assert schema["attributes"][-1]["type"] == "text"                # enum без значений → text
    storage = schema["attributes"][1]
    assert storage["values"] == [64, 128, 256] and storage["unit"] == "GB"

    path = save_draft(schema, "tablets", str(tmp_path))
    assert path.endswith("tablets.draft.yaml")
    loaded = yaml.safe_load(open(path, encoding="utf-8"))
    assert loaded["category"] == "tablets" and loaded["name_ru"] == "Планшеты"
    assert {"code", "name_ru", "type", "required"} <= set(loaded["attributes"][0])


def test_save_draft_never_touches_approved(tmp_path):
    approved = tmp_path / "tablets.yaml"
    approved.write_text("category: tablets\n# утверждено\n", encoding="utf-8")
    save_draft({"category": "tablets", "attributes": []}, "tablets", str(tmp_path))
    assert approved.read_text(encoding="utf-8") == "category: tablets\n# утверждено\n"
    assert (tmp_path / "tablets.draft.yaml").exists()


def test_save_draft_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        save_draft({}, "../evil", str(tmp_path))


def test_null_llm_heuristic_draft(tmp_path):
    schema = bootstrap_schema(tablets(20), "tablets", NullLLM())
    assert schema["draft_source"] == "heuristic"
    assert "планшет" in schema["keywords"]
    codes = {a["code"] for a in schema["attributes"]}
    assert {"brand", "color"} <= codes
    brand = next(a for a in schema["attributes"] if a["code"] == "brand")
    assert {"Apple", "Samsung", "Xiaomi"} <= set(brand["values"])
    path = save_draft(schema, "tablets", str(tmp_path))
    assert yaml.safe_load(open(path, encoding="utf-8"))["draft_source"] == "heuristic"


def test_bad_llm_answer_falls_back_to_heuristic():
    schema = bootstrap_schema(tablets(20), "tablets", MockLLM({}))
    assert schema["draft_source"] == "heuristic"


def test_no_llm_argument_is_heuristic():
    assert bootstrap_schema(tablets(20), "tablets")["draft_source"] == "heuristic"


def test_validate_schema_rejects_garbage():
    with pytest.raises(ValueError):
        validate_schema({"attributes": [{"code": "!!", "name_ru": "x"}]}, "tablets")
    with pytest.raises(ValueError):
        validate_schema("строка", "tablets")


def test_heuristic_schema_small_input():
    s = heuristic_schema([], "x")
    assert [a["code"] for a in s["attributes"]] == ["brand", "color"]


# ───────── очередь и CLI ─────────

def test_queue_ready_and_dedup(tmp_path):
    q = NewCategoryQueue(tmp_path / "q.jsonl")
    assert not q.ready("tablets", 3) and q.count("tablets") == 0
    cards = tablets(3)
    for c in cards:
        q.add("tablets", c)
    q.add("tablets", cards[0])                    # повтор того же product_id не считается
    q.add("other", Card(product_id="o1", title="x"))
    assert q.count("tablets") == 3 and q.ready("tablets", 3) and not q.ready("tablets", 4)
    assert q.counts() == {"other": 1, "tablets": 3}
    assert [c.product_id for c in q.cards("tablets")] == ["t0", "t1", "t2"]


def test_queue_default_threshold_is_20(tmp_path):
    q = NewCategoryQueue(tmp_path / "q.jsonl")
    for c in tablets(19):
        q.add("tablets", c)
    assert not q.ready("tablets")
    q.add("tablets", tablets(20)[19])
    assert q.ready("tablets")


def test_cli_builds_draft_without_llm(tmp_path, capsys):
    q = NewCategoryQueue(tmp_path / "q.jsonl")
    for c in tablets(20):
        q.add("tablets", c)
    code = main(["--queue", str(q.path), "--category", "tablets", "--config-dir", str(tmp_path / "cats")])
    assert code == 0
    assert (tmp_path / "cats" / "tablets.draft.yaml").exists()
    assert "heuristic" in capsys.readouterr().out


def test_cli_not_ready_yet(tmp_path, capsys):
    q = NewCategoryQueue(tmp_path / "q.jsonl")
    for c in tablets(3):
        q.add("tablets", c)
    code = main(["--queue", str(q.path), "--category", "tablets", "--config-dir", str(tmp_path / "cats")])
    assert code == 2 and not (tmp_path / "cats").exists()
    assert main(["--queue", str(q.path), "--list"]) == 0
    assert "tablets: 3/20" in capsys.readouterr().out
