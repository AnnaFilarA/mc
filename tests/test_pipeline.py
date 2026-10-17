"""Пакетный режим: загрузка карточек из разных форматов и запись predictions.jsonl / predictions.csv."""
import csv
import json

import pytest

from src.contracts import Card, ExtractResponse
from src.pipeline import CSV_COLUMNS, load_cards, main, write_predictions
from tests.conftest import FIXTURES

CARD = {"product_id": "p1", "title": "Смартфон Samsung Galaxy A55 8/256GB синий", "description": "Экран 6,6 дюйма",
        "seller_attributes": {"Цвет": "Синий"}, "category_hint": None}


def test_load_fixtures_dir():
    cards = load_cards(str(FIXTURES))
    assert len(cards) == 26 and all(isinstance(c, Card) for c in cards)


def test_load_jsonl_and_json(tmp_path):
    jl = tmp_path / "c.jsonl"
    jl.write_text(json.dumps(CARD, ensure_ascii=False) + "\n\n" + json.dumps({"card": {**CARD, "product_id": "p2"}}) + "\n",
                  encoding="utf-8")
    assert [c.product_id for c in load_cards(str(jl))] == ["p1", "p2"]
    js = tmp_path / "c.json"
    js.write_text(json.dumps([CARD]), encoding="utf-8")
    assert load_cards(str(js))[0].seller_attributes == {"Цвет": "Синий"}


def test_load_csv_with_json_seller_attributes(tmp_path):
    p = tmp_path / "c.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["product_id", "title", "description", "seller_attributes", "category_hint"])
        w.writeheader()
        w.writerow({"product_id": "p1", "title": CARD["title"], "description": "", "category_hint": "smartphones",
                    "seller_attributes": json.dumps({"Цвет": "Синий"}, ensure_ascii=False)})
        w.writerow({"product_id": "p2", "title": "x", "description": "", "seller_attributes": "не json", "category_hint": ""})
    cards = load_cards(str(p))
    assert cards[0].seller_attributes == {"Цвет": "Синий"} and cards[0].category_hint == "smartphones"
    assert cards[1].seller_attributes == {} and cards[1].category_hint is None


def test_unknown_format(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_cards(str(p))


def test_main_writes_both_files(tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["--input", str(FIXTURES), "--output", str(out)]) == 0
    lines = (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 26
    responses = [ExtractResponse.model_validate_json(l) for l in lines]
    assert "\\u04" not in (out / "predictions.jsonl").read_text(encoding="utf-8")   # ensure_ascii=False
    with (out / "predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert len(rows) == sum(len(r.attributes) for r in responses)
    proposed = [r for r in rows if r["status"] == "proposed"]
    assert proposed and all(r["evidence_text"] and r["evidence_source"] and r["evidence_end"] for r in proposed)
    missing = [r for r in rows if r["status"] == "missing"]
    assert missing and all(json.loads(r["question"])["text"] for r in missing)
    out_text = capsys.readouterr().out
    assert "Карточек: 26" in out_text and "p95" in out_text


@pytest.mark.parametrize("config", ["rules_ml", "full"])
def test_other_configs_run_without_real_models(tmp_path, config):
    assert main(["--input", str(FIXTURES), "--output", str(tmp_path), "--config", config]) == 0
    assert (tmp_path / "predictions.jsonl").exists()


def test_write_predictions_roundtrip(tmp_path):
    from src.cascade import extract
    r = extract(Card.model_validate(CARD))
    write_predictions([r], str(tmp_path / "nested" / "dir"))
    back = ExtractResponse.model_validate_json((tmp_path / "nested" / "dir" / "predictions.jsonl").read_text(encoding="utf-8"))
    assert back == r


def test_unknown_category_goes_to_queue(tmp_path):
    from src.schema_bootstrap import NewCategoryQueue
    q = tmp_path / "queue.jsonl"
    assert main(["--input", str(FIXTURES), "--output", str(tmp_path / "out"), "--new-category-queue", str(q)]) == 0
    assert NewCategoryQueue(q).count("unknown") == 1   # u01 — робот-пылесос
