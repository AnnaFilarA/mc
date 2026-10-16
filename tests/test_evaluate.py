"""Тесты src/evaluate.py (поток D). Предсказания синтетические — ядро пишется параллельно."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src import evaluate as ev
from src.contracts import AttributeResult, Evidence, ExtractResponse, Status

FIXTURES = Path(__file__).resolve().parent / "fixtures"
EVID = Evidence(text="x", start=0, end=1, source="title")


def attr(code, value, status="proposed", conf=0.95, was=None, stage="rules", evidence=True) -> AttributeResult:
    return AttributeResult(attribute=code, was=was, value=value, unit=None, status=Status(status),
                           confidence=conf, stage=stage, evidence=EVID if evidence and value is not None else None,
                           question_to_seller=None)


def resp(pid, cat, attrs, elapsed=10, llm_calls=0) -> ExtractResponse:
    return ExtractResponse(product_id=pid, category=cat, category_confidence=1.0, attributes=attrs,
                           card_status="ready", llm_calls=llm_calls, elapsed_ms=elapsed, text_hash="h")


def write_jsonl(path: Path, rows: list) -> str:
    path.write_text("\n".join(r.model_dump_json() if hasattr(r, "model_dump_json") else json.dumps(r, ensure_ascii=False)
                              for r in rows) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture()
def mini(tmp_path):
    """Мини-пример, посчитанный вручную.

    ram:   c1 TP | c2 FP+FN (8 вместо 16) | c3 FN (low_confidence) | c4 FP (эталон null) | c5 TP (unchanged)
           tp=2 fp=2 fn=2 -> P=0.5 R=0.5 F1=0.5; proposed: c1 ok, c2 нет, c4 нет -> 1/3
           coverage_before = 1/5 (c5), coverage_after = 4/5 (c1, c2, c4, c5)
    color: c1 TP («синий» == «Синий») | c2 FN (missing) -> tp=1 fp=0 fn=1 -> P=1 R=0.5 F1=2/3
    micro: tp=3 fp=2 fn=3 -> 6/11; macro: (0.5 + 2/3)/2 = 7/12
    """
    gold = [
        {"product_id": "c1", "category": "k", "attributes": {"ram": "8", "color": "Синий"}},
        {"product_id": "c2", "category": "k", "attributes": {"ram": "16", "color": "Красный"}},
        {"product_id": "c3", "category": "k", "attributes": {"ram": "4"}},
        {"product_id": "c4", "category": "k", "attributes": {"ram": None}},
        {"product_id": "c5", "category": "k", "attributes": {"ram": "32"}},
    ]
    preds = [
        resp("c1", "k", [attr("ram", "8"), attr("color", "синий")], elapsed=100, llm_calls=1),
        resp("c2", "k", [attr("ram", "8", conf=0.9), attr("color", None, "missing", 0.0)], elapsed=200),
        resp("c3", "k", [attr("ram", "4", "low_confidence", 0.6)], elapsed=300),
        resp("c4", "k", [attr("ram", "6.0", conf=0.85)], elapsed=400),
        resp("c5", "k", [attr("ram", "32", "unchanged", 0.99, was="32")], elapsed=500),
    ]
    return write_jsonl(tmp_path / "gold.jsonl", gold), write_jsonl(tmp_path / "pred.jsonl", preds)


def row(table, level, cat, attribute):
    r = table[(table.level == level) & (table.category == cat) & (table.attribute == attribute)]
    assert len(r) == 1
    return r.iloc[0]


def test_known_prf_hand_computed(mini, tmp_path):
    gold_path, pred_path = mini
    s = ev.evaluate(pred_path, gold_path, str(tmp_path / "out"))
    t = pd.read_csv(tmp_path / "out" / "per_attribute.csv")
    ram = row(t, "category_attribute", "k", "ram")
    assert (ram.n, ram.support, ram.tp, ram.fp, ram.fn) == (5, 4, 2, 2, 2)
    assert ram.precision == pytest.approx(0.5) and ram.recall == pytest.approx(0.5) and ram.f1 == pytest.approx(0.5)
    assert ram.precision_proposed == pytest.approx(1 / 3)
    assert ram.coverage_before == pytest.approx(0.2) and ram.coverage_after == pytest.approx(0.8)
    assert ram.share_low_confidence == pytest.approx(0.2) and ram.share_unchanged == pytest.approx(0.2)
    color = row(t, "category_attribute", "k", "color")
    assert (color.tp, color.fp, color.fn) == (1, 0, 1)
    assert color.f1 == pytest.approx(2 / 3) and color.share_missing == pytest.approx(0.5)
    assert row(t, "total_micro", "ALL", "ALL").f1 == pytest.approx(6 / 11)
    assert row(t, "total_macro", "ALL", "ALL").f1 == pytest.approx(7 / 12)
    assert s["micro_f1"] == pytest.approx(6 / 11) and s["macro_f1"] == pytest.approx(7 / 12)
    assert s["proposed_without_evidence"] == 0
    # без файла весов взвешенный F1 == macro и это помечено явно
    assert s["weighted_f1"] == pytest.approx(7 / 12) and "равные" in s["weights_source"]
    assert "равные" in row(t, "total_weighted", "ALL", "ALL").note


def test_weights_file_changes_weighted_f1(mini, tmp_path):
    gold_path, pred_path = mini
    w = tmp_path / "w.csv"
    w.write_text("category,attribute,weight\nk,ram,3\nk,color,1\n", encoding="utf-8")
    s = ev.evaluate(pred_path, gold_path, str(tmp_path / "out"), weights_path=str(w))
    assert s["weighted_f1"] == pytest.approx((3 * 0.5 + 2 / 3) / 4)
    assert s["macro_f1"] == pytest.approx(7 / 12) and "файл" in s["weights_source"]


def test_canon_comparison():
    assert ev.canon(" Синий ") == ev.canon("синий")
    assert ev.canon("Зелёный") == ev.canon("зеленый")
    assert ev.canon("6,60") == ev.canon("6.6") == ev.canon(6.6)
    assert ev.canon("8") != ev.canon("8 GB") and ev.canon("") is None and ev.canon(None) is None


def test_proposed_without_evidence_counted(tmp_path):
    gold = write_jsonl(tmp_path / "g.jsonl", [{"product_id": "a", "category": "k", "attributes": {"ram": "8"}}])
    bad = attr("ram", "8", evidence=False)
    pred = write_jsonl(tmp_path / "p.jsonl", [resp("a", "k", [bad])])
    assert ev.evaluate(pred, gold, str(tmp_path / "o"))["proposed_without_evidence"] == 1


def test_missing_prediction_counts_as_fn(tmp_path):
    gold = write_jsonl(tmp_path / "g.jsonl", [
        {"product_id": "a", "category": "k", "attributes": {"ram": "8"}},
        {"product_id": "b", "category": "k", "attributes": {"ram": "4"}}])
    pred = write_jsonl(tmp_path / "p.jsonl", [resp("a", "k", [attr("ram", "8")])])
    s = ev.evaluate(pred, gold, str(tmp_path / "o"))
    assert s["cards_without_prediction"] == 1
    assert s["micro_f1"] == pytest.approx(2 / 3)  # tp=1 fn=1


def test_gold_from_fixtures():
    g = ev.load_gold(FIXTURES)
    assert len(g.cards) == len(list(FIXTURES.glob("*.json")))
    known, c, raw = g.cards["s01"]["attrs"]["ram"]
    assert known and c == ev.canon("8") and raw == "8"
    # t01: refresh_rate — missing без value -> эталон null (известный)
    known, c, _ = g.cards["t01"]["attrs"]["refresh_rate"]
    assert known and c is None
    # conflict без value -> исключён из P/R
    unknown = [(p, a) for p, card in g.cards.items() for a, (k, _, _) in card["attrs"].items() if not k]
    expected_unknown = sum(
        1 for f in FIXTURES.glob("*.json") for spec in json.loads(f.read_text(encoding="utf-8"))["expected"]["attributes"].values()
        if "value" not in spec and "conflict" in ([spec["status"]] if isinstance(spec["status"], str) else spec["status"]))
    assert len(unknown) == expected_unknown > 0


def test_gold_csv_wide_and_long(tmp_path):
    wide = tmp_path / "w.csv"
    wide.write_text("product_id,category,gold_ram,color\na,k,8,\n", encoding="utf-8")
    g = ev.load_gold(wide)
    assert g.cards["a"]["attrs"]["ram"][1] == ev.canon("8") and g.cards["a"]["attrs"]["color"][1] is None
    long = tmp_path / "l.csv"
    long.write_text("product_id,category,attribute,value\na,k,ram,16\n", encoding="utf-8")
    assert ev.load_gold(long).cards["a"]["attrs"]["ram"][1] == ev.canon("16")


# ---------------------------------------------------------------- синтетика из фикстур
def synth_predictions(variant: int = 0) -> list[ExtractResponse]:
    """Предсказания из эталона фикстур с намеренными ошибками."""
    out = []
    k = 0
    for f in sorted(FIXTURES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        attrs = []
        for code, spec in d["expected"]["attributes"].items():
            k += 1
            if spec.get("value") is None:
                st = spec["status"] if isinstance(spec["status"], str) else spec["status"][0]
                attrs.append(attr(code, None, st, 0.3))
            elif k % 5 == 0:   # ошибка с высокой уверенностью
                attrs.append(attr(code, "WRONG", conf=0.9 - 0.02 * (k % 7)))
            elif k % 7 == 0:   # верное, но неуверенное
                attrs.append(attr(code, spec["value"], "low_confidence", 0.55))
            elif k % 11 == 0:  # ошибка с низкой уверенностью
                attrs.append(attr(code, "WRONG", conf=0.6))
            else:
                attrs.append(attr(code, spec["value"], conf=0.8 + 0.01 * (k % 15), stage="llm" if k % 13 == 0 else "rules"))
        out.append(resp(d["card"]["product_id"], d["expected"]["category"], attrs,
                        elapsed=50 + 10 * variant + k % 40, llm_calls=1 if variant == 2 else 0))
    return out


def test_all_files_created(tmp_path):
    pred = write_jsonl(tmp_path / "p.jsonl", synth_predictions())
    out = tmp_path / "metrics"
    s = ev.evaluate(pred, str(FIXTURES), str(out))
    for name in ("per_attribute.csv", "configs_comparison.csv", "threshold_cost.csv", "threshold_cost.png",
                 "summary.json"):
        assert (out / name).stat().st_size > 0, name
    assert (out / "threshold_cost.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["n_cards"] == 26
    assert 0 < s["micro_f1"] < 1 and s["category_f1_spread_pp"] is not None
    cmp = pd.read_csv(out / "configs_comparison.csv")
    assert list(cmp.config[:3]) == ["rules", "rules_ml", "full"]
    assert cmp.loc[cmp.config == "rules", "status"].iloc[0] == "ok"
    assert (cmp.loc[cmp.config != "rules", "status"] == ev.NOT_RUN).all()
    assert "Файлы" not in ev.format_summary(s) and "F1" in ev.format_summary(s)


def test_configs_comparison_three_configs(tmp_path):
    paths = {n: write_jsonl(tmp_path / f"{n}.jsonl", synth_predictions(i))
             for i, n in enumerate(("rules", "rules_ml", "full"))}
    out = tmp_path / "m"
    ev.evaluate(None, str(FIXTURES), str(out), compare=paths, llm_call_cost_rub=2.0)
    cmp = pd.read_csv(out / "configs_comparison.csv").set_index("config")
    assert (cmp.status == "ok").all()
    assert cmp.loc["full", "llm_calls"] == 26 and cmp.loc["rules", "llm_calls"] == 0
    assert cmp.loc["full", "llm_cost_per_1000_rub"] == pytest.approx(2.0 * 1000)  # 26 вызовов / 26 карточек
    assert cmp.loc["rules", "llm_cost_per_1000_rub"] == 0
    assert cmp.loc["rules", "elapsed_ms_p95"] >= cmp.loc["rules", "elapsed_ms_p50"]
    assert cmp.loc["rules", "cost_per_1000_rub"] == pytest.approx(
        cmp.loc["rules", "manual_cost_per_1000_rub"] + cmp.loc["rules", "error_cost_per_1000_rub"])
    json.loads((out / "summary.json").read_text(encoding="utf-8"))  # primary = full


def test_cli_compare_partial(tmp_path, capsys):
    p = write_jsonl(tmp_path / "rules.jsonl", synth_predictions())
    out = tmp_path / "m"
    rc = ev.main(["--gold", str(FIXTURES), "--out", str(out), "--compare", f"rules={p}", f"full={tmp_path/'nope.jsonl'}"])
    assert rc == 0
    cmp = pd.read_csv(out / "configs_comparison.csv").set_index("config")
    assert cmp.loc["rules", "status"] == "ok"
    assert cmp.loc["rules_ml", "status"] == ev.NOT_RUN and cmp.loc["full", "status"] == ev.NOT_RUN
    assert "не запускались" in capsys.readouterr().out


def test_threshold_cost_monotonic_and_formula(tmp_path):
    pred = write_jsonl(tmp_path / "p.jsonl", synth_predictions())
    out = tmp_path / "m"
    ev.evaluate(pred, str(FIXTURES), str(out))
    curve = pd.read_csv(out / "threshold_cost.csv")
    assert list(curve[curve.attribute == "ALL"].threshold) == pytest.approx([i * 0.05 for i in range(21)])
    for attribute, sub in curve.groupby("attribute"):
        sub = sub.sort_values("threshold")
        assert sub.human_share.is_monotonic_increasing, attribute
        assert sub.error_share.is_monotonic_decreasing, attribute
    cfg = ev.load_cost_config()
    assert cfg["manual_cost_per_field_rub"] == pytest.approx(765 / 60 * 0.5)
    for _, r in curve.iterrows():
        for s in ev.SCENARIOS:
            assert r[f"cost_{s}"] == pytest.approx(
                r.human_share * cfg["manual_cost_per_field_rub"] + r.error_share * cfg["error_cost_rub"][s])
    # при пороге 1.0 всё уходит человеку и ошибок в авто нет
    top = curve[(curve.threshold == 1.0) & (curve.attribute == "ALL")].iloc[0]
    assert top.human_share == pytest.approx(1.0) and top.error_share == 0
    # ровно один оптимум на (атрибут, сценарий)
    assert (curve.groupby("attribute").is_best_mid.sum() == 1).all()


def test_best_thresholds_tradeoff():
    """Дёшево проверять и дорогая ошибка -> порог выше; нет ошибок -> порог не выше самой низкой уверенности."""
    rows = []
    for i in range(20):  # поле «a»: верные при conf 0.9, неверные при conf 0.6
        rows.append({"product_id": f"p{i}", "category": "k", "attribute": "a", "known": True, "gold_present": True,
                     "status": "proposed", "confidence": 0.9 if i % 2 else 0.6, "stage": "rules", "has_value": True,
                     "had_was": False, "auto": True, "match": bool(i % 2), "has_evidence": True})
        rows.append({"product_id": f"p{i}", "category": "k", "attribute": "b", "known": True, "gold_present": True,
                     "status": "proposed", "confidence": 0.7, "stage": "rules", "has_value": True,
                     "had_was": False, "auto": True, "match": True, "has_evidence": True})
    curve = ev.threshold_cost_table(pd.DataFrame(rows))
    best = ev.best_thresholds(curve, "mid")
    assert set(best) == {"a", "b"} and best["b"] == 0.7  # без ошибок авто выгодно; при равенстве берём строгий порог
    # a: человек 6.375 руб/поле против 20 руб * 0.5 ошибок -> выгоднее отсечь неуверенные (порог в (0.6, 0.9])
    assert 0.6 < best["a"] <= 0.9
    assert ev.best_thresholds(curve, "low")["a"] <= best["a"] <= ev.best_thresholds(curve, "high")["a"]
    assert "ALL" in ev.best_thresholds(curve, "mid", include_total=True)


def test_category_spread(tmp_path):
    gold = write_jsonl(tmp_path / "g.jsonl", [
        {"product_id": "a", "category": "x", "attributes": {"ram": "8"}},
        {"product_id": "b", "category": "y", "attributes": {"ram": "8"}},
        {"product_id": "c", "category": "unknown", "attributes": {"ram": "8"}}])
    pred = write_jsonl(tmp_path / "p.jsonl", [resp("a", "x", [attr("ram", "8")]),
                                              resp("b", "y", [attr("ram", "9")]),
                                              resp("c", "unknown", [attr("ram", "7")])])
    s = ev.evaluate(pred, gold, str(tmp_path / "o"))
    assert set(s["category_f1"]) == {"x", "y"}            # unknown не участвует
    assert s["category_f1_spread_pp"] == pytest.approx(100.0)
    assert s["targets"]["category_spread<=10pp"] is False
    assert not math.isnan(s["category_f1_spread_pp"])
