"""Метрики (SPEC 2.3, INTENT 5.2 / 6). Владелец — поток D.

Запуск::

    python -m src.evaluate --pred outputs/predictions.jsonl --gold <jsonl|csv|tests/fixtures> \\
        --out outputs/metrics/ [--weights weights.csv] \\
        [--compare rules=a.jsonl rules_ml=b.jsonl full=c.jsonl] [--llm-call-cost-rub 0]

Эталон (gold) — один из форматов:
  * jsonl: ``{"product_id", "category", "attributes": {code: value|null}}``; ``null`` = значения нет в тексте;
  * каталог фикстур (``tests/fixtures/*.json``): берётся ``expected.attributes[code]``:
    есть ``value`` -> это эталон; нет ``value`` и статус ``missing`` -> эталон ``null``;
    нет ``value`` и статус ``conflict`` (или иной) -> эталон неизвестен, атрибут ИСКЛЮЧАЕТСЯ из P/R;
  * csv: «широкий» (``product_id,category,<код атрибута>,...``, префикс ``gold_`` у колонок допустим,
    пустая ячейка = ``null``) или «длинный» (``product_id,category,attribute,value``).

Определения (одинаковые для всех таблиц; единица оценки — пара «карточка, атрибут из эталона»):

* **авто-значение** — статус поля ``unchanged`` или ``proposed`` и непустой ``value``
  (то, что применяется без человека, INTENT 5.1);
* **совпадение** — каноны равны: ``strip``, ``lower``, ``ё -> е``, схлопнутые пробелы, числа сравниваются как float
  (``"6,60"`` == ``"6.6"``); единица измерения не сравнивается (канон числа — без единицы, DECISIONS);
* **TP** — авто-значение совпало с эталоном; **FP** — авто-значение не совпало или эталон ``null``;
  **FN** — эталон есть, а авто-значения нет или оно неверное (неверное авто-значение — и FP, и FN);
* ``precision = TP/(TP+FP)``, ``recall = TP/(TP+FN)``, ``F1 = 2TP/(2TP+FP+FN)``; если знаменатель 0 — пусто (NaN);
* ``precision_proposed`` — точность только по полям со статусом ``proposed`` (цель >= 0,90);
* ``coverage_before`` — доля полей, где у продавца было значение (``was``); ``coverage_after`` — доля полей
  с авто-значением; оба — по всем полям группы, включая исключённые из P/R;
* ``llm_share`` — доля полей со ``stage == "llm"`` (значение дал Этап 3);
* ``proposed_without_evidence`` — число ``proposed`` без ``evidence`` (должно быть 0);
* micro F1 — по суммарным TP/FP/FN; macro F1 — среднее F1 по парам (категория, атрибут);
  взвешенный F1 — то же с весами трафика (без файла весов веса равны, это помечено явно);
* разброс по категориям — max-min micro F1 по категориям (без ``unknown``), цель <= 10 п.п.

Стоимость порога (INTENT 5.2), доли считаются от всех оценённых полей::

    стоимость(t) = доля_человеку(t) * цена_ручной_проверки + доля_ошибок_в_авто(t) * цена_ошибки

``цена_ручной_проверки = rate/60 * minutes_per_field`` (config/thresholds.yaml -> cost);
цена ошибки — сценарии low / mid / high оттуда же. Поле уходит человеку, если у него нет кандидата
(``conflict``, ``missing``, ``low_confidence`` без значения) или ``confidence < t``. Кандидат со статусом
``low_confidence`` и непустым ``value`` считается кандидатом: так можно оценить и пороги ниже боевого.
Ошибка в авто — кандидат с ``confidence >= t``, не совпавший с эталоном.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS_YAML = ROOT / "config" / "thresholds.yaml"

AUTO_STATUSES = ("unchanged", "proposed")
CANDIDATE_STATUSES = ("unchanged", "proposed", "low_confidence")
ALL_STATUSES = ("unchanged", "proposed", "low_confidence", "conflict", "missing")
SCENARIOS = ("low", "mid", "high")
CONFIG_ORDER = ("rules", "rules_ml", "full")
NOT_RUN = "не запускалась"
THRESHOLD_GRID = [round(i * 0.05, 2) for i in range(21)]
TARGET_PRECISION_PROPOSED = 0.90
TARGET_F1_KEY = 0.80
TARGET_SPREAD_PP = 10.0
TARGET_LLM_SHARE = 0.20

_DEFAULT_COST = {
    "manager_rate_rub_per_hour": 765.0,
    "minutes_per_field": 0.5,
    "error_cost_rub": {"low": 5.0, "mid": 20.0, "high": 60.0},
}


# --------------------------------------------------------------------------- канон и загрузка
def canon(value: Any) -> tuple | None:
    """Канон значения для сравнения; ``None`` для пустого значения."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    s = str(value).strip().lower().replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    if not s:
        return None
    try:
        return ("n", round(float(s.replace(",", ".")), 9))
    except ValueError:
        return ("s", s)


class Gold:
    """Эталон: product_id -> {"category", "attrs": {code: (known: bool, canon|None, raw|None)}}."""

    def __init__(self) -> None:
        self.cards: dict[str, dict] = {}

    def add(self, pid: str, category: str, code: str, known: bool, value: Any = None) -> None:
        card = self.cards.setdefault(pid, {"category": category or "unknown", "attrs": {}})
        if category and card["category"] == "unknown":
            card["category"] = category
        card["attrs"][code] = (known, canon(value) if known else None, None if value is None else str(value))


def _statuses(spec: dict) -> list[str]:
    st = spec.get("status")
    if st is None:
        return []
    return [str(x) for x in (st if isinstance(st, list) else [st])]


def _gold_from_fixture_dicts(items: list[dict]) -> Gold:
    gold = Gold()
    for d in items:
        pid, exp = d["card"]["product_id"], d["expected"]
        cat = exp.get("category", "unknown")
        gold.cards.setdefault(pid, {"category": cat, "attrs": {}})
        for code, spec in exp.get("attributes", {}).items():
            if spec.get("value") is not None:
                gold.add(pid, cat, code, True, spec["value"])
                continue
            st = _statuses(spec)
            # нет value: missing -> эталон null; conflict (или иное) -> эталон неизвестен, исключаем из P/R
            gold.add(pid, cat, code, "conflict" not in st and "missing" in st)
    return gold


def _load_gold_fixtures(path: Path) -> Gold:
    return _gold_from_fixture_dicts(
        [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))])


def _load_gold_records(records: list[dict]) -> Gold:
    gold = Gold()
    for r in records:
        pid, cat = str(r["product_id"]), r.get("category") or "unknown"
        gold.cards.setdefault(pid, {"category": cat, "attrs": {}})
        for code, v in (r.get("attributes") or {}).items():
            gold.add(pid, cat, code, True, v)
    return gold


def _load_gold_csv(path: Path) -> Gold:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    gold = Gold()
    cols = list(df.columns)
    if "attribute" in cols and ("value" in cols or "gold" in cols):
        vcol = "value" if "value" in cols else "gold"
        for _, r in df.iterrows():
            gold.add(str(r["product_id"]), r.get("category", "") or "unknown", r["attribute"], True,
                     r[vcol].strip() or None)
        return gold
    attr_cols = [c for c in cols if c not in ("product_id", "category")]
    for _, r in df.iterrows():
        pid, cat = str(r["product_id"]), r.get("category", "") or "unknown"
        gold.cards.setdefault(pid, {"category": cat, "attrs": {}})
        for c in attr_cols:
            code = c[5:] if c.startswith("gold_") else c
            gold.add(pid, cat, code, True, r[c].strip() or None)
    return gold


def load_gold(path: str | Path) -> Gold:
    """Загрузка эталона: каталог фикстур, .jsonl, .json (список записей) или .csv."""
    p = Path(path)
    if p.is_dir():
        return _load_gold_fixtures(p)
    suf = p.suffix.lower()
    if suf == ".csv":
        return _load_gold_csv(p)
    text = p.read_text(encoding="utf-8")
    if suf == ".json":
        data = json.loads(text)
        if isinstance(data, dict) and "expected" in data:  # одна фикстура
            return _gold_from_fixture_dicts([data])
        return _load_gold_records(data)
    return _load_gold_records([json.loads(line) for line in text.splitlines() if line.strip()])


def load_predictions(path: str | Path) -> dict[str, dict]:
    """predictions.jsonl (ExtractResponse построчно) -> {product_id: dict}. Дубликаты: побеждает последний."""
    out: dict[str, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[str(d["product_id"])] = d
    return out


def load_weights(path: str | Path | None) -> dict | None:
    """csv ``category,attribute,weight`` (или ``attribute,weight``) -> {(category|None, attribute): weight}."""
    if not path:
        return None
    df = pd.read_csv(path, dtype={"category": str, "attribute": str}, encoding="utf-8-sig")
    has_cat = "category" in df.columns
    return {((str(r["category"]) if has_cat else None), str(r["attribute"])): float(r["weight"])
            for _, r in df.iterrows()}


def load_cost_config(path: str | Path | None = None) -> dict:
    """Блок ``cost`` из config/thresholds.yaml + производные: цена ручной проверки поля и сценарии цены ошибки."""
    import yaml

    cfg = json.loads(json.dumps(_DEFAULT_COST))
    p = Path(path) if path else DEFAULT_THRESHOLDS_YAML
    if p.exists():
        raw = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("cost") or {}
        for k in ("manager_rate_rub_per_hour", "minutes_per_field"):
            if k in raw:
                cfg[k] = float(raw[k])
        cfg["error_cost_rub"].update({k: float(v) for k, v in (raw.get("error_cost_rub") or {}).items()})
    cfg["manual_cost_per_field_rub"] = cfg["manager_rate_rub_per_hour"] / 60.0 * cfg["minutes_per_field"]
    return cfg


# --------------------------------------------------------------------------- таблица полей
def build_fields(preds: dict[str, dict], gold: Gold) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Строка на пару (карточка из эталона, атрибут из эталона) + таблица карточек + служебные счётчики."""
    rows: list[dict] = []
    cards: list[dict] = []
    not_in_gold = 0
    no_pred = 0
    for pid, g in gold.cards.items():
        resp = preds.get(pid)
        if resp is None:
            no_pred += 1
        else:
            cards.append({"product_id": pid, "category": g["category"],
                          "elapsed_ms": float(resp.get("elapsed_ms") or 0),
                          "llm_calls": int(resp.get("llm_calls") or 0)})
        by_attr = {a.get("attribute"): a for a in (resp or {}).get("attributes", [])}
        not_in_gold += sum(1 for a in by_attr if a not in g["attrs"])
        for code, (known, gcanon, _raw) in g["attrs"].items():
            a = by_attr.get(code)
            value = None if a is None else a.get("value")
            status = "missing" if a is None else str(a.get("status") or "missing")
            status = status.split(".")[-1]  # на случай "Status.proposed"
            vcanon = canon(value)
            auto = status in AUTO_STATUSES and vcanon is not None
            rows.append({
                "product_id": pid, "category": g["category"], "attribute": code,
                "known": bool(known), "gold_present": gcanon is not None,
                "status": status, "confidence": float((a or {}).get("confidence") or 0.0),
                "stage": str((a or {}).get("stage") or "none"),
                "has_value": vcanon is not None, "had_was": canon((a or {}).get("was")) is not None,
                "auto": auto, "match": bool(vcanon is not None and gcanon is not None and vcanon == gcanon),
                "has_evidence": bool((a or {}).get("evidence") and (a["evidence"].get("text"))),
            })
    fields = pd.DataFrame(rows)
    if fields.empty:
        fields = pd.DataFrame({c: pd.Series(dtype=t) for c, t in [
            ("product_id", "object"), ("category", "object"), ("attribute", "object"), ("known", bool),
            ("gold_present", bool), ("status", "object"), ("confidence", float), ("stage", "object"),
            ("has_value", bool), ("had_was", bool), ("auto", bool), ("match", bool), ("has_evidence", bool)]})
    cards_df = pd.DataFrame(cards, columns=["product_id", "category", "elapsed_ms", "llm_calls"])
    meta = {"n_gold_cards": len(gold.cards), "cards_without_prediction": no_pred,
            "pred_attributes_not_in_gold": not_in_gold}
    return fields, cards_df, meta


# --------------------------------------------------------------------------- метрики
def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def counts_prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    return _ratio(tp, tp + fp), _ratio(tp, tp + fn), _ratio(2 * tp, 2 * tp + fp + fn)


def row_metrics(sub: pd.DataFrame) -> dict:
    """Все метрики таблицы per_attribute для одной группы полей."""
    n = len(sub)
    ev = sub[sub["known"]]
    tp = int((ev["auto"] & ev["match"]).sum())
    fp = int((ev["auto"] & ~ev["match"]).sum())
    fn = int((ev["gold_present"] & ~(ev["auto"] & ev["match"])).sum())
    p, r, f1 = counts_prf(tp, fp, fn)
    pm = ev[(ev["status"] == "proposed") & ev["has_value"]]
    out = {
        "n": n, "n_excluded": int((~sub["known"]).sum()), "support": int(ev["gold_present"].sum()),
        "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
        "n_proposed": len(pm), "precision_proposed": float(pm["match"].mean()) if len(pm) else float("nan"),
        "coverage_before": float(sub["had_was"].mean()) if n else float("nan"),
        "coverage_after": float(sub["auto"].mean()) if n else float("nan"),
    }
    for s in ALL_STATUSES:
        out[f"share_{s}"] = float((sub["status"] == s).mean()) if n else float("nan")
    out["llm_share"] = float((sub["stage"] == "llm").mean()) if n else float("nan")
    out["proposed_without_evidence"] = int(((sub["status"] == "proposed") & ~sub["has_evidence"]).sum())
    return out


def _weight(weights: dict | None, cat: str, attr: str) -> float:
    if weights is None:
        return 1.0
    if (cat, attr) in weights:
        return weights[(cat, attr)]
    return weights.get((None, attr), 0.0)


def weighted_macro(rows: pd.DataFrame, weights: dict | None) -> float:
    """Взвешенное среднее F1 по парам (категория, атрибут); пары без F1 или с нулевым весом не участвуют."""
    num = den = 0.0
    for _, r in rows.iterrows():
        if pd.isna(r["f1"]):
            continue
        w = _weight(weights, r["category"], r["attribute"])
        num += w * r["f1"]
        den += w
    return num / den if den > 0 else float("nan")


def macro_mean(rows: pd.DataFrame, col: str = "f1") -> float:
    v = rows[col].dropna()
    return float(v.mean()) if len(v) else float("nan")


def per_attribute_table(fields: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """per_attribute.csv: уровни category_attribute / attribute / category / total_micro / total_macro / total_weighted."""
    weights_note = "веса равные (файл весов не задан)" if weights is None else "веса трафика из файла"
    rows: list[dict] = []
    if not fields.empty:
        for (cat, attr), sub in fields.groupby(["category", "attribute"], sort=True):
            rows.append({"level": "category_attribute", "category": cat, "attribute": attr,
                         "weight": _weight(weights, cat, attr), **row_metrics(sub)})
        for attr, sub in fields.groupby("attribute", sort=True):
            rows.append({"level": "attribute", "category": "ALL", "attribute": attr, **row_metrics(sub)})
        for cat, sub in fields.groupby("category", sort=True):
            rows.append({"level": "category", "category": cat, "attribute": "ALL", **row_metrics(sub)})
    table = pd.DataFrame(rows)
    ca = table[table["level"] == "category_attribute"] if rows else pd.DataFrame(columns=["f1"])
    total = row_metrics(fields)
    micro = {"level": "total_micro", "category": "ALL", "attribute": "ALL", **total,
             "note": "micro: суммарные TP/FP/FN"}
    pm, rm = macro_mean(ca, "precision"), macro_mean(ca, "recall")
    macro = {"level": "total_macro", "category": "ALL", "attribute": "ALL", "n": total["n"],
             "precision": pm, "recall": rm, "f1": macro_mean(ca),
             "note": "macro: среднее по парам (категория, атрибут)"}
    wf = {"level": "total_weighted", "category": "ALL", "attribute": "ALL", "n": total["n"],
          "f1": weighted_macro(ca, weights) if rows else float("nan"),
          "weight": float(ca["weight"].sum()) if rows else float("nan"), "note": weights_note}
    table = pd.concat([table, pd.DataFrame([micro, macro, wf])], ignore_index=True)
    lead = ["level", "category", "attribute", "n", "n_excluded", "support", "tp", "fp", "fn", "precision",
            "recall", "f1", "n_proposed", "precision_proposed", "coverage_before", "coverage_after",
            *[f"share_{s}" for s in ALL_STATUSES], "llm_share", "proposed_without_evidence", "weight", "note"]
    for c in lead:
        if c not in table.columns:
            table[c] = np.nan
    table = table[lead]
    for c in ("n", "n_excluded", "support", "tp", "fp", "fn", "n_proposed", "proposed_without_evidence"):
        table[c] = table[c].astype("Int64")
    return table


def category_spread(table: pd.DataFrame, min_fields: int = 1) -> dict:
    """Разброс micro F1 по категориям (без ``unknown``): max-min в процентных пунктах."""
    cats = table[(table["level"] == "category") & (table["category"] != "unknown")
                 & (table["n"] >= min_fields) & table["f1"].notna()]
    f1 = {r["category"]: float(r["f1"]) for _, r in cats.iterrows()}
    if len(f1) < 2:
        return {"category_f1": f1, "spread_pp": float("nan"), "ok": None}
    spread = (max(f1.values()) - min(f1.values())) * 100
    return {"category_f1": f1, "spread_pp": spread, "ok": bool(spread <= TARGET_SPREAD_PP)}


# --------------------------------------------------------------------------- порог -> стоимость
def threshold_cost_table(fields: pd.DataFrame, cost_cfg: dict | None = None,
                         grid: list[float] | None = None) -> pd.DataFrame:
    """Стоимость порога по каждому атрибуту (по всем категориям вместе) и итог (``attribute == "ALL"``).

    Колонки: attribute, threshold, n, human_share, error_share, cost_low/mid/high, is_best_low/mid/high.
    """
    cost_cfg = cost_cfg or load_cost_config()
    grid = grid or THRESHOLD_GRID
    manual = cost_cfg["manual_cost_per_field_rub"]
    ev = fields[fields["known"]] if not fields.empty else fields
    groups: list[tuple[str, pd.DataFrame]] = [(a, s) for a, s in ev.groupby("attribute", sort=True)] if len(ev) else []
    groups.append(("ALL", ev))
    rows: list[dict] = []
    for attr, sub in groups:
        n = len(sub)
        cand = (sub["status"].isin(CANDIDATE_STATUSES) & sub["has_value"]).to_numpy()
        conf = sub["confidence"].to_numpy(dtype=float)
        ok = (sub["match"] & sub["gold_present"]).to_numpy()
        for t in grid:
            auto = cand & (conf + 1e-9 >= t)
            human = _ratio(n - auto.sum(), n)
            err = _ratio((auto & ~ok).sum(), n)
            row = {"attribute": attr, "threshold": t, "n": n, "human_share": human, "error_share": err}
            for s in SCENARIOS:
                row[f"cost_{s}"] = (human * manual + err * cost_cfg["error_cost_rub"][s]) if n else float("nan")
            rows.append(row)
    table = pd.DataFrame(rows)
    for s in SCENARIOS:
        table[f"is_best_{s}"] = False
        for attr, idx in table.groupby("attribute").groups.items():
            sub = table.loc[idx, f"cost_{s}"]
            if sub.notna().any():
                best = table.loc[idx][sub <= sub.min() + 1e-12]["threshold"].max()  # при равенстве — более строгий
                table.loc[idx[table.loc[idx, "threshold"] == best], f"is_best_{s}"] = True
    return table


def best_thresholds(curve: pd.DataFrame, scenario: str = "mid", include_total: bool = False) -> dict[str, float]:
    """Оптимальный порог по каждому атрибуту: минимум стоимости сценария (по умолчанию ``mid``).

    ``curve`` — результат ``threshold_cost_table`` (или прочитанный ``threshold_cost.csv``).
    При равной стоимости берётся более строгий (больший) порог. В ``thresholds.yaml`` функция ничего не пишет.
    """
    col = f"cost_{scenario}"
    out: dict[str, float] = {}
    for attr, sub in curve.groupby("attribute", sort=True):
        if attr == "ALL" and not include_total:
            continue
        sub = sub[sub[col].notna()]
        if sub.empty:
            continue
        out[str(attr)] = float(sub[sub[col] <= sub[col].min() + 1e-12]["threshold"].max())
    return out


def calibrate_thresholds(pred_path: str, gold_path: str, scenario: str = "mid",
                         thresholds_yaml: str | None = None) -> dict[str, float]:
    """Удобная обёртка: предсказания + эталон -> {атрибут: порог}. Использовать только на validation."""
    fields, _, _ = build_fields(load_predictions(pred_path), load_gold(gold_path))
    return best_thresholds(threshold_cost_table(fields, load_cost_config(thresholds_yaml)), scenario)


def plot_threshold_cost(curve: pd.DataFrame, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tot = curve[curve["attribute"] == "ALL"].sort_values("threshold")
    colors = {"low": "#2a78d6", "mid": "#eb6834", "high": "#1baf7a"}
    styles = {"low": "-", "mid": "--", "high": ":"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for s in SCENARIOS:
        ax1.plot(tot["threshold"], tot[f"cost_{s}"], styles[s], color=colors[s], lw=2, label=f"цена ошибки: {s}")
        b = tot[tot[f"is_best_{s}"]]
        if len(b):
            ax1.plot(b["threshold"], b[f"cost_{s}"], "o", ms=9, color=colors[s], mec="white", mew=2)
            ax1.annotate(f"{b['threshold'].iloc[0]:.2f}", (b["threshold"].iloc[0], b[f"cost_{s}"].iloc[0]),
                         textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax1.set_xlabel("порог уверенности")
    ax1.set_ylabel("стоимость на поле, руб.")
    ax1.set_title("Стоимость порога (точка = оптимум)")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    ax2.plot(tot["threshold"], tot["human_share"], "-", color="#2a78d6", lw=2, label="доля полей человеку")
    ax2.plot(tot["threshold"], tot["error_share"], "--", color="#eb6834", lw=2, label="доля ошибок в авто")
    ax2.set_xlabel("порог уверенности")
    ax2.set_ylabel("доля от всех полей")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Человек против ошибок")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- сравнение конфигураций
def config_row(name: str, preds: dict | None, gold: Gold, cost_cfg: dict, llm_call_cost_rub: float = 0.0,
               weights: dict | None = None, pred_path: str | None = None) -> dict:
    cols = ["config", "status", "pred_path", "n_cards", "n_fields", "macro_f1", "micro_f1", "weighted_f1",
            "precision_proposed", "coverage_before", "coverage_after", "elapsed_ms_p50", "elapsed_ms_p95",
            "llm_field_share", "llm_calls", "human_field_share", "auto_error_share", "llm_cost_per_1000_rub",
            "manual_cost_per_1000_rub", "error_cost_per_1000_rub", "cost_per_1000_rub"]
    row = {c: np.nan for c in cols}
    row.update({"config": name, "pred_path": pred_path or ""})
    if preds is None:
        row["status"] = NOT_RUN
        return row
    fields, cards, _ = build_fields(preds, gold)
    table = per_attribute_table(fields, weights)
    tot = table[table["level"] == "total_micro"].iloc[0]
    ca = table[table["level"] == "category_attribute"]
    n_cards = max(len(cards), 1)
    ev = fields[fields["known"]]
    n_ev = max(len(ev), 1)
    human = float((~ev["auto"]).sum()) / n_ev
    err = float((ev["auto"] & ~ev["match"]).sum()) / n_ev
    fields_per_card = len(ev) / n_cards
    llm_calls = int(cards["llm_calls"].sum())
    el = cards["elapsed_ms"].to_numpy()
    row.update({
        "status": "ok", "n_cards": len(cards), "n_fields": int(tot["n"]),
        "macro_f1": macro_mean(ca), "micro_f1": tot["f1"], "weighted_f1": weighted_macro(ca, weights),
        "precision_proposed": tot["precision_proposed"], "coverage_before": tot["coverage_before"],
        "coverage_after": tot["coverage_after"],
        "elapsed_ms_p50": float(np.percentile(el, 50)) if len(el) else np.nan,
        "elapsed_ms_p95": float(np.percentile(el, 95)) if len(el) else np.nan,
        "llm_field_share": tot["llm_share"], "llm_calls": llm_calls,
        "human_field_share": human, "auto_error_share": err,
        "llm_cost_per_1000_rub": llm_calls / n_cards * 1000 * llm_call_cost_rub,
        "manual_cost_per_1000_rub": human * fields_per_card * 1000 * cost_cfg["manual_cost_per_field_rub"],
        "error_cost_per_1000_rub": err * fields_per_card * 1000 * cost_cfg["error_cost_rub"]["mid"],
    })
    row["cost_per_1000_rub"] = (row["llm_cost_per_1000_rub"] + row["manual_cost_per_1000_rub"]
                                + row["error_cost_per_1000_rub"])
    return row


def configs_comparison_table(compare: dict[str, str | None], gold: Gold, cost_cfg: dict,
                             llm_call_cost_rub: float = 0.0, weights: dict | None = None) -> pd.DataFrame:
    names = list(CONFIG_ORDER) + [k for k in compare if k not in CONFIG_ORDER]
    rows = []
    for name in names:
        path = compare.get(name)
        if path and Path(path).exists():
            rows.append(config_row(name, load_predictions(path), gold, cost_cfg, llm_call_cost_rub, weights, path))
        else:
            rows.append(config_row(name, None, gold, cost_cfg, pred_path=path))
    table = pd.DataFrame(rows)
    for c in ("n_cards", "n_fields", "llm_calls"):
        table[c] = table[c].astype("Int64")
    return table


# --------------------------------------------------------------------------- главная функция
def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def evaluate(predictions_path: str | None, gold_path: str, out_dir: str = "outputs/metrics", *,
             weights_path: str | None = None, compare: dict[str, str] | None = None,
             llm_call_cost_rub: float = 0.0, thresholds_path: str | None = None,
             config_name: str = "rules") -> dict:
    """Считает все метрики, пишет файлы в ``out_dir`` и возвращает summary (то же, что в summary.json).

    ``predictions_path`` — основная конфигурация для per_attribute / threshold_cost; если не задан,
    берётся лучшая доступная из ``compare`` (full > rules_ml > rules).
    """
    compare = dict(compare or {})
    if predictions_path is None:
        for name in reversed(CONFIG_ORDER):
            if compare.get(name) and Path(compare[name]).exists():
                predictions_path, config_name = compare[name], name
                break
        if predictions_path is None:
            raise ValueError("нужен --pred или хотя бы одна существующая конфигурация в --compare")
    if not compare:
        compare = {config_name: predictions_path}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gold = load_gold(gold_path)
    weights = load_weights(weights_path)
    cost_cfg = load_cost_config(thresholds_path)
    preds = load_predictions(predictions_path)
    fields, cards, meta = build_fields(preds, gold)

    table = per_attribute_table(fields, weights)
    table.to_csv(out / "per_attribute.csv", index=False, encoding="utf-8")
    comparison = configs_comparison_table(compare, gold, cost_cfg, llm_call_cost_rub, weights)
    comparison.to_csv(out / "configs_comparison.csv", index=False, encoding="utf-8")
    curve = threshold_cost_table(fields, cost_cfg)
    curve.to_csv(out / "threshold_cost.csv", index=False, encoding="utf-8")
    plot_threshold_cost(curve, out / "threshold_cost.png")

    tm = table[table["level"] == "total_micro"].iloc[0]
    tmac = table[table["level"] == "total_macro"].iloc[0]
    twt = table[table["level"] == "total_weighted"].iloc[0]
    ca = table[table["level"] == "category_attribute"]
    spread = category_spread(table)
    tot_curve = curve[curve["attribute"] == "ALL"]
    optimum = {s: (float(tot_curve[tot_curve[f"is_best_{s}"]]["threshold"].max()) if len(tot_curve) else None)
               for s in SCENARIOS}
    below = [f"{r['category']}.{r['attribute']}" for _, r in ca.iterrows()
             if pd.notna(r["f1"]) and r["f1"] < TARGET_F1_KEY]
    summary = {
        "config": config_name, "predictions": str(predictions_path), "gold": str(gold_path),
        "n_cards": len(gold.cards), **meta,
        "n_fields": int(tm["n"]), "n_fields_excluded_from_pr": int(tm["n_excluded"]),
        "micro_precision": tm["precision"], "micro_recall": tm["recall"], "micro_f1": tm["f1"],
        "macro_f1": tmac["f1"], "weighted_f1": twt["f1"],
        "weights_source": "файл весов" if weights is not None else "равные веса (файл весов не задан)",
        "precision_proposed": tm["precision_proposed"], "n_proposed": int(tm["n_proposed"]),
        "proposed_without_evidence": int(tm["proposed_without_evidence"]),
        "coverage_before": tm["coverage_before"], "coverage_after": tm["coverage_after"],
        "llm_field_share": tm["llm_share"],
        "status_shares": {s: tm[f"share_{s}"] for s in ALL_STATUSES},
        "category_f1": spread["category_f1"], "category_f1_spread_pp": spread["spread_pp"],
        "attributes_f1_below_0.8": below,
        "threshold_optimum_total": optimum,
        "best_thresholds_mid": best_thresholds(curve, "mid"),
        "cost_per_field_rub": {"manual": cost_cfg["manual_cost_per_field_rub"], **cost_cfg["error_cost_rub"]},
        "targets": {
            "precision_proposed>=0.90": (None if pd.isna(tm["precision_proposed"])
                                         else bool(tm["precision_proposed"] >= TARGET_PRECISION_PROPOSED)),
            "proposed_without_evidence==0": bool(tm["proposed_without_evidence"] == 0),
            "category_spread<=10pp": spread["ok"],
            "llm_share<=0.20": (None if pd.isna(tm["llm_share"]) else bool(tm["llm_share"] <= TARGET_LLM_SHARE)),
        },
        "configs_run": [r["config"] for _, r in comparison.iterrows() if r["status"] == "ok"],
        "configs_not_run": [r["config"] for _, r in comparison.iterrows() if r["status"] != "ok"],
    }
    summary = _clean(summary)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _pct(x: Any) -> str:
    return "н/д" if x is None else f"{x * 100:.1f}%"


def format_summary(s: dict) -> str:
    t = s["targets"]
    ok = lambda v: "н/д" if v is None else ("да" if v else "НЕТ")  # noqa: E731
    sp = s["category_f1_spread_pp"]
    lines = [
        f"AttrGuard evaluate · конфигурация: {s['config']} · карточек: {s['n_cards']} · полей: {s['n_fields']}"
        f" (исключено из P/R: {s['n_fields_excluded_from_pr']})",
        f"  F1 micro {_pct(s['micro_f1'])} · macro {_pct(s['macro_f1'])} · взвеш. {_pct(s['weighted_f1'])}"
        f" [{s['weights_source']}]",
        f"  precision(proposed) {_pct(s['precision_proposed'])} (цель >= 90%: {ok(t['precision_proposed>=0.90'])})"
        f" · proposed без evidence: {s['proposed_without_evidence']}",
        f"  coverage было -> стало: {_pct(s['coverage_before'])} -> {_pct(s['coverage_after'])}"
        f" · доля LLM-полей {_pct(s['llm_field_share'])}",
        f"  разброс F1 по категориям: {'н/д' if sp is None else f'{sp:.1f} п.п.'} (цель <= 10: {ok(t['category_spread<=10pp'])})",
        f"  оптимальный порог (итог) low/mid/high: "
        + " / ".join(str(s['threshold_optimum_total'][k]) for k in SCENARIOS),
    ]
    if s["attributes_f1_below_0.8"]:
        bl = s["attributes_f1_below_0.8"]
        lines.append(f"  F1 < 0.80 ({len(bl)}): " + ", ".join(bl[:8]) + (" ..." if len(bl) > 8 else ""))
    if s["configs_not_run"]:
        lines.append("  конфигурации: не запускались — " + ", ".join(s["configs_not_run"]))
    if s["cards_without_prediction"]:
        lines.append(f"  ВНИМАНИЕ: карточек эталона без предсказания: {s['cards_without_prediction']}")
    return "\n".join(lines)


def _parse_compare(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--compare: ожидается имя=путь, получено {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AttrGuard: метрики качества, сравнение конфигураций, стоимость порога")
    ap.add_argument("--pred", help="predictions.jsonl основной конфигурации (без него берётся лучшая из --compare)")
    ap.add_argument("--gold", required=True, help="эталон: jsonl | csv | каталог фикстур tests/fixtures")
    ap.add_argument("--out", default="outputs/metrics/")
    ap.add_argument("--weights", help="csv category,attribute,weight (трафик); без него веса равные")
    ap.add_argument("--compare", nargs="+", metavar="ИМЯ=ПУТЬ",
                    help="rules=a.jsonl rules_ml=b.jsonl full=c.jsonl; отсутствующие -> «не запускалась»")
    ap.add_argument("--config", default="rules", choices=list(CONFIG_ORDER),
                    help="к какой конфигурации относится --pred, если --compare не задан")
    ap.add_argument("--llm-call-cost-rub", type=float, default=0.0, help="цена одного LLM-вызова, руб. (по умолч. 0)")
    ap.add_argument("--thresholds", help="thresholds.yaml с блоком cost (по умолч. config/thresholds.yaml)")
    a = ap.parse_args(argv)
    summary = evaluate(a.pred, a.gold, a.out, weights_path=a.weights, compare=_parse_compare(a.compare),
                       llm_call_cost_rub=a.llm_call_cost_rub, thresholds_path=a.thresholds, config_name=a.config)
    print(format_summary(summary))
    print(f"Файлы: {Path(a.out).resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
