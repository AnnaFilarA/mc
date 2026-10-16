"""Пакетный режим (SPEC F11): python -m src.pipeline --input X --output outputs/. Владелец — поток B.

Вход: .jsonl / .json (список карточек) / .csv / каталог фикстур tests/fixtures (берётся поле "card").
Выход: <output>/predictions.jsonl (ExtractResponse на строку) и <output>/predictions.csv (строка на атрибут).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from src.contracts import Card, ExtractResponse

CSV_COLUMNS = ["product_id", "category", "attribute", "was", "value", "unit", "status", "confidence", "stage",
               "evidence_text", "evidence_source", "evidence_start", "evidence_end", "question"]


def _card_from_dict(d: dict) -> Card:
    d = d.get("card", d) if isinstance(d.get("card"), dict) else d
    sa = d.get("seller_attributes") or {}
    if isinstance(sa, str):
        try:
            sa = json.loads(sa) if sa.strip() else {}
        except json.JSONDecodeError:
            sa = {}
    hint = d.get("category_hint")
    return Card(
        product_id=str(d["product_id"]),
        title=str(d.get("title") or ""),
        description=str(d.get("description") or ""),
        seller_attributes={str(k): "" if v is None else str(v) for k, v in dict(sa).items()},
        category_hint=str(hint).strip() if hint not in (None, "") and str(hint).strip() else None,
    )


def load_cards(path: str) -> list[Card]:
    """Карточки из .jsonl / .json (список или объект с ключом card) / .csv / каталога *.json."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        return [_card_from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in files]
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(_card_from_dict(json.loads(line)))
        return out
    if suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("cards", [data])
        return [_card_from_dict(d) for d in items]
    if suffix == ".csv":
        with p.open(encoding="utf-8-sig", newline="") as f:
            return [_card_from_dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"Неизвестный формат входа: {path} (ожидаются .jsonl, .json, .csv или каталог)")


def write_predictions(responses: list[ExtractResponse], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")
    with (out / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in responses:
            for a in r.attributes:
                ev = a.evidence
                w.writerow({
                    "product_id": r.product_id, "category": r.category, "attribute": a.attribute,
                    "was": a.was if a.was is not None else "", "value": a.value if a.value is not None else "",
                    "unit": a.unit or "", "status": a.status.value, "confidence": a.confidence, "stage": a.stage,
                    "evidence_text": ev.text if ev else "", "evidence_source": ev.source if ev else "",
                    "evidence_start": ev.start if ev else "", "evidence_end": ev.end if ev else "",
                    "question": json.dumps(a.question_to_seller, ensure_ascii=False) if a.question_to_seller else "",
                })


def _build_cascade(config: str):
    from src.cascade import Cascade, CascadeConfig
    if config == "rules":
        return Cascade(CascadeConfig(use_ml=False, use_llm=False))
    from src.ml import NullML
    if config == "rules_ml":
        return Cascade(CascadeConfig(use_ml=True, use_llm=False), ml=NullML())
    from src.llm import load_llm
    return Cascade(CascadeConfig(use_ml=True, use_llm=True), ml=NullML(), llm=load_llm())


def run_cards(cards: list[Card], config: str = "rules") -> tuple[list[ExtractResponse], list[float]]:
    cascade = _build_cascade(config)
    responses, times = [], []
    for card in cards:
        t0 = time.perf_counter()
        responses.append(cascade.run(card))
        times.append((time.perf_counter() - t0) * 1000)
    return responses, times


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AttrGuard: пакетное извлечение характеристик")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/")
    parser.add_argument("--config", default="rules", choices=["rules", "rules_ml", "full"])
    args = parser.parse_args(argv)

    cards = load_cards(args.input)
    t0 = time.perf_counter()
    responses, times = run_cards(cards, args.config)
    total = time.perf_counter() - t0
    write_predictions(responses, args.output)

    statuses = Counter(a.status.value for r in responses for a in r.attributes)
    card_statuses = Counter(r.card_status for r in responses)
    print(f"Карточек: {len(responses)} · конфигурация: {args.config} · всего {total:.2f} с")
    print(f"Время на карточку: среднее {sum(times) / max(len(times), 1):.1f} мс, p95 {_p95(times):.1f} мс")
    print("Статусы полей: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())))
    print("Статусы карточек: " + ", ".join(f"{k}={v}" for k, v in sorted(card_statuses.items())))
    print(f"Записано: {Path(args.output) / 'predictions.jsonl'}, {Path(args.output) / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
