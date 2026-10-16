"""Оркестрация этапов, статусы, вопрос продавцу (SPEC 3.4). Владелец — поток B, затем интеграция.

prepare → rules → [ml] → [llm]; после каждого этапа: normalize + check_evidence.
Статус proposed/unchanged невозможен без валидной evidence; значение от LLM — максимум low_confidence.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from src import rules
from src.contracts import AttributeResult, Candidate, Card, ExtractResponse, Status
from src.normalize import check_evidence, normalize
from src.prepare import AttributeSpec, PreparedCard, load_schemas, prepare

_ROOT = Path(__file__).resolve().parent.parent
SOURCE_RU = {"title": "в названии", "description": "в описании", "seller_attributes": "в характеристиках продавца"}
CONF_CAP = 0.99
UNIT_RU = {"GB": "ГБ", "inch": "дюймах", "Hz": "Гц"}
DEFAULT_THRESHOLDS = {"high": 0.80, "low": 0.50}


@dataclass
class CascadeConfig:
    use_ml: bool = False
    use_llm: bool = False
    thresholds_path: str = "config/thresholds.yaml"
    categories_dir: str = "config/categories"


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (_ROOT / path if not path.exists() else path)


class Thresholds:
    """Пороги из thresholds.yaml: default + переопределения по атрибуту («ram») или по категории («laptops.ram»)."""

    def __init__(self, path: Path):
        self.path = path
        self._sig: tuple | None = None
        self.default = dict(DEFAULT_THRESHOLDS)
        self.attrs: dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        try:
            st = self.path.stat()
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        if sig == self._sig and self._sig is not None:
            return
        self._sig = sig
        data = {}
        if sig is not None:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.default = {**DEFAULT_THRESHOLDS, **(data.get("default") or {})}
        self.attrs = data.get("attributes") or {}

    def get(self, category: str, attribute: str) -> tuple[float, float]:
        o = self.attrs.get(f"{category}.{attribute}") or self.attrs.get(attribute) or {}
        return float(o.get("high", self.default["high"])), float(o.get("low", self.default["low"]))


class Cascade:
    def __init__(self, config: CascadeConfig | None = None, ml=None, llm=None):
        self.config = config or CascadeConfig()
        self.ml = ml
        self.llm = llm
        self._categories_dir = _abs(self.config.categories_dir)
        self.thresholds = Thresholds(_abs(self.config.thresholds_path))
        self.schemas = load_schemas(self._categories_dir)

    def reload(self) -> None:
        """Перечитать схемы и пороги (например, после утверждения новой категории). Обычно не нужно: кэш сам видит изменения."""
        self.thresholds.reload()
        self.schemas = load_schemas(self._categories_dir)

    # ------------------------------------------------------------ основной метод
    def run(self, card: Card) -> ExtractResponse:
        t0 = time.perf_counter()
        self.thresholds.reload()
        self.schemas = load_schemas(self._categories_dir)   # дёшево: кэш по mtime
        prepared = prepare(card, self.schemas)
        specs = {a.code: a for a in prepared.schema.attributes}
        was = rules.seller_values(prepared)
        calls0 = int(getattr(self.llm, "calls", 0) or 0) if self.llm is not None else 0

        cands = self._finish(prepared, rules.extract(prepared), specs)          # Этап 1
        results = {s.code: self._decide(prepared, s, was.get(s.code), cands) for s in prepared.schema.attributes}

        if self.config.use_ml and self.ml is not None:                          # Этап 2
            need = [c for c, r in results.items() if r.status in (Status.missing, Status.low_confidence, Status.conflict)]
            if need:
                extra = self._finish(prepared, self.ml.predict(prepared, need, list(cands)) or [], specs, only=set(need))
                if extra:
                    cands += extra
                    for code in {c.attribute for c in extra}:
                        results[code] = self._decide(prepared, specs[code], was.get(code), cands)

        if self.config.use_llm and self.llm is not None:                        # Этап 3
            need = [c for c, r in results.items() if r.status in (Status.conflict, Status.low_confidence)]
            if need:
                extra = self._finish(prepared, self.llm.resolve(prepared, need, list(cands)) or [], specs, only=set(need))
                for code in {c.attribute for c in extra}:
                    llm_c = [c for c in extra if c.attribute == code]
                    results[code] = self._from_llm(prepared, specs[code], was.get(code), llm_c, results[code])

        attributes = [results[a.code] for a in prepared.schema.attributes]
        calls = int(getattr(self.llm, "calls", 0) or 0) - calls0 if self.llm is not None else 0
        return ExtractResponse(
            product_id=card.product_id,
            category=prepared.category,
            category_confidence=round(prepared.category_confidence, 3),
            attributes=attributes,
            card_status=self._card_status(prepared.schema.attributes, results),
            llm_calls=max(calls, 0),
            elapsed_ms=int(round((time.perf_counter() - t0) * 1000)),
            text_hash=prepared.text_hash,
        )

    # ------------------------------------------------------------ normalize + check_evidence
    @staticmethod
    def _finish(prepared: PreparedCard, cands: list[Candidate], specs: dict[str, AttributeSpec],
                only: set[str] | None = None) -> list[Candidate]:
        out = []
        for c in cands:
            spec = specs.get(c.attribute)
            if spec is None or (only is not None and c.attribute not in only):
                continue
            n = normalize(c, spec)
            if n.value is None or not check_evidence(prepared.card, n):
                continue
            out.append(n)
        return out

    # ------------------------------------------------------------ статус атрибута (SPEC 3.4)
    def _decide(self, prepared: PreparedCard, spec: AttributeSpec, was: str | None,
                cands: list[Candidate]) -> AttributeResult:
        high, low = self.thresholds.get(prepared.category, spec.code)
        mine = [c for c in cands if c.attribute == spec.code and c.value is not None]
        if not mine:
            return self._missing(spec, was)

        by_value: dict[str, list[Candidate]] = {}
        for c in mine:
            by_value.setdefault(c.value, []).append(c)  # type: ignore[arg-type]

        def conf(items: list[Candidate]) -> float:
            best_per_source: dict[str, float] = {}
            for c in items:
                s = c.evidence.source  # type: ignore[union-attr]
                best_per_source[s] = max(best_per_source.get(s, 0.0), c.score)
            p = 1.0
            for s in best_per_source.values():
                p *= 1.0 - s
            return min(1.0 - p, CONF_CAP)

        ranked = sorted(by_value.items(), key=lambda kv: (-conf(kv[1]), -max(c.score for c in kv[1])))
        top_value, top_items = ranked[0]
        top_conf = conf(top_items)
        best = self._best(top_items)

        # 2. конфликт: разные источники дают разные значения, оба кандидата уверенные (> low)
        for other_value, other_items in ranked[1:]:
            if any(a.evidence.source != b.evidence.source and a.score > low and b.score > low  # type: ignore[union-attr]
                   for a in top_items for b in other_items):
                options = [v for v, _ in ranked if any(c.score > low for c in by_value[v])]
                return AttributeResult(
                    attribute=spec.code, was=was, value=top_value, unit=spec.unit or None,
                    status=Status.conflict, confidence=round(top_conf, 3), stage=best.stage,
                    evidence=best.evidence,
                    question_to_seller=self._conflict_question(spec, [(v, by_value[v]) for v in options]),
                )

        # 3. лучший кандидат выше порога
        if top_conf >= high and best.stage != "llm" and check_evidence(prepared.card, best):
            status = Status.unchanged if self._same_as_was(was, top_value, spec.unit) else Status.proposed
        else:
            status = Status.low_confidence   # 4.
        return AttributeResult(
            attribute=spec.code, was=was, value=top_value, unit=spec.unit or None, status=status,
            confidence=round(top_conf, 3), stage=best.stage, evidence=best.evidence, question_to_seller=None,
        )

    @staticmethod
    def _best(items: list[Candidate]) -> Candidate:
        # при равной уверенности предпочитаем подтверждение в тексте карточки, а не в поле продавца
        return max(items, key=lambda c: (c.score, c.evidence.source != "seller_attributes"))  # type: ignore[union-attr]

    @staticmethod
    def _same_as_was(was: str | None, value: str, unit: str | None) -> bool:
        if not was:
            return False
        w = was.strip().lower()
        v = value.strip().lower()
        return w == v or (bool(unit) and w == f"{v} {unit.lower()}")  # type: ignore[union-attr]

    def _from_llm(self, prepared: PreparedCard, spec: AttributeSpec, was: str | None,
                  llm_cands: list[Candidate], prev: AttributeResult) -> AttributeResult:
        """Значение от LLM: максимум low_confidence (SPEC 3.5), с проверенной evidence."""
        high, _ = self.thresholds.get(prepared.category, spec.code)
        best = max(llm_cands, key=lambda c: c.score)
        return AttributeResult(
            attribute=spec.code, was=was, value=best.value, unit=spec.unit or None, status=Status.low_confidence,
            confidence=round(min(best.score, high - 0.01), 3), stage="llm", evidence=best.evidence,
            question_to_seller=None,
        )

    # ------------------------------------------------------------ вопросы продавцу
    @staticmethod
    def _fmt(spec: AttributeSpec, value: str) -> str:
        return f"{value} {spec.unit}" if spec.unit else value

    def _options(self, spec: AttributeSpec) -> list[str]:
        if spec.values:
            return [self._fmt(spec, str(v)) for v in spec.values]
        return []

    def _missing(self, spec: AttributeSpec, was: str | None) -> AttributeResult:
        unit = f" (в {UNIT_RU.get(spec.unit, spec.unit)})" if spec.unit else ""
        if was and was.strip():
            text = f"Не удалось распознать значение «{was.strip()}» в поле «{spec.name_ru}». Укажите верное значение{unit}."
        else:
            text = f"В карточке нет значения поля «{spec.name_ru}». Укажите его{unit}."
        return AttributeResult(
            attribute=spec.code, was=was, value=None, unit=None, status=Status.missing, confidence=0.0,
            stage="none", evidence=None, question_to_seller={"text": text, "options": self._options(spec)},
        )

    def _conflict_question(self, spec: AttributeSpec, values: list[tuple[str, list[Candidate]]]) -> dict:
        parts = []
        for v, items in values:
            sources = []
            for c in sorted(items, key=lambda c: -c.score):
                s = SOURCE_RU.get(c.evidence.source, c.evidence.source)  # type: ignore[union-attr]
                if s not in sources:
                    sources.append(s)
            parts.append(f"{' и '.join(sources)} — «{self._fmt(spec, v)}»")
        text = f"Поле «{spec.name_ru}»: " + "; ".join(parts) + ". Какое значение верное?"
        return {"text": text, "options": [self._fmt(spec, v) for v, _ in values]}

    # ------------------------------------------------------------ статус карточки
    @staticmethod
    def _card_status(attrs: list[AttributeSpec], results: dict[str, AttributeResult]) -> str:
        req = [results[a.code] for a in attrs if a.required]
        if any(r.status in (Status.conflict, Status.missing) for r in req):
            return "needs_seller"
        if any(r.status not in (Status.unchanged, Status.proposed) for r in req):
            return "needs_manager"
        return "ready"


_default: Cascade | None = None


def extract(card: Card) -> ExtractResponse:
    """Удобная точка входа: каскад с конфигурацией по умолчанию (только правила)."""
    global _default
    if _default is None:
        _default = Cascade()
    return _default.run(card)


__all__ = ["Cascade", "CascadeConfig", "extract", "AttributeResult"]
