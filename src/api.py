"""FastAPI: API + веб-интерфейс (SPEC 3.1, раздел 4). Владелец — поток C.

Запуск: uvicorn src.api:app
Переменные окружения: ATTRGUARD_DB, ATTRGUARD_FEEDBACK, ATTRGUARD_CATEGORIES, ATTRGUARD_DEMO.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError

from src.contracts import (
    AttributeResult,
    Card,
    CardStatus,
    ExtractResponse,
    Status,
    evidence_is_valid,
    seller_attributes_text,
    source_texts,
)
from src.storage import Storage

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

STATUS_LABELS = {
    "proposed": "Предложено",
    "low_confidence": "Низкая уверенность",
    "conflict": "Конфликт",
    "missing": "Нет значения",
    "unchanged": "Без изменений",
}
CARD_STATUS_LABELS = {"ready": "Готово", "needs_manager": "Ждёт менеджера", "needs_seller": "Ждёт продавца"}
STAGE_LABELS = {"rules": "Правила", "ml": "ML", "llm": "LLM", "none": "—"}
SOURCE_LABELS = {"title": "Название", "description": "Описание", "seller_attributes": "Характеристики"}
UNIT_LABELS = {"GB": "ГБ", "inch": "″", "Hz": "Гц"}
ACTOR_LABELS = {"manager": "менеджер", "seller": "продавец"}
PROBLEM_STATUSES = {"low_confidence", "conflict", "missing"}
CATEGORY_ID = re.compile(r"^[A-Za-z0-9_]+$")


# ---------------------------------------------------------------------------
# Схемы категорий (чтение config/categories/*.yaml)
# ---------------------------------------------------------------------------
@dataclass
class AttrSpec:
    code: str
    name_ru: str
    type: str = "enum"
    unit: str | None = None
    required: bool = False
    values: list[str] = field(default_factory=list)
    range: list[float] | None = None
    seller_keys: list[str] = field(default_factory=list)

    @property
    def hint(self) -> str:
        unit = UNIT_LABELS.get(self.unit or "", self.unit or "")
        if self.type == "number":
            s = f"Число{', ' + unit if unit else ''}"
            if self.values:
                s += f", например {self.values[min(2, len(self.values) - 1)]}"
            if self.range:
                s += f" (от {self.range[0]:g} до {self.range[1]:g})"
            return s
        return "Выберите значение из справочника"


@dataclass
class Schema:
    category: str
    name_ru: str
    keywords: list[str]
    attributes: list[AttrSpec]

    def get(self, code: str) -> AttrSpec | None:
        return next((a for a in self.attributes if a.code == code), None)

    @property
    def required_codes(self) -> set[str]:
        return {a.code for a in self.attributes if a.required}


def _parse_schema(data: dict) -> Schema:
    attrs = []
    for a in data.get("attributes", []):
        attrs.append(AttrSpec(
            code=a["code"], name_ru=a.get("name_ru", a["code"]), type=a.get("type", "enum"), unit=a.get("unit"),
            required=bool(a.get("required", False)), values=[str(v) for v in a.get("values", [])],
            range=a.get("range"), seller_keys=[str(k) for k in a.get("seller_keys", [])],
        ))
    return Schema(data.get("category", ""), data.get("name_ru", data.get("category", "")),
                  [str(k) for k in data.get("keywords", [])], attrs)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------
def card_hash(card: Card) -> str:
    """Хэш title + description + seller_attributes (INTENT: ночной прогон и идемпотентность)."""
    payload = json.dumps([card.title, card.description, card.seller_attributes], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def seller_value(card: Card, spec: AttrSpec) -> str | None:
    keys = {spec.code.casefold(), *(k.casefold() for k in spec.seller_keys)}
    for k, v in card.seller_attributes.items():
        if k.strip().casefold() in keys and str(v).strip():
            return str(v).strip()
    return None


def same_as_was(value: str | None, unit: str | None, was: str | None) -> bool:
    if value is None or not was:
        return False
    w, v = was.strip().casefold(), value.strip().casefold()
    return w == v or (bool(unit) and w == f"{v} {unit}".casefold())


def compute_card_status(statuses: list[tuple[str, bool]], required: set[str]) -> CardStatus:
    """statuses: (status, asked_seller) для обязательных полей без решения человека (SPEC 3.4)."""
    pending = [(s, asked) for s, asked in statuses if s not in ("proposed", "unchanged") or asked]
    if not pending:
        return "ready"
    if any(s in ("conflict", "missing") or asked for s, asked in pending):
        return "needs_seller"
    return "needs_manager"


def fmt_dt(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).strftime("%d.%m.%Y %H:%M")
    except ValueError:
        return iso


def fmt_value(value: str | None, unit: str | None) -> str:
    if value in (None, ""):
        return "—"
    u = UNIT_LABELS.get(unit or "", unit or "")
    return f"{value} {u}".strip()


def parse_characteristics(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (text or "").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out


def characteristics_text(seller_attributes: dict[str, str]) -> str:
    return seller_attributes_text(seller_attributes)


# ---------------------------------------------------------------------------
# Представления (итоговое состояние карточки с учётом решений людей)
# ---------------------------------------------------------------------------
@dataclass
class AttrView:
    spec: AttrSpec
    base: AttributeResult
    res: AttributeResult
    resolved_by: str | None = None
    resolved_at: str | None = None
    asked: bool = False
    asked_at: str | None = None
    answered_at: str | None = None
    question: dict | None = None
    problem: bool = False

    @property
    def code(self) -> str:
        return self.spec.code


@dataclass
class CardView:
    card: Card
    owner: str
    created_at: str
    base: ExtractResponse
    category: str
    category_name: str
    attrs: list[AttrView]
    card_status: CardStatus
    problems: list[AttrView]
    questions: list[AttrView]
    answered: list[AttrView]
    min_conf: float | None
    request_state: str | None
    request_at: str | None
    schema: Schema

    @property
    def product_id(self) -> str:
        return self.card.product_id

    @property
    def is_exception(self) -> bool:
        return self.card_status != "ready"


# ---------------------------------------------------------------------------
# Контекст приложения: хранилище, схемы, ядро (с фоллбеком), представления
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self, db_path=None, feedback_path=None, categories_dir=None, demo=None):
        self.storage = Storage(db_path or os.environ.get("ATTRGUARD_DB"),
                               feedback_path or os.environ.get("ATTRGUARD_FEEDBACK"))
        self.categories_dir = Path(categories_dir or os.environ.get("ATTRGUARD_CATEGORIES") or ROOT / "config" / "categories")
        self.demo = Path(demo or os.environ.get("ATTRGUARD_DEMO") or ROOT / "tests" / "fixtures")
        self.fixtures_dir = ROOT / "tests" / "fixtures"
        self._schemas: dict[str, tuple[float, Schema]] = {}
        self._expected: dict[str, dict] | None = None
        self.fallback_used = False
        self.core_calls = 0
        self.load_demo()

    # --- схемы ----------------------------------------------------------
    def schema(self, category: str | None) -> Schema:
        name = category if category and category != "unknown" and CATEGORY_ID.match(category) else "_common"
        path = self.categories_dir / f"{name}.yaml"
        if not path.exists():
            path = self.categories_dir / "_common.yaml"
        mtime = path.stat().st_mtime
        cached = self._schemas.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        sch = _parse_schema(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        self._schemas[str(path)] = (mtime, sch)
        return sch

    def detect_category(self, card: Card) -> str:
        known = {p.stem: p for p in self.categories_dir.glob("*.yaml")
                 if not p.stem.startswith("_") and not p.name.endswith(".draft.yaml")}
        if card.category_hint and card.category_hint in known:
            return card.category_hint
        title = card.title.casefold()
        for cat in known:
            if any(k.casefold() in title for k in self.schema(cat).keywords):
                return cat
        return "unknown"

    # --- ядро и фоллбек -------------------------------------------------
    def _expected_map(self) -> dict[str, dict]:
        if self._expected is None:
            self._expected = {}
            for p in sorted(self.fixtures_dir.glob("*.json")):
                d = json.loads(p.read_text(encoding="utf-8"))
                if "card" in d and "expected" in d:
                    self._expected[d["card"]["product_id"]] = d
        return self._expected

    def fallback_extract(self, card: Card) -> ExtractResponse:
        """ФОЛЛБЕК, пока ядро (src.cascade) не реализовано. После слияния с ядром не срабатывает.

        Карточки фикстур: значения и статусы берутся из `expected`; proposed/unchanged понижаются до
        low_confidence (нет evidence — нет proposed). Остальные карточки: все поля схемы missing.
        """
        t0 = time.perf_counter()
        fx = self._expected_map().get(card.product_id)
        if fx and (fx["card"].get("title"), fx["card"].get("description")) != (card.title, card.description):
            fx = None
        category = fx["expected"]["category"] if fx else self.detect_category(card)
        sch = self.schema(category)
        expected = fx["expected"]["attributes"] if fx else {c.code: {"status": "missing"} for c in sch.attributes}
        results: list[AttributeResult] = []
        for code, exp in expected.items():
            spec = sch.get(code)
            if spec is None:
                continue
            st = exp["status"][0] if isinstance(exp["status"], list) else exp["status"]
            value, unit = exp.get("value"), exp.get("unit") or spec.unit if exp.get("value") else None
            if st in ("proposed", "unchanged"):
                st = "low_confidence"
            conf = {"low_confidence": 0.6, "conflict": 0.55, "missing": 0.0}[st]
            question = None
            if st in ("conflict", "missing"):
                question = {"text": f"Уточните: {spec.name_ru.lower()}", "options": spec.values[:12]}
            results.append(AttributeResult(
                attribute=code, was=seller_value(card, spec), value=value, unit=unit, status=Status(st),
                confidence=conf, stage="none" if st == "missing" else "rules", evidence=None,
                question_to_seller=question))
        status = compute_card_status([(r.status.value, False) for r in results if r.attribute in sch.required_codes],
                                     sch.required_codes)
        return ExtractResponse(
            product_id=card.product_id, category=category, category_confidence=0.9 if fx else 0.5,
            attributes=results, card_status=status, llm_calls=0,
            elapsed_ms=int((time.perf_counter() - t0) * 1000), text_hash=card_hash(card))

    def run_core(self, card: Card) -> ExtractResponse:
        from src import cascade
        try:
            resp = cascade.extract(card)
        except NotImplementedError:
            self.fallback_used = True
            return self.fallback_extract(card)
        self.core_calls += 1
        return resp

    def extract(self, card: Card) -> ExtractResponse:
        """Идемпотентно по (product_id, хэш текста) — F16."""
        h = card_hash(card)
        cached = self.storage.get_cached(card.product_id, h)
        if cached is not None:
            return cached
        resp = self.run_core(card)
        self.storage.put_cached(card.product_id, h, resp)
        return resp

    # --- демо-данные ----------------------------------------------------
    def load_demo(self) -> None:
        for card in self._read_demo_cards():
            if self.storage.get_card(card.product_id) is None:
                self.storage.save_card(card, owner="demo", replace=False)

    def _read_demo_cards(self) -> list[Card]:
        paths: list[Path] = []
        if self.demo.is_dir():
            paths = sorted(self.demo.glob("*.json")) + sorted(self.demo.glob("*.jsonl"))
        elif self.demo.exists():
            paths = [self.demo]
        cards: list[Card] = []
        for p in paths:
            try:
                if p.suffix == ".jsonl":
                    items = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
                else:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    items = data if isinstance(data, list) else [data]
                for it in items:
                    cards.append(Card.model_validate(it["card"] if "card" in it else it))
            except (ValueError, KeyError, ValidationError):
                continue
        return cards

    # --- представления --------------------------------------------------
    def build_view(self, card: Card, owner: str, created_at: str, feedback: list[dict], reads: dict[str, str]) -> CardView:
        base = self.extract(card)
        sch = self.schema(base.category)
        events: dict[str, dict] = {}
        for fb in feedback:
            st = events.setdefault(fb["attribute"], {"asked": None, "resolved": None, "answered_at": None, "answer": None})
            if fb["action"] == "ask_seller":
                st["asked"], st["resolved"] = fb["ts"], None
            else:
                st["resolved"] = fb
                if fb["action"] == "seller_answer":
                    st["answered_at"], st["answer"] = fb["ts"], fb["value"]
        attrs: list[AttrView] = []
        for r in base.attributes:
            spec = sch.get(r.attribute) or AttrSpec(code=r.attribute, name_ru=r.attribute)
            ev = events.get(r.attribute, {})
            res = r.model_copy(deep=True)
            av = AttrView(spec=spec, base=r, res=res, asked=bool(ev.get("asked")), asked_at=ev.get("asked"),
                          answered_at=ev.get("answered_at"))
            fb = ev.get("resolved")
            if fb:
                value = r.value if fb["action"] == "accept" else fb["value"]
                res.value = value
                if fb["action"] != "accept" or value != r.value:
                    res.evidence = None
                res.unit = r.unit or spec.unit if spec.type == "number" else None
                res.status = Status.unchanged if same_as_was(value, res.unit, r.was) else Status.proposed
                res.confidence = 1.0
                av.resolved_by, av.resolved_at = fb["actor"], fb["ts"]
            av.question = r.question_to_seller
            if av.asked and not av.question:
                av.question = {"text": f"Уточните: {spec.name_ru.lower()}", "options": spec.values[:12]}
            av.problem = spec.required and not av.resolved_by and (res.status.value in PROBLEM_STATUSES or av.asked)
            attrs.append(av)
        problems = [a for a in attrs if a.problem]
        card_status = compute_card_status(
            [(a.res.status.value, a.asked) for a in attrs if a.spec.required and not a.resolved_by], sch.required_codes)
        questions = [a for a in attrs if a.question and not a.resolved_by
                     and (a.asked or (a.spec.required and a.res.status.value in ("conflict", "missing")))]
        answered = [a for a in attrs if a.answered_at and a.resolved_by == "seller"]
        confs = [a.res.confidence for a in problems]
        req_state, req_at = None, None
        if questions:
            read_at = reads.get(card.product_id)
            req_state, req_at = ("прочитан", read_at) if read_at else ("отправлен", max(
                [a.asked_at for a in questions if a.asked_at] or [created_at]))
        elif answered:
            req_state, req_at = "ответ получен", max(a.answered_at for a in answered)
        return CardView(card, owner, created_at, base, base.category, "Новая категория" if base.category == "unknown" else sch.name_ru,
                        attrs, card_status, problems, questions, answered, min(confs) if confs else None,
                        req_state, req_at, sch)

    def views(self) -> list[CardView]:
        fb_all = self.storage.feedback_for()
        by_card: dict[str, list[dict]] = {}
        for fb in fb_all:
            by_card.setdefault(fb["product_id"], []).append(fb)
        reads = self.storage.reads()
        return [self.build_view(c, o, t, by_card.get(c.product_id, []), reads) for c, o, t in self.storage.list_cards()]

    def view(self, product_id: str) -> CardView | None:
        row = self.storage.get_card(product_id)
        if row is None:
            return None
        return self.build_view(row[0], row[1], row[2], self.storage.feedback_for(product_id), self.storage.reads())

    def transient_view(self, card: Card) -> CardView:
        return self.build_view(card, "seller", "", [], {})


# ---------------------------------------------------------------------------
# Модели запросов
# ---------------------------------------------------------------------------
class Feedback(BaseModel):
    product_id: str
    attribute: str
    action: Literal["accept", "fix", "ask_seller", "seller_answer"]
    value: str | None = None
    actor: Literal["manager", "seller"] = "manager"


def _ctx_for(app: FastAPI) -> Ctx:
    if app.state.ctx is None:
        with app.state.lock:
            if app.state.ctx is None:
                app.state.ctx = Ctx(**app.state.ctx_args)
    return app.state.ctx


def _ctx(request: Request) -> Ctx:
    return _ctx_for(request.app)


def _render(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
    ctx = _ctx(request)
    path = request.url.path
    role = "manager" if path.startswith("/manager") else "seller" if path.startswith("/seller") else request.cookies.get("role")
    context.update(role=role, fallback_used=ctx.fallback_used, path=path)
    return request.app.state.templates.TemplateResponse(request, name, context, status_code=status_code)


def _highlight(view: CardView) -> list[dict]:
    """Разбивает исходные тексты на сегменты с подсветкой evidence (по start/end)."""
    texts = source_texts(view.card)
    out = []
    for source in ("title", "description", "seller_attributes"):
        text = texts[source]
        if not text:
            continue
        spans = []
        for a in view.attrs:
            ev = a.base.evidence
            if ev and ev.source == source and evidence_is_valid(view.card, ev):
                spans.append((ev.start, ev.end, a.spec.name_ru, a.base.status.value))
        spans.sort()
        segs, pos = [], 0
        for s, e, name, st in spans:
            if s < pos:
                continue
            if s > pos:
                segs.append({"text": text[pos:s], "mark": None})
            segs.append({"text": text[s:e], "mark": name, "status": st})
            pos = e
        if pos < len(text):
            segs.append({"text": text[pos:], "mark": None})
        out.append({"label": SOURCE_LABELS[source], "segments": segs})
    return out


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------
router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --- API -------------------------------------------------------------------
@router.post("/v1/extract", response_model=ExtractResponse)
def v1_extract(card: Card, request: Request) -> ExtractResponse:
    return _ctx(request).extract(card)


@router.post("/v1/extract/batch", response_model=list[ExtractResponse])
def v1_extract_batch(cards: list[Card], request: Request) -> list[ExtractResponse]:
    ctx = _ctx(request)
    return [ctx.extract(c) for c in cards]


@router.post("/v1/feedback")
async def v1_feedback(request: Request):
    ctx = _ctx(request)
    if "application/json" in request.headers.get("content-type", ""):
        data = await request.json()
    else:
        data = dict(await request.form())
    view_name = data.pop("view", None) if isinstance(data, dict) else None
    try:
        fb = Feedback.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, detail=json.loads(e.json()))
    view = ctx.view(fb.product_id)
    if view is None:
        raise HTTPException(404, detail="карточка не найдена")
    av = next((a for a in view.attrs if a.code == fb.attribute), None)
    if av is None:
        raise HTTPException(404, detail="атрибут не найден")
    value = (fb.value or "").strip() or None
    if fb.action in ("fix", "seller_answer") and value is None:
        raise HTTPException(422, detail="для fix и seller_answer нужно value")
    if fb.action == "accept" and av.base.value is None:
        raise HTTPException(422, detail="нечего принимать: у поля нет предложенного значения")
    if fb.action == "seller_answer" and fb.actor != "seller":
        raise HTTPException(422, detail="seller_answer возможен только с actor=seller")
    if fb.action in ("accept", "ask_seller"):
        value = av.base.value if fb.action == "accept" else None
    row = ctx.storage.add_feedback(
        fb.product_id, fb.attribute, fb.action, value, fb.actor,
        extra={"category": view.category, "text_hash": view.base.text_hash, "was": av.base.was,
               "proposed_value": av.base.value, "proposed_status": av.base.status.value})
    new_view = ctx.view(fb.product_id)
    if request.headers.get("HX-Request"):
        if view_name == "seller_q":
            return _render(request, "_questions_card.html", v=new_view, answered_attr=fb.attribute)
        return _render(request, "_card_panel.html", v=new_view, flash=fb.attribute)
    return {"ok": True, "feedback": row, "card_status": new_view.card_status}


@router.get("/v1/queue")
def v1_queue(request: Request) -> list[dict]:
    """Очередь «новая категория»: карточки с category == unknown."""
    return [{"product_id": v.product_id, "title": v.card.title, "category": v.category,
             "card_status": v.card_status, "created_at": v.created_at}
            for v in _ctx(request).views() if v.category == "unknown"]


def _draft_path(ctx: Ctx, cid: str) -> Path:
    if not CATEGORY_ID.match(cid):
        raise HTTPException(404, detail="черновик не найден")
    return ctx.categories_dir / f"{cid}.draft.yaml"


@router.post("/v1/categories/{cid}/approve")
def v1_approve(cid: str, request: Request):
    ctx = _ctx(request)
    draft = _draft_path(ctx, cid)
    if not draft.exists():
        raise HTTPException(404, detail=f"черновик {cid}.draft.yaml не найден")
    target = ctx.categories_dir / f"{cid}.yaml"
    os.replace(draft, target)
    if request.headers.get("HX-Request"):
        return HTMLResponse(f'<div class="notice ok">Категория «{cid}» утверждена: {target.name}</div>')
    return {"ok": True, "category": cid, "file": target.name}


# --- Роли ------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _render(request, "role.html")


@router.get("/role/{role}")
def set_role(role: Literal["manager", "seller"]):
    resp = RedirectResponse(f"/{role}", status_code=303)
    resp.set_cookie("role", role, max_age=60 * 60 * 24 * 30, samesite="lax")
    return resp


# --- Менеджер --------------------------------------------------------------
@router.get("/manager", response_class=HTMLResponse)
def manager(request: Request):
    ctx = _ctx(request)
    views = ctx.views()
    exceptions = sorted((v for v in views if v.is_exception),
                        key=lambda v: (v.min_conf if v.min_conf is not None else 1.0, v.product_id))
    auto_fixed = sum(1 for v in views for a in v.attrs if a.res.status == Status.proposed and not a.resolved_by)
    llm_cards = sum(1 for v in views if v.base.llm_calls > 0)
    summary = {
        "checked": len(views),
        "auto_fixed": auto_fixed,
        "wait_manager": sum(1 for v in views if v.card_status == "needs_manager"),
        "wait_seller": sum(1 for v in views if v.card_status == "needs_seller"),
        "llm_share": round(100 * llm_cards / len(views)) if views else 0,
    }
    answers = sorted(((v, x) for v in views for x in v.answered), key=lambda t: t[1].answered_at, reverse=True)[:8]
    return _render(request, "manager.html", views=exceptions, summary=summary, answers=answers)


@router.get("/manager/card/{pid}", response_class=HTMLResponse)
def manager_card(pid: str, request: Request):
    v = _ctx(request).view(pid)
    if v is None:
        raise HTTPException(404, detail="карточка не найдена")
    return _render(request, "manager_card.html", v=v, source_blocks=_highlight(v), flash=None)


def _drafts(ctx: Ctx) -> list[dict]:
    out = []
    for p in sorted(ctx.categories_dir.glob("*.draft.yaml")):
        out.append({"id": p.name[: -len(".draft.yaml")], "yaml": p.read_text(encoding="utf-8")})
    return out


@router.get("/manager/categories", response_class=HTMLResponse)
def manager_categories(request: Request, saved: str | None = None, error: str | None = None):
    ctx = _ctx(request)
    queue = [v for v in ctx.views() if v.category == "unknown"]
    return _render(request, "manager_categories.html", queue=queue, drafts=_drafts(ctx), saved=saved, error=error)


@router.post("/manager/categories/{cid}/draft")
async def manager_save_draft(cid: str, request: Request):
    ctx = _ctx(request)
    draft = _draft_path(ctx, cid)
    if not draft.exists():
        raise HTTPException(404, detail="черновик не найден")
    text = (await request.form()).get("yaml", "")
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or not isinstance(data.get("attributes"), list):
            raise ValueError("в YAML нет списка attributes")
    except (yaml.YAMLError, ValueError) as e:
        return RedirectResponse(f"/manager/categories?error={cid}: {str(e)[:120]}", status_code=303)
    draft.write_text(text, encoding="utf-8")
    return RedirectResponse(f"/manager/categories?saved={cid}", status_code=303)


# --- Продавец --------------------------------------------------------------
@router.get("/seller", response_class=HTMLResponse)
def seller(request: Request, saved: str | None = None):
    views = _ctx(request).views()
    views.sort(key=lambda v: (v.owner != "seller",))
    return _render(request, "seller.html", views=views, saved=saved)


def _preview(ctx: Ctx, form: dict, check_overrides: bool) -> dict:
    card = Card(product_id=form.get("product_id") or "draft", title=form.get("title", "").strip(),
                description=form.get("description", "").strip(),
                seller_attributes=parse_characteristics(form.get("characteristics", "")))
    v = ctx.transient_view(card)
    rows, filled = [], 0
    for a in v.attrs:
        if not a.spec.required:
            continue
        key = f"f_{a.code}"
        prefill = a.res.value or a.base.was or ""
        overridden = check_overrides and key in form
        value = (form.get(key) or "").strip() if overridden else prefill
        rows.append({"a": a, "prefill": prefill, "value": value,
                     "overridden": overridden and value != prefill,
                     "options": list(a.spec.values)})
        filled += 1 if value else 0
    return {"card": card, "v": v, "rows": rows, "filled": filled, "total": len(rows),
            "chars": characteristics_text(card.seller_attributes)}


@router.get("/seller/new", response_class=HTMLResponse)
def seller_new(request: Request):
    return _render(request, "seller_new.html", pid="new-" + uuid.uuid4().hex[:6])


@router.post("/seller/check", response_class=HTMLResponse)
async def seller_check(request: Request):
    form = dict(await request.form())
    if not (form.get("title") or "").strip():
        return _render(request, "_preview.html", p=None)
    return _render(request, "_preview.html", p=_preview(_ctx(request), form, check_overrides="f_marker" in form))


@router.post("/seller/save")
async def seller_save(request: Request):
    ctx = _ctx(request)
    form = dict(await request.form())
    if not (form.get("title") or "").strip():
        return HTMLResponse('<div class="notice err">Укажите название товара</div>', status_code=200)
    p = _preview(ctx, form, check_overrides=False)
    card: Card = p["card"]
    ctx.storage.save_card(card, owner="seller")
    base_by_code = {r["a"].code: r for r in p["rows"]}
    for code, r in base_by_code.items():
        submitted = (form.get(f"f_{code}") or "").strip()
        if submitted and submitted != r["prefill"]:
            ctx.storage.add_feedback(card.product_id, code, "seller_answer", submitted, "seller", extra={
                "category": p["v"].category, "text_hash": p["v"].base.text_hash, "was": r["a"].base.was,
                "proposed_value": r["a"].base.value, "proposed_status": r["a"].base.status.value})
    resp = Response(status_code=200)
    resp.headers["HX-Redirect"] = f"/seller?saved={card.product_id}"
    return resp


@router.get("/seller/questions", response_class=HTMLResponse)
def seller_questions(request: Request):
    ctx = _ctx(request)
    views = ctx.views()
    shown = [v for v in views if v.questions or v.answered]
    for v in shown:
        if v.questions:
            ctx.storage.mark_read(v.product_id)
    return _render(request, "seller_questions.html", views=shown)


# ---------------------------------------------------------------------------
def create_app(db_path=None, feedback_path=None, categories_dir=None, demo=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = _ctx_for(app)
        for v in ctx.views():  # прогрев кэша извлечения на демо-карточках
            pass
        yield

    app = FastAPI(title="AttrGuard", lifespan=lifespan)
    app.state.ctx = None
    app.state.lock = threading.Lock()
    app.state.ctx_args = dict(db_path=db_path, feedback_path=feedback_path, categories_dir=categories_dir, demo=demo)
    templates = Jinja2Templates(directory=str(WEB / "templates"))
    templates.env.filters.update(
        fmt_value=fmt_value, fmt_dt=fmt_dt, pct=lambda x: "—" if x is None else f"{round(x * 100)}%")
    templates.env.globals.update(
        STATUS_LABELS=STATUS_LABELS, CARD_STATUS_LABELS=CARD_STATUS_LABELS, STAGE_LABELS=STAGE_LABELS,
        SOURCE_LABELS=SOURCE_LABELS, ACTOR_LABELS=ACTOR_LABELS, UNIT_LABELS=UNIT_LABELS)
    app.state.templates = templates
    app.mount("/static", StaticFiles(directory=str(WEB / "static")), name="static")
    app.include_router(router)
    return app


app = create_app()
