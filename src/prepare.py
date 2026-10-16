"""Шаг 0: очистка текста, определение категории, загрузка схемы. Владелец — поток B.

Схемы категорий: config/categories/<category>.yaml (формат — см. smartphones.yaml).
Файлы *.draft.yaml не загружаются (черновики новых категорий, INTENT 4.2).
Неизвестные ключи верхнего уровня в YAML (draft_source, source_cards и т.п.) игнорируются.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.contracts import Card, source_texts

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
UNKNOWN_CATEGORY = "unknown"   # категория не найдена → общие атрибуты из _common.yaml + очередь «новая категория»
COMMON = "_common"


@dataclass
class AttributeSpec:
    code: str
    name_ru: str
    type: str                          # "enum" | "number" | "text"
    required: bool = False
    unit: str | None = None
    values: list = field(default_factory=list)       # справочник допустимых значений
    seller_keys: list[str] = field(default_factory=list)
    range: list[float] | None = None   # [min, max] для number


@dataclass
class CategorySchema:
    category: str
    name_ru: str
    keywords: list[str]
    attributes: list[AttributeSpec]

    def attr(self, code: str) -> AttributeSpec | None:
        return next((a for a in self.attributes if a.code == code), None)


@dataclass
class PreparedCard:
    card: Card
    category: str
    category_confidence: float
    schema: CategorySchema
    texts: dict[str, str]              # source -> исходный текст (source_texts), evidence ссылается на него
    clean: dict[str, str]              # source -> очищенный текст (для классификации/ML; не для evidence)
    text_hash: str


# ---------------------------------------------------------------- схемы

_SCHEMA_CACHE: dict[str, tuple[tuple, dict[str, CategorySchema]]] = {}


def _schema_files(config_dir: Path) -> list[Path]:
    return sorted(p for p in config_dir.glob("*.yaml") if not p.name.endswith(".draft.yaml"))


def _parse_schema(data: dict, fallback_code: str) -> CategorySchema:
    attrs = []
    for a in data.get("attributes") or []:
        rng = a.get("range")
        attrs.append(AttributeSpec(
            code=str(a["code"]),
            name_ru=str(a.get("name_ru") or a["code"]),
            type=str(a.get("type") or "text"),
            required=bool(a.get("required", False)),
            unit=a.get("unit"),
            values=list(a.get("values") or []),
            seller_keys=[str(k) for k in (a.get("seller_keys") or [])],
            range=[float(rng[0]), float(rng[1])] if rng else None,
        ))
    return CategorySchema(
        category=str(data.get("category") or fallback_code),
        name_ru=str(data.get("name_ru") or data.get("category") or fallback_code),
        keywords=[str(k) for k in (data.get("keywords") or [])],
        attributes=attrs,
    )


def load_schemas(config_dir: Path | str = CONFIG_DIR / "categories") -> dict[str, CategorySchema]:
    """Загрузить все утверждённые схемы (*.yaml, кроме *.draft.yaml). Ключ — код категории; включает '_common'.

    Результат кэшируется по (имя, mtime, размер) файлов: повторный вызов не читает YAML,
    а утверждённая менеджером схема подхватывается автоматически."""
    d = Path(config_dir)
    files = _schema_files(d)
    sig = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    key = str(d.resolve())
    hit = _SCHEMA_CACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    schemas: dict[str, CategorySchema] = {}
    for p in files:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        schema = _parse_schema(data, p.stem)
        schemas[schema.category] = schema
    _SCHEMA_CACHE[key] = (sig, schemas)
    return schemas


# ---------------------------------------------------------------- текст

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Нормализация для поиска: lower, ё→е, схлопывание пробелов. Длина может меняться — не для evidence."""
    return _WS.sub(" ", (text or "").lower().replace("ё", "е")).strip()


def text_hash(card: Card) -> str:
    """sha256 от title + description + seller_attributes (INTENT 4.1, идемпотентность)."""
    payload = json.dumps([card.title, card.description, card.seller_attributes], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- категория

_KW_CACHE: dict[str, re.Pattern] = {}


def _kw_pattern(kw: str) -> re.Pattern:
    p = _KW_CACHE.get(kw)
    if p is None:
        # ключевое слово — начало слова (допускает окончания: «ноутбуки», «телефона»)
        p = re.compile(r"(?<![^\W\d_])" + re.escape(clean_text(kw)).replace(r"\ ", r"\s+"))
        _KW_CACHE[kw] = p
    return p


def _score_text(text: str, schema: CategorySchema) -> tuple[int, int, int]:
    """(вес попаданий, число разных ключевых слов, позиция первого попадания)."""
    weight, distinct, first = 0, 0, 10 ** 9
    for i, kw in enumerate(schema.keywords):
        m = _kw_pattern(kw).search(text)
        if m:
            distinct += 1
            weight += 2 if i == 0 else 1   # первое слово списка — название категории, весит больше
            first = min(first, m.start())
    return weight, distinct, first


def detect_category(card: Card, schemas: dict[str, CategorySchema]) -> tuple[str, float]:
    """category_hint, если он есть в схемах; иначе классификатор по title. Не уверены → (UNKNOWN_CATEGORY, conf)."""
    hint = (card.category_hint or "").strip()
    if hint and hint != COMMON and hint in schemas:
        return hint, 1.0
    cats = {c: s for c, s in schemas.items() if c != COMMON and s.keywords}
    title = clean_text(card.title)
    scored = {c: _score_text(title, s) for c, s in cats.items()}
    scored = {c: v for c, v in scored.items() if v[0] > 0}
    if scored:
        order = sorted(scored, key=lambda c: (-scored[c][0], scored[c][2]))
        conf = 0.9 if len(order) == 1 else 0.7
        return order[0], conf
    # по описанию — осторожнее: нужны минимум два разных ключевых слова (в описании часто «управление со смартфона»)
    desc = clean_text(card.description)
    scored = {c: _score_text(desc, s) for c, s in cats.items()}
    scored = {c: v for c, v in scored.items() if v[1] >= 2}
    if scored:
        order = sorted(scored, key=lambda c: (-scored[c][0], scored[c][2]))
        conf = 0.6 if len(order) == 1 else 0.5
        return order[0], conf
    return UNKNOWN_CATEGORY, 0.3


def prepare(card: Card, schemas: dict[str, CategorySchema]) -> PreparedCard:
    category, conf = detect_category(card, schemas)
    if category == UNKNOWN_CATEGORY:
        schema = schemas.get(COMMON) or CategorySchema(COMMON, "Общие атрибуты", [], [])
    else:
        schema = schemas[category]
    texts = source_texts(card)
    return PreparedCard(
        card=card,
        category=category,
        category_confidence=conf,
        schema=schema,
        texts=texts,
        clean={k: clean_text(v) for k, v in texts.items()},
        text_hash=text_hash(card),
    )


__all__ = ["AttributeSpec", "CategorySchema", "PreparedCard", "UNKNOWN_CATEGORY", "load_schemas",
           "clean_text", "text_hash", "detect_category", "prepare", "source_texts"]
