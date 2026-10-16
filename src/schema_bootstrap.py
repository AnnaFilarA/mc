"""Черновик схемы новой категории (INTENT 4.2, SPEC F15). Владелец — поток E.

N карточек из очереди «новая категория» → один вызов LLM → config/categories/<new>.draft.yaml.
Менеджер утверждает: правит и переименовывает в .yaml (draft-файлы load_schemas не загружает).
Без LLM (NullLLM, нет ключа, невалидный ответ) строится эвристический черновик: draft_source: heuristic.

CLI:
    python -m src.schema_bootstrap --queue data/new_category_queue.jsonl --category tablets
    python -m src.schema_bootstrap --queue data/new_category_queue.jsonl --list
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

from src.contracts import Card
from src.llm import NullLLM, load_llm

log = logging.getLogger("attrguard.schema_bootstrap")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUEUE = ROOT / "data" / "new_category_queue.jsonl"
DEFAULT_CONFIG_DIR = "config/categories"
DEFAULT_MIN_CARDS = 20           # INTENT 10, вопрос 8; переопределяется llm.yaml → schema_bootstrap.min_cards
MAX_CARDS_IN_PROMPT = 30         # дороже не нужно: схему видно по 20-30 карточкам
TITLE_CHARS, DESC_CHARS = 200, 300
MAX_ATTRIBUTES = 12
ATTR_TYPES = ("enum", "number", "text")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_CATEGORY_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,40}$")

# JSON-схема ответа LLM — формат схемы категории (как config/categories/smartphones.yaml).
SCHEMA_JSON: dict = {
    "type": "object",
    "properties": {
        "category": {"type": "string"},
        "name_ru": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "name_ru": {"type": "string"},
                    "type": {"type": "string", "enum": list(ATTR_TYPES)},
                    "required": {"type": "boolean"},
                    "unit": {"type": ["string", "null"]},
                    "values": {"type": "array", "items": {"type": ["string", "number"]}},
                    "seller_keys": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code", "name_ru", "type", "required"],
            },
        },
    },
    "required": ["category", "name_ru", "keywords", "attributes"],
}

_STOP = {  # слова, которые не годятся в keywords категории
    "для", "что", "как", "это", "или", "при", "под", "над", "без", "все", "его", "ее", "ещё", "еще", "уже", "его",
    "купить", "цена", "новый", "новая", "новое", "модель", "товар", "цвет", "белый", "черный", "серый", "синий",
    "красный", "зеленый", "розовый", "золотой", "серебристый", "the", "and", "with", "for", "pro", "max", "plus",
}
_COLORS = ["Черный", "Белый", "Серый", "Серебристый", "Синий", "Голубой", "Зеленый", "Красный", "Золотой",
           "Розовый", "Фиолетовый", "Бежевый", "Желтый"]   # как в _common.yaml
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")


# ───────────────────────── очередь «новая категория» ─────────────────────────

class NewCategoryQueue:
    """Очередь карточек неизвестных категорий: JSONL, одна строка = одна карточка (append-only).

    Повторы по (category, product_id) при чтении схлопываются (берётся последняя запись)."""

    def __init__(self, path: Path | str = DEFAULT_QUEUE):
        self.path = Path(path)

    def add(self, category: str, card: Card) -> None:
        row = {"category": category, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.model_dump()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                row["card"]["product_id"], row["category"]
            except (ValueError, KeyError, TypeError):
                continue
            rows.append(row)
        return rows

    def cards(self, category: str) -> list[Card]:
        by_id: dict[str, Card] = {}
        for row in self._rows():
            if row["category"] == category:
                by_id[row["card"]["product_id"]] = Card(**row["card"])
        return list(by_id.values())

    def count(self, category: str) -> int:
        return len(self.cards(category))

    def counts(self) -> dict[str, int]:
        cats = {row["category"] for row in self._rows()}
        return {c: self.count(c) for c in sorted(cats)}

    def ready(self, category: str, n: int | None = None) -> bool:
        """Накопилось ли хотя бы n (по умолчанию schema_bootstrap.min_cards) карточек категории."""
        return self.count(category) >= (n if n is not None else _min_cards())


def _min_cards() -> int:
    try:
        cfg = yaml.safe_load((ROOT / "config" / "llm.yaml").read_text(encoding="utf-8")) or {}
        return int((cfg.get("schema_bootstrap") or {}).get("min_cards", DEFAULT_MIN_CARDS))
    except (OSError, ValueError, yaml.YAMLError):
        return DEFAULT_MIN_CARDS


# ───────────────────────── валидация ответа ─────────────────────────

def _str_list(x) -> list[str]:
    if not isinstance(x, list):
        return []
    out: list[str] = []
    for v in x:
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return out


def validate_schema(raw: dict, category: str) -> dict:
    """Привести ответ LLM к формату схемы категории. Пригодных атрибутов нет → ValueError."""
    if not isinstance(raw, dict) or not isinstance(raw.get("attributes"), list):
        raise ValueError("ответ не похож на схему категории")
    attrs: list[dict] = []
    seen: set[str] = set()
    for a in raw["attributes"]:
        if not isinstance(a, dict):
            continue
        code = str(a.get("code", "")).strip().lower().replace("-", "_").replace(" ", "_")
        name = str(a.get("name_ru", "")).strip()
        if not _CODE_RE.match(code) or not name or code in seen:
            continue
        typ = a.get("type") if a.get("type") in ATTR_TYPES else "text"
        values = _str_list(a.get("values"))
        unit = a.get("unit")
        unit = str(unit).strip() if unit not in (None, "") else None
        if typ == "number":
            values = [v.replace(",", ".") for v in values if re.fullmatch(r"\d+(?:[.,]\d+)?", v)]
        if typ == "enum" and not values:
            typ = "text"   # перечисление без справочника бесполезно
        item: dict = {"code": code, "name_ru": name, "type": typ, "required": bool(a.get("required", False))}
        if typ == "number" and unit:
            item["unit"] = unit
        item["seller_keys"] = _str_list(a.get("seller_keys")) or [code, name.lower()]
        if typ in ("enum", "number") and values:
            item["values"] = [float(v) if "." in v else int(v) for v in values] if typ == "number" else values
        attrs.append(item)
        seen.add(code)
        if len(attrs) >= MAX_ATTRIBUTES:
            break
    if not attrs:
        raise ValueError("в ответе нет пригодных атрибутов")
    return {
        "category": category,
        "name_ru": str(raw.get("name_ru") or category).strip(),
        "keywords": _str_list(raw.get("keywords"))[:12],
        "attributes": attrs,
    }


# ───────────────────────── эвристика без LLM ─────────────────────────

def heuristic_schema(cards: list[Card], category: str) -> dict:
    """Черновик без LLM: частые слова title → keywords; общие атрибуты brand/color."""
    df: Counter[str] = Counter()
    brand_df: Counter[str] = Counter()
    for c in cards:
        words = _WORD_RE.findall(c.title)
        df.update({w.lower().replace("ё", "е") for w in words})
        brand_df.update({w for w in words if w[0].isupper() and w.isascii() and w.lower() not in _STOP})
    keywords = [w for w, n in df.most_common() if n >= 2 and w not in _STOP][:8]
    brands = [w for w, n in brand_df.most_common(12) if n >= 2]
    brand = {"code": "brand", "name_ru": "Бренд", "type": "enum" if brands else "text", "required": True,
             "seller_keys": ["brand", "бренд", "производитель", "марка"]}
    if brands:
        brand["values"] = brands
    color = {"code": "color", "name_ru": "Цвет", "type": "enum", "required": True,
             "seller_keys": ["color", "цвет", "цвет корпуса"], "values": list(_COLORS)}
    return {"category": category, "name_ru": category.replace("_", " ").capitalize(),
            "keywords": keywords, "attributes": [brand, color]}


# ───────────────────────── основной вызов ─────────────────────────

def _truncate(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def build_prompt(cards: list[Card], category: str, click_stats: dict | None) -> str:
    lines = [
        f"Ты проектируешь схему характеристик для новой категории интернет-магазина электроники: «{category}».",
        "Ниже карточки товаров этой категории. Предложи список характеристик, которые покупатели используют как "
        "фильтры: код (латиница, snake_case), название по-русски, тип (enum — выбор из справочника, number — число + "
        "единица, text), обязательность, единица (для number), допустимые значения (для enum), ключи, под которыми "
        "поставщики называют характеристику (seller_keys, в нижнем регистре), и ключевые слова категории (keywords). "
        f"Не более {MAX_ATTRIBUTES} характеристик. Используй только то, что видно в карточках.",
        "",
        "Карточки:",
    ]
    for i, c in enumerate(cards[:MAX_CARDS_IN_PROMPT], 1):
        lines.append(f"{i}. {_truncate(c.title, TITLE_CHARS)} | {_truncate(c.description, DESC_CHARS)}")
    if click_stats:
        top = sorted(click_stats.items(), key=lambda kv: (-float(kv[1]) if isinstance(kv[1], (int, float)) else 0))[:30]
        lines += ["", "Популярные поисковые запросы (запрос: клики):"]
        lines += [f"- {k}: {v}" for k, v in top]
    return "\n".join(lines)


def bootstrap_schema(cards: list[Card], category: str, llm=None, click_stats: dict | None = None,
                     min_cards: int | None = None) -> dict:
    """Черновик схемы категории по N карточкам. Меньше min_cards (по умолчанию 20) → ValueError.

    С LLM — один вызов complete_json, ответ валидируется; draft_source: llm. С NullLLM/без LLM/при невалидном ответе
    — эвристика, draft_source: heuristic."""
    need = min_cards if min_cards is not None else _min_cards()
    if len(cards) < need:
        raise ValueError(f"для категории {category!r} нужно минимум {need} карточек, есть {len(cards)}")
    if not _CATEGORY_RE.match(category):
        raise ValueError(f"код категории {category!r} должен быть латиницей: [a-z0-9_]")
    llm = llm if llm is not None else NullLLM()
    schema: dict | None = None
    if not isinstance(llm, NullLLM):
        try:
            schema = validate_schema(llm.complete_json(build_prompt(cards, category, click_stats), SCHEMA_JSON),
                                     category)
            schema["draft_source"] = "llm"
        except ValueError as e:
            log.warning("ответ LLM не годится как схема (%s) — эвристический черновик", e)
    if schema is None:
        schema = heuristic_schema(cards, category)
        schema["draft_source"] = "heuristic"
    schema["source_cards"] = len(cards)
    return schema


def save_draft(schema: dict, category: str, config_dir: str = DEFAULT_CONFIG_DIR) -> str:
    """Записать config/categories/<category>.draft.yaml; утверждённый <category>.yaml не трогается."""
    if not _CATEGORY_RE.match(category):
        raise ValueError(f"код категории {category!r} должен быть латиницей: [a-z0-9_]")
    d = Path(config_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{category}.draft.yaml"
    header = ("# ЧЕРНОВИК схемы категории — не загружается пайплайном, пока не утверждён.\n"
              f"# Проверьте и поправьте, затем переименуйте в {category}.yaml (и удалите draft_source/source_cards).\n")
    tmp = path.with_suffix(".yaml.tmp")
    tmp.write_text(header + yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    return str(path)


# ───────────────────────── CLI ─────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Черновик схемы новой категории из очереди карточек")
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE), help="JSONL-очередь (data/new_category_queue.jsonl)")
    ap.add_argument("--category", help="код категории; без него — все готовые категории очереди")
    ap.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--llm-config", default="config/llm.yaml")
    ap.add_argument("--min-cards", type=int, default=None, help="порог готовности (по умолчанию из llm.yaml)")
    ap.add_argument("--list", action="store_true", help="показать, сколько карточек накоплено по категориям")
    args = ap.parse_args(argv)

    queue = NewCategoryQueue(args.queue)
    need = args.min_cards if args.min_cards is not None else _min_cards()
    if args.list:
        for cat, n in queue.counts().items():
            print(f"{cat}: {n}/{need}{' — готово' if n >= need else ''}")
        return 0
    cats = [args.category] if args.category else [c for c in queue.counts() if queue.ready(c, need)]
    if not cats:
        print("Нет готовых категорий в очереди.")
        return 0
    llm = load_llm(args.llm_config)
    code = 0
    for cat in cats:
        if not queue.ready(cat, need):
            print(f"{cat}: накоплено {queue.count(cat)} из {need} — рано, жду ещё карточек.")
            code = 2
            continue
        schema = bootstrap_schema(queue.cards(cat), cat, llm, min_cards=need)
        print(f"{cat}: черновик ({schema['draft_source']}) -> {save_draft(schema, cat, args.config_dir)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
