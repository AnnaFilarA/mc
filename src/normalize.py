"""Нормализация к канону и проверка evidence (SPEC F4, F5). Владелец — поток B.

Канон: number — строка числа без единицы ("8", "6.67", 1 ТБ → "1024"), unit из схемы;
enum — значение из справочника схемы (values). Значение вне справочника не может быть proposed.

Здесь же живёт движок словарей синонимов (config/rules/<attr>.yaml), общий для rules и normalize.

Формат словаря (config/rules/<код атрибута>.yaml):
    синоним: Каноническое значение          # ключ — слово/фраза, регистр и ё/е не важны
    синоним: [Каноническое значение, -0.1]  # [значение, поправка к уверенности]
    _groups: {Каноническое: [синоним, ...]} # то же самое, сгруппировано по значению
    _models: {Бренд: [модель, ...]}         # модели → бренд, поправка -0.05
    _seller: {да: Да, нет: Нет}             # синонимы только для значений из seller_attributes
Спецсимволы в ключе: пробел = «пробел/дефис/ничего»; `*` = до 4 букв окончания, `**` = до 8;
цифра x цифра = разделитель размеров; префикс `re:` — сырая регулярка.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.contracts import Candidate, Card, evidence_is_valid
from src.prepare import CONFIG_DIR, AttributeSpec

RULES_DIR = CONFIG_DIR / "rules"

# ---------------------------------------------------------------- вспомогательное


def fold(s: object) -> str:
    """lower + ё→е + схлопывание пробелов (для сравнения значений)."""
    return re.sub(r"\s+", " ", str(s).lower().replace("ё", "е")).strip()


def format_number(x: float) -> str:
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


# ---------------------------------------------------------------- словари синонимов

_LETTER_B = r"(?<![^\W\d_])"   # не после буквы
_LETTER_A = r"(?![^\W\d_])"    # не перед буквой


def key_to_regex(key: str) -> str:
    key = key.strip()
    if key.startswith("re:"):
        return key[3:]
    out: list[str] = []
    i, n = 0, len(key)
    while i < n:
        ch = key[i]
        if ch == "*":
            if i + 1 < n and key[i + 1] == "*":
                out.append(r"\w{0,8}")
                i += 1
            else:
                out.append(r"\w{0,4}")
        elif ch.isspace():
            while i + 1 < n and key[i + 1].isspace():
                i += 1
            out.append(r"[\s\-]*")
        elif ch in "еЕёЁ":
            out.append("[её]")
        elif ch in "xX" and 0 < i < n - 1 and key[i - 1].isdigit() and key[i + 1].isdigit():
            out.append(r"\s*[xх×*]\s*")
        else:
            out.append(re.escape(ch))
        i += 1
    return "".join(out)


@dataclass
class _Compiled:
    regex: re.Pattern | None
    canon: list[str]
    delta: list[float]


@dataclass
class Match:
    start: int
    end: int
    canonical: str
    delta: float


class SynDict:
    """Словарь синонимов атрибута. find() возвращает непересекающиеся совпадения слева направо, длинные первыми."""

    def __init__(self, entries: list[tuple[str, str, float]], seller_entries: list[tuple[str, str, float]]):
        self.main = self._compile(entries)
        self.seller = self._compile(entries + seller_entries)
        self.empty = not entries and not seller_entries

    @staticmethod
    def _compile(entries: list[tuple[str, str, float]]) -> _Compiled:
        if not entries:
            return _Compiled(None, [], [])
        seen: dict[str, tuple[str, float]] = {}
        for key, canon, delta in entries:
            seen.setdefault(key.strip() if key.startswith("re:") else fold(key), (canon, delta))
        # порядок: длинные ключи первыми, чтобы «серый космос» побеждал «серый»
        items = sorted(seen.items(), key=lambda kv: -len(kv[0]))
        parts, canon, delta = [], [], []
        for k, (c, d) in items:
            parts.append("(" + key_to_regex(k) + ")")
            canon.append(c)
            delta.append(d)
        rx = re.compile(_LETTER_B + "(?:" + "|".join(parts) + ")" + _LETTER_A, re.IGNORECASE)
        return _Compiled(rx, canon, delta)

    def find(self, text: str, seller: bool = False) -> list[Match]:
        comp = self.seller if seller else self.main
        if comp.regex is None or not text:
            return []
        out = []
        for m in comp.regex.finditer(text):
            idx = (m.lastindex or 1) - 1
            out.append(Match(m.start(), m.end(), comp.canon[idx], comp.delta[idx]))
        return out

    def fullmatch(self, text: str, seller: bool = True) -> str | None:
        t = text.strip()
        for m in self.find(t, seller=seller):
            if m.start == 0 and m.end == len(t):
                return m.canonical
        return None


def _load_entries(code: str) -> tuple[list, list]:
    path = RULES_DIR / f"{code}.yaml"
    entries: list[tuple[str, str, float]] = []
    seller: list[tuple[str, str, float]] = []
    if not path.exists():
        return entries, seller
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def add(target, key, val, delta0=0.0):
        if isinstance(val, (list, tuple)):
            canon, delta = str(val[0]), float(val[1]) if len(val) > 1 else delta0
        else:
            canon, delta = str(val), delta0
        target.append((str(key), canon, delta))

    for key, val in data.items():
        if key == "_groups":
            for canon, aliases in (val or {}).items():
                for a in aliases:
                    add(entries, a, canon)
        elif key == "_models":
            for canon, aliases in (val or {}).items():
                for a in aliases:
                    add(entries, a, canon, -0.05)
        elif key == "_seller":
            for a, canon in (val or {}).items():
                add(seller, a, canon)
        elif not str(key).startswith("_"):
            add(entries, key, val)
    return entries, seller


_DICT_CACHE: dict[tuple, SynDict] = {}
_NO_SELF = {"да", "нет"}


def get_dictionary(spec: AttributeSpec) -> SynDict:
    """Словарь атрибута: config/rules/<code>.yaml + значения справочника схемы как самосинонимы."""
    key = (spec.code, tuple(str(v) for v in spec.values))
    d = _DICT_CACHE.get(key)
    if d is not None:
        return d
    entries, seller = _load_entries(spec.code)
    values = [str(v) for v in spec.values]
    if spec.type == "enum" and not ({fold(v) for v in values} <= _NO_SELF):
        have = {fold(k) for k, _, _ in entries}
        for v in values:
            if fold(v) not in have and len(v) >= 2:
                entries.append((v, v, 0.0))
    d = SynDict(entries, seller)
    _DICT_CACHE[key] = d
    return d


def clear_caches() -> None:
    _DICT_CACHE.clear()


# ---------------------------------------------------------------- числа

_NUM = re.compile(r"(\d+(?:[.,]\d+)?)\s*([^\W\d_]+|[\"”″“']{1,2})?")
_GB_MULT = [(("тб", "tb", "терабайт"), 1024.0), (("гб", "gb", "гиг", "гига"), 1.0), (("мб", "mb", "мегабайт"), 1 / 1024)]
_INCH = ("дюйм", "inch", "\"", "”", "″", "“", "''", "'", "in")
_HZ = ("гц", "hz", "герц")


def _unit_factor(spec_unit: str | None, token: str | None) -> float | None:
    """Множитель к единице схемы; None — единица чужая, привести нельзя."""
    t = fold(token or "")
    if not t:
        return 1.0
    if spec_unit == "GB":
        for names, mult in _GB_MULT:
            if any(t.startswith(n) for n in names):
                return mult
        return None
    if spec_unit == "inch":
        if t.startswith(("см", "cm")):
            return 1 / 2.54
        return 1.0 if any(t.startswith(n) for n in _INCH) else None
    if spec_unit == "Hz":
        return 1.0 if any(t.startswith(n) for n in _HZ) else None
    return 1.0 if fold(spec_unit or "") == t or not spec_unit else None


def _normalize_number(src: str, spec: AttributeSpec) -> str | None:
    m = _NUM.search(src)
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    k = _unit_factor(spec.unit, m.group(2))
    if k is None:
        return None
    x = x * k
    if k == 1 / 2.54:
        x = round(x, 1)
    x = round(x, 2)
    if spec.range and not (spec.range[0] <= x <= spec.range[1]):
        return None
    if spec.values:
        try:
            ok = any(abs(float(v) - x) < 1e-6 for v in spec.values)
        except (TypeError, ValueError):
            ok = True
        if not ok:
            return None
    return format_number(x)


def _normalize_enum(src: str, spec: AttributeSpec) -> str | None:
    f = fold(src)
    by_fold = {fold(v): str(v) for v in spec.values}
    if f in by_fold:
        return by_fold[f]
    canon = get_dictionary(spec).fullmatch(src, seller=True)
    if canon is not None:
        return by_fold.get(fold(canon))
    return None


def normalize(candidate: Candidate, spec: AttributeSpec) -> Candidate:
    """Вернуть копию кандидата с value/unit в каноне; value=None, если привести нельзя.

    Источник значения — candidate.value (если этап уже предложил канон), иначе raw_value.
    Любое значение проверяется по справочнику / диапазону схемы."""
    src = candidate.value if candidate.value is not None else candidate.raw_value
    value: str | None = None
    unit: str | None = None
    if src is not None and str(src).strip():
        src = str(src)
        if spec.type == "number":
            value = _normalize_number(src, spec)
            if value is None and candidate.value is not None and candidate.raw_value:
                value = _normalize_number(candidate.raw_value, spec)
            unit = spec.unit if value is not None else None
        elif spec.type == "enum":
            value = _normalize_enum(src, spec)
            if value is None and candidate.value is not None and candidate.raw_value:
                value = _normalize_enum(candidate.raw_value, spec)
        else:
            value = src.strip()
    return candidate.model_copy(update={"value": value, "unit": unit})


def check_evidence(card: Card, candidate: Candidate) -> bool:
    """True, если у кандидата есть evidence и он реально есть в исходном тексте."""
    return evidence_is_valid(card, candidate.evidence)
