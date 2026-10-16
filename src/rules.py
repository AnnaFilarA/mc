"""Этап 1: правила и словари (SPEC 3.5). Владелец — поток B.

Каждое правило возвращает Candidate с evidence (span в исходном тексте источника).
Уверенность: точное совпадение со словарём 0.95; регулярка 0.85; нечёткое совпадение 0.6.
Поправки за контекст: подпись атрибута рядом (+), неоднозначность (−), описание вместо названия (−).
Словари синонимов — config/rules/*.yaml (движок — src/normalize.py).

Правила работают по ИСХОДНОМУ тексту источника (prepared.texts), не по очищенному, чтобы span совпал.
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.contracts import Candidate, Evidence
from src.normalize import fold, get_dictionary, key_to_regex, normalize
from src.prepare import AttributeSpec, PreparedCard

SELLER = "seller_attributes"
TEXT_SOURCES = ("title", "description")

S_DICT = 0.95      # точное совпадение со словарём
S_REGEX = 0.85     # регулярка «число + единица»
S_LABELED = 0.88   # число с подписью атрибута рядом
S_SELLER = 0.90    # значение из поля продавца (ключ сопоставлен по seller_keys)
S_WEAK = 0.65      # неоднозначная догадка (ниже порога proposed)
AMBIGUITY_PENALTY = 0.25

_LB = r"(?<![^\W\d_])"   # не после буквы
_LA = r"(?![^\W\d_])"    # не перед буквой


def _cand(attr: str, raw: str, score: float, source: str, text: str, start: int, end: int,
          value: str | None = None) -> Candidate:
    return Candidate(
        attribute=attr, raw_value=raw, value=value,
        score=round(max(0.0, min(score, 0.98)), 2), stage="rules",
        evidence=Evidence(text=text[start:end], start=start, end=end, source=source),  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------- подписи атрибутов

_LABEL_CACHE: dict[tuple, re.Pattern] = {}


def _label_re(spec: AttributeSpec) -> re.Pattern:
    key = (spec.code, spec.name_ru, tuple(spec.seller_keys))
    rx = _LABEL_CACHE.get(key)
    if rx is None:
        labels = {fold(x) for x in (spec.code, spec.name_ru, *spec.seller_keys) if len(str(x)) >= 2}
        parts = sorted((key_to_regex(l) for l in labels), key=len, reverse=True)
        rx = re.compile(_LB + "(?:" + "|".join(parts) + r")\w*[\s:=\-–—]*$", re.IGNORECASE)
        _LABEL_CACHE[key] = rx
    return rx


# ---------------------------------------------------------------- словарные атрибуты (enum)

_FOR_CTX = re.compile(r"(?:для|совместим\w*\s+с|подходит\s+(?:для|к)|for|with|compatible\s+with)\s*$", re.IGNORECASE)


def _dict_candidates(spec: AttributeSpec, source: str, text: str) -> list[Candidate]:
    sd = get_dictionary(spec)
    if sd.empty:
        return []
    label = _label_re(spec)
    out = []
    for m in sd.find(text):
        left = text[max(0, m.start - 40):m.start]
        if spec.code == "brand" and _FOR_CTX.search(left):
            continue   # «чехол для iPhone» — это не бренд товара
        score = S_DICT + m.delta
        if label.search(left):
            score += 0.03
        elif source == "description":
            score -= 0.05
        out.append(_cand(spec.code, text[m.start:m.end], score, source, text, m.start, m.end, value=m.canonical))
    return out


# ---------------------------------------------------------------- память: ОЗУ / накопитель

_UNIT = r"(?:тб|tb|гб|gb|гиг\w*|терабайт\w*)"
_QTY = re.compile(rf"(?<![\w.,])(\d{{1,4}}(?:[.,]\d)?)\s*({_UNIT})" + _LA, re.IGNORECASE)
_PAIR = re.compile(rf"(?<![\w.,])(\d{{1,2}})\s*(?:гб|gb)?\s*[/+]\s*(\d{{1,4}}(?:[.,]\d)?)\s*(тб|tb|гб|gb)" + _LA, re.IGNORECASE)

_RAM_L = r"(?:озу|ram|оперативн\w*(?:\s+памят\w*)?|ddr\d?\w*|lpddr\d?\w*)"
_STO_L = (r"(?:ssd|hdd|emmc|e-mmc|nvme|накопител\w*|встроенн\w*(?:\s+памят\w*)?|пзу|rom|storage|"
          r"памят\w*|диск\w*|хранени\w*)")
_R_RAM = re.compile(rf"\s*[-–—]?\s*{_RAM_L}" + _LA, re.IGNORECASE)
_R_STO = re.compile(rf"\s*[-–—]?\s*{_STO_L}" + _LA, re.IGNORECASE)
_R_EX = re.compile(r"\s*(?:gddr\w*|видеопамят\w*|vram)", re.IGNORECASE)
_L_RAM = re.compile(_LB + rf"{_RAM_L}[\s:=\-–—]*$", re.IGNORECASE)
_L_STO = re.compile(_LB + rf"{_STO_L}[\s:=\-–—]*$", re.IGNORECASE)
# видеокарта/видеопамять и карты памяти — не ОЗУ и не накопитель
_L_EX = re.compile(r"rtx|gtx|geforce|radeon|\brx\s?\d|quadro|видеокарт\w*|видеопамят\w*|gddr|vram|"
                   r"micro\s*sd|microsd|sdxc|sdhc|\bsd\b|карт\w*\s+памят\w*|расширен\w*|слот\w*|кэш\w*|cache|"
                   r"графическ\w*", re.IGNORECASE)
_DELIM = re.compile(r"[,;()|\n]|\.(?=\s)|\s/\s")
_ADJ_GAP = re.compile(r"^[\s,;/+\-–—]{0,3}$")


def _gb(num: str, unit: str) -> float:
    x = float(num.replace(",", "."))
    return x * 1024 if unit.lower().startswith(("тб", "tb", "терабайт")) else x


def _memory(source: str, text: str, want_ram: bool, want_sto: bool) -> list[Candidate]:
    out: list[Candidate] = []
    consumed: list[tuple[int, int]] = []

    # «8/256GB», «12/1TB», «16GB/512GB», «8Gb/256Gb»: первое число — ОЗУ, второе — накопитель (DECISIONS)
    for m in _PAIR.finditer(text):
        a, b = float(m.group(1)), _gb(m.group(2), m.group(3))
        if not (1 <= a <= 64 and b >= 16 and a < b):
            continue
        consumed.append(m.span())
        if want_ram:
            out.append(_cand("ram", m.group(1), S_LABELED, source, text, m.start(), m.end()))
        if want_sto:
            out.append(_cand("storage", f"{m.group(2)} {m.group(3)}", S_LABELED, source, text, m.start(), m.end()))

    qs = [m for m in _QTY.finditer(text) if not any(s <= m.start() < e for s, e in consumed)]
    info = []
    prev_end = 0
    for m in qs:
        r_ram, r_sto = _R_RAM.match(text, m.end()), _R_STO.match(text, m.end())
        r_ex = _R_EX.match(text, m.end())
        right = r_ram or r_sto
        full_end = right.end() if right else m.end()
        win = max(prev_end, m.start() - 60)
        last = None
        for d in _DELIM.finditer(text, win, m.start()):
            last = d
        seg_start = last.end() if last else win
        seg = text[seg_start:m.start()]
        kind, start, end = None, m.start(), m.end()
        if _L_EX.search(seg) or r_ex:
            kind = "skip"
        elif (lm := _L_RAM.search(seg)):
            kind, start = "ram", seg_start + lm.start()
        elif (lm := _L_STO.search(seg)):
            kind, start = "storage", seg_start + lm.start()
        elif r_ram:
            kind, end = "ram", r_ram.end()
        elif r_sto:
            kind, end = "storage", r_sto.end()
        info.append({"m": m, "kind": kind, "start": start, "end": end, "full_end": full_end,
                     "gb": _gb(m.group(1), m.group(2))})
        prev_end = full_end

    for i, q in enumerate(info):
        m, kind = q["m"], q["kind"]
        if kind == "skip":
            continue
        score = S_LABELED
        if kind is None:
            nxt = info[i + 1] if i + 1 < len(info) else None
            prv = info[i - 1] if i > 0 else None
            adj_next = bool(nxt and nxt["kind"] != "skip" and _ADJ_GAP.match(text[q["full_end"]:nxt["m"].start()])
                            and q["gb"] < nxt["gb"])
            adj_prev = bool(prv and prv["kind"] != "skip" and _ADJ_GAP.match(text[prv["full_end"]:m.start()])
                            and prv["gb"] < q["gb"])
            if adj_next:
                kind, score = "ram", 0.82           # «8 ГБ 512 ГБ SSD»: меньшее подряд — ОЗУ
            elif adj_prev:
                kind, score = "storage", 0.82
            elif q["gb"] >= 128:
                kind, score = "storage", 0.82       # одиночное крупное значение без подписи — накопитель
            elif q["gb"] >= 32:
                kind, score = "storage", 0.7        # 32/64 ГБ: бывает и ОЗУ ноутбука
            else:
                kind, score = "ram", S_WEAK         # одиночное «8 ГБ» без подписи — неоднозначно
        if (kind == "ram" and not want_ram) or (kind == "storage" and not want_sto):
            continue
        out.append(_cand(kind, text[m.start():m.end()], score, source, text, q["start"], q["end"]))
    return out


# ---------------------------------------------------------------- диагональ, частота

_QUOTE = r"(?:[\"”″“]|''|’’|´´)"
_SCR_UNIT = re.compile(rf"(?<![\w.,])(\d{{1,3}}(?:[.,]\d{{1,2}})?)\s*(?:{_QUOTE}|дюйм\w*|inch\w*)", re.IGNORECASE)
_SCR_LABEL = re.compile(
    r"(?:диагонал\w*|экран\w*|дисплей\w*|screen|display)(?:\s+(?:экрана|дисплея))?\s*[:\-–]?\s*"
    r"(\d{1,3}(?:[.,]\d{1,2})?)(?![\d])(?![.,]\d)"
    r"(?!\s*(?:см|cm|мм|mm|гц|hz|гб|gb|mah|мач|мп|mp|px|пикс|nit|нит|ppi|%))", re.IGNORECASE)
_SCR_FHD = re.compile(r"(?<![\w.,])(\d{2}[.,]\d)\s*(?:fhd|full\s*hd|hd|ips|oled|qhd|wqhd|uhd|wuxga|2k|4k|retina)" + _LA,
                      re.IGNORECASE)
_SCR_NEAR = re.compile(r"(?:диагонал\w*|экран\w*|дисплей\w*|screen|display)[\s:\-–]*$", re.IGNORECASE)


def _screen(source: str, text: str) -> list[Candidate]:
    out = []
    for m in _SCR_UNIT.finditer(text):
        near = bool(_SCR_NEAR.search(text[max(0, m.start() - 20):m.start()]))
        out.append(_cand("screen_size", m.group(1), S_LABELED + (0.04 if near else 0), source, text, m.start(), m.end()))
    for m in _SCR_LABEL.finditer(text):
        out.append(_cand("screen_size", m.group(1), S_REGEX, source, text, m.start(), m.end()))
    for m in _SCR_FHD.finditer(text):
        out.append(_cand("screen_size", m.group(1), 0.72, source, text, m.start(1), m.end()))   # «15.6 FHD» — догадка
    return out


_HZ = re.compile(r"(?<![\w.,])(\d{2,3})\s*(?:гц|hz)" + _LA, re.IGNORECASE)
_HZ_LABEL = re.compile(r"частот\w*[^.,;]{0,25}$", re.IGNORECASE)
_HZ_EX = re.compile(r"сет[иь]|питани|напряжен", re.IGNORECASE)


def _refresh(source: str, text: str) -> list[Candidate]:
    out = []
    for m in _HZ.finditer(text):
        left = text[max(0, m.start() - 30):m.start()]
        if _HZ_EX.search(left):
            continue
        score = 0.92 if _HZ_LABEL.search(left) else S_REGEX
        out.append(_cand("refresh_rate", m.group(1), score, source, text, m.start(), m.end()))
    return out


def _generic_number(spec: AttributeSpec, source: str, text: str) -> list[Candidate]:
    """Числовой атрибут без особых правил: «<подпись> <число> [единица]»."""
    labels = {fold(x) for x in (spec.name_ru, *spec.seller_keys) if len(str(x)) >= 3}
    if not labels:
        return []
    unit = f"(?:\\s*{re.escape(spec.unit)})?" if spec.unit else ""
    rx = re.compile(_LB + "(?:" + "|".join(key_to_regex(l) for l in labels) + r")\w*[\s:=\-–—]*(\d+(?:[.,]\d+)?)" + unit
                    + _LA, re.IGNORECASE)
    return [_cand(spec.code, m.group(1), S_REGEX, source, text, m.start(), m.end()) for m in rx.finditer(text)]


# ---------------------------------------------------------------- поля продавца

def _key_norm(key: str) -> str:
    k = fold(key)
    k = re.sub(r"\(.*?\)", "", k)
    k = k.split(",")[0]
    return k.strip(" :.")


def _seller_lines(prepared: PreparedCard) -> list[tuple[str, str, int]]:
    """(ключ, значение, позиция значения в строке seller_attributes_text)."""
    out, pos = [], 0
    for k, v in prepared.card.seller_attributes.items():
        v = "" if v is None else str(v)
        out.append((str(k), v, pos + len(str(k)) + 2))
        pos += len(str(k)) + 2 + len(v) + 1
    return out


def _unit_re(unit: str | None) -> str:
    return {"GB": _UNIT, "inch": rf"(?:{_QUOTE}|дюйм\w*|inch\w*)", "Hz": r"(?:гц|hz)"}.get(unit or "", "")


def seller_key_map(schema_attrs: list[AttributeSpec]) -> dict[str, AttributeSpec]:
    """Нормализованное имя поля продавца → атрибут схемы (seller_keys, имя, код)."""
    mapping: dict[str, AttributeSpec] = {}
    for spec in schema_attrs:
        for k in (spec.code, spec.name_ru, *spec.seller_keys):
            mapping.setdefault(_key_norm(k), spec)
    return mapping


def seller_values(prepared: PreparedCard) -> dict[str, str]:
    """Код атрибута → сырое значение продавца (первое подходящее поле), для поля `was`."""
    km = seller_key_map(prepared.schema.attributes)
    out: dict[str, str] = {}
    for k, v, _ in _seller_lines(prepared):
        spec = km.get(_key_norm(k))
        if spec and v.strip():
            out.setdefault(spec.code, v)
    return out


def _seller_candidates(prepared: PreparedCard) -> list[Candidate]:
    text = prepared.texts[SELLER]
    km = seller_key_map(prepared.schema.attributes)
    out: list[Candidate] = []
    for k, v, vstart in _seller_lines(prepared):
        spec = km.get(_key_norm(k))
        if spec is None or not v.strip():
            continue
        if spec.type == "enum":
            for m in get_dictionary(spec).find(v, seller=True):
                out.append(_cand(spec.code, v[m.start:m.end], S_SELLER + m.delta, SELLER, text,
                                 vstart + m.start, vstart + m.end, value=m.canonical))
        elif spec.type == "number":
            unit = _unit_re(spec.unit)
            rx = re.compile(r"(\d+(?:[.,]\d+)?)" + (rf"(?:\s*{unit})?" if unit else ""), re.IGNORECASE)
            m = rx.search(v)
            if m:
                out.append(_cand(spec.code, m.group(0), S_SELLER, SELLER, text, vstart + m.start(), vstart + m.end()))
    return out


# ---------------------------------------------------------------- постобработка

def _dedupe_overlaps(cands: list[Candidate]) -> list[Candidate]:
    """Перекрывающиеся spans одного атрибута и источника: оставить кандидата с большей уверенностью."""
    groups: dict[tuple, list[Candidate]] = defaultdict(list)
    for c in cands:
        groups[(c.attribute, c.evidence.source)].append(c)  # type: ignore[union-attr]
    out = []
    for items in groups.values():
        items.sort(key=lambda c: -c.score)
        kept: list[Candidate] = []
        for c in items:
            e = c.evidence
            if any(e.start < k.evidence.end and k.evidence.start < e.end for k in kept):  # type: ignore[union-attr]
                continue
            kept.append(c)
        out.extend(kept)
    return out


def _penalize_ambiguity(cands: list[Candidate], specs: dict[str, AttributeSpec]) -> list[Candidate]:
    """В одном источнике несколько разных значений атрибута → уверенность каждого падает (не угадываем)."""
    keyed = []
    for c in cands:
        spec = specs.get(c.attribute)
        n = normalize(c, spec) if spec else c
        keyed.append((c, n.value))
    values: dict[tuple, set] = defaultdict(set)
    for c, v in keyed:
        if v is not None:
            values[(c.attribute, c.evidence.source)].add(v)   # type: ignore[union-attr]
    out = []
    for c, v in keyed:
        if v is not None and len(values[(c.attribute, c.evidence.source)]) > 1:   # type: ignore[union-attr]
            c = c.model_copy(update={"score": round(max(0.0, c.score - AMBIGUITY_PENALTY), 2)})
        out.append(c)
    return out


def _drop_description_brand(cands: list[Candidate], specs: dict[str, AttributeSpec]) -> list[Candidate]:
    """Бренд назван в названии → упоминания других брендов в описании («совместимо с…») игнорируем."""
    spec = specs.get("brand")
    if spec is None:
        return cands
    title_has = any(c.attribute == "brand" and c.evidence.source == "title"  # type: ignore[union-attr]
                    and normalize(c, spec).value is not None for c in cands)
    if not title_has:
        return cands
    return [c for c in cands if not (c.attribute == "brand" and c.evidence.source == "description")]  # type: ignore[union-attr]


_SPECIAL_NUMBERS = {"ram", "storage", "screen_size", "refresh_rate"}


def extract(prepared: PreparedCard) -> list[Candidate]:
    """Кандидаты для всех атрибутов схемы категории (несколько на атрибут допустимо)."""
    schema = prepared.schema
    specs = {a.code: a for a in schema.attributes}
    cands: list[Candidate] = []
    for source in TEXT_SOURCES:
        text = prepared.texts.get(source) or ""
        if not text.strip():
            continue
        for spec in schema.attributes:
            if spec.type == "enum":
                cands += _dict_candidates(spec, source, text)
            elif spec.type == "number" and spec.code not in _SPECIAL_NUMBERS:
                cands += _generic_number(spec, source, text)
        if "ram" in specs or "storage" in specs:
            cands += _memory(source, text, "ram" in specs, "storage" in specs)
        if "screen_size" in specs:
            cands += _screen(source, text)
        if "refresh_rate" in specs:
            cands += _refresh(source, text)
    cands += _seller_candidates(prepared)
    cands = _drop_description_brand(cands, specs)
    cands = _dedupe_overlaps(cands)
    cands = _penalize_ambiguity(cands, specs)
    return cands
