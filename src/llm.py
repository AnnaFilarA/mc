"""Этап 3: LLM (SPEC 3.5). По умолчанию выключен. Владелец — поток E.

Правила (INTENT 7, SPEC N4/N8/F9):
- ключ только из переменной окружения LLM_API_KEY, в коде и конфигах секретов нет;
- в промпте только спорные атрибуты, их допустимые значения и релевантные куски текста (±context_chars);
- ответ — строгий JSON; значение без цитаты, найденной в исходном тексте, отбрасывается без исключений;
- кэш ответов (JSONL) и лимит реальных вызовов на прогон (max_calls_per_run);
- статус LLM-значения в каскаде максимум low_confidence: кандидат получает score <= max_score (по умолчанию 0.7,
  ниже порога proposed 0.80 из thresholds.yaml). Поле config/llm.yaml `max_status` документирует это правило;
  применяет его каскад (поток B), здесь — только ограничение score.
- нет провайдера, ключа или enabled: false → NullLLM (каскад получает []).

Провайдеры подключаются адаптером (_ADAPTERS). Из коробки: openai_compatible (любой OpenAI-совместимый
chat/completions: vLLM, llama.cpp server, шлюзы), gigachat (тот же формат, адрес по умолчанию Сбера),
yandexgpt (нативный Foundation Models API). Транспорт инъецируется (transport=...), тесты идут без сети.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Callable

import yaml

from src.contracts import Candidate, Evidence
from src.prepare import PreparedCard

log = logging.getLogger("attrguard.llm")

ROOT = Path(__file__).resolve().parent.parent
API_KEY_ENV = "LLM_API_KEY"

# Транспорт: (url, headers, payload, timeout_s) -> разобранный JSON-ответ сервера.
Transport = Callable[[str, dict, dict, float], dict]

# JSON-схема ответа Этапа 3 (SPEC 3.5).
RESOLVE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "attributes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "value": {"type": "string"},
                    "quote": {"type": "string"},
                    "source": {"type": "string", "enum": ["title", "description", "seller_attributes"]},
                },
                "required": ["attribute", "value", "quote", "source"],
            },
        }
    },
    "required": ["attributes"],
}

_SYSTEM_PROMPT = (
    "Ты извлекаешь характеристики товара ТОЛЬКО из приведённых фрагментов текста карточки. "
    "Ничего не выдумывай и не используй свои знания о товаре. Если значения во фрагментах нет — не включай атрибут. "
    "Для каждого значения приведи дословную цитату из фрагмента (quote) и источник (source). "
    "Отвечай одним JSON-объектом без пояснений."
)

# Сколько начала текста брать, если у кандидатов нет evidence (символов на источник).
_FALLBACK_CHARS = {"title": 200, "description": 400, "seller_attributes": 300}


# ───────────────────────── интерфейс ─────────────────────────

class LLMClient:
    """Интерфейс Этапа 3. calls — число реальных (не из кэша) вызовов за прогон."""

    calls: int = 0

    def resolve(self, prepared: PreparedCard, attributes: list[str], candidates: list[Candidate]) -> list[Candidate]:
        """Кандидаты stage="llm" только для переданных спорных атрибутов, каждый с проверенной evidence."""
        raise NotImplementedError

    def complete_json(self, prompt: str, schema: dict) -> dict:
        """Сырой вызов со строгой JSON-схемой ответа (нужен schema_bootstrap)."""
        raise NotImplementedError


class NullLLM(LLMClient):
    """Заглушка: ничего не вызывает, ничего не возвращает."""

    def resolve(self, prepared, attributes, candidates):
        return []

    def complete_json(self, prompt, schema):
        return {}


# ───────────────────────── провайдеры ─────────────────────────

class _OpenAICompatible:
    """POST {base_url}/chat/completions, Authorization: Bearer <ключ>."""

    default_base_url: str | None = None
    default_json_mode = True

    def build(self, cfg: dict, key: str, model: str, system: str, user: str) -> tuple[str, dict, dict]:
        base = (cfg.get("base_url") or self.default_base_url or "").rstrip("/")
        if not base:
            raise ValueError("llm.yaml: для openai_compatible нужен base_url")
        payload: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": cfg.get("temperature", 0),
            "max_tokens": cfg.get("max_tokens", 600),
        }
        json_mode = cfg.get("json_mode")
        if json_mode is None:
            json_mode = self.default_json_mode
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        return f"{base}/chat/completions", headers, payload

    def parse(self, resp: dict) -> str:
        return resp["choices"][0]["message"]["content"]


class _GigaChat(_OpenAICompatible):
    """GigaChat: chat/completions-формат; ключ — действующий access token (получение токена вне модуля)."""

    default_base_url = "https://gigachat.devices.sberbank.ru/api/v1"
    default_json_mode = False


class _YandexGPT:
    """YandexGPT Foundation Models API. model — имя модели (yandexgpt-lite), каталог — cfg.folder_id."""

    default_base_url = "https://llm.api.cloud.yandex.net/foundationModels/v1"

    def build(self, cfg: dict, key: str, model: str, system: str, user: str) -> tuple[str, dict, dict]:
        folder = cfg.get("folder_id")
        if not model.startswith("gpt://") and not folder:
            raise ValueError("llm.yaml: для yandexgpt нужен folder_id (или model в виде gpt://...)")
        model_uri = model if model.startswith("gpt://") else f"gpt://{folder}/{model}/latest"
        base = (cfg.get("base_url") or self.default_base_url).rstrip("/")
        payload = {
            "modelUri": model_uri,
            "completionOptions": {"stream": False, "temperature": cfg.get("temperature", 0),
                                  "maxTokens": str(cfg.get("max_tokens", 600))},
            "messages": [{"role": "system", "text": system}, {"role": "user", "text": user}],
        }
        headers = {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}
        return f"{base}/completion", headers, payload

    def parse(self, resp: dict) -> str:
        return resp["result"]["alternatives"][0]["message"]["text"]


_ADAPTERS: dict[str, object] = {
    "openai_compatible": _OpenAICompatible(),
    "gigachat": _GigaChat(),
    "yandexgpt": _YandexGPT(),
}


def _httpx_post(url: str, headers: dict, payload: dict, timeout: float) -> dict:
    import httpx

    r = httpx.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ───────────────────────── вспомогательное ─────────────────────────

def _parse_json_object(text: str) -> dict | None:
    """Достать JSON-объект из ответа модели (допускаем ```json-ограждение). Не объект → None."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    for chunk in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        if not chunk:
            continue
        try:
            obj = json.loads(chunk)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def find_quote(texts: dict[str, str], quote, source: str | None) -> tuple[str, int, int] | None:
    """Найти цитату в исходном тексте: точно, затем без учёта регистра. → (source, start, end) или None.

    Если source указан и известен — ищем только в нём; иначе перебираем источники по порядку.
    Срез texts[source][start:end] — точный текст из карточки (регистр исходного)."""
    if not isinstance(quote, str) or not quote.strip():
        return None
    names = [source] if source in texts else list(texts)
    for lower in (False, True):
        for name in names:
            src = texts.get(name) or ""
            hay, needle = (src, quote) if not lower else (src.lower(), quote.lower())
            if lower and len(hay) != len(src):   # lower() сдвинул позиции — не доверяем
                continue
            i = hay.find(needle)
            if i >= 0:
                return name, i, i + len(needle)
    return None


def _merge_windows(spans: list[tuple[int, int]], n: int, ctx: int) -> list[tuple[int, int]]:
    wins = sorted((max(0, s - ctx), min(n, e + ctx)) for s, e in spans)
    out: list[list[int]] = []
    for s, e in wins:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _format_number(value) -> str | None:
    try:
        f = float(str(value).replace(",", ".").strip())
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return str(int(f)) if f == int(f) else str(f)


# ───────────────────────── HttpLLM ─────────────────────────

class HttpLLM(LLMClient):
    """Провайдеро-независимый HTTP-клиент Этапа 3. Ключ — только из os.environ["LLM_API_KEY"]."""

    def __init__(self, cfg: dict, transport: Transport | None = None):
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise RuntimeError(f"не задана переменная окружения {API_KEY_ENV}")
        provider = cfg.get("provider")
        if provider not in _ADAPTERS:
            raise ValueError(f"llm.yaml: неизвестный provider {provider!r}; доступны {sorted(_ADAPTERS)}")
        if not cfg.get("model"):
            raise ValueError("llm.yaml: не задана model")
        self._key = key
        self.cfg = cfg
        self.adapter = _ADAPTERS[provider]
        self.model: str = cfg["model"]
        # модель для редких вызовов complete_json (черновик схемы): schema_bootstrap.model, иначе основная
        self.bootstrap_model: str = (cfg.get("schema_bootstrap") or {}).get("model") or self.model
        self.transport: Transport = transport or _httpx_post
        self.max_calls: int = int(cfg.get("max_calls_per_run", 200))
        self.context_chars: int = int(cfg.get("context_chars", 200))
        self.max_score: float = float(cfg.get("max_score", 0.7))
        self.timeout: float = float(cfg.get("timeout_s", 20))
        self.calls = 0
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._cache_path: Path | None = None
        if cfg.get("cache_path"):
            p = Path(cfg["cache_path"])
            self._cache_path = p if p.is_absolute() else ROOT / p
            self._load_cache()

    # ---- кэш ----
    def _load_cache(self) -> None:
        try:
            if self._cache_path and self._cache_path.exists():
                for line in self._cache_path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                        self._cache[row["key"]] = row["response"]
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError as e:
            log.warning("кэш LLM не прочитан: %s", e)

    def _cache_put(self, key: str, response: dict) -> None:
        self._cache[key] = response
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("кэш LLM не записан: %s", e)

    @staticmethod
    def cache_key(model: str, prompt: str) -> str:
        return hashlib.sha256((model + prompt).encode("utf-8")).hexdigest()

    # ---- вызов ----
    def _complete(self, prompt: str, schema: dict, model: str) -> dict:
        """Кэш → (лимит) → HTTP → JSON. Любая ошибка → {} без исключения. Невалидный JSON: один повтор."""
        full_prompt = (f"{prompt}\n\nФормат ответа — один JSON-объект по схеме:\n"
                       f"{json.dumps(schema, ensure_ascii=False)}")
        key = self.cache_key(model, full_prompt)
        if key in self._cache:
            return self._cache[key]
        for attempt in range(2):
            with self._lock:
                if self.calls >= self.max_calls:
                    log.info("лимит вызовов LLM на прогон исчерпан (%d)", self.max_calls)
                    return {}
                self.calls += 1
            try:
                url, headers, payload = self.adapter.build(self.cfg, self._key, model, _SYSTEM_PROMPT, full_prompt)
                text = self.adapter.parse(self.transport(url, headers, payload, self.timeout))
            except Exception as e:  # сеть, HTTP-ошибка, неожиданный формат ответа — не роняем каскад
                log.warning("вызов LLM не удался: %s", type(e).__name__)
                return {}
            obj = _parse_json_object(text)
            if obj is not None:
                self._cache_put(key, obj)
                return obj
            log.warning("LLM вернула невалидный JSON (попытка %d)", attempt + 1)
        return {}

    def complete_json(self, prompt: str, schema: dict) -> dict:
        return self._complete(prompt, schema, self.bootstrap_model)

    # ---- Этап 3 ----
    def build_prompt(self, prepared: PreparedCard, attributes: list[str], candidates: list[Candidate]) -> str | None:
        """Промпт только по спорным атрибутам с известной схемой. Нет таких атрибутов → None."""
        specs = [s for s in (prepared.schema.attr(a) for a in attributes) if s is not None]
        if not specs:
            return None
        wanted = {s.code for s in specs}

        spans: dict[str, list[tuple[int, int]]] = {}
        hints: list[str] = []
        for c in candidates:
            if c.attribute in wanted and c.evidence is not None and c.evidence.source in prepared.texts:
                spans.setdefault(c.evidence.source, []).append((c.evidence.start, c.evidence.end))
                hints.append(f"- {c.attribute}: «{c.evidence.text}» (этап {c.stage})")

        fragments: list[str] = []
        for source in ("title", "description", "seller_attributes"):
            text = prepared.texts.get(source) or ""
            if not text:
                continue
            if spans:
                for s, e in _merge_windows(spans.get(source, []), len(text), self.context_chars):
                    fragments.append(f"[{source}] {text[s:e]}")
            else:
                fragments.append(f"[{source}] {text[:_FALLBACK_CHARS[source]]}")

        attr_lines = []
        for s in specs:
            line = f"- {s.code} ({s.name_ru}), тип {s.type}"
            if s.type == "number":
                line += f", число в {s.unit}, без единицы, точка как разделитель" if s.unit else ", число"
            if s.values:
                line += ", допустимые значения: " + ", ".join(str(v) for v in s.values)
            elif s.range:
                line += f", диапазон {s.range[0]}..{s.range[1]}"
            attr_lines.append(line)

        parts = [f"Категория: {prepared.category}", "Нужные атрибуты:", *attr_lines]
        if hints:
            parts += ["Возможные значения, найденные дешёвыми методами (могут быть неверны):", *hints]
        parts += ["Фрагменты текста карточки (в квадратных скобках — источник):", *fragments,
                  'Верни JSON {"attributes":[{"attribute":код,"value":значение,"quote":дословная цитата из '
                  'фрагмента,"source":источник}]}. Для перечислимых атрибутов value строго из допустимых значений.']
        return "\n".join(parts)

    def resolve(self, prepared: PreparedCard, attributes: list[str], candidates: list[Candidate]) -> list[Candidate]:
        prompt = self.build_prompt(prepared, attributes, candidates)
        if prompt is None:
            return []
        data = self._complete(prompt, RESOLVE_SCHEMA, self.model)
        return self._validate(prepared, attributes, data)

    def _validate(self, prepared: PreparedCard, attributes: list[str], data: dict) -> list[Candidate]:
        items = data.get("attributes") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return []
        out: dict[str, Candidate] = {}
        for it in items:
            if not isinstance(it, dict):
                continue
            code, raw, quote = it.get("attribute"), it.get("value"), it.get("quote")
            if code not in attributes or code in out or raw is None or isinstance(raw, (dict, list)):
                continue
            spec = prepared.schema.attr(code)
            raw = str(raw).strip()
            if spec is None or not raw:
                continue
            value = self._canonical(spec, raw)
            if value is None:
                continue          # вне справочника / диапазона / не число
            hit = find_quote(prepared.texts, quote, it.get("source"))
            if hit is None:
                continue          # цитаты нет в тексте → значение отбрасываем
            source, s, e = hit
            out[code] = Candidate(
                attribute=code, raw_value=raw, value=value, unit=spec.unit if spec.type == "number" else None,
                score=self.max_score, stage="llm",
                evidence=Evidence(text=prepared.texts[source][s:e], start=s, end=e, source=source),
            )
        return list(out.values())

    @staticmethod
    def _canonical(spec, raw: str) -> str | None:
        """Значение в каноне схемы или None, если не проходит проверку справочника."""
        if spec.type == "enum":
            by_lower = {str(v).lower().replace("ё", "е"): str(v) for v in spec.values}
            return by_lower.get(raw.lower().replace("ё", "е"))
        if spec.type == "number":
            num = _format_number(raw)
            if num is None:
                return None
            if spec.values and num not in {_format_number(v) for v in spec.values}:
                return None
            if spec.range and not (spec.range[0] <= float(num) <= spec.range[1]):
                return None
            return num
        return raw  # text: свободная строка, подтверждена цитатой


# ───────────────────────── загрузка ─────────────────────────

def load_llm(config_path: str = "config/llm.yaml", transport: Transport | None = None) -> LLMClient:
    """enabled: false, нет провайдера/модели или нет LLM_API_KEY → NullLLM (с логом)."""
    p = Path(config_path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    try:
        cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        log.warning("LLM выключена: конфиг %s не прочитан (%s)", p, e)
        return NullLLM()
    if not cfg.get("enabled"):
        log.info("LLM выключена (enabled: false) — Этап 3 пропускается")
        return NullLLM()
    if not os.environ.get(API_KEY_ENV):
        log.warning("LLM выключена: не задана переменная окружения %s", API_KEY_ENV)
        return NullLLM()
    try:
        return HttpLLM(cfg, transport=transport)
    except (ValueError, RuntimeError) as e:
        log.warning("LLM выключена: %s", e)
        return NullLLM()
