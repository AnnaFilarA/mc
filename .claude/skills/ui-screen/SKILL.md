---
name: ui-screen
description: Добавить или изменить экран веб-интерфейса AttrGuard (FastAPI + Jinja2 + HTMX, web/).
---
# ui-screen

1. Шаблон `web/templates/<screen>.html`, наследуется от `base.html` (шапка с переключателем роли).
2. Роут в `src/api.py` (только чтение/вызов ядра и `/v1/*`, без бизнес-логики в шаблоне).
3. Стиль (SPEC 4.1): белый фон, акцент `#E30611`, тёмный `#15171A`, Inter / системный sans, текст ≥ 14 px. Один CSS — `web/static/style.css`.
4. Статус — **цветом и словом**: `proposed` зелёный «Предложено», `low_confidence` жёлтый «Низкая уверенность», `conflict` красный «Конфликт», `missing` красный «Нет значения», `unchanged` серый «Без изменений».
5. Интерактив — HTMX (`hx-post`, `hx-target`), без сборки фронтенда.
6. Проверка в браузере: `uvicorn src.api:app --port 8000` + Playwright/Chromium (`executablePath: /opt/pw-browsers/chromium` при необходимости), скриншот, 0 ошибок в консоли. Тест роутов — `fastapi.testclient`.
