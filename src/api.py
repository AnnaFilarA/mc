"""FastAPI: API + веб-интерфейс (SPEC 3.1, раздел 4). Владелец — поток C.

Запуск: uvicorn src.api:app
"""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="AttrGuard")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
