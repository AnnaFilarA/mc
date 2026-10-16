"""E2E: оба сценария SPEC 4.5 в браузере (не входит в pytest).

Запуск (лучше на временной БД):
  ATTRGUARD_DB=/tmp/a.db ATTRGUARD_FEEDBACK=/tmp/fb.jsonl uvicorn src.api:app --port 8765 &
  python tests/e2e/ui_scenarios.py outputs/screenshots
CHROMIUM_PATH — путь к Chromium, если Playwright не находит его сам.
"""
import os
import sys
from playwright.sync_api import sync_playwright

B, OUT, errors = os.environ.get("ATTRGUARD_URL", "http://127.0.0.1:8765"), sys.argv[1], []
T = "Ноутбук Acer Swift Go 14 16GB/512GB i5"
with sync_playwright() as p:
    exe = os.environ.get("CHROMIUM_PATH")
    br = p.chromium.launch(**({"executable_path": exe} if exe else {}))
    pg = br.new_page(viewport={"width": 1280, "height": 900})
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # Сценарий 1: продавец вводит карточку → предзаполнение и вопросы → отвечает → менеджер видит «ответ получен»
    pg.goto(B + "/role/seller"); pg.goto(B + "/seller/new")
    pg.fill("#f-title", T); pg.fill("#f-description", "Экран 14 дюймов, черный.")
    pg.fill("#f-characteristics", "Оперативная память: 8 ГБ")
    pg.click("#check-btn"); pg.wait_for_selector("#save-btn")
    pg.screenshot(path=f"{OUT}/e2e_01_seller_preview.png", full_page=True)
    pg.click("#save-btn"); pg.wait_for_timeout(800)
    pg.goto(B + "/seller/questions")
    q = pg.locator(".qcard").filter(has_text="Swift Go")
    print("S1 вопрос продавцу:", q.locator(".q-text").all_inner_texts())
    q.locator("button.option").filter(has_text="16").first.click(); pg.wait_for_timeout(800)
    pg.screenshot(path=f"{OUT}/e2e_02_seller_answered.png", full_page=True)
    pg.goto(B + "/role/manager"); pg.goto(B + "/manager")
    print("S1 менеджер видит:", " | ".join(pg.locator("tr").filter(has_text="Swift Go").all_inner_texts()).replace("\n", " "))
    pg.screenshot(path=f"{OUT}/e2e_03_manager_list.png", full_page=True)
    # Сценарий 2: менеджер открывает спорную карточку → исправляет поле → карточка уходит из исключений
    print("S2 l05 в исключениях до:", pg.locator("a[href='/manager/card/l05']").count())
    pg.goto(B + "/manager/card/l05")
    print("S2 подсветок evidence:", pg.locator("mark").count())
    pg.screenshot(path=f"{OUT}/e2e_04_manager_card.png", full_page=True)
    row = pg.locator("#row-ram")
    row.locator("summary").click()
    row.locator("input[name=value]").fill("16")
    row.locator("button[type=submit]").click(); pg.wait_for_timeout(800)
    pg.screenshot(path=f"{OUT}/e2e_05_manager_card_resolved.png", full_page=True)
    pg.goto(B + "/manager")
    print("S2 l05 в исключениях после:", pg.locator("a[href='/manager/card/l05']").count())
    br.close()
print("ошибки консоли:", errors)
