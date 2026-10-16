"""Этап 1 (правила и словари) и нормализация. По скиллу add-rule: ≥3 позитивных и ≥2 негативных на ключевое правило."""
import pytest

from src import rules
from src.contracts import Candidate, Card, evidence_is_valid
from src.normalize import normalize
from src.prepare import load_schemas, prepare
from tests.conftest import load_fixtures

SCHEMAS = load_schemas()


def run(title="", description="", seller=None, hint="smartphones") -> list[Candidate]:
    """Кандидаты Этапа 1 после нормализации (value=None — отброшены нормализатором)."""
    card = Card(product_id="t", title=title, description=description, seller_attributes=seller or {}, category_hint=hint)
    pc = prepare(card, SCHEMAS)
    out = []
    for c in rules.extract(pc):
        assert evidence_is_valid(card, c.evidence), (c.attribute, c.evidence)   # span точно совпадает с текстом
        out.append(normalize(c, pc.schema.attr(c.attribute)))
    return out


def vals(cands, attr) -> set[str]:
    return {c.value for c in cands if c.attribute == attr and c.value is not None}


# ---------------------------------------------------------------- evidence на всех фикстурах

@pytest.mark.parametrize("fx", load_fixtures(), ids=lambda f: f["card"]["product_id"])
def test_every_candidate_has_valid_evidence(fx):
    card = Card.model_validate(fx["card"])
    pc = prepare(card, SCHEMAS)
    for c in rules.extract(pc):
        assert c.stage == "rules" and 0 < c.score <= 0.98
        assert evidence_is_valid(card, c.evidence)


# ---------------------------------------------------------------- память: «ОЗУ/накопитель»

@pytest.mark.parametrize("title,ram,storage", [
    ("Смартфон Samsung Galaxy A55 8/256GB синий", "8", "256"),
    ("Samsung Galaxy S24 Ultra 12/1TB", "12", "1024"),
    ("Xiaomi Redmi Note 13 6/128 Гб", "6", "128"),
    ("Ноутбук Acer Aspire 7 16GB/512GB i5", "16", "512"),
    ("Ноутбук Lenovo i3-1315U/8Gb/256Gb SSD/15.6\"/DOS", "8", "256"),
])
def test_ram_storage_pair(title, ram, storage):
    c = run(title, hint=None)
    assert vals(c, "ram") == {ram}
    assert vals(c, "storage") == {storage}


@pytest.mark.parametrize("title,ram,storage", [
    ("Ноутбук ASUS 16 ГБ ОЗУ + 512 ГБ SSD", "16", "512"),
    ("Смартфон realme GT 6 16 гигов ОЗУ, 512 ГБ памяти", "16", "512"),
    ("Ноутбук HP Ryzen 5 7520U 8 ГБ 512 ГБ SSD", "8", "512"),
    ("Ноутбук Dell Core i7 16 ГБ 1 ТБ HDD", "16", "1024"),
])
def test_ram_storage_labeled(title, ram, storage):
    c = run(title, hint="laptops" if "Ноутбук" in title else "smartphones")
    assert vals(c, "ram") == {ram}
    assert vals(c, "storage") == {storage}


def test_labeled_description():
    c = run("Смартфон Huawei nova 12 SE", "Оперативная память 8 ГБ, встроенная память 256 ГБ.")
    assert vals(c, "ram") == {"8"} and vals(c, "storage") == {"256"}


def test_microsd_is_not_storage():
    c = run("Смартфон Tecno Spark 20", "Поддержка карт памяти microSD до 1 ТБ.")
    assert vals(c, "storage") == set()
    c = run("Смартфон X", "Слот расширения памяти до 512 ГБ")
    assert vals(c, "storage") == set()


def test_gpu_memory_is_not_ram():
    c = run("Игровой ноутбук ASUS", "Видеокарта RTX 4060 8 ГБ, оперативная память 16 ГБ DDR5, SSD 1 ТБ.", hint="laptops")
    assert vals(c, "ram") == {"16"}
    assert vals(c, "storage") == {"1024"}


def test_5g_and_dates_are_not_memory():
    c = run("Смартфон Honor 5G", "Выпущен 12/05/2024. Поддержка 5G.")
    assert vals(c, "ram") == set() and vals(c, "storage") == set()


def test_lone_small_memory_is_weak():
    c = [x for x in run("Смартфон Honor X8b 8 ГБ", hint="smartphones") if x.attribute == "ram"]
    assert c and all(x.score < 0.8 for x in c)   # «8 ГБ» без подписи — неоднозначно, не выше порога


# ---------------------------------------------------------------- диагональ

@pytest.mark.parametrize("text,expected", [
    ("Экран 6,6 дюйма Super AMOLED", "6.6"),
    ("Дисплей 6.1\" Super Retina XDR", "6.1"),
    ("Диагональ 6,78\".", "6.78"),
    ("Экран 6.7'' LTPO OLED", "6.7"),
    ("Диагональ экрана: 6.4", "6.4"),
])
def test_screen_size_phone(text, expected):
    assert vals(run("Смартфон X", text), "screen_size") == {expected}


@pytest.mark.parametrize("title,expected", [
    ("Телевизор Samsung UE55CU7100U 55\" 4K UHD", "55"),
    ("Телевизор TCL 65C745 65'' QLED", "65"),
    ("Телевизор LG 43UR78006LK 43 дюйма Ultra HD", "43"),
])
def test_screen_size_tv(title, expected):
    assert vals(run(title, hint="tvs"), "screen_size") == {expected}


def test_screen_not_from_model_name_or_cm():
    assert vals(run("Apple MacBook Air 13 M2 8/256 ГБ", hint="laptops"), "screen_size") == set()
    c = run("Телевизор Xiaomi TV A 32", "Диагональ 81 см, 120 Гц", hint="tvs")
    assert vals(c, "screen_size") == set()


def test_decimal_before_panel_type_is_low():
    c = [x for x in run("Ноутбук HP 15s-fq5000 Ryzen 5 15.6 FHD", hint="laptops") if x.attribute == "screen_size"]
    assert {x.value for x in c} == {"15.6"} and all(x.score < 0.8 for x in c)


# ---------------------------------------------------------------- словари: бренд, цвет, ОС, разрешение

@pytest.mark.parametrize("title,brand", [
    ("Apple iPhone 15 128 ГБ", "Apple"),
    ("Смартфон Galaxy A55", "Samsung"),
    ("Смартфон Redmi Note 13", "Xiaomi"),
    ("Смартфон POCO X6 Pro", "Poco"),
    ("Google Pixel 8 Pro", "Google"),
])
def test_brand_by_name_and_model(title, brand):
    assert vals(run(title), "brand") == {brand}


def test_brand_not_from_compatibility_phrase_or_google_tv():
    assert vals(run("Смартфон Honor X8b чехол для iPhone"), "brand") == {"Honor"}
    c = run("Телевизор Xiaomi TV A 32", "Google TV, Wi-Fi", hint="tvs")
    assert vals(c, "brand") == {"Xiaomi"}


@pytest.mark.parametrize("text,color", [
    ("Смартфон X титановый серый", "Серый"),
    ("Apple MacBook серый космос", "Серый"),
    ("Tecno Spark 20 Gravity Black", "Черный"),
    ("Корпус зелёного цвета", "Зеленый"),
    ("Белые наушники", "Белый"),
])
def test_color_synonyms(text, color):
    assert vals(run(text), "color") == {color}


def test_color_not_inside_other_words():
    assert vals(run("Смартфон Redmi Note 13, сервис в подарок"), "color") == set()
    assert vals(run("Смартфон X", "Содержит белки и чернила"), "color") == set()


def test_color_two_values_in_one_source_is_ambiguous():
    c = [x for x in run("Смартфон X чёрный или белый") if x.attribute == "color"]
    assert len({x.value for x in c}) == 2 and all(x.score < 0.8 for x in c)


@pytest.mark.parametrize("text,os_", [
    ("Lenovo IdeaPad i3/8Gb/256Gb SSD/DOS", "Без ОС"),
    ("Windows 11 Home", "Windows 11"),
    ("macOS Sonoma", "macOS"),
    ("Android 14", "Android"),
])
def test_os(text, os_):
    cat = "smartphones" if "Android" in text else "laptops"
    assert vals(run("Ноутбук X", text, hint=cat), "os") == {os_}


def test_os_requires_version_for_windows():
    assert vals(run("Ноутбук X", "Совместим с Windows", hint="laptops"), "os") == set()
    assert vals(run("Ноутбук X", "Доставка по всей стране", hint="laptops"), "os") == set()


@pytest.mark.parametrize("text,res", [
    ("разрешение 1366x768", "HD"),
    ("43 дюйма Ultra HD", "4K UHD"),
    ("TCL QLED 4K 144 Гц", "4K UHD"),
    ("Full HD 1920x1080", "Full HD"),
    ("Xiaomi TV A 32 2025 HD", "HD"),
])
def test_resolution(text, res):
    assert vals(run("Телевизор " + text, hint="tvs"), "resolution") == {res}


def test_resolution_not_from_hdmi_hdr():
    assert vals(run("Телевизор LG", "HDMI 2.1, HDR10", hint="tvs"), "resolution") == set()


def test_refresh_rate():
    assert vals(run("Телевизор TCL 144 Гц", hint="tvs"), "refresh_rate") == {"144"}
    assert vals(run("Телевизор LG", "частота обновления 60 Гц", hint="tvs"), "refresh_rate") == {"60"}
    assert vals(run("Телевизор LG", "питание 50 Гц", hint="tvs"), "refresh_rate") == set()


def test_cpu():
    c = run("Ноутбук ASUS Intel Core i5-1335U", hint="laptops")
    assert vals(c, "cpu") == {"Intel Core i5"}
    assert vals(run("Ноутбук HP Ryzen 5 7520U", hint="laptops"), "cpu") == {"AMD Ryzen 5"}
    assert vals(run("Ноутбук Apple MacBook Air 13 M2", hint="laptops"), "cpu") == {"Apple M2"}
    assert vals(run("Ноутбук X", "накопитель M.2 SSD и M2 SSD", hint="laptops"), "cpu") == set()


def test_headphones_dictionaries():
    c = run("Беспроводные наушники Sony", "Активное шумоподавление, внутриканальные.", hint="headphones")
    assert vals(c, "connection") == {"Беспроводные"} and vals(c, "form_factor") == {"Внутриканальные"}
    assert vals(c, "anc") == {"Да"}
    c = run("Проводные наушники JBL вкладыши", "Без шумоподавления", hint="headphones")
    assert vals(c, "connection") == {"Проводные"} and vals(c, "form_factor") == {"Вкладыши"}
    assert vals(c, "anc") == {"Нет"}


# ---------------------------------------------------------------- поля продавца

def test_seller_keys_case_insensitive_and_evidence():
    seller = {"ОПЕРАТИВНАЯ ПАМЯТЬ": "8 Гб", "цвет": "Черный", "Диагональ (дюймы)": "6.1"}
    card = Card(product_id="t", title="Смартфон X", seller_attributes=seller, category_hint="smartphones")
    pc = prepare(card, SCHEMAS)
    got = {c.attribute: c for c in rules.extract(pc) if c.evidence.source == "seller_attributes"}
    assert {"ram", "color", "screen_size"} <= set(got)
    for c in got.values():
        assert evidence_is_valid(card, c.evidence)
    assert got["ram"].evidence.text == "8 Гб" and got["color"].evidence.text == "Черный"


def test_unknown_seller_key_is_ignored():
    assert run("Смартфон X", seller={"Гарантия": "12 месяцев", "Вес": "180 г"}) == []


# ---------------------------------------------------------------- нормализация

def _n(attr, raw, category="laptops", value=None):
    schema = SCHEMAS[category]
    return normalize(Candidate(attribute=attr, raw_value=raw, value=value), schema.attr(attr))


def test_normalize_numbers():
    assert _n("storage", "1 ТБ").value == "1024"
    assert _n("storage", "512 Gb").value == "512"
    assert _n("ram", "16 гигов").value == "16" and _n("ram", "16 гигов").unit == "GB"
    assert _n("screen_size", "15,6").value == "15.6"
    assert _n("screen_size", "6.67", "smartphones").value == "6.67"
    assert _n("screen_size", "40 см").value == "15.7"   # см → дюймы
    assert _n("screen_size", "81 см", "tvs").value == "31.9"


def test_normalize_rejects_out_of_range_and_unknown_values():
    assert _n("ram", "10 ГБ").value is None             # нет в values
    assert _n("screen_size", "55", "smartphones").value is None   # вне range
    assert _n("os", "Plan9").value is None
    assert _n("color", "Мокрый асфальт").value is None
    assert _n("refresh_rate", "75", "tvs").value is None


def test_normalize_enum_case_and_yo():
    assert _n("color", "чёрный").value == "Черный"
    assert _n("color", "СЕРЕБРИСТЫЙ").value == "Серебристый"
    assert _n("brand", "asus").value == "ASUS"
    assert _n("os", "dos").value == "Без ОС"
