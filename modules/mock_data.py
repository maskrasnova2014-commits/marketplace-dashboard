"""
Генератор реалистичных тестовых (мок) данных для дашборда.

Используется, когда пользователь не ввёл API-ключи или когда запрос
к API маркетплейса завершился ошибкой. Позволяет интерфейсу работать
"из коробки" без реальных ключей доступа.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

# Каталог артикулов: селлер -> маркетплейсы -> категория -> себестоимость (ориентир)
CATALOG: list[dict] = [
    {"Артикул": "MASON-A-02-14-1", "Категория": "Стулья", "Селлер": "Лебедев", "Цена_база": 4200},
    {"Артикул": "MASON-A-03-11-2", "Категория": "Стулья барные", "Селлер": "Лебедев", "Цена_база": 5600},
    {"Артикул": "NORD", "Категория": "Столы", "Селлер": "Лебедев", "Цена_база": 12500},
    {"Артикул": "NORD-COMPACT", "Категория": "Столы", "Селлер": "Лебедев", "Цена_база": 8900},
    {"Артикул": "NORD-BAR", "Категория": "Стулья полубарные", "Селлер": "Лебедев", "Цена_база": 5100},
    {"Артикул": "PIXEL", "Категория": "Диваны", "Селлер": "Госович", "Цена_база": 34900},
    {"Артикул": "PIXEL-MINI", "Категория": "Диваны", "Селлер": "Госович", "Цена_база": 24900},
    {"Артикул": "PIXEL-CHAIR", "Категория": "Стулья", "Селлер": "Госович", "Цена_база": 3900},
    {"Артикул": "PIXEL-SEMI", "Категория": "Стулья полубарные", "Селлер": "Госович", "Цена_база": 4800},
    {"Артикул": "PIXEL-BAR", "Категория": "Стулья барные", "Селлер": "Госович", "Цена_база": 5300},
]

MARKETPLACES = ["Ozon", "Wildberries"]
STATUSES = ["Доставлен", "В пути", "Отменён"]
CANCEL_REASONS = [
    "Отказ покупателя",
    "Брак/повреждение при доставке",
    "Не забрали с ПВЗ",
    "Отмена продавцом (нет в наличии)",
    "Долгая доставка",
]

# Комиссии и логистика для категории "Мебель" по маркетплейсам (%, руб)
COMMISSION_RATES = {"Ozon": 0.19, "Wildberries": 0.17}
LOGISTICS_MAGISTRAL = {"Ozon": 350, "Wildberries": 300}
LOGISTICS_LAST_MILE = {"Ozon": 220, "Wildberries": 180}


def _daterange(start: date, end: date) -> list[date]:
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days + 1)]


def generate_orders(start_date: date, end_date: date, seed: int = 42) -> pd.DataFrame:
    """Генерирует таблицу заказов (FBO/FBS) по дням, селлерам и артикулам."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = _daterange(start_date, end_date)

    for d in dates:
        # Не каждый день по каждому артикулу есть заказы — имитируем разреженность
        n_orders_today = rng.integers(15, 45)
        for _ in range(n_orders_today):
            item = CATALOG[rng.integers(0, len(CATALOG))]
            marketplace = rng.choice(MARKETPLACES, p=[0.55, 0.45])
            qty = int(rng.choice([1, 1, 1, 2, 2, 3], p=[0.45, 0.2, 0.15, 0.1, 0.06, 0.04]))
            price = item["Цена_база"] * rng.uniform(0.9, 1.08)
            status = rng.choice(STATUSES, p=[0.78, 0.10, 0.12])
            reason = rng.choice(CANCEL_REASONS) if status == "Отменён" else None

            rows.append(
                {
                    "Дата": d,
                    "Маркетплейс": marketplace,
                    "Селлер": item["Селлер"],
                    "Артикул": item["Артикул"],
                    "Категория": item["Категория"],
                    "Количество": qty,
                    "Цена": round(price, 2),
                    "Сумма": round(price * qty, 2),
                    "Статус": status,
                    "Причина_отмены": reason,
                }
            )

    df = pd.DataFrame(rows)
    df["Дата"] = pd.to_datetime(df["Дата"])
    df["Месяц"] = df["Дата"].dt.to_period("M").astype(str)
    return df


def generate_advertising(start_date: date, end_date: date, seed: int = 43) -> pd.DataFrame:
    """Генерирует данные по рекламным кампаниям (Трафареты Ozon / АВТО-кампании WB)."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = _daterange(start_date, end_date)

    for d in dates:
        for marketplace in MARKETPLACES:
            for seller in {c["Селлер"] for c in CATALOG}:
                spend = rng.uniform(800, 5000)
                revenue = spend * rng.uniform(3, 9)  # ДРР обычно 10-30%
                orders = int(revenue / rng.uniform(4000, 9000)) or 1
                rows.append(
                    {
                        "Дата": d,
                        "Маркетплейс": marketplace,
                        "Селлер": seller,
                        "Расходы_на_рекламу": round(spend, 2),
                        "Заказы_с_рекламы": orders,
                        "Выручка_с_рекламы": round(revenue, 2),
                    }
                )

    df = pd.DataFrame(rows)
    df["Дата"] = pd.to_datetime(df["Дата"])
    df["Месяц"] = df["Дата"].dt.to_period("M").astype(str)
    return df


def generate_cost_reference() -> pd.DataFrame:
    """Возвращает дефолтный справочник себестоимости (используется, если файл не загружен)."""
    return pd.DataFrame(
        [
            {"Артикул": c["Артикул"], "Себестоимость": round(c["Цена_база"] * 0.42, 2)}
            for c in CATALOG
        ]
    )
