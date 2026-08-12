"""
Логика обработки данных: объединение API + себестоимости, расчёт KPI
и юнит-экономики. Вся тяжёлая обработка ведётся в Pandas.
"""

from __future__ import annotations

import pandas as pd

from modules.mock_data import COMMISSION_RATES, LOGISTICS_LAST_MILE, LOGISTICS_MAGISTRAL

DELIVERED_STATUSES = {"Доставлен"}
CANCELLED_STATUSES = {"Отменён"}

OZON_STATUS_MAP = {
    "delivered": "Доставлен",
    "cancelled": "Отменён",
    "delivering": "В пути",
    "awaiting_deliver": "В пути",
    "awaiting_packaging": "В пути",
}

FURNITURE_CATEGORIES = ["Стулья полубарные", "Стулья барные", "Стулья", "Столы", "Диваны"]


def infer_furniture_category(text: str) -> str:
    """
    Определяет подкатегорию мебели (Стулья / Стулья барные / Стулья полубарные /
    Столы / Диваны) по названию товара или предмету (subject) из API.
    Порядок проверки важен: "полубарный" содержит подстроку "барный".
    """
    t = (text or "").lower()
    if "полубар" in t:
        return "Стулья полубарные"
    if "барны" in t or "барн" in t:
        return "Стулья барные"
    if "диван" in t:
        return "Диваны"
    if "стол" in t:
        return "Столы"
    if "стул" in t:
        return "Стулья"
    return "Мебель (не определено)"


def transform_ozon_postings(postings: list[dict], seller_label: str) -> pd.DataFrame:
    """Приводит отправления Ozon (FBO/FBS) к внутренней схеме заказов."""
    rows = []
    for posting in postings:
        status_raw = posting.get("status", "")
        status = OZON_STATUS_MAP.get(status_raw, "В пути")
        order_date = posting.get("in_process_at") or posting.get("created_at")
        for product in posting.get("products", []):
            qty = int(product.get("quantity", 1))
            price = float(product.get("price", 0))
            rows.append(
                {
                    "Дата": pd.to_datetime(order_date, utc=True).tz_localize(None),
                    "Маркетплейс": "Ozon",
                    "Селлер": seller_label,
                    "Артикул": product.get("offer_id", "N/A"),
                    "Категория": infer_furniture_category(product.get("name", "")),
                    "Количество": qty,
                    "Цена": price,
                    "Сумма": round(price * qty, 2),
                    "Статус": status,
                    "Причина_отмены": "Отмена (см. Ozon Seller)" if status == "Отменён" else None,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Месяц"] = df["Дата"].dt.to_period("M").astype(str)
    return df


def transform_wb_sales(sales: list[dict], seller_label: str) -> pd.DataFrame:
    """Приводит продажи Wildberries (API Статистики) к внутренней схеме заказов."""
    rows = []
    for sale in sales:
        is_return = bool(sale.get("isReturn") or sale.get("isRealization") is False)
        status = "Отменён" if is_return else "Доставлен"
        price = float(sale.get("finishedPrice", sale.get("totalPrice", 0)))
        order_date = sale.get("date")
        rows.append(
            {
                "Дата": pd.to_datetime(order_date, utc=True).tz_localize(None),
                "Маркетплейс": "Wildberries",
                "Селлер": seller_label,
                "Артикул": str(sale.get("supplierArticle", "N/A")),
                "Категория": infer_furniture_category(f"{sale.get('subject', '')} {sale.get('category', '')}"),
                "Количество": 1,
                "Цена": price,
                "Сумма": round(price, 2),
                "Статус": status,
                "Причина_отмены": "Возврат покупателем" if is_return else None,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["Месяц"] = df["Дата"].dt.to_period("M").astype(str)
    return df


def _parse_ozon_money(value) -> float:
    """Ozon Performance API отдаёт суммы строкой с запятой как разделителем ('185,59')."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace(",", ".").replace(" ", "") or 0)


def transform_ozon_ads(daily_stats: list[dict], seller_label: str) -> pd.DataFrame:
    """
    Приводит статистику Ozon Performance API к внутренней схеме рекламы.

    Проверено на реальном ответе /api/client/statistics/daily/json: поля
    'moneySpent', 'orders', 'ordersMoney', 'date' — на уровне отдельных
    кампаний/товаров за день. Суммы приходят строкой с запятой (напр. "185,59").
    """
    rows = []
    for row in daily_stats:
        spend = _parse_ozon_money(row.get("moneySpent") or row.get("spend") or row.get("sum"))
        orders = int(float(row.get("orders") or row.get("ordersCount") or 0))
        revenue = _parse_ozon_money(row.get("ordersMoney") or row.get("revenue"))
        row_date = row.get("date") or row.get("day")
        rows.append(
            {
                "Дата": pd.to_datetime(row_date, utc=True).tz_localize(None) if row_date else pd.NaT,
                "Маркетплейс": "Ozon",
                "Селлер": seller_label,
                "Расходы_на_рекламу": round(spend, 2),
                "Заказы_с_рекламы": orders,
                "Выручка_с_рекламы": round(revenue, 2),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["Дата"])
        df["Месяц"] = df["Дата"].dt.to_period("M").astype(str)
    return df


def transform_wb_ads(advert_costs: list[dict], seller_label: str) -> pd.DataFrame:
    """
    Приводит списания WB Advert API (/adv/v1/upd) к внутренней схеме рекламы.

    Этот эндпоинт даёт только сумму списания по кампаниям — атрибуции
    заказов/выручки к рекламе в нём нет, поэтому эти поля остаются
    нулевыми (честно, а не выдуманы) — ДРР по WB рекламе не считается,
    пока не подключён более полный отчёт по эффективности кампаний.
    """
    rows = []
    for row in advert_costs:
        spend = float(row.get("updSum", 0))
        row_date = row.get("updTime")
        rows.append(
            {
                "Дата": pd.to_datetime(row_date, utc=True).tz_localize(None) if row_date else pd.NaT,
                "Маркетплейс": "Wildberries",
                "Селлер": seller_label,
                "Расходы_на_рекламу": round(spend, 2),
                "Заказы_с_рекламы": 0,
                "Выручка_с_рекламы": 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.dropna(subset=["Дата"])
        df["Месяц"] = df["Дата"].dt.to_period("M").astype(str)
    return df


def filter_orders(
    orders: pd.DataFrame,
    marketplaces: list[str] | None = None,
    sellers: list[str] | None = None,
    date_from=None,
    date_to=None,
) -> pd.DataFrame:
    """Применяет фильтры сайдбара (маркетплейс, селлер, диапазон дат) к заказам."""
    df = orders.copy()
    if marketplaces:
        df = df[df["Маркетплейс"].isin(marketplaces)]
    if sellers:
        df = df[df["Селлер"].isin(sellers)]
    if date_from is not None:
        df = df[df["Дата"] >= pd.Timestamp(date_from)]
    if date_to is not None:
        df = df[df["Дата"] <= pd.Timestamp(date_to)]
    return df


def calculate_kpis(orders: pd.DataFrame) -> dict:
    """Считает верхнеуровневые KPI: GMV, доставлено, отмены, AOV."""
    total_qty = int(orders["Количество"].sum())
    total_sum = float(orders["Сумма"].sum())

    delivered = orders[orders["Статус"].isin(DELIVERED_STATUSES)]
    delivered_qty = int(delivered["Количество"].sum())
    delivered_sum = float(delivered["Сумма"].sum())

    cancelled = orders[orders["Статус"].isin(CANCELLED_STATUSES)]
    cancelled_qty = int(cancelled["Количество"].sum())
    cancelled_sum = float(cancelled["Сумма"].sum())

    cancel_rate_qty = (cancelled_qty / total_qty * 100) if total_qty else 0.0
    cancel_rate_sum = (cancelled_sum / total_sum * 100) if total_sum else 0.0
    aov = (total_sum / total_qty) if total_qty else 0.0

    cancel_reasons = (
        cancelled.groupby("Причина_отмены")
        .agg(Количество=("Количество", "sum"), Сумма=("Сумма", "sum"))
        .reset_index()
        .sort_values("Сумма", ascending=False)
    )

    return {
        "gmv_sum": total_sum,
        "gmv_qty": total_qty,
        "delivered_sum": delivered_sum,
        "delivered_qty": delivered_qty,
        "cancelled_sum": cancelled_sum,
        "cancelled_qty": cancelled_qty,
        "cancel_rate_qty": cancel_rate_qty,
        "cancel_rate_sum": cancel_rate_sum,
        "aov": aov,
        "cancel_reasons": cancel_reasons,
    }


def build_dynamics_table(orders: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица динамики по селлерам и месяцам."""
    def agg_group(g: pd.DataFrame) -> pd.Series:
        delivered = g[g["Статус"].isin(DELIVERED_STATUSES)]
        cancelled = g[g["Статус"].isin(CANCELLED_STATUSES)]
        total_sum = g["Сумма"].sum()
        cancel_sum = cancelled["Сумма"].sum()
        return pd.Series(
            {
                "Принято (шт)": g["Количество"].sum(),
                "Принято (руб)": total_sum,
                "Доставлено (шт)": delivered["Количество"].sum(),
                "Доставлено (руб)": delivered["Сумма"].sum(),
                "Отменено (шт)": cancelled["Количество"].sum(),
                "Отменено (руб)": cancel_sum,
                "% Отмен (по сумме)": round(cancel_sum / total_sum * 100, 1) if total_sum else 0.0,
            }
        )

    result = (
        orders.groupby(["Селлер", "Месяц"])
        .apply(agg_group, include_groups=False)
        .reset_index()
    )
    return result.sort_values(["Селлер", "Месяц"])


def category_breakdown(orders: pd.DataFrame) -> pd.DataFrame:
    """Продажи (не отменённые заказы) по категориям мебели."""
    sales = orders[~orders["Статус"].isin(CANCELLED_STATUSES)]
    result = (
        sales.groupby("Категория", as_index=False)
        .agg(**{"Количество (шт)": ("Количество", "sum"), "Сумма (руб)": ("Сумма", "sum")})
        .sort_values("Сумма (руб)", ascending=False)
    )
    return result


def advertising_summary(ads: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    """Сводка по рекламе с расчётом ДРР (%)."""
    keys = group_by or ["Маркетплейс"]
    result = (
        ads.groupby(keys, as_index=False)
        .agg(
            **{
                "Расходы на рекламу": ("Расходы_на_рекламу", "sum"),
                "Заказы с рекламы": ("Заказы_с_рекламы", "sum"),
                "Выручка с рекламы": ("Выручка_с_рекламы", "sum"),
            }
        )
    )
    result["ДРР (%)"] = (
        (result["Расходы на рекламу"] / result["Выручка с рекламы"].replace(0, pd.NA) * 100)
        .astype(float)
        .round(1)
    )
    return result


def merge_cost_data(orders: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    """Объединяет заказы со справочником себестоимости по артикулу."""
    return orders.merge(cost_df[["Артикул", "Себестоимость"]], on="Артикул", how="left")


def calculate_unit_economics(
    orders: pd.DataFrame,
    cost_df: pd.DataFrame,
    ads: pd.DataFrame,
) -> pd.DataFrame:
    """
    Считает юнит-экономику по каждому артикулу: от выручки до чистой прибыли.

    Складывается из:
      Выручка - Себестоимость - Комиссия МП - Логистика - Реклама (ДРР на ед.)
    """
    sales = orders[orders["Статус"] != "Отменён"].copy()

    per_sku = (
        sales.groupby(["Артикул", "Категория", "Селлер", "Маркетплейс"], as_index=False)
        .agg(
            **{
                "Продано (шт)": ("Количество", "sum"),
                "Выручка": ("Сумма", "sum"),
            }
        )
    )
    per_sku = per_sku.merge(cost_df[["Артикул", "Себестоимость"]], on="Артикул", how="left")
    per_sku["Себестоимость"] = per_sku["Себестоимость"].fillna(0.0)
    per_sku["Себестоимость (итого)"] = per_sku["Себестоимость"] * per_sku["Продано (шт)"]

    per_sku["Комиссия МП"] = per_sku.apply(
        lambda r: r["Выручка"] * COMMISSION_RATES.get(r["Маркетплейс"], 0.18), axis=1
    )
    per_sku["Логистика"] = per_sku.apply(
        lambda r: (
            LOGISTICS_MAGISTRAL.get(r["Маркетплейс"], 300)
            + LOGISTICS_LAST_MILE.get(r["Маркетплейс"], 200)
        )
        * r["Продано (шт)"],
        axis=1,
    )

    ad_by_mp = ads.groupby("Маркетплейс", as_index=False).agg(
        **{"Расходы_на_рекламу": ("Расходы_на_рекламу", "sum"), "Выручка_с_рекламы": ("Выручка_с_рекламы", "sum")}
    )
    ad_by_mp["ДРР_доля"] = (
        ad_by_mp["Расходы_на_рекламу"] / ad_by_mp["Выручка_с_рекламы"].replace(0, pd.NA)
    ).fillna(0.0)
    drr_map = dict(zip(ad_by_mp["Маркетплейс"], ad_by_mp["ДРР_доля"]))

    per_sku["Реклама (ДРР)"] = per_sku.apply(
        lambda r: r["Выручка"] * drr_map.get(r["Маркетплейс"], 0.0), axis=1
    )

    per_sku["Чистая прибыль"] = (
        per_sku["Выручка"]
        - per_sku["Себестоимость (итого)"]
        - per_sku["Комиссия МП"]
        - per_sku["Логистика"]
        - per_sku["Реклама (ДРР)"]
    )
    per_sku["Маржинальность (%)"] = (
        per_sku["Чистая прибыль"] / per_sku["Выручка"] * 100
    ).round(1)

    for col in ["Выручка", "Себестоимость (итого)", "Комиссия МП", "Логистика", "Реклама (ДРР)", "Чистая прибыль"]:
        per_sku[col] = per_sku[col].round(2)

    return per_sku.sort_values("Чистая прибыль", ascending=False)
