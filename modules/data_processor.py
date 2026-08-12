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
                    "Отправление": posting.get("posting_number", ""),
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
                "Отправление": None,
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


def aggregate_ozon_finance_by_posting(transactions: list[dict]) -> pd.DataFrame:
    """
    Сворачивает операции /v3/finance/transaction/list в реальную комиссию и
    логистику по каждому posting_number.

    Проверено на реальных данных: sale_commission — комиссия маркетплейса
    (уже отрицательная в ответе). services[] — построчные услуги (курьер
    последней мили, магистраль, эквайринг и т.п.); delivery_charge/
    return_delivery_charge — доп. поля логистики у некоторых типов операций.
    Суммируется по всем операциям за период для каждого отправления —
    комиссия и разные логистические начисления по одному заказу могут
    приходить отдельными операциями.
    """
    totals: dict[str, dict[str, float]] = {}
    for op in transactions:
        posting_number = (op.get("posting") or {}).get("posting_number")
        if not posting_number:
            continue
        commission = -(op.get("sale_commission") or 0)
        services_cost = -sum(s.get("price", 0) for s in (op.get("services") or []))
        delivery_cost = -((op.get("delivery_charge") or 0) + (op.get("return_delivery_charge") or 0))

        entry = totals.setdefault(posting_number, {"Комиссия_реальная": 0.0, "Логистика_реальная": 0.0})
        entry["Комиссия_реальная"] += commission
        entry["Логистика_реальная"] += services_cost + delivery_cost

    if not totals:
        return pd.DataFrame(columns=["Отправление", "Комиссия_реальная", "Логистика_реальная"])

    return pd.DataFrame(
        [{"Отправление": k, **v} for k, v in totals.items()]
    )


# Группировка реальных операций Ozon (/v3/finance/transaction/list) в статьи
# P&L-сводки, по образцу раздела "Финансы -> Экономика магазина" в кабинете
# Ozon. Список operation_type собран по факту из ответа реального аккаунта —
# необязательно исчерпывающий: всё, что не попало в маппинг, уходит в
# "Прочее (неклассифицировано)" и остаётся видно в детальной таблице, а не
# теряется молча.
OZON_FINANCE_CATEGORY_MAP: dict[str, str] = {
    "OperationAgentDeliveredToCustomer": "Продажи (доставлено покупателям)",
    "ClientReturnAgentOperation": "Возвраты, отмены, невыкупы",
    "OperationReturnGoodsFBSofRMS": "Возвраты, отмены, невыкупы",
    "OperationItemReturn": "Возвраты, отмены, невыкупы",
    "MarketplaceServiceRedistributionOfDeliveryServicesRFBS": "Логистика и доставка",
    "MarketplaceAgencyFeeAggregator3plRFBS": "Логистика и доставка",
    "MarketplaceSellerReexposureDeliveryReturnOperation": "Логистика и доставка",
    "OperationMarketplaceCostPerClick": "Реклама и продвижение (основной кабинет)",
    "MarketplaceServiceBrandCommission": "Реклама и продвижение (основной кабинет)",
    "OperationPromotionWithCostPerOrder": "Реклама и продвижение (основной кабинет)",
    "OperationSubscriptionPremiumPlus": "Реклама и продвижение (основной кабинет)",
    "OperationMarketplaceAcceleratedProductReviews": "Реклама и продвижение (основной кабинет)",
    "OperationLabelOriginal": "Прочие услуги Ozon",
    "OperationLabelBrandVerified": "Прочие услуги Ozon",
    "MarketplaceRedistributionOfAcquiringOperation": "Прочие услуги Ozon",
    "InsuranceServiceSellerItem": "Прочие услуги Ozon",
    "OperationMarketplaceServiceVolumeWeightCharacsProcessing": "Прочие услуги Ozon",
    "OperationMarketplaceItemTemporaryStorageRedistribution": "Прочие услуги Ozon",
    "DefectFineCancellation": "Штрафы и корректировки",
    "DefectFineCancellationCancelled": "Штрафы и корректировки",
    "AccrualWithoutDocs": "Компенсации",
}
FINANCE_ROW_ORDER = [
    "Продажи (доставлено покупателям)",
    "Возвраты, отмены, невыкупы",
    "Логистика и доставка",
    "Реклама и продвижение (основной кабинет)",
    "Прочие услуги Ozon",
    "Штрафы и корректировки",
    "Компенсации",
    "Прочее (неклассифицировано)",
]


def build_ozon_finance_waterfall(
    transactions: list[dict],
    ozon_sales: pd.DataFrame,
    cost_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    P&L-сводка по реальным операциям Ozon: от начислений по статьям до
    суммы, реально причитающейся к перечислению на расчётный счёт
    ("ИТОГО начислено"), и дальше — до чистой прибыли после себестоимости.

    ozon_sales — заказы Ozon (не отменённые), используются только для
    расчёта себестоимости проданного за период; сама выручка/комиссия/
    логистика берутся из транзакций, а не из заказов.
    """
    if not transactions:
        return pd.DataFrame(columns=["Статья", "Сумма"])

    tx_df = pd.DataFrame(transactions)
    tx_df["amount"] = tx_df["amount"].astype(float)
    tx_df["Статья"] = tx_df["operation_type"].map(OZON_FINANCE_CATEGORY_MAP).fillna("Прочее (неклассифицировано)")

    by_category = tx_df.groupby("Статья", as_index=False)["amount"].sum().rename(columns={"amount": "Сумма"})
    by_category["_order"] = by_category["Статья"].apply(
        lambda c: FINANCE_ROW_ORDER.index(c) if c in FINANCE_ROW_ORDER else len(FINANCE_ROW_ORDER)
    )
    by_category = by_category.sort_values("_order").drop(columns="_order")

    total_accrued = float(tx_df["amount"].sum())

    cost_merged = ozon_sales.merge(cost_df[["Артикул", "Себестоимость"]], on="Артикул", how="left")
    cost_merged["Себестоимость"] = cost_merged["Себестоимость"].fillna(0.0)
    cogs = float((cost_merged["Себестоимость"] * cost_merged["Количество"]).sum())

    rows = by_category.to_dict("records")
    rows.append({"Статья": "ИТОГО начислено (к расчётному счёту)", "Сумма": total_accrued})
    rows.append({"Статья": "Себестоимость проданных товаров", "Сумма": -cogs})
    rows.append({"Статья": "Чистая прибыль (после себестоимости)", "Сумма": total_accrued - cogs})

    return pd.DataFrame(rows)


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
    ozon_finance_by_posting: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Считает юнит-экономику по каждому артикулу: от выручки до чистой прибыли.

    Складывается из:
      Выручка - Себестоимость - Комиссия МП - Логистика - Реклама (ДРР на ед.)

    Комиссия и логистика берутся из реального финансового отчёта Ozon
    (ozon_finance_by_posting, см. aggregate_ozon_finance_by_posting), если он
    передан — построчно, с распределением суммы отправления между товарами
    пропорционально их доле в сумме отправления. Для строк без реального
    совпадения (WB, ещё не сведённые Ozon-отправления, mock-данные) — плоская
    оценка по ставкам из mock_data (COMMISSION_RATES/LOGISTICS_*).
    """
    sales = orders[orders["Статус"] != "Отменён"].copy()

    flat_commission = sales.apply(
        lambda r: r["Сумма"] * COMMISSION_RATES.get(r["Маркетплейс"], 0.18), axis=1
    )
    flat_logistics = sales.apply(
        lambda r: (
            LOGISTICS_MAGISTRAL.get(r["Маркетплейс"], 300) + LOGISTICS_LAST_MILE.get(r["Маркетплейс"], 200)
        )
        * r["Количество"],
        axis=1,
    )

    has_real_ozon_data = (
        ozon_finance_by_posting is not None
        and not ozon_finance_by_posting.empty
        and "Отправление" in sales.columns
    )
    if has_real_ozon_data:
        posting_totals = sales.groupby("Отправление")["Сумма"].transform("sum")
        row_share = (sales["Сумма"] / posting_totals.replace(0, pd.NA)).fillna(1.0)
        real = sales[["Отправление"]].merge(ozon_finance_by_posting, on="Отправление", how="left")

        sales["Комиссия МП"] = (real["Комиссия_реальная"] * row_share).where(
            real["Комиссия_реальная"].notna(), flat_commission
        )
        sales["Логистика"] = (real["Логистика_реальная"] * row_share).where(
            real["Логистика_реальная"].notna(), flat_logistics
        )
    else:
        sales["Комиссия МП"] = flat_commission
        sales["Логистика"] = flat_logistics

    per_sku = (
        sales.groupby(["Артикул", "Категория", "Селлер", "Маркетплейс"], as_index=False)
        .agg(
            **{
                "Продано (шт)": ("Количество", "sum"),
                "Выручка": ("Сумма", "sum"),
                "Комиссия МП": ("Комиссия МП", "sum"),
                "Логистика": ("Логистика", "sum"),
            }
        )
    )
    per_sku = per_sku.merge(cost_df[["Артикул", "Себестоимость"]], on="Артикул", how="left")
    per_sku["Себестоимость"] = per_sku["Себестоимость"].fillna(0.0)
    per_sku["Себестоимость (итого)"] = per_sku["Себестоимость"] * per_sku["Продано (шт)"]

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
