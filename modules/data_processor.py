"""
Логика обработки данных: объединение API + себестоимости, расчёт KPI
и юнит-экономики. Вся тяжёлая обработка ведётся в Pandas.
"""

from __future__ import annotations

import numpy as np
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

# Операции Ozon, относящиеся к рекламе/продвижению через ОСНОВНОЙ кабинет
# (не Performance API). Реальный расход по ним уже учитывается отдельно
# через Performance API ("Реклама (ДРР)" в юнит-экономике) — их нужно
# исключать из "Логистика_реальная" в aggregate_ozon_finance_by_posting,
# иначе реклама вычитается дважды. Проверено на реальных данных: у всех
# операций этого списка sale_commission и accruals_for_sale всегда равны 0,
# то есть их можно полностью пропускать без потери комиссии.
AD_PROMOTION_OPERATION_TYPES = {
    "OperationMarketplaceCostPerClick",
    "MarketplaceServiceBrandCommission",
    "OperationPromotionWithCostPerOrder",
    "OperationSubscriptionPremiumPlus",
    "OperationLabelOriginal",
    "OperationLabelBrandVerified",
    "OperationMarketplaceAcceleratedProductReviews",
}


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


def transform_ozon_stocks(items: list[dict]) -> pd.DataFrame:
    """Приводит остатки Ozon (/v4/product/info/stocks) к плоской таблице по артикулу."""
    rows = []
    for item in items:
        stocks = item.get("stocks", [])
        present_total = sum(s.get("present", 0) for s in stocks)
        reserved_total = sum(s.get("reserved", 0) for s in stocks)
        rows.append(
            {
                "Артикул": item.get("offer_id", "N/A"),
                "Остаток (всего)": present_total,
                "Резерв": reserved_total,
                "Доступно": present_total - reserved_total,
            }
        )
    return pd.DataFrame(rows)


def build_production_forecast(
    stocks: pd.DataFrame,
    orders: pd.DataFrame,
    weeks_lookback: int = 8,
    production_weeks: float = 2,
    order_cycle_weeks: float = 1,
) -> pd.DataFrame:
    """
    Прогноз заказа в производство по каждому артикулу.

    Логика: цикл производства — production_weeks недель, заказ делается раз в
    order_cycle_weeks недель. Партия, заказанная сегодня, доедет только через
    production_weeks недель — а следующая возможность доказать нехватку
    появится только через order_cycle_weeks после этого. Поэтому запаса на
    складе должно хватать на весь цикл (production_weeks + order_cycle_weeks),
    иначе перед следующей поставкой возникнет дефицит.

    Скорость продаж — среднее количество проданных (доставленных) штук в
    неделю за последние weeks_lookback недель, по фактическим заказам.
    """
    columns = [
        "Артикул", "Остаток (всего)", "Резерв", "Доступно",
        "Продажи/нед", "Хватит на (нед)", "К заказу (шт)", "Статус",
    ]
    if stocks.empty:
        return pd.DataFrame(columns=columns)

    cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=weeks_lookback)
    recent_sales = orders[(orders["Статус"] == "Доставлен") & (orders["Дата"] >= cutoff)]
    weekly_sales = (
        recent_sales.groupby("Артикул")["Количество"].sum() / weeks_lookback
    ).reset_index().rename(columns={"Количество": "Продажи/нед"})

    forecast = stocks.merge(weekly_sales, on="Артикул", how="left")
    forecast["Продажи/нед"] = forecast["Продажи/нед"].fillna(0.0).round(1)

    cycle_weeks = production_weeks + order_cycle_weeks
    sales_per_week = forecast["Продажи/нед"].to_numpy(dtype=float)
    available = forecast["Доступно"].to_numpy(dtype=float)
    weeks_left = np.divide(
        available, sales_per_week, out=np.full_like(available, np.nan), where=sales_per_week > 0
    )
    forecast["Хватит на (нед)"] = pd.Series(weeks_left, index=forecast.index).round(1)
    forecast["К заказу (шт)"] = (
        (forecast["Продажи/нед"] * cycle_weeks - forecast["Доступно"]).clip(lower=0)
    ).round(0).astype(int)

    def _status(row: pd.Series) -> str:
        weeks_left = row["Хватит на (нед)"]
        if row["Продажи/нед"] == 0 or pd.isna(weeks_left):
            return "⚪ Нет продаж"
        if weeks_left < production_weeks:
            return "🔴 Срочно"
        if weeks_left < cycle_weeks:
            return "🟡 Пора заказывать"
        return "🟢 Достаточно"

    forecast["Статус"] = forecast.apply(_status, axis=1)

    priority = {"🔴": 0, "🟡": 1, "🟢": 2, "⚪": 3}
    forecast["_priority"] = forecast["Статус"].str[0].map(priority).fillna(9)
    forecast = forecast.sort_values(["_priority", "Хватит на (нед)"]).drop(columns="_priority")

    return forecast[columns]


def aggregate_ozon_finance_by_posting(transactions: list[dict]) -> pd.DataFrame:
    """
    Сворачивает операции /v3/finance/transaction/list в реальную комиссию и
    логистику по каждому posting_number.

    Комиссия — sale_commission (уже отрицательная в ответе). Логистика — это
    ОСТАТОК (amount - accruals_for_sale - sale_commission), а не только
    services[]/delivery_charge: проверено на реальных данных, что часть
    операций (например, "Услуги доставки Партнерами Ozon на схеме realFBS")
    несёт всю свою стоимость прямо в "amount", с ПУСТЫМИ services[] и
    delivery_charge=0 — при подсчёте только по services[]/delivery_charge
    терялось ~95% реальной логистики (49 620₽ вместо 972 201₽ за проверенный
    месяц). Остаток — тот же метод, что уже сверен построчно с кабинетом Ozon
    в build_ozon_finance_waterfall.

    Рекламные операции (AD_PROMOTION_OPERATION_TYPES) исключены полностью:
    несмотря на название "по отправлению", у Ozon клики/продвижение бренда
    почти всегда привязаны к конкретному posting_number (проверено: 17514
    из 17610 таких операций за 90 дней). Если их не исключить, реклама
    задваивается — один раз здесь как "Логистика", второй раз в юнит-
    экономике как "Реклама (ДРР)" из Performance API. У этих операций
    sale_commission/accruals_for_sale всегда 0, так что пропуск безопасен.
    """
    totals: dict[str, dict[str, float]] = {}
    for op in transactions:
        posting_number = (op.get("posting") or {}).get("posting_number")
        if not posting_number:
            continue
        if op.get("operation_type") in AD_PROMOTION_OPERATION_TYPES:
            continue
        accruals = float(op.get("accruals_for_sale") or 0)
        raw_commission = float(op.get("sale_commission") or 0)
        commission = -raw_commission
        # residual положительный = реальный расход (услуги/логистика), знак Ozon инвертирован,
        # так же, как commission/services_cost/delivery_cost выше — по этой же причине минус.
        residual_cost = -(float(op.get("amount") or 0) - accruals - raw_commission)

        entry = totals.setdefault(posting_number, {"Комиссия_реальная": 0.0, "Логистика_реальная": 0.0})
        entry["Комиссия_реальная"] += commission
        entry["Логистика_реальная"] += residual_cost

    if not totals:
        return pd.DataFrame(columns=["Отправление", "Комиссия_реальная", "Логистика_реальная"])

    return pd.DataFrame(
        [{"Отправление": k, **v} for k, v in totals.items()]
    )


# Разложение реальных операций Ozon (/v3/finance/transaction/list) на статьи
# P&L-сводки, по образцу раздела "Финансы -> Экономика магазина -> Детализация
# начислений" в кабинете Ozon (см. seller-edu.ozon.ru, "Работа с финансами").
# Официальная формула Ozon для "Итого":
#   Продажи и возвраты + отрицательные начисления + положительные начисления
# У нас это total_accrued = сумма всех статей ниже — структурно то же самое.
# "Продажи" и "Возвраты" по документации — РАЗНЫЕ строки (начисления по
# доставленным заказам отдельно от начисленного/списанного по возвращённым),
# различаем их через поле "type" операции ('returns' vs остальное).
# "Вознаграждение Ozon" по документации считается ОБЩИМ по доставленным и
# возвращённым заказам (не делится) — так и оставляем одной суммой.
# Каждая операция раскладывается на accruals_for_sale (валовая продажа —
# выручка + баллы за скидки + программы партнёров, единым числом: см. сверку
# ниже) + sale_commission (комиссия Ozon) + остаток (услуги/логистика/реклама).
# Сверено построчно с реальным кабинетом (скриншот "Финансы" за календарный
# месяц): "Продажи и возвраты" (= accruals_sum), "Вознаграждение Ozon",
# "Реклама и продвижение", "Прочие поступления" и "Итого" совпадают точно;
# несколько мелких статей Ozon (услуги партнёров/доставки/штрафы) у нас
# объединены в одну строку "Логистика и прочие услуги", т.к. Ozon не
# публикует точное правило их разделения через этот API — сумма объединённой
# строки сверена и совпадает с точностью до рублей.
OTHER_INCOME_TRANSACTION_TYPES = {"transfer_delivery", "compensation"}


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

    sales_accrual_sum = 0.0
    returns_accrual_sum = 0.0
    commission_sum = 0.0
    ad_promotion_sum = 0.0
    other_income_sum = 0.0
    other_services_sum = 0.0

    for t in transactions:
        amount = float(t.get("amount") or 0)
        accruals = float(t.get("accruals_for_sale") or 0)
        commission = float(t.get("sale_commission") or 0)
        residual = amount - accruals - commission  # услуги/логистика/реклама, встроенные в эту же операцию

        op_type = t.get("operation_type", "")
        t_type = t.get("type", "")

        if t_type == "returns":
            returns_accrual_sum += accruals
        else:
            sales_accrual_sum += accruals
        commission_sum += commission

        if t_type in OTHER_INCOME_TRANSACTION_TYPES:
            other_income_sum += residual
        elif op_type in AD_PROMOTION_OPERATION_TYPES:
            ad_promotion_sum += residual
        else:
            other_services_sum += residual

    total_accrued = (
        sales_accrual_sum + returns_accrual_sum + commission_sum
        + ad_promotion_sum + other_income_sum + other_services_sum
    )

    cost_merged = ozon_sales.merge(cost_df[["Артикул", "Себестоимость"]], on="Артикул", how="left")
    cost_merged["Себестоимость"] = cost_merged["Себестоимость"].fillna(0.0)
    cogs = float((cost_merged["Себестоимость"] * cost_merged["Количество"]).sum())

    rows = [
        {"Статья": "Продажи (начислено по доставленным)", "Сумма": sales_accrual_sum},
        {"Статья": "Возвраты (по возвращённым заказам)", "Сумма": returns_accrual_sum},
        {"Статья": "Вознаграждение Ozon (комиссия)", "Сумма": commission_sum},
        {"Статья": "Продвижение и реклама", "Сумма": ad_promotion_sum},
        {"Статья": "Логистика и прочие услуги", "Сумма": other_services_sum},
        {"Статья": "Прочие поступления (компенсации, перерасчёты)", "Сумма": other_income_sum},
        {"Статья": "ИТОГО начислено (к расчётному счёту)", "Сумма": total_accrued},
        {"Статья": "Себестоимость проданных товаров", "Сумма": -cogs},
        {"Статья": "Чистая прибыль (после себестоимости)", "Сумма": total_accrued - cogs},
    ]
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


def build_status_cost_breakdown(orders: pd.DataFrame, cost_df: pd.DataFrame) -> pd.DataFrame:
    """
    Разбивка выручки и себестоимости по статусу: "Доставлен" (деньги уже
    подтверждены) отдельно от "В пути" (заказ ещё может быть отменён/возвращён
    до фактической доставки — считать эту выручку окончательной прибылью рано).
    Отменённые заказы не входят.
    """
    sales = orders[orders["Статус"] != "Отменён"].copy()
    merged = sales.merge(cost_df[["Артикул", "Себестоимость"]], on="Артикул", how="left")
    merged["Себестоимость_найдена"] = merged["Себестоимость"].notna()
    merged["Себестоимость"] = merged["Себестоимость"].fillna(0.0)
    merged["Себестоимость (итого)"] = merged["Себестоимость"] * merged["Количество"]

    result = (
        merged.groupby("Статус", as_index=False)
        .agg(
            **{
                "Количество (шт)": ("Количество", "sum"),
                "Выручка": ("Сумма", "sum"),
                "Себестоимость (итого)": ("Себестоимость (итого)", "sum"),
                "Без себестоимости (шт)": ("Себестоимость_найдена", lambda s: int((~s).sum())),
            }
        )
    )
    result["Валовая прибыль"] = result["Выручка"] - result["Себестоимость (итого)"]
    result["Валовая маржа (%)"] = (result["Валовая прибыль"] / result["Выручка"].replace(0, pd.NA) * 100).round(1)
    status_order = {"Доставлен": 0, "В пути": 1}
    return result.sort_values(by="Статус", key=lambda s: s.map(status_order).fillna(99))


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

    Только по ДОСТАВЛЕННЫМ заказам — так же, как Ozon считает "Продажи" в
    своём отчёте (см. build_ozon_finance_waterfall). Заказы "В пути" сюда не
    входят: у Ozon по ним ещё нет реальных начислений (комиссия/логистика
    появляются только при доставке), а заказ ещё может быть отменён — включать
    их значило бы смешивать подтверждённую прибыль с ещё не наступившей. Они
    отдельно видны в таблице "Доставлено vs В пути" (build_status_cost_breakdown).

    Комиссия и логистика берутся из реального финансового отчёта Ozon
    (ozon_finance_by_posting, см. aggregate_ozon_finance_by_posting), если он
    передан — построчно, с распределением суммы отправления между товарами
    пропорционально их доле в сумме отправления. Для строк без реального
    совпадения (WB, ещё не сведённые Ozon-отправления, mock-данные) — плоская
    оценка по ставкам из mock_data (COMMISSION_RATES/LOGISTICS_*).
    """
    sales = orders[orders["Статус"] == "Доставлен"].reset_index(drop=True).copy()

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
    per_sku["Себестоимость_найдена"] = per_sku["Себестоимость"].notna()
    per_sku["Себестоимость"] = per_sku["Себестоимость"].fillna(0.0)
    per_sku["Себестоимость (итого)"] = per_sku["Себестоимость"] * per_sku["Продано (шт)"]
    per_sku["Цена (1 шт)"] = (per_sku["Выручка"] / per_sku["Продано (шт)"]).round(2)
    per_sku = per_sku.rename(columns={"Себестоимость": "Себестоимость (1 шт)"})

    # Делим реальный расход на рекламу пропорционально ВЫРУЧКЕ, а не по
    # "Выручка_с_рекламы" из Performance API — та учитывает только заказы,
    # которые сама Ozon-статистика атрибутировала рекламе, и обычно меньше
    # общей выручки. Деление на неё завышало бы сумму (проверено: 1 427 663₽
    # разнесённой рекламы против 1 114 712₽ реального расхода). Так сумма
    # "Реклама (ДРР)" по всем артикулам всегда точно равна реальному расходу.
    ad_spend_by_mp = ads.groupby("Маркетплейс")["Расходы_на_рекламу"].sum()
    revenue_by_mp = per_sku.groupby("Маркетплейс")["Выручка"].sum()
    drr_map = (ad_spend_by_mp / revenue_by_mp.replace(0, pd.NA)).fillna(0.0).to_dict()

    per_sku["Реклама (ДРР)"] = per_sku.apply(
        lambda r: r["Выручка"] * drr_map.get(r["Маркетплейс"], 0.0), axis=1
    )

    # Порядок вычетов повторяет реальный порядок денег: сначала то, что
    # удерживает сам Ozon (комиссия, логистика, реклама) — это отражено в
    # "Начислено Ozon" (сверяется с "ИТОГО начислено" в финансовой сводке
    # выше). Себестоимость — отдельный, уже не-Ozon вычет поверх этого.
    per_sku["Начислено Ozon (после его вычетов)"] = (
        per_sku["Выручка"] - per_sku["Комиссия МП"] - per_sku["Логистика"] - per_sku["Реклама (ДРР)"]
    )
    per_sku["Чистая прибыль"] = per_sku["Начислено Ozon (после его вычетов)"] - per_sku["Себестоимость (итого)"]
    per_sku["Маржинальность (%)"] = (
        per_sku["Чистая прибыль"] / per_sku["Выручка"] * 100
    ).round(1)

    for col in [
        "Выручка", "Комиссия МП", "Логистика", "Реклама (ДРР)",
        "Начислено Ozon (после его вычетов)", "Себестоимость (итого)", "Чистая прибыль",
    ]:
        per_sku[col] = per_sku[col].round(2)

    column_order = [
        "Артикул",
        "Категория",
        "Селлер",
        "Маркетплейс",
        "Продано (шт)",
        "Цена (1 шт)",
        "Выручка",
        "Комиссия МП",
        "Логистика",
        "Реклама (ДРР)",
        "Начислено Ozon (после его вычетов)",
        "Себестоимость (1 шт)",
        "Себестоимость (итого)",
        "Чистая прибыль",
        "Маржинальность (%)",
        "Себестоимость_найдена",
    ]
    per_sku = per_sku[column_order]

    return per_sku.sort_values("Чистая прибыль", ascending=False)
