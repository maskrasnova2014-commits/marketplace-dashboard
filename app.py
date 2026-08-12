"""
Дашборд аналитики маркетплейсов (Ozon + Wildberries) для ниши "Мебель".

Запуск: streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from modules import data_processor as dp
from modules import mock_data
from modules.ozon_api import OzonAPIError, OzonClient
from modules.ozon_ads_api import OzonAdsAPIError, OzonAdsClient
from modules.wb_api import WBAPIError, WBAdsClient, WBClient

load_dotenv()  # подхватывает ключи из локального .env, если он есть

st.set_page_config(page_title="Аналитика маркетплейсов — Мебель", layout="wide")

# При деплое на Streamlit Community Cloud ключи задаются в Settings -> Secrets
# (st.secrets), а не в .env. Подмешиваем их в os.environ, чтобы весь остальной
# код (os.environ.get(...)) работал одинаково и локально, и в облаке.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass


def _check_password() -> bool:
    """
    Простой пароль на вход. Обязателен при публичном деплое (Streamlit Cloud
    даёт публичную ссылку) — иначе реальные продажи и маржу увидит любой,
    у кого есть URL. Если APP_PASSWORD не задан (напр. при локальном
    запуске для разработки), экран входа не показывается.
    """
    app_password = os.environ.get("APP_PASSWORD", "")
    if not app_password:
        return True
    if st.session_state.get("_authed"):
        return True

    st.title("🔒 Вход")
    pwd = st.text_input("Пароль", type="password", key="_login_pwd")
    if st.button("Войти"):
        if pwd == app_password:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("Неверный пароль")
    return False


if not _check_password():
    st.stop()


def _load_ozon_accounts_from_env() -> list[dict]:
    """
    Читает кабинеты Ozon из .env. Поддерживает несколько кабинетов через
    суффикс _1, _2, ... (OZON_CLIENT_ID_1, OZON_CLIENT_ID_2, ...), а также
    старый формат без суффикса (OZON_CLIENT_ID) как кабинет №1 — чтобы уже
    настроенный .env продолжал работать без изменений.
    """
    accounts = []
    for i in range(1, 11):
        suffix = f"_{i}"
        client_id = os.environ.get(f"OZON_CLIENT_ID{suffix}") or (os.environ.get("OZON_CLIENT_ID") if i == 1 else "")
        api_key = os.environ.get(f"OZON_API_KEY{suffix}") or (os.environ.get("OZON_API_KEY") if i == 1 else "")
        seller_label = os.environ.get(f"OZON_SELLER_LABEL{suffix}") or (
            os.environ.get("OZON_SELLER_LABEL") if i == 1 else ""
        )
        ads_client_id = os.environ.get(f"OZON_ADS_CLIENT_ID{suffix}", "")
        ads_client_secret = os.environ.get(f"OZON_ADS_CLIENT_SECRET{suffix}", "")
        if not (client_id or api_key or seller_label or ads_client_id or ads_client_secret):
            continue
        accounts.append(
            {
                "uid": f"env_{i}",
                "client_id": client_id or "",
                "api_key": api_key or "",
                "seller_label": seller_label or f"Ozon-кабинет {i}",
                "ads_client_id": ads_client_id,
                "ads_client_secret": ads_client_secret,
            }
        )
    if not accounts:
        accounts.append(
            {
                "uid": "env_1",
                "client_id": "",
                "api_key": "",
                "seller_label": "Ozon-кабинет 1",
                "ads_client_id": "",
                "ads_client_secret": "",
            }
        )
    return accounts


# --------------------------------------------------------------------------
# Инициализация session_state (ключи по умолчанию берутся из .env, если заданы)
# --------------------------------------------------------------------------
DEFAULTS = {
    "ozon_accounts": _load_ozon_accounts_from_env(),
    "ozon_account_seq": 100,  # счётчик для uid новых кабинетов, добавленных вручную
    "wb_stats_token": os.environ.get("WB_STATS_TOKEN") or os.environ.get("WB_API_TOKEN", ""),
    "wb_ads_token": os.environ.get("WB_ADS_TOKEN", ""),
    "wb_seller_label": os.environ.get("WB_SELLER_LABEL", "WB-кабинет"),
    "cost_df": None,
    "date_from": date.today() - timedelta(days=90),
    "date_to": date.today(),
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Загрузка данных (с кэшированием и безопасным fallback на моки)
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Загрузка заказов...")
def load_mock_orders(date_from: date, date_to: date) -> pd.DataFrame:
    return mock_data.generate_orders(date_from, date_to)


@st.cache_data(ttl=3600, show_spinner="Загрузка данных по рекламе...")
def load_mock_ads(date_from: date, date_to: date) -> pd.DataFrame:
    return mock_data.generate_advertising(date_from, date_to)


RETRY_COOLDOWN_SECONDS = 65  # WB/Ozon отдают 429 чаще ~1 запроса/мин — не долбим их каждый rerun


@st.cache_data(ttl=3600, show_spinner="Загрузка заказов Ozon...")
def _fetch_ozon_orders_raw(client_id: str, api_key: str, seller_label: str, date_from: date, date_to: date):
    """Сырой запрос к Ozon. Кэшируется ТОЛЬКО при успехе — исключения Streamlit не кэширует."""
    client = OzonClient(client_id, api_key)
    postings = client.get_fbo_postings(date_from, date_to)
    postings += client.get_fbs_postings(date_from, date_to)
    return dp.transform_ozon_postings(postings, seller_label)


@st.cache_data(ttl=3600, show_spinner="Загрузка заказов Wildberries...")
def _fetch_wb_orders_raw(api_token: str, seller_label: str, date_from: date):
    """Сырой запрос к WB. Кэшируется ТОЛЬКО при успехе — исключения Streamlit не кэширует."""
    client = WBClient(api_token)
    sales = client.get_sales(date_from)
    return dp.transform_wb_sales(sales, seller_label)


@st.cache_data(ttl=3600, show_spinner="Загрузка рекламы Ozon...")
def _fetch_ozon_ads_raw(ads_client_id: str, ads_client_secret: str, seller_label: str, date_from: date, date_to: date):
    """Сырой запрос к Ozon Performance API (реклама)."""
    client = OzonAdsClient(ads_client_id, ads_client_secret)
    stats = client.get_daily_statistics(date_from, date_to)
    return dp.transform_ozon_ads(stats, seller_label)


@st.cache_data(ttl=3600, show_spinner="Загрузка рекламы Wildberries...")
def _fetch_wb_ads_raw(api_token: str, seller_label: str, date_from: date, date_to: date):
    """Сырой запрос к WB Advert API (реклама)."""
    client = WBAdsClient(api_token)
    costs = client.get_advert_costs(date_from, date_to)
    return dp.transform_wb_ads(costs, seller_label)


@st.cache_data(ttl=3600, show_spinner="Загрузка финансовых данных Ozon...")
def _fetch_ozon_finance_raw(client_id: str, api_key: str, date_from: date, date_to: date):
    """Сырые финансовые операции Ozon за период (используются и для юнит-экономики, и для сводки P&L)."""
    client = OzonClient(client_id, api_key)
    return client.get_finance_transactions(date_from, date_to)


def _fetch_with_cooldown(source_key: str, fetch_fn, error_types, label: str):
    """
    Тонкая (не кэшируемая) обвязка: ловит ошибки API и не даёт им "залипать"
    на час, как было бы при кэшировании самого исключения. Хранит время
    последней неудачи и реальный retry_after (если его прислал маркетплейс,
    напр. заголовок X-Ratelimit-Reset у WB) в session_state, чтобы не бить
    по API чаще, чем реально разрешено.
    """
    fail_key = f"_last_fail_{source_key}"
    msg_key = f"_last_fail_msg_{source_key}"
    cooldown_key = f"_cooldown_{source_key}"
    now = pd.Timestamp.now()

    last_fail = st.session_state.get(fail_key)
    cooldown = st.session_state.get(cooldown_key, RETRY_COOLDOWN_SECONDS)
    if last_fail is not None and (now - last_fail).total_seconds() < cooldown:
        wait = int(cooldown - (now - last_fail).total_seconds())
        wait_str = f"{wait // 60} мин {wait % 60} сек" if wait >= 60 else f"{wait} сек"
        st.warning(f"{st.session_state.get(msg_key, label + ': временная ошибка API')} "
                   f"Повторная попытка возможна через ~{wait_str}. Использую тестовые данные для {label}.")
        return None

    try:
        result = fetch_fn()
        st.session_state.pop(fail_key, None)
        st.session_state.pop(cooldown_key, None)
        return result if len(result) > 0 else None
    except error_types as exc:
        st.session_state[fail_key] = now
        st.session_state[cooldown_key] = getattr(exc, "retry_after_seconds", None) or RETRY_COOLDOWN_SECONDS
        st.session_state[msg_key] = f"{label} API: {exc}."
        st.warning(f"{label} API: {exc}. Использую тестовые данные для {label}.")
        return None
    except Exception as exc:  # непредвиденный формат ответа API
        st.session_state[fail_key] = now
        st.session_state[cooldown_key] = RETRY_COOLDOWN_SECONDS
        st.session_state[msg_key] = f"Не удалось обработать ответ {label} API ({exc})."
        st.warning(f"Не удалось обработать ответ {label} API ({exc}). Использую тестовые данные для {label}.")
        return None


def fetch_ozon_orders(client_id: str, api_key: str, seller_label: str, date_from: date, date_to: date):
    return _fetch_with_cooldown(
        f"ozon_orders_{client_id}",
        lambda: _fetch_ozon_orders_raw(client_id, api_key, seller_label, date_from, date_to),
        OzonAPIError,
        f"Ozon ({seller_label})",
    )


def fetch_wb_orders(api_token: str, seller_label: str, date_from: date):
    return _fetch_with_cooldown(
        "wb_orders",
        lambda: _fetch_wb_orders_raw(api_token, seller_label, date_from),
        WBAPIError,
        "Wildberries",
    )


def fetch_ozon_ads(ads_client_id: str, ads_client_secret: str, seller_label: str, date_from: date, date_to: date):
    return _fetch_with_cooldown(
        f"ozon_ads_{ads_client_id}",
        lambda: _fetch_ozon_ads_raw(ads_client_id, ads_client_secret, seller_label, date_from, date_to),
        OzonAdsAPIError,
        f"Ozon Performance ({seller_label})",
    )


def fetch_wb_ads(api_token: str, seller_label: str, date_from: date, date_to: date):
    return _fetch_with_cooldown(
        "wb_ads",
        lambda: _fetch_wb_ads_raw(api_token, seller_label, date_from, date_to),
        WBAPIError,
        "Wildberries Advert",
    )


def fetch_ozon_finance(client_id: str, api_key: str, seller_label: str, date_from: date, date_to: date):
    return _fetch_with_cooldown(
        f"ozon_finance_{client_id}",
        lambda: _fetch_ozon_finance_raw(client_id, api_key, date_from, date_to),
        OzonAPIError,
        f"Ozon Finance ({seller_label})",
    )


def load_orders(date_from: date, date_to: date) -> tuple[pd.DataFrame, str]:
    """
    Собирает заказы со всех настроенных кабинетов Ozon + WB. Если ключи не
    заданы или запрос не удался — безопасно переключается на mock_data.py.
    Возвращает (df, статус), где статус — "real" (всё по API), "partial"
    (часть по API, часть тестовые) или "mock" (полностью тестовые).
    """
    frames = []
    used_mock = False
    any_ozon_configured = False

    for account in st.session_state.ozon_accounts:
        if not (account["client_id"] and account["api_key"]):
            continue
        any_ozon_configured = True
        ozon_df = fetch_ozon_orders(
            account["client_id"],
            account["api_key"],
            account["seller_label"] or "Ozon-кабинет",
            date_from,
            date_to,
        )
        if ozon_df is not None:
            frames.append(ozon_df)
        else:
            used_mock = True
    if not any_ozon_configured:
        used_mock = True

    if st.session_state.wb_stats_token:
        wb_df = fetch_wb_orders(st.session_state.wb_stats_token, st.session_state.wb_seller_label, date_from)
        if wb_df is not None:
            frames.append(wb_df)
        else:
            used_mock = True
    else:
        used_mock = True

    if frames:
        real_df = pd.concat(frames, ignore_index=True)
    else:
        real_df = pd.DataFrame()

    if real_df.empty:
        return load_mock_orders(date_from, date_to), "mock"

    if used_mock:
        # Часть источников недоступна — дополняем мок-данными для полноты картины,
        # но помечаем это явно пользователю.
        st.info("Часть данных получена по API, часть — тестовые (нет ключей/ошибка для одного из источников).")
        return real_df, "partial"

    return real_df, "real"


def load_ads(date_from: date, date_to: date) -> pd.DataFrame:
    """
    Собирает рекламные расходы со всех кабинетов Ozon Performance API + WB
    Advert API. Для источников без ключей/с ошибкой честно подставляет
    только недостающий кусок из mock_data — реальные данные не подменяются
    целиком, если хотя бы часть кабинетов доступна.
    """
    frames = []
    got_ozon = False
    got_wb = False

    for account in st.session_state.ozon_accounts:
        if not (account["ads_client_id"] and account["ads_client_secret"]):
            continue
        df = fetch_ozon_ads(
            account["ads_client_id"],
            account["ads_client_secret"],
            account["seller_label"] or "Ozon-кабинет",
            date_from,
            date_to,
        )
        if df is not None:
            frames.append(df)
            got_ozon = True

    if st.session_state.wb_ads_token:
        df = fetch_wb_ads(st.session_state.wb_ads_token, st.session_state.wb_seller_label, date_from, date_to)
        if df is not None:
            frames.append(df)
            got_wb = True

    mock = load_mock_ads(date_from, date_to)
    if not got_ozon:
        frames.append(mock[mock["Маркетплейс"] == "Ozon"])
    if not got_wb:
        frames.append(mock[mock["Маркетплейс"] == "Wildberries"])

    if not got_ozon or not got_wb:
        missing = ", ".join(
            m for m, ok in [("Ozon Performance", got_ozon), ("WB Advert", got_wb)] if not ok
        )
        st.info(f"Реклама: тестовые данные используются для — {missing} (нет ключа рекламного API или ошибка).")

    return pd.concat(frames, ignore_index=True)


def load_ozon_finance_transactions(date_from: date, date_to: date) -> list[dict] | None:
    """
    Сырые финансовые операции Ozon (/v3/finance/transaction/list) по всем
    настроенным кабинетам сразу. Возвращает None, если ни один кабинет не
    настроен или данные недоступны — тогда используется плоская оценка.
    """
    all_transactions: list[dict] = []
    got_any = False
    for account in st.session_state.ozon_accounts:
        if not (account["client_id"] and account["api_key"]):
            continue
        transactions = fetch_ozon_finance(
            account["client_id"], account["api_key"], account["seller_label"] or "Ozon-кабинет", date_from, date_to
        )
        if transactions:
            all_transactions.extend(transactions)
            got_any = True

    return all_transactions if got_any else None


# --------------------------------------------------------------------------
# Сайдбар: ключи, файл себестоимости, фильтры
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    st.subheader("🔑 Кабинеты Ozon")
    st.caption(
        "Seller API (Client-Id/Api-Key) — заказы. Performance API (Client-Id/Client-Secret "
        "рекламного кабинета, раздел «Продвижение → API») — реклама. Это разные ключи."
    )
    for account in list(st.session_state.ozon_accounts):
        uid = account["uid"]
        with st.expander(f"🏪 {account['seller_label'] or 'Ozon-кабинет'}", expanded=False):
            account["seller_label"] = st.text_input("Название кабинета (селлер)", value=account["seller_label"], key=f"label_{uid}")
            account["client_id"] = st.text_input("Client-Id (Seller API)", value=account["client_id"], key=f"cid_{uid}")
            account["api_key"] = st.text_input(
                "Api-Key (Seller API)", value=account["api_key"], type="password", key=f"akey_{uid}"
            )
            st.caption("Реклама (Ozon Performance API) — необязательно:")
            account["ads_client_id"] = st.text_input(
                "Client-Id (Performance API)", value=account["ads_client_id"], key=f"adscid_{uid}"
            )
            account["ads_client_secret"] = st.text_input(
                "Client-Secret (Performance API)", value=account["ads_client_secret"], type="password", key=f"adssecret_{uid}"
            )
            if len(st.session_state.ozon_accounts) > 1:
                if st.button("🗑️ Удалить кабинет", key=f"del_{uid}"):
                    st.session_state.ozon_accounts = [
                        a for a in st.session_state.ozon_accounts if a["uid"] != uid
                    ]
                    st.rerun()

    if st.button("➕ Добавить кабинет Ozon"):
        st.session_state.ozon_account_seq += 1
        new_uid = f"manual_{st.session_state.ozon_account_seq}"
        st.session_state.ozon_accounts.append(
            {
                "uid": new_uid,
                "client_id": "",
                "api_key": "",
                "seller_label": f"Ozon-кабинет {len(st.session_state.ozon_accounts) + 1}",
                "ads_client_id": "",
                "ads_client_secret": "",
            }
        )
        st.rerun()

    st.divider()
    st.subheader("🔑 Wildberries")
    st.caption("Statistics API и Advert API — разные токены (выбираются при создании ключа в кабинете WB).")
    with st.expander("🔑 API-ключи Wildberries", expanded=False):
        st.session_state.wb_stats_token = st.text_input(
            "Токен Statistics API (заказы)", value=st.session_state.wb_stats_token, type="password", key="wb_stats_input"
        )
        st.session_state.wb_ads_token = st.text_input(
            "Токен Advert API (реклама, необязательно)", value=st.session_state.wb_ads_token, type="password", key="wb_ads_input"
        )
        st.session_state.wb_seller_label = st.text_input(
            "Название кабинета (селлер)", value=st.session_state.wb_seller_label, key="wb_label_input"
        )

    st.divider()
    st.subheader("📁 Себестоимость")
    uploaded_file = st.file_uploader("Файл (.xlsx / .csv): Артикул, Себестоимость", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                parsed = pd.read_csv(uploaded_file)
            else:
                parsed = pd.read_excel(uploaded_file)
            parsed.columns = parsed.columns.astype(str).str.strip()

            required_cols = {"Артикул", "Себестоимость"}
            missing = required_cols - set(parsed.columns)
            if missing:
                st.error(
                    f"В файле нет колонок: {', '.join(sorted(missing))}. "
                    f"Найдены колонки: {', '.join(parsed.columns)}. "
                    "Использую справочник по умолчанию."
                )
            else:
                st.session_state.cost_df = parsed
                st.success(f"Загружено строк: {len(parsed)}")
        except Exception as exc:
            st.error(f"Не удалось прочитать файл: {exc}. Использую справочник по умолчанию.")

    st.divider()
    st.subheader("🔎 Фильтры")
    date_range = st.date_input(
        "Период",
        value=(st.session_state.date_from, st.session_state.date_to),
        max_value=date.today(),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        st.session_state.date_from, st.session_state.date_to = date_range

# --------------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------------
orders_raw, orders_source = load_orders(st.session_state.date_from, st.session_state.date_to)
ads_raw = load_ads(st.session_state.date_from, st.session_state.date_to)
ozon_finance_transactions = load_ozon_finance_transactions(st.session_state.date_from, st.session_state.date_to)
ozon_finance_by_posting = (
    dp.aggregate_ozon_finance_by_posting(ozon_finance_transactions)
    if ozon_finance_transactions is not None
    else None
)

all_marketplaces = sorted(orders_raw["Маркетплейс"].unique().tolist())
all_sellers = sorted(orders_raw["Селлер"].unique().tolist())

with st.sidebar:
    selected_marketplaces = st.multiselect("Маркетплейс", all_marketplaces, default=all_marketplaces)
    selected_sellers = st.multiselect("Селлер", all_sellers, default=all_sellers)

orders = dp.filter_orders(
    orders_raw,
    marketplaces=selected_marketplaces,
    sellers=selected_sellers,
    date_from=st.session_state.date_from,
    date_to=st.session_state.date_to,
)
ads = ads_raw[ads_raw["Маркетплейс"].isin(selected_marketplaces)]

cost_df = st.session_state.cost_df if st.session_state.cost_df is not None else mock_data.generate_cost_reference()

def format_money(value) -> str:
    """Денежный формат: округление только дробной части, разряды через пробел, ₽."""
    if pd.isna(value):
        return "—"
    return f"{value:,.0f} ₽".replace(",", " ")


def with_money_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Копия df с указанными колонками, отформатированными как деньги (только для отображения)."""
    display_df = df.copy()
    for col in columns:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_money)
    return display_df


# --------------------------------------------------------------------------
# Заголовок
# --------------------------------------------------------------------------
st.title("📊 Аналитика маркетплейсов — Мебель")
if orders_source == "mock":
    st.caption("⚠️ Показаны полностью тестовые (mock) данные — введите API-ключи в сайдбаре для реальных данных.")
elif orders_source == "partial":
    st.caption("ℹ️ Заказы получены по API частично — по одному из источников (см. предупреждение выше) используются тестовые данные.")

if orders.empty:
    st.warning("Нет данных за выбранный период/фильтры.")
    st.stop()

# --------------------------------------------------------------------------
# 1. Блок KPI
# --------------------------------------------------------------------------
kpis = dp.calculate_kpis(orders)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GMV (₽)", f"{kpis['gmv_sum']:,.0f}".replace(",", " "), f"{kpis['gmv_qty']} шт")
c2.metric("Доставлено (₽)", f"{kpis['delivered_sum']:,.0f}".replace(",", " "), f"{kpis['delivered_qty']} шт")
c3.metric("Отменено (₽)", f"{kpis['cancelled_sum']:,.0f}".replace(",", " "), f"{kpis['cancelled_qty']} шт")
c4.metric("% Отмен (по сумме / шт)", f"{kpis['cancel_rate_sum']:.1f}%", f"{kpis['cancel_rate_qty']:.1f}% по шт")
c5.metric("Средний чек (AOV)", f"{kpis['aov']:,.0f} ₽".replace(",", " "))

with st.expander("Причины отмен"):
    st.dataframe(with_money_columns(kpis["cancel_reasons"], ["Сумма"]), use_container_width=True, hide_index=True)

st.divider()

# --------------------------------------------------------------------------
# 2. Сводная динамика
# --------------------------------------------------------------------------
st.subheader("📈 Сводная динамика по селлерам и месяцам")
dynamics = dp.build_dynamics_table(orders)
st.dataframe(
    with_money_columns(dynamics, ["Принято (руб)", "Доставлено (руб)", "Отменено (руб)"]),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --------------------------------------------------------------------------
# 3. Продажи по категориям
# --------------------------------------------------------------------------
st.subheader("🛋️ Продажи по категориям товаров")
categories = dp.category_breakdown(orders)

col_table, col_pie, col_bar = st.columns([1.2, 1, 1])
with col_table:
    st.dataframe(with_money_columns(categories, ["Сумма (руб)"]), use_container_width=True, hide_index=True)
with col_pie:
    fig_pie = px.pie(categories, names="Категория", values="Сумма (руб)", title="Доля выручки по категориям")
    st.plotly_chart(fig_pie, use_container_width=True)
with col_bar:
    fig_bar = px.bar(categories, x="Категория", y="Количество (шт)", title="Продажи по категориям, шт")
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------
# 4. Реклама
# --------------------------------------------------------------------------
st.subheader("📣 Реклама и ДРР")
ad_summary = dp.advertising_summary(ads, group_by=["Маркетплейс"])
st.dataframe(
    with_money_columns(ad_summary, ["Расходы на рекламу", "Выручка с рекламы"]),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# --------------------------------------------------------------------------
# 5. Финансовая сводка Ozon (по образцу "Финансы -> Экономика магазина")
# --------------------------------------------------------------------------
if ozon_finance_transactions:
    st.subheader("🧾 Финансовая сводка Ozon: от начислений к чистой прибыли")
    st.caption(
        "Реальные данные из финансового отчёта Ozon (/v3/finance/transaction/list) — "
        "то же, что видно в кабинете Ozon в разделе «Финансы». «ИТОГО начислено» — "
        "сумма, которую Ozon фактически должен перечислить на расчётный счёт за период "
        "(до вычета себестоимости, которую Ozon не знает)."
    )
    ozon_sales_for_cogs = orders[(orders["Маркетплейс"] == "Ozon") & (orders["Статус"] != "Отменён")]
    finance_waterfall = dp.build_ozon_finance_waterfall(ozon_finance_transactions, ozon_sales_for_cogs, cost_df)
    st.dataframe(with_money_columns(finance_waterfall, ["Сумма"]), use_container_width=True, hide_index=True)
    st.divider()

# --------------------------------------------------------------------------
# 6. Юнит-экономика
# --------------------------------------------------------------------------
st.subheader("💰 Юнит-экономика (от выручки к чистой прибыли)")
if ozon_finance_by_posting is not None:
    st.caption(
        "✅ Комиссия и логистика Ozon — реальные данные из финансового отчёта "
        "(/v3/finance/transaction/list) по отправлениям. Для WB и непривязанных строк — оценка по средним ставкам."
    )
else:
    st.caption("⚠️ Комиссия и логистика — оценка по средним ставкам (нет данных Ozon Finance API).")

st.markdown("**Доставлено vs В пути** — выручка «В пути» ещё не окончательна: заказ может быть отменён/возвращён.")
status_breakdown = dp.build_status_cost_breakdown(orders, cost_df)
st.dataframe(
    with_money_columns(status_breakdown, ["Выручка", "Себестоимость (итого)", "Валовая прибыль"]),
    use_container_width=True,
    hide_index=True,
)

unit_econ = dp.calculate_unit_economics(orders, cost_df, ads, ozon_finance_by_posting)

missing_cost = unit_econ[~unit_econ["Себестоимость_найдена"]]
if not missing_cost.empty:
    missing_qty = int(missing_cost["Продано (шт)"].sum())
    missing_revenue = missing_cost["Выручка"].sum()
    st.warning(
        f"⚠️ Себестоимость не найдена для {len(missing_cost)} артикулов ({missing_qty} шт, "
        f"{format_money(missing_revenue)} выручки) — по ним прибыль в таблице ниже завышена (посчитана как если "
        f"бы себестоимость = 0). Артикулы: {', '.join(missing_cost['Артикул'].tolist())}"
    )

loss_making = unit_econ[unit_econ["Чистая прибыль"] < 0]
if not loss_making.empty:
    loss_qty = int(loss_making["Продано (шт)"].sum())
    loss_total = loss_making["Чистая прибыль"].sum()
    st.error(
        f"🚨 Продажа в минус: {len(loss_making)} артикулов ({loss_qty} шт) дают отрицательную чистую прибыль, "
        f"суммарно {format_money(loss_total)}."
    )
    st.dataframe(
        with_money_columns(
            loss_making[
                [
                    "Артикул", "Категория", "Селлер", "Маркетплейс", "Продано (шт)",
                    "Цена (1 шт)", "Себестоимость (1 шт)", "Выручка", "Чистая прибыль", "Маржинальность (%)",
                ]
            ],
            ["Цена (1 шт)", "Себестоимость (1 шт)", "Выручка", "Чистая прибыль"],
        ),
        use_container_width=True,
        hide_index=True,
    )

total_profit = unit_econ["Чистая прибыль"].sum()
total_revenue = unit_econ["Выручка"].sum()

st.dataframe(
    with_money_columns(
        unit_econ.drop(columns=["Себестоимость_найдена"]),
        [
            "Цена (1 шт)",
            "Себестоимость (1 шт)",
            "Выручка",
            "Себестоимость (итого)",
            "Комиссия МП",
            "Логистика",
            "Реклама (ДРР)",
            "Чистая прибыль",
        ],
    ),
    use_container_width=True,
    hide_index=True,
)
avg_margin = (total_profit / total_revenue * 100) if total_revenue else 0
mc1, mc2 = st.columns(2)
mc1.metric("Чистая прибыль (итого)", f"{total_profit:,.0f} ₽".replace(",", " "))
mc2.metric("Средняя маржинальность", f"{avg_margin:.1f}%")
