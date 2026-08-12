"""
Клиент для работы с API Wildberries (API Статистики v1 / v5).

Используемые методы:
- GET /api/v1/supplier/incomes               — поставки продавца
- GET /api/v1/supplier/sales                 — продажи и возвраты
- GET /api/v5/supplier/reportDetailByPeriod   — детальный финансовый отчёт
                                                 (комиссии, логистика, реклама)

Документация: https://openapi.wildberries.ru/
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

STATS_BASE_URL = "https://statistics-api.wildberries.ru"
ADVERT_BASE_URL = "https://advert-api.wildberries.ru"
TIMEOUT = 15


class WBAPIError(Exception):
    """Ошибка при обращении к API Wildberries (сеть, 401/403/500 и т.п.)."""

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


def _wb_get(base_url: str, endpoint: str, api_token: str, params: dict[str, Any]) -> Any:
    """Общая логика запроса + обработки ошибок для Statistics и Advert API Wildberries."""
    try:
        response = requests.get(
            f"{base_url}{endpoint}",
            headers={"Authorization": api_token},
            params=params,
            timeout=TIMEOUT,
        )
        if response.status_code in (401, 403):
            raise WBAPIError(f"Ошибка авторизации WB ({response.status_code}): проверьте API-токен и его доступ (скоуп)")
        if response.status_code == 429:
            retry_after = response.headers.get("X-Ratelimit-Reset") or response.headers.get("Retry-After")
            retry_after_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
            minutes = f"{retry_after_seconds // 60} мин" if retry_after_seconds else "некоторое время"
            raise WBAPIError(
                f"Превышен лимит запросов WB (429). Повтор возможен через {minutes}.",
                retry_after_seconds=retry_after_seconds,
            )
        if response.status_code >= 500:
            raise WBAPIError(f"Сервер Wildberries вернул ошибку {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise WBAPIError(f"Сетевая ошибка при запросе к Wildberries: {exc}") from exc


class WBClient:
    """Тонкая обёртка над API Статистики Wildberries (токен со скоупом «Статистика»)."""

    def __init__(self, api_token: str) -> None:
        self.api_token = api_token

    def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        return _wb_get(STATS_BASE_URL, endpoint, self.api_token, params)

    def get_incomes(self, date_from: date) -> list[dict[str, Any]]:
        """Поставки продавца начиная с указанной даты."""
        params = {"dateFrom": date_from.isoformat()}
        data = self._get("/api/v1/supplier/incomes", params)
        return data or []

    def get_sales(self, date_from: date) -> list[dict[str, Any]]:
        """Продажи и возвраты начиная с указанной даты."""
        params = {"dateFrom": date_from.isoformat()}
        data = self._get("/api/v1/supplier/sales", params)
        return data or []

    def get_report_detail_by_period(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Детальный финансовый отчёт: комиссии, логистика, хранение, реклама."""
        params = {
            "dateFrom": date_from.isoformat(),
            "dateTo": date_to.isoformat(),
            "limit": 100000,
            "rrdid": 0,
        }
        data = self._get("/api/v5/supplier/reportDetailByPeriod", params)
        return data or []


class WBAdsClient:
    """
    Клиент WB Advert API (реклама: Автоматические кампании, Аукцион).

    ВАЖНО: это ОТДЕЛЬНЫЙ токен — в личном кабинете WB при создании API-ключа
    нужно выбрать категорию доступа «Продвижение», а не «Статистика».
    Токен для Statistics API сюда не подходит (вернёт 401/403).
    """

    def __init__(self, api_token: str) -> None:
        self.api_token = api_token

    def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        return _wb_get(ADVERT_BASE_URL, endpoint, self.api_token, params)

    def get_advert_costs(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """История списаний по рекламным кампаниям (сумма, дата, кампания)."""
        params = {"from": date_from.isoformat(), "to": date_to.isoformat()}
        data = self._get("/adv/v1/upd", params)
        return data or []
