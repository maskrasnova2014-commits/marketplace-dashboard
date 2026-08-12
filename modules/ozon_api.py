"""
Клиент для работы с API Ozon Seller.

Используемые методы:
- POST /v2/posting/fbo/list        — список отправлений FBO (склад Ozon)
- POST /v3/posting/fbs/list        — список отправлений FBS (склад продавца)
- POST /v1/finance/realization     — отчёт о реализации (факт продаж, комиссии)

Документация: https://docs.ozon.ru/api/seller/
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

BASE_URL = "https://api-seller.ozon.ru"
TIMEOUT = 15


class OzonAPIError(Exception):
    """Ошибка при обращении к API Ozon (сеть, 401/403/500 и т.п.)."""


class OzonClient:
    """Тонкая обёртка над Ozon Seller API."""

    def __init__(self, client_id: str, api_key: str) -> None:
        self.client_id = client_id
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Client-Id": self.client_id,
            "Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{BASE_URL}{endpoint}",
                headers=self._headers(),
                json=payload,
                timeout=TIMEOUT,
            )
            if response.status_code in (401, 403):
                raise OzonAPIError(f"Ошибка авторизации Ozon ({response.status_code}): проверьте Client-Id/Api-Key")
            if response.status_code >= 500:
                raise OzonAPIError(f"Сервер Ozon вернул ошибку {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise OzonAPIError(f"Сетевая ошибка при запросе к Ozon: {exc}") from exc

    def get_fbo_postings(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Список отправлений FBO за период."""
        payload = {
            "dir": "ASC",
            "filter": {
                "since": f"{date_from.isoformat()}T00:00:00.000Z",
                "to": f"{date_to.isoformat()}T23:59:59.999Z",
            },
            "limit": 1000,
            "offset": 0,
            "translit": True,
            "with": {"analytics_data": True, "financial_data": True},
        }
        data = self._post("/v2/posting/fbo/list", payload)
        return data.get("result", [])

    def get_fbs_postings(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Список отправлений FBS за период."""
        payload = {
            "dir": "ASC",
            "filter": {
                "since": f"{date_from.isoformat()}T00:00:00.000Z",
                "to": f"{date_to.isoformat()}T23:59:59.999Z",
            },
            "limit": 1000,
            "offset": 0,
            "with": {"analytics_data": True, "financial_data": True},
        }
        data = self._post("/v3/posting/fbs/list", payload)
        return data.get("result", {}).get("postings", [])

    def get_realization_report(self, year: int, month: int) -> list[dict[str, Any]]:
        """Отчёт о реализации товаров (факт продаж и комиссии) за месяц."""
        payload = {"year": year, "month": month}
        data = self._post("/v1/finance/realization", payload)
        return data.get("result", {}).get("rows", [])
