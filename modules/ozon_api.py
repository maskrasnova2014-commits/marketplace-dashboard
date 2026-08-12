"""
Клиент для работы с API Ozon Seller.

Используемые методы:
- POST /v2/posting/fbo/list           — список отправлений FBO (склад Ozon)
- POST /v3/posting/fbs/list           — список отправлений FBS (склад продавца)
- POST /v3/finance/transaction/list   — реальные начисления по отправлениям:
                                         комиссия, логистика, прочие услуги
                                         (используется вместо /v1/finance/realization,
                                         который на практике возвращает 404 — видимо,
                                         устарел в текущей версии API)

Документация: https://docs.ozon.ru/api/seller/
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import requests

BASE_URL = "https://api-seller.ozon.ru"
TIMEOUT = 15
MAX_TRANSACTION_PERIOD_DAYS = 30  # реальное ограничение Ozon: "too long period, only one month allowed"


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
        """Список отправлений FBO за период. Пагинируется — без этого молча обрезалось бы на 1000."""
        postings: list[dict[str, Any]] = []
        offset = 0
        limit = 1000
        while True:
            payload = {
                "dir": "ASC",
                "filter": {
                    "since": f"{date_from.isoformat()}T00:00:00.000Z",
                    "to": f"{date_to.isoformat()}T23:59:59.999Z",
                },
                "limit": limit,
                "offset": offset,
                "translit": True,
                "with": {"analytics_data": True, "financial_data": True},
            }
            data = self._post("/v2/posting/fbo/list", payload)
            batch = data.get("result", [])
            postings.extend(batch)
            if len(batch) < limit:
                break
            offset += limit
        return postings

    def get_fbs_postings(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Список отправлений FBS за период. Пагинируется — без этого молча обрезалось бы на 1000."""
        postings: list[dict[str, Any]] = []
        offset = 0
        limit = 1000
        while True:
            payload = {
                "dir": "ASC",
                "filter": {
                    "since": f"{date_from.isoformat()}T00:00:00.000Z",
                    "to": f"{date_to.isoformat()}T23:59:59.999Z",
                },
                "limit": limit,
                "offset": offset,
                "with": {"analytics_data": True, "financial_data": True},
            }
            data = self._post("/v3/posting/fbs/list", payload)
            result = data.get("result", {})
            batch = result.get("postings", [])
            postings.extend(batch)
            has_next = result.get("has_next")
            if has_next is False or (has_next is None and len(batch) < limit):
                break
            offset += limit
        return postings

    def _get_finance_transactions_chunk(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Пагинированный запрос за период не длиннее месяца (см. get_finance_transactions)."""
        operations: list[dict[str, Any]] = []
        page = 1
        page_size = 1000
        while True:
            payload = {
                "filter": {
                    "date": {
                        "from": f"{date_from.isoformat()}T00:00:00.000Z",
                        "to": f"{date_to.isoformat()}T23:59:59.999Z",
                    },
                    "operation_type": [],
                    "posting_number": "",
                    "transaction_type": "all",
                },
                "page": page,
                "page_size": page_size,
            }
            data = self._post("/v3/finance/transaction/list", payload)
            result = data.get("result", {})
            batch = result.get("operations", [])
            operations.extend(batch)
            if len(batch) < page_size or page >= result.get("page_count", page):
                break
            page += 1
        return operations

    def get_finance_transactions(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """
        Реальные финансовые операции по отправлениям за период: комиссия
        (sale_commission), логистика и прочие услуги (services[]) по каждому
        posting_number. Пагинируется автоматически (до 1000 строк за раз).

        Ozon ограничивает период одним месяцем ("too long period, only one
        month allowed") — более длинные периоды разбиваются на куски по 30 дней.
        """
        operations: list[dict[str, Any]] = []
        chunk_start = date_from
        while chunk_start <= date_to:
            chunk_end = min(chunk_start + timedelta(days=MAX_TRANSACTION_PERIOD_DAYS - 1), date_to)
            operations.extend(self._get_finance_transactions_chunk(chunk_start, chunk_end))
            chunk_start = chunk_end + timedelta(days=1)
        return operations
