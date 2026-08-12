"""
Клиент Ozon Performance API (реклама: Трафареты, Продвижение товаров).

ВАЖНО: это ОТДЕЛЬНЫЙ контур авторизации от Ozon Seller API. Client-Id и
Client-Secret для него берутся не в разделе "API-ключи" личного кабинета,
а в разделе "Продвижение → API" (Performance). Api-Key от Seller API сюда
не подходит.

Документация: https://docs.ozon.ru/api/performance/

Точная форма ответа статистики может отличаться в зависимости от версии
кабинета — при несовпадении полей приложение (app.py) безопасно
откатывается на тестовые данные, поэтому даже неточный маппинг не ломает
дашборд. Если структура ответа вашего кабинета отличается — поправьте
get_daily_statistics().
"""

from __future__ import annotations

from datetime import date
from typing import Any

import requests

ADS_BASE_URL = "https://api-performance.ozon.ru"
TIMEOUT = 20


class OzonAdsAPIError(Exception):
    """Ошибка при обращении к Ozon Performance API."""


class OzonAdsClient:
    """Тонкая обёртка над Ozon Performance API (реклама)."""

    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: str | None = None

    def _authenticate(self) -> str:
        try:
            response = requests.post(
                f"{ADS_BASE_URL}/api/client/token",
                json={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=TIMEOUT,
            )
            if response.status_code in (401, 403):
                raise OzonAdsAPIError(
                    f"Ошибка авторизации Ozon Performance API ({response.status_code}): "
                    "проверьте Client-Id/Client-Secret рекламного кабинета"
                )
            response.raise_for_status()
            token = response.json().get("access_token")
            if not token:
                raise OzonAdsAPIError("Ozon Performance API не вернул access_token")
            self._token = token
            return token
        except requests.RequestException as exc:
            raise OzonAdsAPIError(f"Сетевая ошибка при авторизации в Ozon Performance API: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        token = self._token or self._authenticate()
        return {"Authorization": f"Bearer {token}", "Client-Id": self.client_id, "Content-Type": "application/json"}

    def get_daily_statistics(self, date_from: date, date_to: date) -> list[dict[str, Any]]:
        """Расходы на рекламу по дням (Трафареты / Продвижение товаров)."""
        try:
            response = requests.get(
                f"{ADS_BASE_URL}/api/client/statistics/daily/json",
                headers=self._headers(),
                params={"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()},
                timeout=TIMEOUT,
            )
            if response.status_code in (401, 403):
                raise OzonAdsAPIError(f"Ошибка авторизации Ozon Performance API ({response.status_code})")
            if response.status_code >= 500:
                raise OzonAdsAPIError(f"Сервер Ozon Performance API вернул ошибку {response.status_code}")
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
            return data.get("rows") or data.get("result") or []
        except requests.RequestException as exc:
            raise OzonAdsAPIError(f"Сетевая ошибка при запросе статистики Ozon Performance API: {exc}") from exc
