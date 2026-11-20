from __future__ import annotations

import ipaddress
from datetime import datetime, timezone
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator


def validate_public_http_url(url: str) -> str:
    """Reject non-HTTP schemes, loopback/private IPs, and internal hostnames (SSRF guard)."""
    parsed = urlsplit(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")

    host = parsed.hostname.lower().rstrip(".")
    if host in ("localhost", "ip6-localhost") or host.endswith((".local", ".internal", ".localhost")):
        raise ValueError(f"Blocked internal hostname: {host!r}")

    # shortcut: validates literal IPs and internal suffixes offline; resolve DNS per-socket before live HTTP fetches
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return url.strip()

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise ValueError(f"Blocked non-public IP address: {host!r}")
    return url.strip()


class ProductSnapshot(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=240)
    supplier_url: str
    supplier_price_eur: float = Field(gt=0)
    target_resale_eur: float = Field(gt=0)
    shipping_cost_eur: float = Field(ge=0, default=0.0)
    in_stock: bool = True
    ships_from_eu: bool = True
    delivery_days: int = Field(ge=0, le=90, default=3)
    rating: float = Field(ge=0.0, le=5.0, default=4.5)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("supplier_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return validate_public_http_url(value)


class SieveConfig(BaseModel):
    marketplace_fee_pct: float = Field(ge=0.0, lt=1.0, default=0.12)
    return_reserve_pct: float = Field(ge=0.0, lt=1.0, default=0.04)
    min_net_profit_eur: float = Field(ge=0.0, default=8.0)
    min_net_margin_pct: float = Field(ge=0.0, lt=1.0, default=0.18)
    max_delivery_days: int = Field(ge=1, default=5)
    min_rating: float = Field(ge=0.0, le=5.0, default=4.2)
    require_eu_dispatch: bool = True
    price_drop_alert_pct: float = Field(gt=0.0, lt=1.0, default=0.10)


class PriceAlert(BaseModel):
    sku: str
    title: str
    previous_price_eur: float
    current_price_eur: float
    drop_pct: float
    recorded_at: datetime


class FilterVerdict(BaseModel):
    sku: str
    title: str
    approved: bool
    stage_reached: str
    reasons: list[str]
    net_profit_eur: float
    net_margin_pct: float
    score: float
    price_alert: PriceAlert | None = None
