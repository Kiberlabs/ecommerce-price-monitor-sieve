from __future__ import annotations

import ipaddress
import math
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_HOST_CHARS_RE = re.compile(r"^[a-z0-9.-]+$")
_INTERNAL_HOSTS = frozenset({
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    "metadata",
    "metadata.google.internal",
    "instance-data",
})
_INTERNAL_SUFFIXES = (
    ".local",
    ".localdomain",
    ".internal",
    ".localhost",
    ".lan",
    ".home.arpa",
    ".corp",
    ".intranet",
)
_FORBIDDEN_NETS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("192.88.99.0/24"),
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
)


_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("Boolean value is not valid for numeric field.")
    return value


def _normalize_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _parse_inet_aton_part(part: str) -> int | None:
    if not part:
        return None
    try:
        if part.startswith(("0x", "0X")):
            return int(part, 16)
        if len(part) > 1 and part.startswith("0"):
            return int(part, 8)
        if part.isdigit():
            return int(part, 10)
    except ValueError:
        return None
    return None


def _parse_obfuscated_ipv4(host: str) -> ipaddress.IPv4Address | None:
    parts = host.split(".")
    if not (1 <= len(parts) <= 4):
        return None
    vals = [_parse_inet_aton_part(p) for p in parts]
    if any(v is None or v < 0 for v in vals):
        return None
    nums = [v for v in vals if v is not None]
    if len(nums) == 1 and nums[0] <= 0xFFFFFFFF:
        return ipaddress.IPv4Address(nums[0])
    if len(nums) == 2 and nums[0] <= 0xFF and nums[1] <= 0xFFFFFF:
        return ipaddress.IPv4Address((nums[0] << 24) | nums[1])
    if len(nums) == 3 and nums[0] <= 0xFF and nums[1] <= 0xFF and nums[2] <= 0xFFFF:
        return ipaddress.IPv4Address((nums[0] << 24) | (nums[1] << 16) | nums[2])
    if len(nums) == 4 and all(n <= 0xFF for n in nums):
        return ipaddress.IPv4Address((nums[0] << 24) | (nums[1] << 16) | (nums[2] << 8) | nums[3])
    return None


def _is_forbidden_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv6Address) and ip.is_site_local)
        or not ip.is_global
    ):
        return True
    if any(ip in net for net in _FORBIDDEN_NETS if ip.version == net.version):
        return True
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None and _is_forbidden_ip(ip.ipv4_mapped):
            return True
        if ip.sixtofour is not None and _is_forbidden_ip(ip.sixtofour):
            return True
        if ip.teredo is not None and any(_is_forbidden_ip(addr) for addr in ip.teredo):
            return True
    return False


def validate_public_http_url(url: str) -> str:
    """Enforce SSRF boundaries on supplier URLs.

    Accepts only HTTP(S) URLs with public hosts; rejects localhost, internal
    suffixes (.local/.internal), credentials, control chars, obfuscated IPs,
    and non-public/link-local/loopback/transition IP literals.
    """
    if not isinstance(url, str) or not url or len(url) > 2048:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in url):
        raise ValueError(f"URL contains whitespace or control characters: {url!r}")

    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"Invalid HTTP(S) URL: {url!r}")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc or "\\" in url:
        raise ValueError(f"Blocked URL credentials or unsafe authority: {url!r}")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid port in URL: {url!r}") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Invalid port range in URL: {url!r}")

    host = parsed.hostname.lower()
    if host.endswith("."):
        host = host[:-1]
    if not host or "%" in host:
        raise ValueError(f"Blocked invalid hostname: {host!r}")
    if host in _INTERNAL_HOSTS or host.endswith(_INTERNAL_SUFFIXES):
        raise ValueError(f"Blocked internal hostname: {host!r}")

    # shortcut: validates literal IPs and internal suffixes offline; resolve DNS per-socket before live HTTP fetches
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = _parse_obfuscated_ipv4(host)

    if ip is not None:
        if _is_forbidden_ip(ip):
            raise ValueError(f"Blocked non-public IP address: {host!r}")
        return url

    labels = host.split(".")
    if "." not in host or not _HOST_CHARS_RE.match(host) or "" in labels or labels[-1].isdigit():
        raise ValueError(f"Blocked non-public or malformed hostname: {host!r}")
    return url


class ProductSnapshot(BaseModel):
    """Single validated supplier product observation ingested from a catalog feed."""

    model_config = ConfigDict(allow_inf_nan=False)

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

    @field_validator("sku", "title")
    @classmethod
    def _check_non_blank(cls, value: str) -> str:
        if not _ZERO_WIDTH_RE.sub("", value).strip():
            raise ValueError("Field must not be blank.")
        return value

    @field_validator("supplier_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return validate_public_http_url(value)

    @field_validator(
        "supplier_price_eur",
        "target_resale_eur",
        "shipping_cost_eur",
        "delivery_days",
        "rating",
        mode="before",
    )
    @classmethod
    def _reject_bool_numerics(cls, value: object) -> object:
        return _reject_bool(value)

    @field_validator("recorded_at")
    @classmethod
    def _check_utc(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class SieveConfig(BaseModel):
    """Operational thresholds, fee percentages, and alert ratios controlling the 3-stage sieve."""

    model_config = ConfigDict(allow_inf_nan=False)

    marketplace_fee_pct: float = Field(ge=0.0, lt=1.0, default=0.12)
    return_reserve_pct: float = Field(ge=0.0, lt=1.0, default=0.04)
    min_net_profit_eur: float = Field(ge=0.0, default=8.0)
    min_net_margin_pct: float = Field(ge=0.0, lt=1.0, default=0.18)
    max_delivery_days: int = Field(ge=1, default=5)
    min_rating: float = Field(ge=0.0, le=5.0, default=4.2)
    require_eu_dispatch: bool = True
    price_drop_alert_pct: float = Field(gt=0.0, lt=1.0, default=0.10)

    @field_validator(
        "marketplace_fee_pct",
        "return_reserve_pct",
        "min_net_profit_eur",
        "min_net_margin_pct",
        "max_delivery_days",
        "min_rating",
        "price_drop_alert_pct",
        mode="before",
    )
    @classmethod
    def _reject_bool_numerics(cls, value: object) -> object:
        return _reject_bool(value)

    @model_validator(mode="after")
    def _check_combined_fees(self) -> SieveConfig:
        if self.marketplace_fee_pct + self.return_reserve_pct >= 1.0:
            raise ValueError("Combined marketplace_fee_pct and return_reserve_pct must be < 1.0.")
        return self


class PriceAlert(BaseModel):
    """Emitted when a SKU's current supplier price drops beyond the configured threshold."""

    model_config = ConfigDict(allow_inf_nan=False)

    sku: str
    title: str
    previous_price_eur: float
    current_price_eur: float
    drop_pct: float
    recorded_at: datetime

    @field_validator("previous_price_eur", "current_price_eur", "drop_pct", mode="before")
    @classmethod
    def _reject_bool_numerics(cls, value: object) -> object:
        return _reject_bool(value)

    @field_validator("recorded_at")
    @classmethod
    def _check_utc(cls, value: datetime) -> datetime:
        return _normalize_utc(value)


class FilterVerdict(BaseModel):
    """Final per-SKU decision, net economics, ranking score (0-100), and optional price-drop alert."""

    model_config = ConfigDict(allow_inf_nan=False)

    sku: str
    title: str
    approved: bool
    stage_reached: str
    reasons: list[str]
    net_profit_eur: float
    net_margin_pct: float
    score: float
    price_alert: PriceAlert | None = None

    @field_validator("net_profit_eur", "net_margin_pct", "score", mode="before")
    @classmethod
    def _reject_bool_numerics(cls, value: object) -> object:
        return _reject_bool(value)


class MarketValuation(BaseModel):
    """Outlier-trimmed comparable market valuation and liquidity score (0-100)."""

    model_config = ConfigDict(allow_inf_nan=False)

    sample_count: int
    trimmed_count: int
    median_eur: float | None = None
    fast_sell_p40_eur: float | None = None
    spread_ratio: float | None = None
    liquidity_score: int = Field(ge=0, le=100)
    low_liquidity: bool

    @field_validator(
        "sample_count",
        "trimmed_count",
        "median_eur",
        "fast_sell_p40_eur",
        "spread_ratio",
        "liquidity_score",
        mode="before",
    )
    @classmethod
    def _reject_bool_numerics(cls, value: object) -> object:
        return _reject_bool(value)
