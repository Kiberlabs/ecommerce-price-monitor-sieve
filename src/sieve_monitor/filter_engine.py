from __future__ import annotations

import math
from datetime import datetime

from .models import FilterVerdict, MarketValuation, PriceAlert, ProductSnapshot, SieveConfig, _normalize_utc
from .storage import SnapshotStore


def _validate_finite_prices(prices: list[float]) -> list[float]:
    positive: list[float] = []
    for raw in prices:
        if isinstance(raw, bool):
            raise ValueError(f"Boolean value is not valid for price: {raw!r}")
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite price value: {raw!r}")
        if val > 0:
            positive.append(val)
    return positive


def _percentile(sorted_vals: list[float], p: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def trim_outliers_iqr(prices: list[float]) -> list[float]:
    """Remove extreme low/high noisy listings using Tukey's 1.5x interquartile range rule."""
    clean = sorted(_validate_finite_prices(prices))
    if len(clean) < 4:
        return clean
    q1 = _percentile(clean, 0.25)
    q3 = _percentile(clean, 0.75)
    iqr = q3 - q1
    if iqr <= 0:
        med = _percentile(clean, 0.50)
        trimmed = [p for p in clean if abs(p - med) <= 0.5 * med]
        return trimmed or clean
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    trimmed = [p for p in clean if lo <= p <= hi]
    return trimmed or clean


def compute_market_valuation(
    comp_prices: list[float],
    *,
    min_liquidity_score: int = 20,
    anchor_cap_eur: float | None = None,
    anchor_cap_ratio: float = 0.78,
) -> MarketValuation:
    """Compute IQR-trimmed median, P40 fast-sell target, and 0-100 market liquidity score."""
    if isinstance(min_liquidity_score, bool):
        raise ValueError(f"Boolean value is not valid for min_liquidity_score: {min_liquidity_score!r}")
    if isinstance(anchor_cap_ratio, bool) or not math.isfinite(anchor_cap_ratio):
        raise ValueError(f"Non-finite anchor_cap_ratio: {anchor_cap_ratio!r}")
    if anchor_cap_eur is not None and (isinstance(anchor_cap_eur, bool) or not math.isfinite(anchor_cap_eur)):
        raise ValueError(f"Non-finite anchor_cap_eur: {anchor_cap_eur!r}")
    raw = _validate_finite_prices(comp_prices)
    trimmed = trim_outliers_iqr(raw)
    if not trimmed:
        return MarketValuation(
            sample_count=len(raw),
            trimmed_count=0,
            liquidity_score=0,
            low_liquidity=True,
        )

    median = round(_percentile(trimmed, 0.50), 2)
    fast_p40 = round(_percentile(trimmed, 0.40), 2)
    if anchor_cap_eur is not None and anchor_cap_eur > 0:
        fast_p40 = round(min(fast_p40, anchor_cap_eur * anchor_cap_ratio), 2)

    q1 = _percentile(trimmed, 0.25)
    q3 = _percentile(trimmed, 0.75)
    spread_ratio = round((q3 - q1) / median, 4) if median > 0 else 1.0

    # Density rewards active comparable depth; tight IQR spread rewards predictable market pricing
    density_pts = min(65.0, len(trimmed) * 8.0)
    stability_pts = max(0.0, 35.0 * (1.0 - min(1.0, spread_ratio)))
    liquidity = int(round(min(100.0, density_pts + stability_pts)))
    return MarketValuation(
        sample_count=len(raw),
        trimmed_count=len(trimmed),
        median_eur=median,
        fast_sell_p40_eur=fast_p40,
        spread_ratio=spread_ratio,
        liquidity_score=liquidity,
        low_liquidity=liquidity < min_liquidity_score,
    )


def compute_net_economics(item: ProductSnapshot, cfg: SieveConfig) -> tuple[float, float]:
    """Calculate net unit profit (EUR) and net margin ratio after all selling costs.

    Deducts supplier price, outbound shipping, marketplace commission, and return reserve
    from target_resale_eur. Returns (rounded_profit_eur, rounded_margin_ratio).
    """
    vals = (
        item.supplier_price_eur,
        item.target_resale_eur,
        item.shipping_cost_eur,
        cfg.marketplace_fee_pct,
        cfg.return_reserve_pct,
    )
    if any(isinstance(v, bool) or not math.isfinite(v) for v in vals) or item.target_resale_eur <= 0:
        raise ValueError("Economic inputs must be finite non-boolean numbers with positive target_resale_eur.")
    if cfg.marketplace_fee_pct + cfg.return_reserve_pct >= 1.0:
        raise ValueError("Combined marketplace_fee_pct and return_reserve_pct must be < 1.0.")
    fees = item.target_resale_eur * (cfg.marketplace_fee_pct + cfg.return_reserve_pct)
    total_cost = item.supplier_price_eur + item.shipping_cost_eur + fees
    net_profit = item.target_resale_eur - total_cost
    net_margin = net_profit / item.target_resale_eur
    if not (math.isfinite(net_profit) and math.isfinite(net_margin)):
        raise ValueError("Computed economic metrics overflowed to non-finite values.")
    return round(net_profit, 2), round(net_margin, 4)


def detect_price_drop(
    item: ProductSnapshot,
    previous_price: float | None,
    min_drop_pct: float,
    previous_recorded_at: datetime | None = None,
) -> PriceAlert | None:
    """Compare current supplier price against the last stored observation.

    Returns a PriceAlert if the relative drop meets or exceeds min_drop_pct; otherwise None.
    """
    if (
        isinstance(item.supplier_price_eur, bool)
        or isinstance(min_drop_pct, bool)
        or not (math.isfinite(item.supplier_price_eur) and math.isfinite(min_drop_pct))
    ):
        raise ValueError("Price drop inputs must be finite non-boolean numbers.")
    if previous_price is not None and (isinstance(previous_price, bool) or not math.isfinite(previous_price)):
        raise ValueError("previous_price must be a finite non-boolean number.")
    if previous_recorded_at is not None and _normalize_utc(item.recorded_at) < _normalize_utc(previous_recorded_at):
        return None
    if previous_price is None or previous_price <= 0 or item.supplier_price_eur >= previous_price:
        return None
    raw_drop = (previous_price - item.supplier_price_eur) / previous_price
    if not math.isfinite(raw_drop):
        raise ValueError("Computed price drop overflowed to a non-finite value.")
    if raw_drop < min_drop_pct:
        return None
    drop_pct = round(raw_drop, 4)
    return PriceAlert(
        sku=item.sku,
        title=item.title,
        previous_price_eur=round(previous_price, 2),
        current_price_eur=round(item.supplier_price_eur, 2),
        drop_pct=drop_pct,
        recorded_at=item.recorded_at,
    )


def evaluate_product(
    item: ProductSnapshot,
    cfg: SieveConfig,
    previous_price: float | None = None,
    previous_recorded_at: datetime | None = None,
) -> FilterVerdict:
    """Run a single SKU through the deterministic 3-stage B2B sieve pipeline.

    Flow: (1) Operational hard gate (stock, EU dispatch, SLA, rating) ->
    (2) Net profit & margin floor check -> (3) Weighted 0-100 priority score
    combining margin, rating, delivery speed, and active price-drop bonus.
    """
    net_profit, net_margin = compute_net_economics(item, cfg)
    alert = detect_price_drop(
        item,
        previous_price,
        cfg.price_drop_alert_pct,
        previous_recorded_at=previous_recorded_at,
    )
    reasons: list[str] = []

    # Stage 1: Hard operational gate
    if not item.in_stock:
        reasons.append("out_of_stock")
    if cfg.require_eu_dispatch and not item.ships_from_eu:
        reasons.append("non_eu_dispatch")
    if item.delivery_days > cfg.max_delivery_days:
        reasons.append(f"delivery_days>{cfg.max_delivery_days}")
    if item.rating < cfg.min_rating:
        reasons.append(f"rating<{cfg.min_rating}")

    if reasons:
        return FilterVerdict(
            sku=item.sku,
            title=item.title,
            approved=False,
            stage_reached="hard_gate",
            reasons=reasons,
            net_profit_eur=net_profit,
            net_margin_pct=net_margin,
            score=0.0,
            price_alert=alert,
        )

    # Stage 2: Unit economics & margin floor
    if net_profit < cfg.min_net_profit_eur:
        reasons.append(f"net_profit<{cfg.min_net_profit_eur:.2f}")
    if net_margin < cfg.min_net_margin_pct:
        reasons.append(f"net_margin<{cfg.min_net_margin_pct:.2%}")

    if reasons:
        return FilterVerdict(
            sku=item.sku,
            title=item.title,
            approved=False,
            stage_reached="margin_sieve",
            reasons=reasons,
            net_profit_eur=net_profit,
            net_margin_pct=net_margin,
            score=round(max(0.0, net_margin * 100), 1),
            price_alert=alert,
        )

    # Stage 3: Priority ranking score (0-100) for survivors
    margin_component = min(55.0, net_margin * 140.0)
    rating_component = (item.rating / 5.0) * 25.0
    speed_component = max(0.0, (cfg.max_delivery_days - item.delivery_days + 1) * 4.0)
    drop_bonus = 10.0 if alert is not None else 0.0
    score = round(min(100.0, margin_component + rating_component + speed_component + drop_bonus), 1)

    return FilterVerdict(
        sku=item.sku,
        title=item.title,
        approved=True,
        stage_reached="approved",
        reasons=["passed_all_gates"],
        net_profit_eur=net_profit,
        net_margin_pct=net_margin,
        score=score,
        price_alert=alert,
    )


def run_sieve_batch(
    snapshots: list[ProductSnapshot],
    store: SnapshotStore,
    cfg: SieveConfig | None = None,
) -> list[FilterVerdict]:
    """Evaluate a catalog batch end-to-end against SQLite history and persist each snapshot.

    Looks up prior SKU price before writing the new observation, collects all verdicts,
    and returns them sorted by (approved, score, net_profit_eur) descending.
    """
    active_cfg = cfg or SieveConfig()
    verdicts: list[FilterVerdict] = []
    with store.transaction():
        for item in sorted(snapshots, key=lambda s: _normalize_utc(s.recorded_at)):
            latest = store.get_latest_snapshot(item.sku)
            prev_price, prev_ts = latest if latest is not None else (None, None)
            verdict = evaluate_product(
                item,
                active_cfg,
                previous_price=prev_price,
                previous_recorded_at=prev_ts,
            )
            store.save_observation(item, verdict.price_alert)
            verdicts.append(verdict)
    verdicts.sort(key=lambda v: (v.approved, v.score, v.net_profit_eur), reverse=True)
    return verdicts
