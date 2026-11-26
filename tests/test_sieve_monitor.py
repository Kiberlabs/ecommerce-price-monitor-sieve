from __future__ import annotations

import csv
import gc
import json
import math
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from sieve_monitor.cli import export_csv, main as cli_main
from sieve_monitor.filter_engine import (
    compute_market_valuation,
    compute_net_economics,
    detect_price_drop,
    evaluate_product,
    run_sieve_batch,
    trim_outliers_iqr,
)
from sieve_monitor.models import FilterVerdict, MarketValuation, PriceAlert, ProductSnapshot, SieveConfig
from sieve_monitor.storage import SnapshotStore


def _sample_product(**overrides: object) -> ProductSnapshot:
    base: dict[str, object] = {
        "sku": "SKU-100",
        "title": "Industrial Barcode Reader",
        "supplier_url": "https://supplier.example.eu/products/SKU-100",
        "supplier_price_eur": 60.0,
        "target_resale_eur": 120.0,
        "shipping_cost_eur": 5.0,
        "in_stock": True,
        "ships_from_eu": True,
        "delivery_days": 3,
        "rating": 4.6,
    }
    base.update(overrides)
    return ProductSnapshot.model_validate(base)


def test_approves_healthy_b2b_sku() -> None:
    cfg = SieveConfig()
    verdict = evaluate_product(_sample_product(), cfg)
    # Fees: 120 * (0.12 + 0.04) = 19.20 -> Net profit: 120 - 60 - 5 - 19.20 = 35.80 (29.83%)
    assert verdict.approved is True
    assert verdict.stage_reached == "approved"
    assert verdict.net_profit_eur == 35.80
    assert verdict.net_margin_pct == pytest.approx(0.2983, abs=1e-4)
    assert verdict.score > 60.0


def test_rejects_hard_gate_and_thin_margin() -> None:
    cfg = SieveConfig()
    oos = evaluate_product(_sample_product(in_stock=False, ships_from_eu=False), cfg)
    assert oos.approved is False
    assert oos.stage_reached == "hard_gate"
    assert "out_of_stock" in oos.reasons
    assert "non_eu_dispatch" in oos.reasons

    thin = evaluate_product(_sample_product(supplier_price_eur=98.0), cfg)
    assert thin.approved is False
    assert thin.stage_reached == "margin_sieve"


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "https://localhost/internal",
        "https://db.internal/catalog.json",
        "file:///etc/passwd",
    ],
)
def test_ssrf_guard_blocks_internal_targets(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        _sample_product(supplier_url=bad_url)


def test_sqlite_wal_persistence_and_price_drop_alert(tmp_path: Path) -> None:
    db_file = tmp_path / "sieve.db"
    cfg = SieveConfig(price_drop_alert_pct=0.10)

    with SnapshotStore(db_file) as store:
        assert store.journal_mode() == "wal"
        first = run_sieve_batch([_sample_product(supplier_price_eur=80.0)], store, cfg)
        assert first[0].price_alert is None

        second = run_sieve_batch([_sample_product(supplier_price_eur=68.0)], store, cfg)
        assert second[0].price_alert is not None
        assert second[0].price_alert.drop_pct == pytest.approx(0.15)
        alerts = store.recent_alerts()
        assert len(alerts) == 1
        assert alerts[0]["sku"] == "SKU-100"


def test_cli_end_to_end_json_and_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = repo_root / "data" / "sample_catalog.json"
    db_file = tmp_path / "cli.db"
    csv_file = tmp_path / "ranked.csv"

    rc = cli_main([
        "run",
        "--input",
        str(catalog),
        "--db",
        str(db_file),
        "--csv-out",
        str(csv_file),
        "--json",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_valid_inputs"] == 7
    assert len(payload["blocked_invalid_inputs"]) == 1
    assert payload["blocked_invalid_inputs"][0]["sku"] == "KB-SSRF-BLOCKED"
    assert payload["approved_count"] == 3
    assert csv_file.exists()


def test_iqr_outlier_trimming_and_market_valuation_liquidity_score() -> None:
    from sieve_monitor.filter_engine import compute_market_valuation, trim_outliers_iqr

    # Removes damaged accessory lowball (3.0) and inflated typo (999.0)
    trimmed = trim_outliers_iqr([3.0, 68.0, 72.0, 75.0, 78.0, 80.0, 84.0, 999.0])
    assert 3.0 not in trimmed and 999.0 not in trimmed
    assert len(trimmed) == 6

    val = compute_market_valuation(
        [3.0, 68.0, 72.0, 75.0, 78.0, 80.0, 84.0, 999.0],
        anchor_cap_eur=90.0,
    )
    assert val.sample_count == 8 and val.trimmed_count == 6
    assert val.median_eur == 76.5
    assert val.fast_sell_p40_eur == 70.2  # Capped at 78% of 90.0 anchor
    assert val.liquidity_score >= 70 and val.low_liquidity is False

    empty = compute_market_valuation([])
    assert empty.liquidity_score == 0 and empty.low_liquidity is True


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0177.0.0.1/",
        "http://127.1/",
        "http://10.1/",
        "http://0/",
        "http://08.0.0.1/",
        "http://127.0.0.256/",
        "http://100.64.0.1/",
        "http://[::ffff:100.64.0.1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::127.0.0.1]/",
        "http://[fec0::1]/",
        "http://[2002:7f00:1::]/",
        "http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/",
        "http://[64:ff9b::127.0.0.1]/",
        "http://[fe80::1%25eth0]/",
        "https://user:pass@supplier.example.eu/p",
        "https://supplier.example.eu\t/p",
        "https://supplier.example.eu/p\r\n",
        "https://supplier.example.eu/p\x00",
        "https:// supplier.example.eu/p",
        "https://supplier.example.eu:99999/p",
        "https://supplier.example.eu:0/p",
        "http://redis:6379/",
        "https://printer.lan/feed",
        "https://metadata.google.internal/computeMetadata/v1/",
    ],
)
def test_ssrf_guard_blocks_obfuscated_and_transition_targets(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        _sample_product(supplier_url=bad_url)


@pytest.mark.parametrize(
    "good_url",
    [
        "https://supplier.example.eu:8443/products/SKU-1?v=1",
        "http://8.8.8.8/feed",
    ],
)
def test_ssrf_guard_allows_valid_public_targets(good_url: str) -> None:
    snap = _sample_product(supplier_url=good_url)
    assert snap.supplier_url == good_url


def test_rejects_non_finite_floats_across_models_and_engine() -> None:
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError):
            _sample_product(supplier_price_eur=bad)
        with pytest.raises(ValidationError):
            SieveConfig(min_net_profit_eur=bad)
        with pytest.raises(ValueError):
            trim_outliers_iqr([10.0, bad])
        with pytest.raises(ValueError):
            compute_market_valuation([10.0, bad])

    with pytest.raises(ValueError):
        compute_market_valuation([10.0, 12.0], anchor_cap_eur=float("inf"))
    with pytest.raises(ValueError):
        compute_market_valuation([10.0, 12.0], anchor_cap_ratio=float("nan"))

    broken = ProductSnapshot.model_construct(
        sku="SKU-INF",
        title="Broken",
        supplier_url="https://supplier.example.eu/p",
        supplier_price_eur=10.0,
        target_resale_eur=float("inf"),
        shipping_cost_eur=0.0,
        in_stock=True,
        ships_from_eu=True,
        delivery_days=2,
        rating=4.5,
        recorded_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValueError):
        compute_net_economics(broken, SieveConfig())
    with pytest.raises(ValueError):
        detect_price_drop(_sample_product(), float("nan"), 0.1)


def test_economics_and_price_drop_reject_finite_overflow_and_subnormal_division() -> None:
    overflow_item = _sample_product(
        supplier_price_eur=1e308,
        target_resale_eur=1e308,
        shipping_cost_eur=1e308,
    )
    with pytest.raises(ValueError):
        compute_net_economics(overflow_item, SieveConfig())
    with pytest.raises(ValueError):
        evaluate_product(overflow_item, SieveConfig())

    subnormal_item = _sample_product(
        supplier_price_eur=10.0,
        target_resale_eur=1e-320,
        shipping_cost_eur=0.0,
    )
    with pytest.raises(ValueError):
        compute_net_economics(subnormal_item, SieveConfig())
    with pytest.raises(ValueError):
        evaluate_product(subnormal_item, SieveConfig())

    neg_extreme = ProductSnapshot.model_construct(
        sku="SKU-DROP-INF",
        title="Extreme",
        supplier_url="https://supplier.example.eu/p",
        supplier_price_eur=-1e308,
        target_resale_eur=100.0,
        shipping_cost_eur=0.0,
        in_stock=True,
        ships_from_eu=True,
        delivery_days=2,
        rating=4.5,
        recorded_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValueError):
        detect_price_drop(neg_extreme, 1e-320, 0.1)


def test_export_csv_neutralizes_formulas_and_preserves_negative_numbers(tmp_path: Path) -> None:
    verdicts = [
        FilterVerdict(
            sku="=CMD|'/C calc'!A0",
            title=" =1+1",
            approved=False,
            stage_reached="-margin_sieve",
            reasons=["\t=bad", "@mention"],
            net_profit_eur=-12.5,
            net_margin_pct=-0.125,
            score=0.0,
            price_alert=None,
        ),
        FilterVerdict(
            sku="+SKU-2",
            title="@Title",
            approved=True,
            stage_reached="approved",
            reasons=["passed_all_gates"],
            net_profit_eur=15.0,
            net_margin_pct=0.25,
            score=75.0,
            price_alert=None,
        ),
    ]
    csv_path = tmp_path / "safe.csv"
    export_csv(verdicts, csv_path)

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[1][0] == "'=CMD|'/C calc'!A0"
    assert rows[1][1] == "' =1+1"
    assert rows[1][3] == "'-margin_sieve"
    assert rows[1][4] == "-12.50"
    assert rows[1][5] == "-0.1250"
    assert rows[1][8] == "'\t=bad;@mention"
    assert rows[2][0] == "'+SKU-2"
    assert rows[2][1] == "'@Title"


def test_snapshot_store_concurrent_batches_thread_safe(tmp_path: Path) -> None:
    db_file = tmp_path / "concurrent.db"
    cfg = SieveConfig(price_drop_alert_pct=0.10)

    def _worker(worker_id: int) -> None:
        with SnapshotStore(db_file) as store:
            run_sieve_batch(
                [_sample_product(sku=f"SKU-W{worker_id}", supplier_price_eur=80.0)],
                store,
                cfg,
            )
            run_sieve_batch(
                [_sample_product(sku=f"SKU-W{worker_id}", supplier_price_eur=60.0)],
                store,
                cfg,
            )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(8)))

    with SnapshotStore(db_file) as store:
        shared_items = [_sample_product(sku="SKU-SHARED", supplier_price_eur=90.0)]
        run_sieve_batch(shared_items, store, cfg)

        def _shared_worker(idx: int) -> None:
            run_sieve_batch(
                [_sample_product(sku=f"SKU-T-{idx}", supplier_price_eur=70.0)],
                store,
                cfg,
            )
            run_sieve_batch(
                [_sample_product(sku=f"SKU-T-{idx}", supplier_price_eur=50.0)],
                store,
                cfg,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_shared_worker, range(8)))

        alerts = store.recent_alerts(limit=100)
        assert len(alerts) == 16


def test_run_sieve_batch_atomic_rollback_on_mid_batch_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "atomic.db"
    items = [
        _sample_product(sku="SKU-ATOMIC-1", supplier_price_eur=75.0),
        _sample_product(sku="SKU-ATOMIC-2", supplier_price_eur=85.0),
    ]
    with SnapshotStore(db_file) as store:
        orig_save = store.save_observation
        calls = 0

        def _failing_save(snapshot: ProductSnapshot, alert=None) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("Simulated mid-batch crash")
            orig_save(snapshot, alert)

        monkeypatch.setattr(store, "save_observation", _failing_save)
        with pytest.raises(RuntimeError, match="Simulated mid-batch crash"):
            run_sieve_batch(items, store)

        assert store.get_previous_price("SKU-ATOMIC-1") is None
        assert store.get_previous_price("SKU-ATOMIC-2") is None


def test_recorded_at_utc_normalization_orders_mixed_timezones(tmp_path: Path) -> None:
    db_file = tmp_path / "tz.db"
    tz_plus2 = timezone(timedelta(hours=2))
    older_local = _sample_product(
        sku="SKU-TZ",
        supplier_price_eur=100.0,
        recorded_at=datetime(2026, 11, 22, 13, 0, tzinfo=tz_plus2),  # 11:00 UTC
    )
    newer_utc = _sample_product(
        sku="SKU-TZ",
        supplier_price_eur=80.0,
        recorded_at=datetime(2026, 11, 22, 11, 30),  # naive -> 11:30 UTC
    )
    with SnapshotStore(db_file) as store:
        store.save_observation(newer_utc)
        store.save_observation(older_local)
        assert store.get_previous_price("SKU-TZ") == pytest.approx(80.0)


def test_recent_alerts_limit_clamped(tmp_path: Path) -> None:
    db_file = tmp_path / "alerts.db"
    with SnapshotStore(db_file) as store:
        for idx in range(5):
            snap = _sample_product(sku=f"SKU-A{idx}", supplier_price_eur=50.0)
            verdict = evaluate_product(snap, SieveConfig(), previous_price=100.0)
            store.save_observation(snap, verdict.price_alert)
        assert len(store.recent_alerts(-1)) == 1
        assert len(store.recent_alerts(0)) == 1
        assert len(store.recent_alerts(999999)) == 5


def test_fuzz_trim_outliers_iqr_invariants() -> None:
    rng = random.Random(20261123)
    # Deterministic zero-IQR flat plateau with extreme outliers
    assert trim_outliers_iqr([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 999.0]) == [50.0] * 7
    assert trim_outliers_iqr([1.0, 50.0, 50.0, 50.0, 999.0]) == [50.0, 50.0, 50.0]

    for _ in range(200):
        base = rng.uniform(10.0, 500.0)
        cluster_n = rng.randint(8, 15)
        if rng.random() < 0.5:
            cluster = [base] * cluster_n
        else:
            cluster = [round(base * rng.uniform(0.95, 1.05), 2) for _ in range(cluster_n)]
        low_outliers = [round(base * 0.01, 2) for _ in range(rng.randint(0, 2))]
        high_outliers = [round(base * 25.0, 2) for _ in range(rng.randint(0, 2))]
        sample = cluster + low_outliers + high_outliers
        rng.shuffle(sample)

        trimmed = trim_outliers_iqr(sample)
        assert trimmed
        assert trimmed == sorted(trimmed)
        assert all(math.isfinite(x) and x > 0 for x in trimmed)
        for lo in low_outliers:
            assert lo not in trimmed
        for hi in high_outliers:
            assert hi not in trimmed
        if not low_outliers and not high_outliers and len(set(cluster)) == 1:
            assert trimmed == sorted(cluster)


def test_fuzz_price_drop_rounding_and_out_of_order(tmp_path: Path) -> None:
    # Exact boundary checks before round(..., 4)
    snap_sub_10 = _sample_product(supplier_price_eur=90.004)
    assert detect_price_drop(snap_sub_10, 100.0, 0.10) is None

    snap_just_over = _sample_product(supplier_price_eur=89.996)
    alert = detect_price_drop(snap_just_over, 100.0, 0.10002)
    assert alert is not None and alert.drop_pct == 0.1

    # Out-of-order delayed backfill must not trigger false drop against newer observation
    t1 = datetime(2026, 11, 20, 10, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 21, 10, 0, tzinfo=timezone.utc)
    assert detect_price_drop(_sample_product(supplier_price_eur=70.0, recorded_at=t1), 100.0, 0.10, previous_recorded_at=t2) is None

    rng = random.Random(20261124)
    cfg = SieveConfig(price_drop_alert_pct=0.10)
    db_file = tmp_path / "fuzz_order.db"
    with SnapshotStore(db_file) as store:
        for trial in range(100):
            sku = f"SKU-ORD-{trial}"
            t0 = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
            points = [
                (t0 + timedelta(hours=i), round(120.0 * (0.85 ** i) if i % 2 == 1 else rng.uniform(80.0, 130.0), 2))
                for i in range(6)
            ]
            shuffled = list(points)
            rng.shuffle(shuffled)
            snaps = [_sample_product(sku=sku, supplier_price_eur=p, recorded_at=ts) for ts, p in shuffled]

            expected_drops = 0
            prev_p: float | None = None
            for _, p in sorted(points, key=lambda x: x[0]):
                if prev_p is not None and (prev_p - p) / prev_p >= cfg.price_drop_alert_pct:
                    expected_drops += 1
                prev_p = p

            verdicts = run_sieve_batch(snaps, store, cfg)
            emitted_drops = sum(1 for v in verdicts if v.price_alert is not None)
            assert emitted_drops == expected_drops
            # Stale arrival older than latest stored timestamp must never emit a false drop
            stale = _sample_product(sku=sku, supplier_price_eur=1.0, recorded_at=t0 - timedelta(hours=1))
            stale_verdict = run_sieve_batch([stale], store, cfg)[0]
            assert stale_verdict.price_alert is None


def test_fuzz_numeric_bool_and_fee_sum_validation() -> None:
    for b in (True, False):
        for field in ("supplier_price_eur", "target_resale_eur", "shipping_cost_eur", "delivery_days", "rating"):
            with pytest.raises(ValidationError):
                _sample_product(**{field: b})
        for cfg_field in (
            "marketplace_fee_pct",
            "return_reserve_pct",
            "min_net_profit_eur",
            "min_net_margin_pct",
            "max_delivery_days",
            "min_rating",
            "price_drop_alert_pct",
        ):
            with pytest.raises(ValidationError):
                SieveConfig(**{cfg_field: b})
        for alert_field in ("previous_price_eur", "current_price_eur", "drop_pct"):
            kwargs = {
                "sku": "S",
                "title": "T",
                "previous_price_eur": 100.0,
                "current_price_eur": 80.0,
                "drop_pct": 0.2,
                "recorded_at": datetime.now(timezone.utc),
            }
            kwargs[alert_field] = b
            with pytest.raises(ValidationError):
                PriceAlert.model_validate(kwargs)
        for v_field in ("net_profit_eur", "net_margin_pct", "score"):
            v_kwargs = {
                "sku": "S",
                "title": "T",
                "approved": True,
                "stage_reached": "approved",
                "reasons": ["ok"],
                "net_profit_eur": 10.0,
                "net_margin_pct": 0.2,
                "score": 50.0,
            }
            v_kwargs[v_field] = b
            with pytest.raises(ValidationError):
                FilterVerdict.model_validate(v_kwargs)
        for mv_field in ("sample_count", "trimmed_count", "median_eur", "fast_sell_p40_eur", "spread_ratio", "liquidity_score"):
            mv_kwargs = {"sample_count": 4, "trimmed_count": 4, "liquidity_score": 60, "low_liquidity": False}
            mv_kwargs[mv_field] = b
            with pytest.raises(ValidationError):
                MarketValuation.model_validate(mv_kwargs)

    with pytest.raises(ValueError):
        trim_outliers_iqr([True, 10.0])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        compute_market_valuation([10.0, 20.0], min_liquidity_score=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_market_valuation([10.0, 20.0], anchor_cap_eur=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        compute_market_valuation([10.0, 20.0], anchor_cap_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        detect_price_drop(_sample_product(), True, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        detect_price_drop(_sample_product(), 100.0, False)  # type: ignore[arg-type]

    rng = random.Random(20261125)
    for _ in range(100):
        f1 = round(rng.uniform(0.0, 0.99), 4)
        f2 = round(rng.uniform(0.0, 0.99), 4)
        if f1 + f2 >= 1.0:
            with pytest.raises(ValidationError):
                SieveConfig(marketplace_fee_pct=f1, return_reserve_pct=f2)
            bad_cfg = SieveConfig.model_construct(marketplace_fee_pct=f1, return_reserve_pct=f2)
            with pytest.raises(ValueError):
                compute_net_economics(_sample_product(), bad_cfg)
        else:
            cfg = SieveConfig(marketplace_fee_pct=f1, return_reserve_pct=f2)
            profit, margin = compute_net_economics(_sample_product(), cfg)
            assert math.isfinite(profit) and math.isfinite(margin)


def test_fuzz_ssrf_bypass_vectors() -> None:
    static_vectors = [
        "http://[fec0::1]/",
        "http://[FEC0:0:0:1::]/admin",
        "http://localhost.localdomain/",
        "https://app.localdomain/api",
        "http://ip6-localhost/",
        "https://example.com../feed",
        r"https://127.0.0.1\@8.8.8.8/",
        r"https://8.8.8.8\@127.0.0.1/",
        "https://example.com/foo\\bar",
        "http://192.88.99.1/",
        "http://0x7f.0.0.1/",
        "http://0177.0000.0000.0001/",
        "http://[::ffff:192.168.1.1]/",
        "http://[::ffff:7f00:1]/",
    ]
    for url in static_vectors:
        with pytest.raises(ValidationError):
            _sample_product(supplier_url=url)

    rng = random.Random(20261126)
    internal_hosts = ["localhost", "ip6-localhost", "app.localdomain", "db.internal", "svc.local", "[fec0::1]", "192.88.99.1"]
    for _ in range(80):
        host = rng.choice(internal_hosts)
        if rng.random() < 0.5:
            host = "".join(c.upper() if rng.random() < 0.5 else c for c in host)
        url = f"http://{host}:{rng.choice([80, 443, 8080])}/feed"
        with pytest.raises(ValidationError):
            _sample_product(supplier_url=url)


def test_fuzz_csv_injection_neutralization(tmp_path: Path) -> None:
    rng = random.Random(20261127)
    prefixes = ["=", "+", "-", "@", "\t", "\r", "\n", "\x0b", " ", "  "]
    verdicts: list[FilterVerdict] = []
    for idx in range(100):
        p = rng.choice(prefixes)
        q = rng.choice(["=", "+", "-", "@"]) if p in (" ", "  ", "\x0b", "\t", "\r", "\n") else ""
        verdicts.append(
            FilterVerdict(
                sku=f"{p}{q}SKU_{idx}",
                title=f"{p}{q}Title_{idx}",
                approved=False,
                stage_reached=f"{p}{q}stage",
                reasons=[f"{p}{q}reason"],
                net_profit_eur=-10.25,
                net_margin_pct=-0.15,
                score=0.0,
                price_alert=None,
            )
        )
    csv_path = tmp_path / "fuzz_safe.csv"
    export_csv(verdicts, csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 101
    for row in rows[1:]:
        assert row[0].startswith("'")
        assert row[1].startswith("'")
        assert row[3].startswith("'")
        assert row[8].startswith("'")
        assert row[4] == "-10.25"
        assert row[5] == "-0.1500"


def test_cli_sample_catalog_matches_readme_scores(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = repo_root / "data" / "sample_catalog.json"
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    rc = cli_main(["run", "--input", str(catalog), "--db", str(tmp_path / "readme.db")])
    assert rc == 0
    out = capsys.readouterr().out
    for expected in (
        "KB-DOCK-TB4  score= 73.9",
        "KB-MON-27QHD score= 70.5",
        "KB-SCAN-PRO  score= 58.2",
    ):
        assert expected in out
        assert expected in readme


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://[::1]/",
        "http://[0:0:0:0:0:0:0:1]/admin",
        "http://[0:0:0:0:0:ffff:127.0.0.1]/",
        "https://localh\u043est/feed",
        "http://\uff11\uff12\uff17.\uff10.\uff10.\uff11/",
        "http://127\u200b.0.0.1/",
        "http://2130706433:8080/",
        "http://0177.0.0.1:8080/",
        "http://0x7f000001:8080/",
    ],
)
def test_ssrf_blocks_ipv6_homoglyphs_and_zero_width_bypasses(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        _sample_product(supplier_url=bad_url)


def test_unicode_zero_width_sku_title_and_nfkc_csv_formula_neutralization(tmp_path: Path) -> None:
    for zw in ("\u200b\u200c\u200d", "\ufeff  \u2060"):
        with pytest.raises(ValidationError):
            _sample_product(sku=zw)
        with pytest.raises(ValidationError):
            _sample_product(title=zw)

    verdicts = [
        FilterVerdict(
            sku="\ufeff=1+1",
            title="\uff1dCMD()",
            approved=False,
            stage_reached="\u200b+stage",
            reasons=["\uff20alert", "\r\n-inject"],
            net_profit_eur=-5.0,
            net_margin_pct=-0.1,
            score=0.0,
            price_alert=None,
        ),
        FilterVerdict(
            sku="\u00ad=1+1",
            title="\u200e+1+1",
            approved=False,
            stage_reached="\u202a-stage",
            reasons=["\u00ad=reason"],
            net_profit_eur=-5.0,
            net_margin_pct=-0.1,
            score=0.0,
            price_alert=None,
        ),
    ]
    csv_path = tmp_path / "unicode_safe.csv"
    export_csv(verdicts, csv_path)
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    for row in rows[1:]:
        assert row[0].startswith("'")
        assert row[1].startswith("'")
        assert row[3].startswith("'")
        assert row[8].startswith("'")


def test_recent_alerts_int64_bounds_and_wal_16_thread_stress_and_rmtree(tmp_path: Path) -> None:
    wal_dir = tmp_path / "wal16"
    wal_dir.mkdir()
    db_file = wal_dir / "stress16.db"
    cfg = SieveConfig(price_drop_alert_pct=0.10)
    barrier = threading.Barrier(16)
    errors: list[tuple[int, Exception]] = []
    err_lock = threading.Lock()

    def _worker(tid: int) -> None:
        try:
            barrier.wait(timeout=5.0)
            with SnapshotStore(db_file) as store:
                run_sieve_batch([_sample_product(sku=f"SKU-S16-{tid}", supplier_price_eur=100.0)], store, cfg)
                run_sieve_batch([_sample_product(sku=f"SKU-S16-{tid}", supplier_price_eur=70.0)], store, cfg)
        except Exception as exc:
            with err_lock:
                errors.append((tid, exc))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_worker, range(16)))

    assert errors == []
    with SnapshotStore(db_file) as store:
        assert store.journal_mode() == "wal"
        assert store._ensure_conn().execute("PRAGMA busy_timeout;").fetchone()[0] == 5000
        for bound in (0, 2**63 - 1, 2**63, -(2**63) - 1):
            res = store.recent_alerts(bound)
            assert 1 <= len(res) <= 16

    shutil.rmtree(wal_dir)
    assert not wal_dir.exists()

    # Unclosed SnapshotStore relies on weakref.finalize during GC before rmtree (WinError 32 check)
    leak_dir = tmp_path / "wal_gc"
    leak_dir.mkdir()
    leak_db = leak_dir / "gc.db"

    def _unclosed() -> None:
        s = SnapshotStore(leak_db)
        s.save_observation(_sample_product(sku="SKU-GC"))

    _unclosed()
    gc.collect()
    shutil.rmtree(leak_dir)
    assert not leak_dir.exists()
