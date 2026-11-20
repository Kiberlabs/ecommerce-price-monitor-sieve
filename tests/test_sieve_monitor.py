from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sieve_monitor.cli import main as cli_main
from sieve_monitor.filter_engine import evaluate_product, run_sieve_batch
from sieve_monitor.models import ProductSnapshot, SieveConfig
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
