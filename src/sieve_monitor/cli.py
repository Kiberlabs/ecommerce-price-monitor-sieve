from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from .filter_engine import run_sieve_batch
from .models import FilterVerdict, ProductSnapshot, SieveConfig
from .storage import SnapshotStore


def load_catalog(path: Path) -> tuple[list[ProductSnapshot], list[dict[str, str]]]:
    """Parse and validate a supplier JSON feed, separating valid snapshots from blocked rows.

    Invalid or SSRF-blocked entries are collected in rejected_inputs instead of aborting the batch.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Catalog JSON must be a list of product objects.")
    valid: list[ProductSnapshot] = []
    rejected_inputs: list[dict[str, str]] = []
    for idx, entry in enumerate(raw):
        try:
            valid.append(ProductSnapshot.model_validate(entry))
        except ValidationError as exc:
            sku = str(entry.get("sku", f"row_{idx}")) if isinstance(entry, dict) else f"row_{idx}"
            rejected_inputs.append({"sku": sku, "error": exc.errors()[0]["msg"]})
    return valid, rejected_inputs


def _csv_safe(value: object) -> str:
    s = str(value)
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFKC", s) if unicodedata.category(ch) != "Cf"
    ).lstrip()
    if s[:1] in ("=", "+", "-", "@", "\t", "\r", "\n") or stripped[:1] in ("=", "+", "-", "@"):
        return "'" + s
    return s


def export_csv(verdicts: list[FilterVerdict], csv_path: Path) -> None:
    """Write ranked filter verdicts and price-drop metrics to a UTF-8 CSV report."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sku", "title", "approved", "stage", "net_profit_eur", "net_margin_pct", "score", "alert_drop_pct", "reasons"])
        for v in verdicts:
            writer.writerow([
                _csv_safe(v.sku),
                _csv_safe(v.title),
                v.approved,
                _csv_safe(v.stage_reached),
                f"{v.net_profit_eur:.2f}",
                f"{v.net_margin_pct:.4f}",
                f"{v.score:.1f}",
                f"{v.price_alert.drop_pct:.4f}" if v.price_alert else "",
                _csv_safe(";".join(v.reasons)),
            ])


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: loads catalog -> runs 3-stage sieve with SQLite WAL -> exports CSV/JSON."""
    parser = argparse.ArgumentParser(prog="sieve-monitor", description="KiberLabs B2B Price & Margin Sieve")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Evaluate a supplier catalog JSON and persist snapshots to SQLite WAL")
    run_p.add_argument("--input", type=Path, required=True, help="Path to supplier catalog JSON")
    run_p.add_argument("--db", type=Path, default=Path("data/monitor.db"), help="SQLite WAL database path")
    run_p.add_argument("--csv-out", type=Path, default=None, help="Optional path to export ranked CSV")
    run_p.add_argument("--min-margin", type=float, default=0.18, help="Minimum net margin ratio (default: 0.18)")
    run_p.add_argument("--json", action="store_true", help="Emit structured JSON summary")

    args = parser.parse_args(argv)
    snapshots, invalid_rows = load_catalog(args.input)
    cfg = SieveConfig(min_net_margin_pct=args.min_margin)

    with SnapshotStore(args.db) as store:
        verdicts = run_sieve_batch(snapshots, store, cfg)

    if args.csv_out:
        export_csv(verdicts, args.csv_out)

    approved = [v for v in verdicts if v.approved]
    alerts = [v.price_alert for v in verdicts if v.price_alert is not None]

    if args.json:
        payload = {
            "total_valid_inputs": len(snapshots),
            "blocked_invalid_inputs": invalid_rows,
            "approved_count": len(approved),
            "alerts_triggered": len(alerts),
            "champions": [v.model_dump(mode="json") for v in approved],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"[KiberLabs Sieve] Processed={len(snapshots)} | Approved={len(approved)} | Blocked={len(invalid_rows)} | Alerts={len(alerts)}")
        for v in approved:
            tag = f" [DROP -{v.price_alert.drop_pct:.0%}]" if v.price_alert else ""
            print(f"  + {v.sku:<12} score={v.score:>5.1f}  profit={v.net_profit_eur:>6.2f} EUR ({v.net_margin_pct:.1%}){tag} — {v.title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
