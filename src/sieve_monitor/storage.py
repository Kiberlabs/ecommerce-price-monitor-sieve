from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import PriceAlert, ProductSnapshot

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS price_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL,
    title TEXT NOT NULL,
    supplier_url TEXT NOT NULL,
    supplier_price_eur REAL NOT NULL,
    target_resale_eur REAL NOT NULL,
    in_stock INTEGER NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_sku_time
    ON price_snapshots (sku, recorded_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS alerts_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL,
    title TEXT NOT NULL,
    previous_price_eur REAL NOT NULL,
    current_price_eur REAL NOT NULL,
    drop_pct REAL NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


class SnapshotStore:
    """Concurrent SQLite WAL store for historical supplier snapshots and price-drop alerts."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        with self._conn:
            self._conn.executescript(SCHEMA_SQL)

    def journal_mode(self) -> str:
        row = self._conn.execute("PRAGMA journal_mode;").fetchone()
        return str(row[0]).lower() if row else ""

    def get_previous_price(self, sku: str) -> float | None:
        row = self._conn.execute(
            "SELECT supplier_price_eur FROM price_snapshots WHERE sku = ? ORDER BY recorded_at DESC, id DESC LIMIT 1",
            (sku,),
        ).fetchone()
        return float(row["supplier_price_eur"]) if row is not None else None

    def save_observation(self, snapshot: ProductSnapshot, alert: PriceAlert | None = None) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO price_snapshots (
                    sku, title, supplier_url, supplier_price_eur, target_resale_eur, in_stock, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.sku,
                    snapshot.title,
                    snapshot.supplier_url,
                    snapshot.supplier_price_eur,
                    snapshot.target_resale_eur,
                    int(snapshot.in_stock),
                    snapshot.recorded_at.isoformat(),
                ),
            )
            if alert is not None:
                self._conn.execute(
                    """
                    INSERT INTO alerts_log (
                        sku, title, previous_price_eur, current_price_eur, drop_pct, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        alert.sku,
                        alert.title,
                        alert.previous_price_eur,
                        alert.current_price_eur,
                        alert.drop_pct,
                        alert.recorded_at.isoformat(),
                    ),
                )

    def recent_alerts(self, limit: int = 20) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT sku, title, previous_price_eur, current_price_eur, drop_pct, recorded_at "
            "FROM alerts_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
