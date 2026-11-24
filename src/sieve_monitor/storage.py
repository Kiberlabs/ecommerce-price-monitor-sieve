from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import PriceAlert, ProductSnapshot


def _utc_iso(dt: datetime) -> str:
    utc = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds")


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
        """Open SQLite in WAL mode with 5s busy timeout and bootstrap schema tables/indexes."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        for attempt in range(10):
            try:
                if self.journal_mode() != "wal":
                    self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute("PRAGMA synchronous=NORMAL;")
                self._conn.executescript(SCHEMA_SQL)
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt == 9:
                    raise
                time.sleep(0.05)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize writes across threads and SQLite connections via BEGIN IMMEDIATE."""
        with self._lock:
            is_outer = self._tx_depth == 0
            if is_outer:
                self._conn.execute("BEGIN IMMEDIATE;")
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                if is_outer and self._conn.in_transaction:
                    self._conn.execute("ROLLBACK;")
                raise
            else:
                if is_outer and self._conn.in_transaction:
                    self._conn.execute("COMMIT;")
            finally:
                self._tx_depth -= 1

    def journal_mode(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row else ""

    def get_latest_snapshot(self, sku: str) -> tuple[float, datetime] | None:
        """Return (supplier_price_eur, recorded_at) for the most recent SKU observation, or None."""
        with self._lock:
            row = self._conn.execute(
                "SELECT supplier_price_eur, recorded_at FROM price_snapshots WHERE sku = ? ORDER BY recorded_at DESC, id DESC LIMIT 1",
                (sku,),
            ).fetchone()
            if row is None:
                return None
            return float(row["supplier_price_eur"]), datetime.fromisoformat(str(row["recorded_at"]))

    def get_previous_price(self, sku: str) -> float | None:
        """Return the most recent recorded supplier price for a SKU, or None if unseen."""
        latest = self.get_latest_snapshot(sku)
        return latest[0] if latest is not None else None

    def save_observation(self, snapshot: ProductSnapshot, alert: PriceAlert | None = None) -> None:
        """Atomically persist a product snapshot and, when present, its triggered price-drop alert."""
        with self.transaction():
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
                    _utc_iso(snapshot.recorded_at),
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
                        _utc_iso(alert.recorded_at),
                    ),
                )

    def recent_alerts(self, limit: int = 20) -> list[dict[str, object]]:
        """Fetch the latest triggered price-drop alerts ordered newest-first."""
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._conn.execute(
                "SELECT sku, title, previous_price_eur, current_price_eur, drop_pct, recorded_at "
                "FROM alerts_log ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
