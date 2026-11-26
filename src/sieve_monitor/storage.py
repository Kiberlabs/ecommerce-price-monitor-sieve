from __future__ import annotations

import random
import sqlite3
import threading
import time
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .models import PriceAlert, ProductSnapshot

_REGISTRY_LOCK = threading.Lock()
_PATH_LOCKS: dict[Path, threading.RLock] = {}


def _close_conn_box(box: list[sqlite3.Connection | None]) -> None:
    conn = box[0]
    if conn is not None:
        box[0] = None
        conn.close()


def _lock_for_path(db_path: Path) -> threading.RLock:
    if str(db_path) == ":memory:":
        return threading.RLock()
    key = db_path.resolve()
    with _REGISTRY_LOCK:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _PATH_LOCKS[key] = lock
        return lock


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
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _lock_for_path(self.db_path)
        self._tx_depth = 0
        self._conn_box: list[sqlite3.Connection | None] = [None]
        self._finalizer = weakref.finalize(self, _close_conn_box, self._conn_box)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=5.0,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.execute("PRAGMA foreign_keys=ON;")
        except BaseException:
            conn.close()
            raise
        return conn

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn_box[0] is None:
            self._conn_box[0] = self._connect()
            if not self._finalizer.alive:
                self._finalizer = weakref.finalize(self, _close_conn_box, self._conn_box)
        return self._conn_box[0]

    def _drop_conn(self) -> None:
        _close_conn_box(self._conn_box)

    def _init_schema(self) -> None:
        with self._lock:
            for attempt in range(20):
                try:
                    conn = self._ensure_conn()
                    row = conn.execute("PRAGMA journal_mode;").fetchone()
                    if not row or str(row[0]).lower() != "wal":
                        conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA synchronous=NORMAL;")
                    conn.executescript(SCHEMA_SQL)
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 19:
                        self._drop_conn()
                        raise
                    time.sleep(0.02 + random.uniform(0.0, 0.03))
                except BaseException:
                    self._drop_conn()
                    raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Serialize writes across threads and SQLite connections via BEGIN IMMEDIATE."""
        with self._lock:
            conn = self._ensure_conn()
            is_outer = self._tx_depth == 0
            if is_outer:
                for attempt in range(20):
                    try:
                        conn.execute("BEGIN IMMEDIATE;")
                        break
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc).lower() or attempt == 19:
                            raise
                        time.sleep(0.02 + random.uniform(0.0, 0.03))
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                if is_outer and conn.in_transaction:
                    conn.execute("ROLLBACK;")
                raise
            else:
                if is_outer and conn.in_transaction:
                    conn.execute("COMMIT;")
            finally:
                self._tx_depth -= 1

    def journal_mode(self) -> str:
        with self._lock:
            row = self._ensure_conn().execute("PRAGMA journal_mode;").fetchone()
            return str(row[0]).lower() if row else ""

    def get_latest_snapshot(self, sku: str) -> tuple[float, datetime] | None:
        """Return (supplier_price_eur, recorded_at) for the most recent SKU observation, or None."""
        with self._lock:
            row = self._ensure_conn().execute(
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
            conn = self._ensure_conn()
            conn.execute(
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
                conn.execute(
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
            rows = self._ensure_conn().execute(
                "SELECT sku, title, previous_price_eur, current_price_eur, drop_pct, recorded_at "
                "FROM alerts_log ORDER BY id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._finalizer()

    def __enter__(self) -> SnapshotStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
