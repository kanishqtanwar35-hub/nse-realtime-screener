"""
SQLite persistence.

Two consumers, one file: the live engine writes, the dashboard reads. That is
why WAL is enabled - it lets the Streamlit process read while the engine is
mid-write instead of blocking on a locked database.

Every write is idempotent. Signals and trades carry a natural key
(symbol + timestamp + side), so re-running a cycle or replaying a backfill
updates rows instead of duplicating them. Same discipline as any ingestion
pipeline: running it twice must be safe.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import settings
from .models import Prediction, Signal, Trade
from .utils import get_logger

log = get_logger(__name__)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    ltp         REAL NOT NULL,
    reason      TEXT,
    features    TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE(symbol, timestamp, side)
);

CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,
    entry_time       TEXT NOT NULL,
    entry_price      REAL NOT NULL,
    exit_time        TEXT,
    exit_price       REAL,
    exit_reason      TEXT,
    gross_pnl_pct    REAL,
    net_pnl_pct      REAL,
    duration_minutes REAL,
    mae_pct          REAL,
    mfe_pct          REAL,
    label            INTEGER,
    features         TEXT,
    created_at       TEXT NOT NULL,
    UNIQUE(symbol, entry_time, side)
);

CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol        TEXT NOT NULL,
    side          TEXT NOT NULL,
    signal_time   TEXT NOT NULL,
    probability   REAL NOT NULL,
    decision      TEXT NOT NULL,
    explanation   TEXT,
    model_version TEXT,
    created_at    TEXT NOT NULL,
    UNIQUE(symbol, signal_time, side)
);

CREATE TABLE IF NOT EXISTS screen_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    ltp        REAL, bid_price REAL, ask_price REAL,
    bid_qty    INTEGER, ask_qty INTEGER, ltq INTEGER,
    spread_pct REAL, imbalance REAL, passed INTEGER
);

CREATE TABLE IF NOT EXISTS live_metrics (
    symbol       TEXT PRIMARY KEY,
    captured_at  TEXT NOT NULL,
    ltp          REAL,
    smma_20      REAL, smma_120 REAL, smma_signal TEXT,
    etq_5m       REAL, etq_20m  REAL, etq_60m     REAL,
    avg_ltp_20m  REAL, avg_ltp_60m REAL,
    bid_price    REAL, bid_qty  INTEGER,
    ask_price    REAL, ask_qty  INTEGER,
    ltq          INTEGER, ltq_avg_2m REAL, ltq_avg_5m REAL, ltq_ratio_2_5 REAL,
    spread_pct   REAL, imbalance REAL
);

-- 1-minute OHLCV with the two SMMA levels already attached.
--
-- The engine holds bars in memory; without this table the dashboard - a
-- separate process - has no way to draw a price chart, so the one visual a
-- trading screener most obviously needs would be impossible. Keyed on
-- (symbol, timestamp) so replaying a session overwrites rather than duplicates.
CREATE TABLE IF NOT EXISTS bars (
    symbol    TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open      REAL, high REAL, low REAL, close REAL, volume REAL,
    smma_fast REAL, smma_slow REAL,
    PRIMARY KEY (symbol, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_signals_time  ON signals(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_bars_symbol   ON bars(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_entry  ON trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_pred_time     ON predictions(signal_time DESC);
CREATE INDEX IF NOT EXISTS idx_snap_time     ON screen_snapshots(captured_at DESC);
"""


class Store:
    """Thin, thread-safe SQLite wrapper. No ORM - the schema is five tables."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.paths.db)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)
        log.debug("store ready at %s", self.path)

    # ------------------------------------------------------------------
    def save_signals(self, signals: list[Signal]) -> int:
        if not signals:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            (s.symbol, s.side.value, s.timestamp.isoformat(), s.ltp, s.reason,
             json.dumps(s.features), now)
            for s in signals
        ]
        with self._lock, self._connect() as conn:
            cur = conn.executemany(
                """INSERT INTO signals (symbol, side, timestamp, ltp, reason, features, created_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, timestamp, side) DO UPDATE SET
                       ltp = excluded.ltp, reason = excluded.reason,
                       features = excluded.features""",
                rows,
            )
            return cur.rowcount

    def save_trades(self, trades: list[Trade]) -> int:
        if not trades:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            (t.symbol, t.side.value, t.entry_time.isoformat(), t.entry_price,
             t.exit_time.isoformat() if t.exit_time else None, t.exit_price, t.exit_reason,
             t.gross_pnl_pct, t.net_pnl_pct, t.duration_minutes, t.mae_pct, t.mfe_pct,
             int(t.is_profitable), json.dumps(t.features), now)
            for t in trades
        ]
        with self._lock, self._connect() as conn:
            cur = conn.executemany(
                """INSERT INTO trades (symbol, side, entry_time, entry_price, exit_time,
                       exit_price, exit_reason, gross_pnl_pct, net_pnl_pct, duration_minutes,
                       mae_pct, mfe_pct, label, features, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, entry_time, side) DO UPDATE SET
                       exit_time = excluded.exit_time, exit_price = excluded.exit_price,
                       exit_reason = excluded.exit_reason, net_pnl_pct = excluded.net_pnl_pct,
                       gross_pnl_pct = excluded.gross_pnl_pct,
                       duration_minutes = excluded.duration_minutes,
                       mae_pct = excluded.mae_pct, mfe_pct = excluded.mfe_pct,
                       label = excluded.label""",
                rows,
            )
            return cur.rowcount

    def save_predictions(self, predictions: list[Prediction], signal_times: list[datetime]) -> int:
        if not predictions:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        rows = [
            (p.symbol, p.side.value, t.isoformat(), p.probability, p.decision.value,
             p.explanation, p.model_version, now)
            for p, t in zip(predictions, signal_times)
        ]
        with self._lock, self._connect() as conn:
            cur = conn.executemany(
                """INSERT INTO predictions (symbol, side, signal_time, probability, decision,
                       explanation, model_version, created_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(symbol, signal_time, side) DO UPDATE SET
                       probability = excluded.probability, decision = excluded.decision,
                       explanation = excluded.explanation,
                       model_version = excluded.model_version""",
                rows,
            )
            return cur.rowcount

    def save_screen_snapshot(self, frame: pd.DataFrame) -> int:
        """Persist the current screen so the dashboard has data between cycles."""
        if frame is None or frame.empty:
            return 0
        now = datetime.now().isoformat(timespec="seconds")
        df = frame.copy()
        df["captured_at"] = now
        df["passed"] = (df.get("status", "PASS") == "PASS").astype(int)

        cols = ["captured_at", "symbol", "ltp", "bid_price", "ask_price",
                "bid_qty", "ask_qty", "ltq", "spread_pct", "imbalance", "passed"]
        for c in cols:
            if c not in df.columns:
                df[c] = None

        with self._lock, self._connect() as conn:
            # Keep only the newest snapshot - this table is a live view, not history.
            conn.execute("DELETE FROM screen_snapshots")
            conn.executemany(
                f"INSERT INTO screen_snapshots ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                df[cols].itertuples(index=False, name=None),
            )
        return len(df)

    def save_live_metrics(self, rows: list[dict]) -> int:
        """
        Upsert the one-row-per-stock dashboard view.

        Keyed on symbol so the table is always the CURRENT picture, never a
        growing log - the dashboard wants latest state, and an append-only table
        would need a window function on every read.
        """
        if not rows:
            return 0
        from .features import DISPLAY_COLUMNS

        now = datetime.now().isoformat(timespec="seconds")
        cols = ["symbol", "captured_at"] + [c for c in DISPLAY_COLUMNS if c != "symbol"]
        payload = [
            tuple([r.get("symbol"), now] + [r.get(c) for c in DISPLAY_COLUMNS if c != "symbol"])
            for r in rows
        ]
        updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "symbol")
        with self._lock, self._connect() as conn:
            conn.executemany(
                f"INSERT INTO live_metrics ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))}) "
                f"ON CONFLICT(symbol) DO UPDATE SET {updates}",
                payload,
            )
        return len(payload)

    def load_live_metrics(self) -> pd.DataFrame:
        return self._read("SELECT * FROM live_metrics ORDER BY symbol")

    # ------------------------------------------------------------------
    def save_bars(self, symbol: str, frame: pd.DataFrame, tail: int | None = None) -> int:
        """
        Persist OHLCV plus the SMMA levels for one symbol.

        `tail` limits the write to the newest N rows, which is what the live
        loop wants: only the bar that just closed has changed, and rewriting a
        1,500-row history every five seconds would make the cycle disk-bound.
        """
        if frame is None or frame.empty:
            return 0

        df = frame.tail(tail) if tail else frame
        cols = ["open", "high", "low", "close", "volume", "smma_fast", "smma_slow"]
        rows = []
        for row in df.itertuples(index=False):
            ts = getattr(row, "timestamp", None)
            if ts is None or pd.isna(ts):
                continue
            values = []
            for c in cols:
                v = getattr(row, c, None)
                values.append(None if v is None or pd.isna(v) else float(v))
            rows.append((symbol, pd.Timestamp(ts).isoformat(), *values))

        if not rows:
            return 0
        with self._lock, self._connect() as conn:
            conn.executemany(
                f"INSERT INTO bars (symbol, timestamp, {','.join(cols)}) "
                f"VALUES ({','.join('?' * (len(cols) + 2))}) "
                "ON CONFLICT(symbol, timestamp) DO UPDATE SET "
                + ", ".join(f"{c}=excluded.{c}" for c in cols),
                rows,
            )
        return len(rows)

    def load_bars(self, symbol: str, limit: int = 600) -> pd.DataFrame:
        """Newest `limit` bars for one symbol, returned oldest-first for plotting."""
        df = self._read(
            "SELECT * FROM bars WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol, limit),
        )
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)

    def bar_symbols(self) -> list[str]:
        df = self._read("SELECT symbol, COUNT(*) AS n FROM bars GROUP BY symbol "
                        "HAVING n > 0 ORDER BY symbol")
        return df["symbol"].tolist() if not df.empty else []

    # ------------------------------------------------------------------
    def _read(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        try:
            with self._connect() as conn:
                return pd.read_sql_query(sql, conn, params=params)
        except Exception as exc:  # noqa: BLE001
            log.error("read failed: %s", exc)
            return pd.DataFrame()

    def load_signals(self, limit: int = 200) -> pd.DataFrame:
        return self._read("SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,))

    def load_trades(self, limit: int | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM trades ORDER BY entry_time DESC"
        return self._read(sql + " LIMIT ?", (limit,)) if limit else self._read(sql)

    def load_predictions(self, limit: int = 200) -> pd.DataFrame:
        return self._read("SELECT * FROM predictions ORDER BY signal_time DESC LIMIT ?", (limit,))

    def load_screen(self) -> pd.DataFrame:
        return self._read("SELECT * FROM screen_snapshots ORDER BY passed DESC, symbol")

    def load_training_set(self) -> pd.DataFrame:
        """
        Closed trades, expanded into a model-ready frame.

        The features JSON blob is exploded into f_-prefixed columns, which is
        exactly the shape TradeClassifier.train() expects.
        """
        df = self._read("SELECT * FROM trades WHERE exit_time IS NOT NULL ORDER BY entry_time")
        if df.empty:
            return df

        feats = df["features"].apply(lambda s: json.loads(s) if isinstance(s, str) and s else {})
        expanded = pd.json_normalize(feats).add_prefix("f_")
        return pd.concat([df.drop(columns=["features"]), expanded], axis=1)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for table in ("signals", "trades", "predictions", "screen_snapshots",
                      "live_metrics", "bars"):
            df = self._read(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(df["n"].iloc[0]) if not df.empty else 0
        return out

    def clear(self, tables: list[str] | None = None) -> None:
        tables = tables or ["signals", "trades", "predictions", "screen_snapshots",
                            "live_metrics", "bars"]
        with self._lock, self._connect() as conn:
            for t in tables:
                conn.execute(f"DELETE FROM {t}")
        log.info("cleared tables: %s", ", ".join(tables))
