"""Shared fixtures. Everything here is deterministic - no clocks, no randomness."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nse_screener.models import Quote, Side, Signal  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def session_start() -> datetime:
    return datetime(2024, 3, 1, 9, 15)


def make_bars(closes, start: datetime, volume: float = 1000.0) -> pd.DataFrame:
    """Build a minimal but valid bar frame from a close series."""
    closes = list(closes)
    return pd.DataFrame(
        {
            "timestamp": [start + timedelta(minutes=i) for i in range(len(closes))],
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [volume] * len(closes),
        }
    )


@pytest.fixture
def bars_factory():
    return make_bars


@pytest.fixture
def crossing_bars(session_start) -> pd.DataFrame:
    """
    A price path engineered to produce genuine SMMA20/SMMA120 crossovers.

    Down, then up, then down. The decline pushes the fast average clearly BELOW
    the slow one, so when price turns the spread crosses from negative to
    positive - a real sign change with a decided prior state. The final decline
    crosses back. Result: exactly one BUY and one SELL.

    Note what does NOT work: starting flat and then ramping. Two SMMAs over a
    constant price are exactly equal, so the spread is 0 - undecided, never
    negative - and the subsequent rise is a divergence from equality, not a
    crossing. `crossover` deliberately refuses to emit there (see
    test_no_signal_from_an_undecided_start).
    """
    down = list(np.linspace(130.0, 100.0, 180))
    up = list(np.linspace(100.0, 135.0, 180))
    down2 = list(np.linspace(135.0, 100.0, 180))
    return make_bars(down + up + down2, session_start)


@pytest.fixture
def quote_factory():
    def _make(symbol="TEST", ltp=100.0, bid_qty=2_000_000, ask_qty=2_000_000,
              ltq=500, spread=0.10, ts: datetime | None = None, volume=0):
        return Quote(
            symbol=symbol,
            ltp=ltp,
            bid_price=ltp - spread / 2,
            ask_price=ltp + spread / 2,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            ltq=ltq,
            volume=volume,
            timestamp=ts or datetime(2024, 3, 1, 10, 0),
        )

    return _make


@pytest.fixture
def signal_factory():
    def _make(symbol="TEST", side=Side.BUY, ts: datetime | None = None,
              ltp=100.0, **features):
        return Signal(
            symbol=symbol,
            side=side,
            timestamp=ts or datetime(2024, 3, 1, 10, 0),
            ltp=ltp,
            features=features or {"smma_spread_pct": 0.5},
        )

    return _make


@pytest.fixture
def tmp_store(tmp_path):
    from nse_screener.store import Store

    return Store(path=tmp_path / "test.db")
