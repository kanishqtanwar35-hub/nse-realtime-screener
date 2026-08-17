"""
Feature engineering.

Two different clocks feed the feature vector, and keeping them straight is the
whole design:

  * BAR features  - computed from the 1-minute OHLCV series (indicators, ETQ
                    totals, momentum, volatility, distance from SMMA).
  * TICK features - computed from the raw quote stream held in memory (LTQ
                    rolling averages and spike ratios, bid-ask spread, order
                    book imbalance). These have no meaningful bar equivalent
                    because LTQ is a per-trade quantity, not a per-minute one.

`FEATURE_COLUMNS` is the contract between this module, the model, and the
dashboard. Anything not in that list is diagnostic only and never reaches the
classifier.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Iterable

import numpy as np
import pandas as pd

from .config import IndicatorConfig
from .indicators import add_indicators, momentum, realised_volatility
from .models import Quote
from .utils import get_logger, safe_div

log = get_logger(__name__)

# The exact feature vector handed to the model, in a fixed order.
FEATURE_COLUMNS: list[str] = [
    # trend / SMMA
    "smma_spread_pct",
    "smma_spread_roc",
    "price_dist_fast_pct",
    "price_dist_slow_pct",
    # momentum & volatility
    "momentum_pct",
    "volatility_pct",
    # LTQ dynamics (tick clock)
    "ltq_avg_2m",
    "ltq_avg_5m",
    "ltq_avg_20m",
    "ltq_ratio_2_5",      # the comparison the brief names explicitly
    "ltq_spike_2_20",
    "ltq_spike_5_20",
    # traded quantity (bar clock)
    "etq_5m",
    "etq_20m",
    "etq_60m",
    "etq_ratio_5_60",
    # microstructure
    "bid_ask_spread_pct",
    "order_book_imbalance",
    "depth_ratio",
]

# Human-readable names used in model explanations.
FEATURE_LABELS: dict[str, str] = {
    "smma_spread_pct": "SMMA20-SMMA120 spread",
    "smma_spread_roc": "SMMA spread rate of change",
    "price_dist_fast_pct": "price distance from SMMA20",
    "price_dist_slow_pct": "price distance from SMMA120",
    "momentum_pct": "short-term momentum",
    "volatility_pct": "realised volatility",
    "ltq_avg_2m": "2-minute average trade size",
    "ltq_avg_5m": "5-minute average trade size",
    "ltq_avg_20m": "20-minute average trade size",
    "ltq_ratio_2_5": "trade-size ratio (2m vs 5m)",
    "ltq_spike_2_20": "trade-size spike (2m vs 20m)",
    "ltq_spike_5_20": "trade-size spike (5m vs 20m)",
    "etq_5m": "5-minute traded quantity",
    "etq_20m": "20-minute traded quantity",
    "etq_60m": "60-minute traded quantity",
    "etq_ratio_5_60": "volume concentration (5m vs 60m)",
    "bid_ask_spread_pct": "bid-ask spread",
    "order_book_imbalance": "order book imbalance",
    "depth_ratio": "bid/ask depth ratio",
}


# --------------------------------------------------------------------------
# Tick-clock state
# --------------------------------------------------------------------------
@dataclass
class TickWindow:
    """
    Rolling in-memory buffer of recent quotes for ONE symbol.

    Bounded by time, not by count: a thin stock produces far fewer ticks per
    minute than a liquid one, so a fixed-length deque would silently mean
    different lookbacks for different symbols.
    """

    symbol: str
    max_minutes: int = 60
    _ticks: Deque[tuple[datetime, int, float]] = field(default_factory=deque)  # (ts, ltq, ltp)

    def add(self, quote: Quote) -> None:
        if quote.ltq <= 0:
            # No trade happened on this poll - the quote is a book update only.
            # Recording a zero would drag every LTQ average toward zero.
            return
        self._ticks.append((quote.timestamp, int(quote.ltq), float(quote.ltp)))
        self._evict(quote.timestamp)

    def _evict(self, now: datetime) -> None:
        cutoff = now - timedelta(minutes=self.max_minutes)
        while self._ticks and self._ticks[0][0] < cutoff:
            self._ticks.popleft()

    def ltq_average(self, minutes: int, now: datetime | None = None) -> float:
        """Mean last-traded-quantity over the trailing `minutes`."""
        if not self._ticks:
            return 0.0
        now = now or self._ticks[-1][0]
        cutoff = now - timedelta(minutes=minutes)
        vals = [q for ts, q, _ in self._ticks if ts >= cutoff]
        return float(np.mean(vals)) if vals else 0.0

    def ltq_sum(self, minutes: int, now: datetime | None = None) -> float:
        if not self._ticks:
            return 0.0
        now = now or self._ticks[-1][0]
        cutoff = now - timedelta(minutes=minutes)
        return float(sum(q for ts, q, _ in self._ticks if ts >= cutoff))

    def __len__(self) -> int:
        return len(self._ticks)


# --------------------------------------------------------------------------
# Bar-clock features
# --------------------------------------------------------------------------
def compute_bar_features(bars: pd.DataFrame, cfg: IndicatorConfig) -> pd.DataFrame:
    """
    Indicators + rolling aggregates over the 1-minute bar frame.

    Expects columns: timestamp (index or column), open/high/low/close/volume.
    Returns a copy with every bar-clock feature attached.
    """
    if bars.empty:
        return bars.copy()

    df = add_indicators(bars, cfg.smma_fast, cfg.smma_slow, price_col="close")

    close = df["close"]
    df["momentum_pct"] = momentum(close, cfg.momentum_window)
    df["volatility_pct"] = realised_volatility(close, cfg.volatility_window)

    # Distance of price from each SMMA, in percent. Sign carries information:
    # positive means price is extended above the average.
    with np.errstate(divide="ignore", invalid="ignore"):
        df["price_dist_fast_pct"] = (close - df["smma_fast"]) / df["smma_fast"].abs() * 100.0
        df["price_dist_slow_pct"] = (close - df["smma_slow"]) / df["smma_slow"].abs() * 100.0
    df[["price_dist_fast_pct", "price_dist_slow_pct"]] = df[
        ["price_dist_fast_pct", "price_dist_slow_pct"]
    ].replace([np.inf, -np.inf], np.nan)

    # Average LTP over trailing windows. On 1-minute bars the close IS the last
    # traded price of that minute, so a rolling mean of close is the average LTP.
    for window in (20, 60):
        df[f"avg_ltp_{window}m"] = close.rolling(window, min_periods=1).mean()

    # ETQ = exchange traded quantity, summed over each window of bars.
    volume = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index)
    for window in cfg.etq_windows:
        df[f"etq_{window}m"] = volume.rolling(window, min_periods=1).sum()

    short, long_ = cfg.etq_windows[0], cfg.etq_windows[-1]
    # Volume concentration: is recent activity a burst or just the usual rate?
    # Normalised by the window ratio so 1.0 means "exactly the average pace".
    expected = df[f"etq_{long_}m"] * (short / long_)
    df["etq_ratio_5_60"] = np.where(expected > 0, df[f"etq_{short}m"] / expected, 0.0)

    return df


# --------------------------------------------------------------------------
# Combined snapshot
# --------------------------------------------------------------------------
def build_feature_snapshot(
    bars: pd.DataFrame,
    quote: Quote,
    ticks: TickWindow,
    cfg: IndicatorConfig,
) -> dict[str, float]:
    """
    Collapse the latest bar row plus live tick state into one flat feature dict.

    Every key in FEATURE_COLUMNS is guaranteed present and finite - the model
    must never receive a NaN it was not trained on. Missing values become 0.0,
    which for these features means "neutral / no information".
    """
    snap: dict[str, float] = {}

    if not bars.empty:
        last = bars.iloc[-1]
        for col in (
            "smma_spread_pct", "smma_spread_roc", "price_dist_fast_pct",
            "price_dist_slow_pct", "momentum_pct", "volatility_pct",
            "etq_ratio_5_60",
        ):
            snap[col] = _finite(last.get(col))
        for window in cfg.etq_windows:
            snap[f"etq_{window}m"] = _finite(last.get(f"etq_{window}m"))

    # --- tick clock -------------------------------------------------------
    now = quote.timestamp
    for window in cfg.ltq_windows:
        snap[f"ltq_avg_{window}m"] = ticks.ltq_average(window, now)

    # The brief singles this out: "compare the average LTQ over the last 2 minutes
    # with the average LTQ over the last 5 minutes". Above 1.0 means size is
    # stepping in right now relative to the recent norm.
    five = snap.get("ltq_avg_5m", 0.0)
    snap["ltq_ratio_2_5"] = safe_div(snap.get("ltq_avg_2m", 0.0), five, 1.0) if five else 1.0

    base = snap.get(f"ltq_avg_{cfg.ltq_windows[-1]}m", 0.0)
    # Spike ratio > 1 means recent trades are larger than the 20-minute norm,
    # i.e. size is stepping in. Guarded: with no baseline the ratio is 1.0
    # (neutral), never a divide-by-zero or a misleading infinity.
    snap["ltq_spike_2_20"] = safe_div(snap.get("ltq_avg_2m", 0.0), base, 1.0) if base else 1.0
    snap["ltq_spike_5_20"] = safe_div(snap.get("ltq_avg_5m", 0.0), base, 1.0) if base else 1.0

    # --- microstructure ---------------------------------------------------
    snap["bid_ask_spread_pct"] = _finite(quote.spread_pct)
    snap["order_book_imbalance"] = _finite(quote.imbalance)
    snap["depth_ratio"] = safe_div(float(quote.bid_qty), float(quote.ask_qty), 1.0)

    # Guarantee the full contract, in order.
    return {col: _finite(snap.get(col, 0.0)) for col in FEATURE_COLUMNS}


def _finite(value) -> float:
    """Coerce anything to a finite float; NaN/inf/None all become 0.0."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(f):
        return 0.0
    return f


def features_to_frame(rows: Iterable[dict[str, float]]) -> pd.DataFrame:
    """Stack feature dicts into a model-ready frame with the canonical columns."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], 0.0).fillna(0.0)


# --------------------------------------------------------------------------
# Dashboard row
# --------------------------------------------------------------------------
# The columns the live dashboard shows, one row per stock. This is the
# assignment's display contract (requirements 3-6) and is deliberately SEPARATE
# from FEATURE_COLUMNS: several of these - SMMA levels, average LTP - are raw
# price levels. They are exactly what a trader needs to see and exactly what a
# classifier must not be fed, because a non-stationary level lets the model
# learn "this symbol" rather than "this setup".
DISPLAY_COLUMNS: list[str] = [
    "symbol",
    "ltp",
    "smma_20", "smma_120", "smma_signal",
    "etq_5m", "etq_20m", "etq_60m",
    "avg_ltp_20m", "avg_ltp_60m",
    "bid_price", "bid_qty", "ask_price", "ask_qty",
    "ltq", "ltq_avg_2m", "ltq_avg_5m", "ltq_ratio_2_5",
    "spread_pct", "imbalance",
]


def build_display_row(
    symbol: str,
    bars: pd.DataFrame,
    quote: Quote,
    ticks: TickWindow,
    cfg: IndicatorConfig,
) -> dict[str, float | str]:
    """
    One dashboard row for one stock: live depth plus the derived indicators.

    `bars` must already carry bar features (i.e. come from compute_bar_features).
    Missing values render as 0.0 rather than NaN so the table stays sortable.
    """
    row: dict[str, float | str] = {"symbol": symbol}
    last = bars.iloc[-1] if bars is not None and not bars.empty else None

    def bar(col: str) -> float:
        return _finite(last.get(col)) if last is not None else 0.0

    row["smma_20"] = bar("smma_fast")
    row["smma_120"] = bar("smma_slow")
    # Which side the fast average is on right now - the standing state, as
    # opposed to the crossover EVENT that fires only on the bar it happens.
    spread = row["smma_20"] - row["smma_120"]
    row["smma_signal"] = "BULLISH" if spread > 0 else ("BEARISH" if spread < 0 else "FLAT")

    for window in cfg.etq_windows:
        row[f"etq_{window}m"] = bar(f"etq_{window}m")
    for window in (20, 60):
        row[f"avg_ltp_{window}m"] = bar(f"avg_ltp_{window}m")

    row["ltp"] = float(quote.ltp)
    row["bid_price"] = float(quote.bid_price)
    row["ask_price"] = float(quote.ask_price)
    row["bid_qty"] = int(quote.bid_qty)
    row["ask_qty"] = int(quote.ask_qty)
    row["ltq"] = int(quote.ltq)
    row["spread_pct"] = _finite(quote.spread_pct)
    row["imbalance"] = _finite(quote.imbalance)

    now = quote.timestamp
    row["ltq_avg_2m"] = ticks.ltq_average(2, now)
    row["ltq_avg_5m"] = ticks.ltq_average(5, now)
    row["ltq_ratio_2_5"] = (
        safe_div(row["ltq_avg_2m"], row["ltq_avg_5m"], 1.0) if row["ltq_avg_5m"] else 1.0
    )
    return row
