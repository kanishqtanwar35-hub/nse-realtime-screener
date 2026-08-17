"""
Ingestion: fetching, cleaning, and turning a quote stream into 1-minute bars.

Missing-data policy, stated once and applied everywhere:

  * A quote that fails structural validation is DROPPED, never repaired. A
    crossed book or a zero LTP is a broken tick, and guessing what it "meant"
    puts fabricated data into the signal path.
  * A gap in the bar series is forward-filled for at most `max_forward_fill`
    bars, and the filled bars carry volume = 0. A one-minute hole in a thin
    stock is normal; a twenty-minute hole means the feed is broken and should
    surface as such rather than being smoothed over.
  * Forward-filled bars are flagged (`is_filled`) so downstream code can choose
    to ignore signals generated on synthetic bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import DataConfig, settings
from .models import Quote
from .providers.base import BAR_COLUMNS, MarketDataProvider
from .utils import Timer, chunked, get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
def clean_bars(bars: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """
    Regularise a bar frame onto a 1-minute grid and fill small gaps.

    Gap filling is SESSION-SCOPED, and that is the whole subtlety. Reindexing
    naively across the full timestamp range would manufacture ~1,065 phantom
    minutes for every overnight break and ~3,000 for every weekend - an enormous
    sparse frame, and worse, it would let a forward fill carry Friday's close
    into Monday's open. Sessions are therefore regularised independently and
    concatenated, so a gap can only ever be filled within one trading day.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=BAR_COLUMNS + ["is_filled"])

    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last")
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS + ["is_filled"])

    freq = f"{max(1, cfg.interval_minutes)}min"
    sessions: list[pd.DataFrame] = []
    total_filled = 0
    total_unfilled = 0

    for _, day in df.groupby(df["timestamp"].dt.date, sort=True):
        day = day.set_index("timestamp")
        grid = pd.date_range(day.index.min(), day.index.max(), freq=freq)
        block = day.reindex(grid)

        missing = block["close"].isna()
        if missing.any():
            # Only fill runs short enough to be plausible micro-gaps; a long
            # hole means the feed was down and must stay visible as a hole.
            run_id = (~missing).cumsum()
            run_len = missing.groupby(run_id).transform("sum")
            fillable = missing & (run_len <= cfg.max_forward_fill)

            block["close"] = block["close"].ffill(limit=cfg.max_forward_fill)
            for col in ("open", "high", "low"):
                block[col] = block[col].fillna(block["close"])
            # A synthetic bar had no trades, so its volume is genuinely zero.
            block["volume"] = block["volume"].fillna(0.0)

            block["is_filled"] = fillable.fillna(False).astype(bool)
            total_filled += int(fillable.sum())
            total_unfilled += int((missing & ~fillable).sum())
        else:
            block["is_filled"] = False

        sessions.append(block.dropna(subset=["close"]))

    if not sessions:
        return pd.DataFrame(columns=BAR_COLUMNS + ["is_filled"])

    out = pd.concat(sessions).rename_axis("timestamp").reset_index()

    if total_filled:
        log.debug("clean_bars: forward-filled %d intra-session minute(s)", total_filled)
    if total_unfilled:
        log.info(
            "clean_bars: %d minute(s) left as gaps (run longer than max_forward_fill=%d)",
            total_unfilled, cfg.max_forward_fill,
        )
    return out[BAR_COLUMNS + ["is_filled"]]


# --------------------------------------------------------------------------
@dataclass
class BarAggregator:
    """
    Folds live quotes into the in-memory 1-minute bar series for one symbol.

    Seeded from the provider's history, then extended tick by tick. The current
    (incomplete) minute is kept separate from closed bars: signals are evaluated
    on CLOSED bars only, because an indicator computed on a half-formed candle
    repaints and would fire signals that vanish seconds later.
    """

    symbol: str
    bars: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=BAR_COLUMNS))
    max_bars: int = 2000

    _cur_minute: datetime | None = None
    _cur: dict | None = None
    _last_cum_volume: int | None = None

    def seed(self, history: pd.DataFrame) -> None:
        self.bars = history.copy() if history is not None else pd.DataFrame(columns=BAR_COLUMNS)
        if not self.bars.empty:
            self.bars["timestamp"] = pd.to_datetime(self.bars["timestamp"])
        log.debug("%s seeded with %d bars", self.symbol, len(self.bars))

    def update(self, quote: Quote) -> bool:
        """
        Add one quote. Returns True if a bar just CLOSED (i.e. new minute began).

        Volume is derived from the delta in the broker's cumulative day volume
        where available, falling back to LTQ. Cumulative is preferable because
        polling misses ticks between calls, and summing LTQ would undercount.
        """
        minute = quote.timestamp.replace(second=0, microsecond=0)
        closed = False

        if self._cur_minute is None:
            self._cur_minute = minute
            self._cur = self._new_bar(quote)
        elif minute > self._cur_minute:
            self._flush()
            closed = True
            self._cur_minute = minute
            self._cur = self._new_bar(quote)
        else:
            bar = self._cur
            assert bar is not None
            bar["high"] = max(bar["high"], quote.ltp)
            bar["low"] = min(bar["low"], quote.ltp)
            bar["close"] = quote.ltp
            bar["volume"] += self._volume_delta(quote)

        return closed

    def _new_bar(self, quote: Quote) -> dict:
        return {
            "timestamp": quote.timestamp.replace(second=0, microsecond=0),
            "open": quote.ltp,
            "high": quote.ltp,
            "low": quote.ltp,
            "close": quote.ltp,
            "volume": float(self._volume_delta(quote)),
        }

    def _volume_delta(self, quote: Quote) -> float:
        if quote.volume and quote.volume > 0:
            if self._last_cum_volume is None:
                self._last_cum_volume = quote.volume
                return float(quote.ltq or 0)
            delta = quote.volume - self._last_cum_volume
            self._last_cum_volume = quote.volume
            # A negative delta means the counter reset (new session) - trust LTQ.
            return float(delta if delta >= 0 else (quote.ltq or 0))
        return float(quote.ltq or 0)

    def _flush(self) -> None:
        if self._cur is None:
            return
        row = pd.DataFrame([self._cur])
        self.bars = (
            pd.concat([self.bars, row], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .tail(self.max_bars)
            .reset_index(drop=True)
        )
        self._cur = None

    def snapshot(self, include_partial: bool = False) -> pd.DataFrame:
        """Closed bars, optionally with the in-progress minute appended."""
        if not include_partial or self._cur is None:
            return self.bars
        return pd.concat([self.bars, pd.DataFrame([self._cur])], ignore_index=True)

    def __len__(self) -> int:
        return len(self.bars)


# --------------------------------------------------------------------------
class IngestionPipeline:
    """Batched quote fetching plus per-symbol history seeding."""

    def __init__(self, provider: MarketDataProvider, cfg: DataConfig | None = None) -> None:
        self.provider = provider
        self.cfg = cfg or settings.data
        self.aggregators: dict[str, BarAggregator] = {}
        self._consecutive_failures = 0

    # ------------------------------------------------------------------
    def seed_history(self, symbols: list[str]) -> dict[str, BarAggregator]:
        """
        Load history for each symbol. Symbols with too few bars are excluded -
        SMMA(120) on 40 bars is not a weak signal, it is no signal.
        """
        ok, skipped = 0, 0
        with Timer("seed_history", log):
            for symbol in symbols:
                try:
                    raw = self.provider.fetch_history(symbol, self.cfg.history_minutes)
                    bars = clean_bars(raw, self.cfg)
                except Exception as exc:  # noqa: BLE001
                    log.error("history failed for %s: %s", symbol, exc)
                    skipped += 1
                    continue

                if len(bars) < self.cfg.min_bars_required:
                    log.warning(
                        "%s: only %d bars (need %d) - excluded from this session",
                        symbol, len(bars), self.cfg.min_bars_required,
                    )
                    skipped += 1
                    continue

                agg = BarAggregator(symbol=symbol)
                agg.seed(bars)
                self.aggregators[symbol] = agg
                ok += 1

        log.info("history seeded: %d symbols ready, %d skipped", ok, skipped)
        return self.aggregators

    # ------------------------------------------------------------------
    def fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        """
        Fetch quotes in provider-sized batches.

        A failing batch is logged and skipped rather than aborting the cycle;
        a partial universe is far more useful than no universe. Repeated total
        failure is escalated so it cannot be silently tolerated forever.
        """
        collected: list[Quote] = []
        failures = 0

        for batch in chunked(symbols, self.cfg.quote_batch_size):
            try:
                quotes = self.provider.fetch_quotes(batch)
                collected.extend(q for q in quotes if q.is_valid())
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.error("quote batch of %d failed: %s", len(batch), exc)

        if collected:
            self._consecutive_failures = 0
        elif symbols:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 5:
                log.critical(
                    "no quotes returned for %d consecutive cycles - the feed is likely down",
                    self._consecutive_failures,
                )

        if failures:
            log.warning("%d quote batch(es) failed this cycle", failures)
        return collected

    # ------------------------------------------------------------------
    def apply_quotes(self, quotes: list[Quote]) -> set[str]:
        """Push quotes into their aggregators; return symbols whose bar closed."""
        closed: set[str] = set()
        for quote in quotes:
            agg = self.aggregators.get(quote.symbol)
            if agg is None:
                continue
            try:
                if agg.update(quote):
                    closed.add(quote.symbol)
            except Exception as exc:  # noqa: BLE001
                log.error("aggregator update failed for %s: %s", quote.symbol, exc)
        return closed
