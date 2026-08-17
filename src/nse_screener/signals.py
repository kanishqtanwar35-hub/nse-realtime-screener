"""
Signal engine: SMMA(20) / SMMA(120) crossovers.

    BUY  when SMMA20 crosses ABOVE SMMA120
    SELL when SMMA20 crosses BELOW SMMA120

Two properties the implementation guarantees:

  * NO REPAINTING. Crossovers are evaluated on CLOSED bars only. An indicator
    computed on the in-progress minute changes as ticks arrive, so a signal
    taken from it can appear and disappear within the same minute.
  * NO DUPLICATES. Per-symbol state remembers the last emitted crossover, so a
    single event fires exactly once even though the same bar is re-evaluated on
    every poll.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .config import IndicatorConfig, settings
from .features import TickWindow, build_feature_snapshot, compute_bar_features
from .models import Quote, Side, Signal
from .utils import get_logger

log = get_logger(__name__)


@dataclass
class SymbolState:
    """Per-symbol memory that makes signal emission idempotent."""

    symbol: str
    last_signal_time: datetime | None = None
    last_side: Side | None = None
    ticks: TickWindow = field(default_factory=lambda: TickWindow(symbol=""))

    def __post_init__(self) -> None:
        if not self.ticks.symbol:
            self.ticks = TickWindow(symbol=self.symbol)

    def already_emitted(self, when: datetime, side: Side) -> bool:
        return self.last_signal_time == when and self.last_side == side

    def record(self, when: datetime, side: Side) -> None:
        self.last_signal_time = when
        self.last_side = side


class SignalEngine:
    """Detects crossovers and attaches a full feature snapshot to each one."""

    def __init__(self, cfg: IndicatorConfig | None = None, *, ignore_filled_bars: bool = True):
        self.cfg = cfg or settings.indicators
        self.ignore_filled_bars = ignore_filled_bars
        self.states: dict[str, SymbolState] = {}
        self.history: list[Signal] = []

    # ------------------------------------------------------------------
    def state_for(self, symbol: str) -> SymbolState:
        if symbol not in self.states:
            self.states[symbol] = SymbolState(symbol=symbol)
        return self.states[symbol]

    def observe(self, quote: Quote) -> None:
        """Feed a live quote into the tick-clock window for its symbol."""
        self.state_for(quote.symbol).ticks.add(quote)

    # ------------------------------------------------------------------
    def evaluate(self, symbol: str, bars: pd.DataFrame, quote: Quote) -> Signal | None:
        """
        Check the most recent CLOSED bar for a crossover.

        Returns a Signal on a fresh crossover, else None.
        """
        min_bars = self.cfg.smma_slow + 2
        if bars is None or len(bars) < min_bars:
            return None

        try:
            featured = compute_bar_features(bars, self.cfg)
        except Exception as exc:  # noqa: BLE001
            log.error("feature computation failed for %s: %s", symbol, exc)
            return None

        last = featured.iloc[-1]
        marker = int(last.get("crossover", 0) or 0)
        if marker == 0:
            return None

        # A crossover detected on a synthetic (forward-filled) bar is an artefact
        # of gap filling, not a market event.
        if self.ignore_filled_bars and bool(last.get("is_filled", False)):
            log.debug("%s: crossover on a forward-filled bar - ignored", symbol)
            return None

        side = Side.BUY if marker > 0 else Side.SELL
        when = pd.Timestamp(last["timestamp"]).to_pydatetime()

        state = self.state_for(symbol)
        if state.already_emitted(when, side):
            return None

        features = build_feature_snapshot(featured, quote, state.ticks, self.cfg)
        signal = Signal(
            symbol=symbol,
            side=side,
            timestamp=when,
            ltp=float(quote.ltp),
            features=features,
            reason=(
                f"SMMA{self.cfg.smma_fast} crossed "
                f"{'above' if side is Side.BUY else 'below'} SMMA{self.cfg.smma_slow} "
                f"(spread {features.get('smma_spread_pct', 0.0):+.3f}%)"
            ),
        )

        state.record(when, side)
        self.history.append(signal)
        log.info(
            "SIGNAL %s %s @ %.2f | %s", side.value, symbol, quote.ltp, signal.reason
        )
        return signal

    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        bars_by_symbol: dict[str, pd.DataFrame],
        quotes_by_symbol: dict[str, Quote],
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, bars in bars_by_symbol.items():
            quote = quotes_by_symbol.get(symbol)
            if quote is None:
                continue
            signal = self.evaluate(symbol, bars, quote)
            if signal is not None:
                signals.append(signal)
        return signals

    def recent(self, limit: int = 50) -> list[Signal]:
        return self.history[-limit:]

    def to_frame(self, limit: int | None = None) -> pd.DataFrame:
        rows = self.history[-limit:] if limit else self.history
        return pd.DataFrame([s.to_dict() for s in rows])


# --------------------------------------------------------------------------
def extract_historical_signals(
    bars: pd.DataFrame,
    symbol: str,
    cfg: IndicatorConfig | None = None,
) -> list[Signal]:
    """
    Replay a bar frame and return every crossover in it.

    This is how the training set is manufactured: the live engine only ever sees
    the newest bar, so historical crossovers must be harvested in bulk.

    Note the feature snapshot here is BAR-ONLY - LTQ and order book features are
    zero, because tick data is not retained historically. That is a real and
    acknowledged train/serve skew: the model sees richer microstructure live
    than it was trained on. Options are to train only on bar features, or to
    persist live snapshots and retrain on those. `MODEL_NOTES` in the README
    covers this; the shipped default trains on what history can supply.
    """
    cfg = cfg or settings.indicators
    if bars is None or len(bars) < cfg.smma_slow + 2:
        return []

    featured = compute_bar_features(bars, cfg)
    events = featured[featured["crossover"] != 0]
    signals: list[Signal] = []

    for _, row in events.iterrows():
        if bool(row.get("is_filled", False)):
            continue
        side = Side.BUY if row["crossover"] > 0 else Side.SELL
        features = {
            col: float(row[col]) if col in row and pd.notna(row[col]) else 0.0
            for col in (
                "smma_spread_pct", "smma_spread_roc", "price_dist_fast_pct",
                "price_dist_slow_pct", "momentum_pct", "volatility_pct",
                "etq_5m", "etq_20m", "etq_60m", "etq_ratio_5_60",
            )
        }
        signals.append(
            Signal(
                symbol=symbol,
                side=side,
                timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                ltp=float(row["close"]),
                features=features,
                reason=f"historical SMMA{cfg.smma_fast}/{cfg.smma_slow} crossover",
            )
        )
    return signals
