"""
Trade simulation.

Rules, exactly as specified:
    enter at the crossover LTP, exit at the reverse crossover.

The system is therefore always-in-the-market and stop-and-reverse: every exit is
someone else's entry. Two guards on top of the pure rule:

  * `max_hold_minutes` force-closes a position that never sees a reverse signal
    (a stock that trends all session would otherwise hold one trade forever and
    contribute a single, useless training row).
  * Costs and slippage are charged on BOTH legs. Labelling trades profitable on
    a gross basis would train the model to accept trades that lose money after
    friction, which is the most expensive mistake available here.

Excursion metrics (MAE/MFE) are computed from the bars spanned by the trade and
are direction-aware, so a short is measured correctly rather than sign-flipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from .config import IndicatorConfig, TradingConfig, settings
from .features import compute_bar_features
from .models import Side, Signal, Trade
from .utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BacktestStats:
    trades: int
    wins: int
    losses: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    avg_duration_min: float
    profit_factor: float
    max_drawdown_pct: float

    def summary(self) -> str:
        return (
            f"{self.trades} trades | win rate {self.win_rate:.1f}% | "
            f"avg {self.avg_pnl_pct:+.3f}% | total {self.total_pnl_pct:+.2f}% | "
            f"PF {self.profit_factor:.2f} | maxDD {self.max_drawdown_pct:.2f}%"
        )

    @classmethod
    def empty(cls) -> "BacktestStats":
        return cls(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TradeSimulator:
    """Turns a crossover sequence into a set of closed, labelled round trips."""

    def __init__(self, cfg: TradingConfig | None = None) -> None:
        self.cfg = cfg or settings.trading

    # ------------------------------------------------------------------
    @property
    def round_trip_cost_pct(self) -> float:
        """Total friction charged per round trip, in percent."""
        return (self.cfg.cost_bps + self.cfg.slippage_bps) * 2.0 / 100.0

    def _pnl(self, side: Side, entry: float, exit_: float) -> tuple[float, float]:
        """Return (gross_pct, net_pct), direction-aware."""
        if entry <= 0:
            return 0.0, 0.0
        gross = (exit_ - entry) / entry * 100.0 * side.sign
        return gross, gross - self.round_trip_cost_pct

    @staticmethod
    def _pnl_points(side: Side, entry: float, exit_: float) -> float:
        """
        P&L in RUPEE POINTS, which is how the brief states it:
            Buy  -> exit LTP - entry LTP
            Sell -> entry LTP - exit LTP   (a short profits as price falls)
        Percentage is what the model and the stats use, because points are not
        comparable across a Rs 35 stock and a Rs 480 one. Both are recorded.
        """
        return (exit_ - entry) * side.sign

    # ------------------------------------------------------------------
    def _excursions(
        self, bars: pd.DataFrame, side: Side, entry_price: float,
        entry_time: datetime, exit_time: datetime,
    ) -> tuple[float, float]:
        """
        Maximum adverse and favourable excursion over the life of the trade.

        MAE is the worst unrealised loss, MFE the best unrealised gain - both as
        positive-magnitude percentages. They answer "would a stop have been hit"
        and "did we give back a winner", which raw P&L cannot.
        """
        if bars is None or bars.empty or entry_price <= 0:
            return 0.0, 0.0

        window = bars[(bars["timestamp"] >= entry_time) & (bars["timestamp"] <= exit_time)]
        if window.empty:
            return 0.0, 0.0

        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)

        if side is Side.BUY:
            best, worst = highs.max(), lows.min()
        else:
            # A short profits as price falls, so the roles swap.
            best, worst = lows.min(), highs.max()

        mfe = abs((best - entry_price) / entry_price * 100.0) if _favourable(side, best, entry_price) else 0.0
        mae = abs((worst - entry_price) / entry_price * 100.0) if not _favourable(side, worst, entry_price) else 0.0
        return float(mae), float(mfe)

    # ------------------------------------------------------------------
    def simulate(self, signals: list[Signal], bars: pd.DataFrame | None = None) -> list[Trade]:
        """
        Walk a chronological signal list and pair each entry with its reverse.

        Signals for a single symbol. Consecutive same-side signals are ignored -
        the position is already on and the rule says exit only on the reverse.
        """
        if not signals:
            return []

        ordered = sorted(signals, key=lambda s: s.timestamp)
        trades: list[Trade] = []
        open_trade: Trade | None = None

        for signal in ordered:
            if open_trade is None:
                if signal.side is Side.SELL and not self.cfg.allow_short:
                    continue
                open_trade = Trade(
                    symbol=signal.symbol,
                    side=signal.side,
                    entry_time=signal.timestamp,
                    entry_price=signal.ltp,
                    features=dict(signal.features),
                )
                continue

            if signal.side is open_trade.side:
                continue                      # already positioned this way

            self._close(open_trade, signal.timestamp, signal.ltp, "reverse_crossover", bars)
            trades.append(open_trade)

            if signal.side is Side.SELL and not self.cfg.allow_short:
                open_trade = None
            else:
                open_trade = Trade(
                    symbol=signal.symbol,
                    side=signal.side,
                    entry_time=signal.timestamp,
                    entry_price=signal.ltp,
                    features=dict(signal.features),
                )

        # Force-close a position left open at the end of the data.
        if open_trade is not None and bars is not None and not bars.empty:
            last = bars.iloc[-1]
            last_time = pd.Timestamp(last["timestamp"]).to_pydatetime()
            held = (last_time - open_trade.entry_time).total_seconds() / 60.0
            if held >= self.cfg.max_hold_minutes:
                self._close(open_trade, last_time, float(last["close"]), "max_hold", bars)
                trades.append(open_trade)

        return trades

    def _close(
        self, trade: Trade, exit_time: datetime, exit_price: float,
        reason: str, bars: pd.DataFrame | None,
    ) -> None:
        trade.exit_time = exit_time
        trade.exit_price = float(exit_price)
        trade.exit_reason = reason
        trade.gross_pnl_pct, trade.net_pnl_pct = self._pnl(
            trade.side, trade.entry_price, trade.exit_price
        )
        trade.pnl_points = self._pnl_points(trade.side, trade.entry_price, trade.exit_price)
        trade.duration_minutes = max(
            0.0, (exit_time - trade.entry_time).total_seconds() / 60.0
        )
        if bars is not None:
            trade.mae_pct, trade.mfe_pct = self._excursions(
                bars, trade.side, trade.entry_price, trade.entry_time, exit_time
            )

    # ------------------------------------------------------------------
    def run_symbol(
        self, symbol: str, bars: pd.DataFrame, cfg: IndicatorConfig | None = None
    ) -> list[Trade]:
        """Extract every historical crossover for one symbol and simulate them."""
        from .signals import extract_historical_signals

        cfg = cfg or settings.indicators
        featured = compute_bar_features(bars, cfg)
        signals = extract_historical_signals(bars, symbol, cfg)
        if not signals:
            return []
        return self.simulate(signals, featured)


def _favourable(side: Side, price: float, entry: float) -> bool:
    return (price > entry) if side is Side.BUY else (price < entry)


# --------------------------------------------------------------------------
def compute_stats(trades: list[Trade]) -> BacktestStats:
    """Headline performance of a closed trade list."""
    closed = [t for t in trades if not t.is_open]
    if not closed:
        return BacktestStats.empty()

    pnl = np.array([t.net_pnl_pct for t in closed], dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    gross_win = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(abs(losses.sum())) if losses.size else 0.0
    # Profit factor is undefined with no losses; inf is honest but unplottable,
    # so report the gross win instead of a sentinel.
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win else 0.0)

    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity)) if equity.size else 0.0

    return BacktestStats(
        trades=len(closed),
        wins=int(wins.size),
        losses=int(losses.size),
        win_rate=float(wins.size / len(closed) * 100.0),
        avg_pnl_pct=float(pnl.mean()),
        total_pnl_pct=float(pnl.sum()),
        avg_duration_min=float(np.mean([t.duration_minutes for t in closed])),
        profit_factor=profit_factor,
        max_drawdown_pct=max_dd,
    )


def trades_to_frame(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.to_dict() for t in trades])
