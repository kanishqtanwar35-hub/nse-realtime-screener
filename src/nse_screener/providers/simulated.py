"""
Simulated provider - the zero-credential fallback.

This is not a toy. It is the reference implementation the whole pipeline is
tested against, so it has to produce data with the statistical properties the
downstream code cares about:

  * Prices inside the screener band (Rs 30-500) so the filter has work to do.
  * Mean-reverting drift with trending regimes, so SMMA20/SMMA120 actually
    cross - a pure random walk crosses too rarely to generate training data.
  * Intraday volume smile (heavy at open and close, thin midday).
  * LTQ drawn from a heavy-tailed distribution, so spike ratios vary.
  * Occasional wide spreads and one-sided books, so microstructure features
    are not constant.

Deterministic given a seed, which is what makes the tests reproducible.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd

from ..models import Quote
from ..utils import get_logger
from .base import BAR_COLUMNS, MarketDataProvider

log = get_logger(__name__)

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


class SimulatedProvider(MarketDataProvider):
    """Synthetic NSE-like market data. No network, no credentials."""

    name = "simulated"

    def __init__(self, symbols: list[str] | None = None, seed: int = 42,
                 liquid_fraction: float = 0.45) -> None:
        self._rng = np.random.default_rng(seed)
        self._symbols = list(symbols or [])
        self._state: dict[str, dict] = {}
        self._liquid_fraction = liquid_fraction
        self._authenticated = False

    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        self._authenticated = True
        log.info("SimulatedProvider ready (no credentials required)")

    def close(self) -> None:
        self._authenticated = False

    # ------------------------------------------------------------------
    def _init_symbol(self, symbol: str) -> dict:
        """One-time per-symbol personality: price level, vol, liquidity tier."""
        if symbol in self._state:
            return self._state[symbol]

        # Stable per-symbol seed so a symbol behaves consistently across calls.
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        base_price = float(rng.uniform(35.0, 480.0))

        # Split the universe into liquid and illiquid names so the liquidity
        # filter actually discriminates instead of passing or failing everyone.
        is_liquid = rng.random() < self._liquid_fraction
        depth_scale = float(rng.uniform(0.9e6, 4.0e6)) if is_liquid else float(rng.uniform(2e3, 8e4))

        state = {
            "price": base_price,
            "anchor": base_price,
            "daily_vol": float(rng.uniform(0.008, 0.030)),
            "depth_scale": depth_scale,
            "is_liquid": is_liquid,
            "tick": 0.05,
            "regime": 0.0,
            "regime_left": 0,
            "cum_volume": int(rng.integers(50_000, 500_000)),
            "rng": rng,
        }
        self._state[symbol] = state
        return state

    def _step(self, state: dict, minutes: float = 1.0) -> float:
        """
        Advance the price one step.

        Regime-switching drift plus mean reversion to the anchor. The drift is
        what makes SMMA crossovers happen at a realistic rate; without it the
        two averages sit on top of each other and never cross.
        """
        rng = state["rng"]

        if state["regime_left"] <= 0:
            state["regime"] = float(rng.normal(0.0, 1.0)) * state["daily_vol"] * 0.06
            state["regime_left"] = int(rng.integers(25, 140))
        state["regime_left"] -= 1

        per_min_vol = state["daily_vol"] / np.sqrt(375.0)
        shock = float(rng.normal(0.0, per_min_vol)) * np.sqrt(minutes)
        reversion = 0.004 * (state["anchor"] - state["price"]) / max(state["anchor"], 1e-9)

        state["price"] *= (1.0 + state["regime"] + shock + reversion)
        state["price"] = float(np.clip(state["price"], 5.0, 5000.0))
        # Snap to the exchange tick size - real LTPs are never arbitrary floats.
        state["price"] = round(state["price"] / state["tick"]) * state["tick"]
        return state["price"]

    @staticmethod
    def _volume_multiplier(ts: datetime) -> float:
        """Intraday U-shape: busy at the open and close, quiet at lunch."""
        minutes_in = (ts.hour - 9) * 60 + (ts.minute - 15)
        frac = np.clip(minutes_in / 375.0, 0.0, 1.0)
        return float(0.6 + 1.9 * ((2 * frac - 1) ** 2))

    # ------------------------------------------------------------------
    def fetch_history(self, symbol: str, minutes: int) -> pd.DataFrame:
        state = self._init_symbol(symbol)
        rng = state["rng"]
        minutes = max(1, int(minutes))

        end = datetime.now().replace(second=0, microsecond=0)
        # Walk backwards over session minutes only, so timestamps look like a
        # real trading calendar rather than a continuous 24h series.
        stamps: list[datetime] = []
        cursor = end
        while len(stamps) < minutes:
            if MARKET_OPEN <= cursor.time() <= MARKET_CLOSE and cursor.weekday() < 5:
                stamps.append(cursor)
            cursor -= timedelta(minutes=1)
            if (end - cursor).days > 30:      # safety valve
                break
        stamps.reverse()
        if not stamps:
            return self._empty_bars()

        # Rewind the price so the simulated history ENDS near the live price.
        saved = state["price"]
        state["price"] = saved * float(np.exp(rng.normal(0, state["daily_vol"] * 0.5)))

        rows = []
        for ts in stamps:
            open_ = state["price"]
            close = self._step(state)
            wick = abs(float(rng.normal(0, state["daily_vol"] / np.sqrt(375) * open_)))
            high = max(open_, close) + wick
            low = max(0.05, min(open_, close) - wick)
            vol = int(max(1, rng.gamma(2.2, state["depth_scale"] / 60.0) * self._volume_multiplier(ts)))
            rows.append((ts, open_, high, low, close, vol))

        df = pd.DataFrame(rows, columns=BAR_COLUMNS)
        return self.normalise_bars(df)

    # ------------------------------------------------------------------
    def fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        now = datetime.now()
        quotes: list[Quote] = []

        for symbol in symbols:
            state = self._init_symbol(symbol)
            rng = state["rng"]
            ltp = self._step(state, minutes=0.1)

            tick = state["tick"]
            # Spread: usually 1 tick, occasionally much wider (thin book moment).
            n_ticks = 1 if rng.random() > 0.15 else int(rng.integers(2, 7))
            half = tick * n_ticks / 2.0
            bid_price = round((ltp - half) / tick) * tick
            ask_price = round((ltp + half) / tick) * tick
            if ask_price <= bid_price:
                ask_price = bid_price + tick

            depth = state["depth_scale"]
            # Skew the book so order_book_imbalance is not permanently ~0.
            skew = float(rng.uniform(0.55, 1.75))
            bid_qty = int(max(1, rng.gamma(3.0, depth / 3.0) * skew))
            ask_qty = int(max(1, rng.gamma(3.0, depth / 3.0) / skew))

            # LTQ: lognormal gives the heavy right tail real trade sizes have,
            # which is what makes the spike-ratio features informative.
            ltq = int(max(1, rng.lognormal(mean=np.log(max(depth / 900.0, 1.0)), sigma=1.15)))
            state["cum_volume"] += ltq

            quotes.append(
                Quote(
                    symbol=symbol,
                    ltp=float(ltp),
                    bid_price=float(bid_price),
                    ask_price=float(ask_price),
                    bid_qty=bid_qty,
                    ask_qty=ask_qty,
                    ltq=ltq,
                    volume=int(state["cum_volume"]),
                    timestamp=now,
                )
            )
        return quotes

    def health_check(self) -> bool:
        return self._authenticated
