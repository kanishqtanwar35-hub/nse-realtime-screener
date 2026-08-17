"""
Provider interface.

Every data source - Fyers, Angel One, or the simulator - implements this and
nothing above it knows which one is live. The engine depends on the abstract
contract, so swapping brokers is a config change, not a code change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from ..models import Quote
from ..utils import get_logger

log = get_logger(__name__)

# The canonical bar frame every provider must return.
BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class MarketDataProvider(ABC):
    """
    Contract for a market data source.

    Implementations must be defensive: the engine calls these in a tight loop
    and expects them to raise TransientError (retryable) or PermanentError
    (fatal) rather than leaking library-specific exceptions.
    """

    name: str = "base"

    @abstractmethod
    def authenticate(self) -> None:
        """Establish a session. Must raise PermanentError on bad credentials."""

    @abstractmethod
    def fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        """
        Top-of-book snapshot for each symbol.

        Partial results are acceptable and expected - a symbol that fails should
        be omitted with a warning, not bring down the whole batch.
        """

    @abstractmethod
    def fetch_history(self, symbol: str, minutes: int) -> pd.DataFrame:
        """
        Intraday 1-minute OHLCV, oldest first, with columns == BAR_COLUMNS.

        Returns an EMPTY frame (not None, not an exception) when the symbol has
        no data - callers filter on emptiness.
        """

    # -- optional lifecycle ------------------------------------------------
    def close(self) -> None:
        """Release sockets/sessions. Safe to call more than once."""

    def health_check(self) -> bool:
        """Cheap liveness probe. Default: assume healthy."""
        return True

    # -- shared helpers ----------------------------------------------------
    @staticmethod
    def _empty_bars() -> pd.DataFrame:
        return pd.DataFrame(columns=BAR_COLUMNS)

    @staticmethod
    def normalise_bars(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce any provider's frame into the canonical shape.

        Sorting and de-duplicating here rather than in each adapter means a
        broker returning out-of-order or repeated candles - which happens
        around reconnects - cannot corrupt the indicator series downstream.
        """
        if df is None or df.empty:
            return MarketDataProvider._empty_bars()

        out = df.copy()
        for col in BAR_COLUMNS:
            if col not in out.columns:
                out[col] = 0.0

        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        out = out.dropna(subset=["timestamp"])

        for col in ("open", "high", "low", "close", "volume"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = (
            out[BAR_COLUMNS]
            .sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="last")
            .reset_index(drop=True)
        )
        # A bar with no close is unusable; volume may legitimately be zero.
        out = out.dropna(subset=["close"])
        out["volume"] = out["volume"].fillna(0.0)
        return out

    def __enter__(self):
        self.authenticate()
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} name={self.name!r}>"
