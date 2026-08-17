"""Typed records passed between layers. Plain dataclasses - no ORM, no magic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @property
    def sign(self) -> int:
        """+1 for a long, -1 for a short. Lets P&L be direction-agnostic."""
        return 1 if self is Side.BUY else -1


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    AVOID = "AVOID"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class Quote:
    """One top-of-book snapshot."""

    symbol: str
    ltp: float
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    ltq: int                      # last traded quantity (this tick)
    volume: int = 0               # cumulative traded quantity for the day
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def spread(self) -> float:
        return self.ask_price - self.bid_price

    @property
    def spread_pct(self) -> float:
        mid = self.mid
        return (self.spread / mid * 100.0) if mid > 0 else 0.0

    @property
    def mid(self) -> float:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2.0
        return self.ltp

    @property
    def imbalance(self) -> float:
        """(bid - ask) / (bid + ask). +1 = all bid, -1 = all ask, 0 = balanced."""
        total = self.bid_qty + self.ask_qty
        return ((self.bid_qty - self.ask_qty) / total) if total > 0 else 0.0

    def is_valid(self) -> bool:
        """Reject structurally impossible ticks before they poison the pipeline."""
        return (
            self.ltp > 0
            and self.bid_price >= 0
            and self.ask_price >= 0
            and self.bid_qty >= 0
            and self.ask_qty >= 0
            and not (self.bid_price > 0 and self.ask_price > 0 and self.bid_price > self.ask_price)
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass(slots=True)
class Signal:
    """A crossover event plus the full feature snapshot that produced it."""

    symbol: str
    side: Side
    timestamp: datetime
    ltp: float
    features: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "timestamp": self.timestamp.isoformat(),
            "ltp": self.ltp,
            "reason": self.reason,
            **{f"f_{k}": v for k, v in self.features.items()},
        }


@dataclass(slots=True)
class Trade:
    """A simulated round trip: entry at a crossover, exit at the reverse."""

    symbol: str
    side: Side
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    gross_pnl_pct: float = 0.0
    net_pnl_pct: float = 0.0
    pnl_points: float = 0.0      # exit LTP - entry LTP, as the brief states it
    duration_minutes: float = 0.0
    mae_pct: float = 0.0          # maximum adverse excursion
    mfe_pct: float = 0.0          # maximum favourable excursion
    features: dict[str, float] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    @property
    def is_profitable(self) -> bool:
        return self.net_pnl_pct > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "gross_pnl_pct": self.gross_pnl_pct,
            "net_pnl_pct": self.net_pnl_pct,
            "pnl_points": self.pnl_points,
            "duration_minutes": self.duration_minutes,
            "mae_pct": self.mae_pct,
            "mfe_pct": self.mfe_pct,
            "label": int(self.is_profitable),
            **{f"f_{k}": v for k, v in self.features.items()},
        }


@dataclass(slots=True)
class Prediction:
    """Model verdict on one signal."""

    symbol: str
    side: Side
    probability: float
    decision: Decision
    explanation: str
    contributions: list[tuple[str, float]] = field(default_factory=list)
    model_version: str = "n/a"

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "probability": self.probability,
            "decision": self.decision.value,
            "explanation": self.explanation,
            "model_version": self.model_version,
        }
