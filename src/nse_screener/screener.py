"""
Universe screening: price band + liquidity.

Filters are evaluated per quote and the REASON for each rejection is retained.
That matters operationally - when the screener returns zero symbols you need to
know instantly whether the price band or the depth threshold is responsible,
otherwise you are guessing at an empty dashboard.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from .config import ScreenerConfig, settings
from .models import Quote
from .utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ScreenResult:
    passed: list[Quote]
    rejected: dict[str, str]        # symbol -> reason
    stats: Counter

    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.rejected)
        return (len(self.passed) / total * 100.0) if total else 0.0

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.stats.items())]
        return (
            f"{len(self.passed)} passed / {len(self.passed) + len(self.rejected)} "
            f"({self.pass_rate:.1f}%) | " + ", ".join(parts)
        )


def screen_quotes(quotes: list[Quote], cfg: ScreenerConfig | None = None) -> ScreenResult:
    """
    Apply the price band and the two-sided liquidity floor.

    Both bid AND ask quantity must clear the threshold, per the specification -
    a book that is deep on one side only is not tradeable in both directions.
    """
    cfg = cfg or settings.screener
    passed: list[Quote] = []
    rejected: dict[str, str] = {}
    stats: Counter = Counter()

    for quote in quotes:
        if not quote.is_valid():
            rejected[quote.symbol] = "malformed quote"
            stats["malformed"] += 1
            continue

        if not (cfg.min_ltp <= quote.ltp <= cfg.max_ltp):
            rejected[quote.symbol] = (
                f"LTP {quote.ltp:.2f} outside [{cfg.min_ltp:.0f}, {cfg.max_ltp:.0f}]"
            )
            stats["price_band"] += 1
            continue

        if quote.bid_qty < cfg.min_bid_qty:
            rejected[quote.symbol] = f"bid qty {quote.bid_qty:,} < {cfg.min_bid_qty:,}"
            stats["thin_bid"] += 1
            continue

        if quote.ask_qty < cfg.min_ask_qty:
            rejected[quote.symbol] = f"ask qty {quote.ask_qty:,} < {cfg.min_ask_qty:,}"
            stats["thin_ask"] += 1
            continue

        passed.append(quote)
        stats["passed"] += 1

    result = ScreenResult(passed=passed, rejected=rejected, stats=stats)

    # An empty universe is nearly always the liquidity floor, not a market event.
    if not passed and quotes:
        thin = stats["thin_bid"] + stats["thin_ask"]
        if thin >= len(quotes) * 0.8:
            log.warning(
                "screen passed 0/%d - %d rejected on depth. The 1,000,000 two-sided "
                "floor is far above typical NSE cash top-of-book depth; lower "
                "MIN_BID_QTY / MIN_ASK_QTY to see the pipeline work.",
                len(quotes), thin,
            )
    log.info("screen: %s", result.summary())
    return result


def screen_to_frame(result: ScreenResult) -> pd.DataFrame:
    """Tabular view of the screen, for the dashboard."""
    rows = [
        {
            "symbol": q.symbol,
            "ltp": q.ltp,
            "bid_price": q.bid_price,
            "ask_price": q.ask_price,
            "bid_qty": q.bid_qty,
            "ask_qty": q.ask_qty,
            "ltq": q.ltq,
            "spread_pct": round(q.spread_pct, 4),
            "imbalance": round(q.imbalance, 4),
            "status": "PASS",
        }
        for q in result.passed
    ]
    rows.extend(
        {"symbol": sym, "status": "REJECT", "reason": reason}
        for sym, reason in result.rejected.items()
    )
    return pd.DataFrame(rows)
