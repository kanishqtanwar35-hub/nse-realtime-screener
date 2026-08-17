#!/usr/bin/env python
"""
Run the live screening loop.

    python scripts/run_live.py                          # simulated, forever
    python scripts/run_live.py --cycles 5               # 5 cycles then exit
    python scripts/run_live.py --provider fyers         # live feed
    python scripts/run_live.py --relax-liquidity        # usable depth threshold
    python scripts/run_live.py --symbols IDEA,PNB,SAIL  # explicit universe

Ctrl-C shuts down cleanly: the loop finishes the cycle it is in, closes the
provider session, and reports any still-open paper positions.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make `src/` importable when run directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NSE real-time screener + signal engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", choices=["simulated", "fyers", "angelone"],
                        help="data source (default: DATA_PROVIDER env, else simulated)")
    parser.add_argument("--symbols", help="comma-separated universe override")
    parser.add_argument("--cycles", type=int, help="stop after N cycles (default: run forever)")
    parser.add_argument("--interval", type=float, help="seconds between cycles")
    parser.add_argument("--relax-liquidity", action="store_true",
                        help="lower the depth floor to 5,000 so the screen actually passes "
                             "symbols; the 1,000,000 default is above real NSE book depth")
    parser.add_argument("--history-minutes", type=int, help="history to seed per symbol")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--no-fallback", action="store_true",
                        help="fail instead of falling back to simulated data")
    args = parser.parse_args()

    # Environment must be set BEFORE importing the package - settings are frozen
    # dataclasses resolved at import time.
    if args.provider:
        os.environ["DATA_PROVIDER"] = args.provider
    if args.symbols:
        os.environ["SYMBOLS"] = args.symbols
    if args.interval:
        os.environ["POLL_SECONDS"] = str(args.interval)
    if args.history_minutes:
        os.environ["HISTORY_MINUTES"] = str(args.history_minutes)
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    if args.relax_liquidity:
        os.environ["MIN_BID_QTY"] = "5000"
        os.environ["MIN_ASK_QTY"] = "5000"

    from nse_screener.config import resolve_universe, settings
    from nse_screener.engine import ScreenerEngine
    from nse_screener.providers import get_provider
    from nse_screener.utils import get_logger

    log = get_logger("run_live")

    symbols = resolve_universe()
    log.info("=" * 70)
    log.info("NSE SCREENER")
    log.info("  provider   : %s", settings.provider)
    log.info("  symbols    : %d", len(symbols))
    log.info("  price band : Rs %.0f - %.0f", settings.screener.min_ltp, settings.screener.max_ltp)
    log.info("  depth floor: bid >= %s and ask >= %s",
             f"{settings.screener.min_bid_qty:,}", f"{settings.screener.min_ask_qty:,}")
    log.info("  SMMA       : %d / %d on %d-min bars",
             settings.indicators.smma_fast, settings.indicators.smma_slow,
             settings.data.interval_minutes)
    log.info("  poll       : %.1fs", settings.data.poll_seconds)
    log.info("=" * 70)

    if settings.screener.min_bid_qty >= 1_000_000:
        log.warning(
            "depth floor is %d on BOTH sides. Real NSE cash top-of-book depth in the "
            "Rs 30-500 band is typically hundreds to low thousands, so expect an empty "
            "screen. Re-run with --relax-liquidity to exercise the full pipeline.",
            settings.screener.min_bid_qty,
        )

    try:
        provider = get_provider(symbols=symbols, allow_fallback=not args.no_fallback)
    except Exception as exc:  # noqa: BLE001
        log.critical("could not obtain a data provider: %s", exc)
        return 2

    engine = ScreenerEngine(provider=provider, symbols=symbols)

    tradeable = engine.warm_up()
    if tradeable == 0:
        log.critical("no symbols have sufficient history - nothing to do")
        return 3

    results = engine.run(max_cycles=args.cycles)

    # ---- session summary --------------------------------------------------
    total_signals = sum(len(r.signals) for r in results)
    total_errors = sum(len(r.errors) for r in results)
    stats = engine.stats()

    log.info("=" * 70)
    log.info("SESSION SUMMARY")
    log.info("  cycles        : %d", len(results))
    log.info("  signals       : %d", total_signals)
    log.info("  closed trades : %d", stats.trades)
    if stats.trades:
        log.info("  performance   : %s", stats.summary())
    log.info("  open positions: %d", len(engine.open_trades))
    log.info("  errors        : %d", total_errors)
    log.info("  database      : %s", settings.paths.db)
    log.info("=" * 70)

    if total_signals == 0 and results:
        log.info(
            "No crossovers fired. That is normal over a short run - SMMA20/SMMA120 "
            "cross a few times a day per symbol. Use scripts/train_model.py to harvest "
            "historical crossovers in bulk."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
