#!/usr/bin/env python
"""
Build a training set and fit the trade-quality classifier.

    python scripts/train_model.py                      # backfill + train
    python scripts/train_model.py --from-db            # reuse stored trades
    python scripts/train_model.py --algorithm xgboost
    python scripts/train_model.py --history-minutes 3000 --symbols IDEA,PNB

Pipeline:
    seed history -> replay every crossover -> simulate the round trips
                 -> label profitable / not (NET of costs) -> fit -> save

Why backfill rather than wait for live signals: a crossover fires a few times a
day per symbol, so live collection would need weeks to reach a usable sample.
Replaying history gets you hundreds of labelled trades in a minute.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _rescore_signals(store, classifier, log) -> int:
    """
    Re-run every stored crossover through the freshly fitted model.

    Predictions are upserted on (symbol, signal_time, side), so this replaces
    the earlier verdicts rather than piling up a second set beside them.
    """
    import json

    import pandas as pd

    from nse_screener.models import Side

    signals = store.load_signals(limit=1_000_000)
    if signals is None or signals.empty:
        return 0

    predictions, times = [], []
    for row in signals.itertuples(index=False):
        try:
            features = json.loads(row.features) if isinstance(row.features, str) else {}
            predictions.append(
                classifier.predict(features, symbol=row.symbol, side=Side(row.side))
            )
            # SQLite hands timestamps back as TEXT; save_predictions calls
            # .isoformat() on them, so they have to go back as datetimes.
            times.append(pd.Timestamp(row.timestamp).to_pydatetime())
        except Exception as exc:  # noqa: BLE001 - one bad row must not lose the batch
            log.debug("could not rescore %s at %s: %s", row.symbol, row.timestamp, exc)

    return store.save_predictions(predictions, times) if predictions else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the trade-quality classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--provider", choices=["simulated", "fyers", "angelone"])
    parser.add_argument("--symbols", help="comma-separated universe override")
    parser.add_argument("--history-minutes", type=int, default=3000,
                        help="more history = more crossovers = a better model")
    parser.add_argument("--algorithm", choices=["random_forest", "xgboost"])
    parser.add_argument("--threshold", type=float, help="ACCEPT probability cutoff")
    parser.add_argument("--from-db", action="store_true",
                        help="skip backfill and train on trades already in the database")
    parser.add_argument("--output", help="model output path")
    parser.add_argument("--min-trades", type=int, default=100,
                        help="warn if the dataset is smaller than this")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    if args.provider:
        os.environ["DATA_PROVIDER"] = args.provider
    if args.symbols:
        os.environ["SYMBOLS"] = args.symbols
    if args.history_minutes:
        os.environ["HISTORY_MINUTES"] = str(args.history_minutes)
    if args.algorithm:
        os.environ["ML_ALGORITHM"] = args.algorithm
    if args.threshold is not None:
        os.environ["ACCEPT_THRESHOLD"] = str(args.threshold)
    os.environ["LOG_LEVEL"] = args.log_level

    from nse_screener.backtest import compute_stats
    from nse_screener.config import resolve_universe, settings
    from nse_screener.engine import ScreenerEngine
    from nse_screener.ml.model import TradeClassifier
    from nse_screener.store import Store
    from nse_screener.utils import get_logger

    log = get_logger("train_model")
    store = Store()

    # ---- 1. dataset -------------------------------------------------------
    if args.from_db:
        log.info("loading trades from %s", settings.paths.db)
        dataset = store.load_training_set()
        if dataset.empty:
            log.error("no closed trades in the database - run without --from-db first")
            return 1
    else:
        symbols = resolve_universe()
        log.info("backfilling %d symbols with %d minutes of history ...",
                 len(symbols), settings.data.history_minutes)
        engine = ScreenerEngine(symbols=symbols)
        frame = engine.backfill_training_data(min_trades=args.min_trades)
        if frame.empty:
            log.error(
                "backfill produced no trades. Either history is too short for "
                "SMMA(%d), or no crossovers occurred. Try --history-minutes 5000.",
                settings.indicators.smma_slow,
            )
            return 1
        dataset = store.load_training_set()

    log.info("dataset: %d labelled trades", len(dataset))
    if len(dataset) < args.min_trades:
        log.warning(
            "only %d trades (< %d). The model will fit, but treat its scores as "
            "indicative rather than reliable.", len(dataset), args.min_trades,
        )

    # ---- 2. what the data actually looks like -----------------------------
    if "net_pnl_pct" in dataset.columns:
        wins = int((dataset["net_pnl_pct"] > 0).sum())
        print("\n" + "=" * 68)
        print("DATASET")
        print("=" * 68)
        print(f"  trades          : {len(dataset)}")
        print(f"  winners         : {wins} ({wins / len(dataset):.1%})")
        print(f"  mean net P&L    : {dataset['net_pnl_pct'].mean():+.4f}%")
        print(f"  median net P&L  : {dataset['net_pnl_pct'].median():+.4f}%")
        print(f"  mean duration   : {dataset['duration_minutes'].mean():.1f} min")
        if "symbol" in dataset.columns:
            print(f"  symbols covered : {dataset['symbol'].nunique()}")
        print(f"  round-trip cost : {(settings.trading.cost_bps + settings.trading.slippage_bps) * 2 / 100:.3f}%")

    # ---- 3. train ---------------------------------------------------------
    classifier = TradeClassifier()
    try:
        report = classifier.train(dataset)
    except ValueError as exc:
        log.error("training failed: %s", exc)
        return 1

    print("\n" + "=" * 68)
    print("MODEL")
    print("=" * 68)
    print(report.summary())

    path = classifier.save(Path(args.output) if args.output else None)
    print(f"\nsaved  : {path}")
    print(f"report : {path.with_suffix('.report.json')}")

    # ---- 3b. re-score every stored crossover ------------------------------
    # The backfill scored its signals with whatever model existed BEFORE this
    # run - on a fresh install, none, so every verdict was UNKNOWN. Re-scoring
    # now means the dashboard shows the model that was actually just fitted.
    rescored = _rescore_signals(store, classifier, log)
    if rescored:
        print(f"rescored: {rescored} stored crossovers with the new model")

    # ---- 4. an honest read on whether this model is worth anything --------
    print("\n" + "=" * 68)
    print("INTERPRETATION")
    print("=" * 68)
    if report.roc_auc < 0.55:
        print("  ROC AUC is at or near chance. The features carry little signal about")
        print("  which crossovers pay - which is a genuine result, not a bug. Options:")
        print("    - collect more history (--history-minutes 10000)")
        print("    - train on live-captured snapshots so LTQ/order-book features are")
        print("      populated (see the train/serve skew note in the README)")
        print("    - accept that SMMA crossovers alone may not be predictable here")
    elif report.roc_auc < 0.65:
        print("  Modest but real discrimination. Usable as a filter, not as a green light.")
    else:
        print("  Strong in-sample discrimination. Before believing it, check that the")
        print("  test split is genuinely out-of-time and that the sample is not dominated")
        print("  by a handful of symbols.")

    if report.warnings:
        print("\n  warnings:")
        for w in report.warnings:
            print(f"    ! {w}")

    print("\nnext: python scripts/run_analysis.py       # what the data actually supports")
    print("      python scripts/run_live.py --relax-liquidity --cycles 10")
    print("      python scripts/run_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
