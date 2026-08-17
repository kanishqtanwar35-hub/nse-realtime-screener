"""
Orchestration: the loop that ties every layer together.

One cycle:

    fetch quotes -> screen -> update bars -> detect crossovers
                 -> score with the model -> persist -> repeat

Design rules:

  * ONE cycle failure never kills the loop. Each stage is guarded; a stage that
    throws is logged and the cycle moves on. A screener that dies at 09:20 and
    stays dead until someone notices is worse than one that skips a cycle.
  * State lives in the engine, not in globals, so it is testable and two
    engines can coexist in one process.
  * The loop is cooperative: `stop()` sets an event and the loop exits at the
    next boundary rather than being killed mid-write.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from .backtest import TradeSimulator, compute_stats, trades_to_frame
from .config import resolve_universe, settings
from .features import build_display_row, compute_bar_features
from .ingestion import IngestionPipeline
from .ml.model import TradeClassifier
from .models import Prediction, Quote, Signal, Trade
from .providers import get_provider
from .providers.base import MarketDataProvider
from .screener import screen_quotes, screen_to_frame
from .signals import SignalEngine
from .store import Store
from .utils import Timer, get_logger

log = get_logger(__name__)


@dataclass
class CycleResult:
    """What one pass through the loop produced. Returned for tests and the CLI."""

    cycle: int
    started_at: datetime
    quotes_fetched: int = 0
    symbols_passed: int = 0
    bars_closed: int = 0
    signals: list[Signal] = field(default_factory=list)
    predictions: list[Prediction] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"cycle {self.cycle}: {self.quotes_fetched} quotes, "
            f"{self.symbols_passed} passed screen, {self.bars_closed} bars closed, "
            f"{len(self.signals)} signals, {len(self.errors)} errors "
            f"({self.duration_ms:.0f} ms)"
        )


class ScreenerEngine:
    """The live application."""

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        symbols: list[str] | None = None,
        store: Store | None = None,
        classifier: TradeClassifier | None = None,
    ) -> None:
        self.symbols = symbols or resolve_universe()
        self.provider = provider or get_provider(symbols=self.symbols)
        self.store = store or Store()
        self.classifier = classifier or TradeClassifier.load()

        self.ingestion = IngestionPipeline(self.provider)
        self.signal_engine = SignalEngine()
        self.simulator = TradeSimulator()

        self.latest_quotes: dict[str, Quote] = {}
        self.latest_screen: pd.DataFrame = pd.DataFrame()
        self.latest_metrics: pd.DataFrame = pd.DataFrame()
        self.open_trades: dict[str, Trade] = {}
        self.closed_trades: list[Trade] = []

        self.cycle_count = 0
        self._stop = threading.Event()
        self._seeded = False

    # ------------------------------------------------------------------
    def warm_up(self) -> int:
        """Seed history. Returns the number of tradeable symbols."""
        if self._seeded:
            return len(self.ingestion.aggregators)
        log.info("warming up %d symbols from %s ...", len(self.symbols), self.provider.name)
        with Timer("warm_up", log):
            self.ingestion.seed_history(self.symbols)
        self._seeded = True
        n = len(self.ingestion.aggregators)
        if n == 0:
            log.critical("no symbols have enough history - check the provider and MIN_BARS_REQUIRED")
        self._persist_bars(self.ingestion.aggregators.keys())
        return n

    # ------------------------------------------------------------------
    def _persist_bars(self, symbols, tail: int | None = None) -> int:
        """
        Write bars to the store so the dashboard process can chart them.

        Bars live in memory inside the aggregators, and the dashboard is a
        separate process that only ever sees SQLite. Without this the price /
        SMMA / crossover chart - the one picture a screener is expected to have
        - cannot be drawn at all.

        Guarded as a whole: charting is a convenience, and a disk problem here
        must never take down the trading loop.
        """
        written = 0
        for symbol in list(symbols):
            agg = self.ingestion.aggregators.get(symbol)
            if agg is None or len(agg) == 0:
                continue
            try:
                featured = compute_bar_features(agg.snapshot(), self.signal_engine.cfg)
                written += self.store.save_bars(symbol, featured, tail=tail)
            except Exception as exc:  # noqa: BLE001
                log.debug("bar persistence failed for %s: %s", symbol, exc)
        if written:
            log.debug("persisted %d bar rows", written)
        return written

    # ------------------------------------------------------------------
    def run_cycle(self) -> CycleResult:
        """One full pass. Never raises; failures land in result.errors."""
        self.cycle_count += 1
        result = CycleResult(cycle=self.cycle_count, started_at=datetime.now())
        t0 = time.perf_counter()

        if not self._seeded:
            self.warm_up()

        # --- 1. quotes ----------------------------------------------------
        try:
            quotes = self.ingestion.fetch_quotes(self.symbols)
            result.quotes_fetched = len(quotes)
            self.latest_quotes = {q.symbol: q for q in quotes}
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"quote fetch: {exc}")
            log.exception("quote fetch failed")
            quotes = []

        # --- 2. screen ----------------------------------------------------
        passed: list[Quote] = []
        try:
            screen = screen_quotes(quotes)
            passed = screen.passed
            result.symbols_passed = len(passed)
            self.latest_screen = screen_to_frame(screen)
            self.store.save_screen_snapshot(self.latest_screen)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"screen: {exc}")
            log.exception("screening failed")

        # --- 3. bars + tick windows --------------------------------------
        closed: set[str] = set()
        try:
            # Tick features need EVERY quote, not just the screened ones - the
            # LTQ baseline would be full of holes if we only fed passers.
            for quote in quotes:
                self.signal_engine.observe(quote)
            closed = self.ingestion.apply_quotes(quotes)
            result.bars_closed = len(closed)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"bar update: {exc}")
            log.exception("bar aggregation failed")

        # --- 3b. the one-row-per-stock dashboard view ---------------------
        # Built for every symbol that PASSED the screen, whether or not a bar
        # closed - the table must stay live even on cycles with no new bar.
        try:
            rows = []
            for quote in passed:
                agg = self.ingestion.aggregators.get(quote.symbol)
                if agg is None or len(agg) == 0:
                    continue
                featured = compute_bar_features(agg.snapshot(), self.signal_engine.cfg)
                rows.append(
                    build_display_row(
                        quote.symbol, featured, quote,
                        self.signal_engine.state_for(quote.symbol).ticks,
                        self.signal_engine.cfg,
                    )
                )
            if rows:
                self.store.save_live_metrics(rows)
                self.latest_metrics = pd.DataFrame(rows)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"live metrics: {exc}")
            log.exception("live metrics build failed")

        # --- 3c. persist the bars that just closed ------------------------
        # Only the closed ones, and only their tail: a bar closes about once a
        # minute per symbol, so this is a handful of upserts, not a rewrite of
        # the whole history on every five-second poll.
        if closed:
            self._persist_bars(closed, tail=3)

        # --- 4. signals ---------------------------------------------------
        # Only symbols that (a) passed the screen and (b) just closed a bar.
        # Re-evaluating an unchanged bar wastes CPU and cannot produce anything new.
        candidates = [q.symbol for q in passed if q.symbol in closed]
        for symbol in candidates:
            try:
                agg = self.ingestion.aggregators.get(symbol)
                quote = self.latest_quotes.get(symbol)
                if agg is None or quote is None:
                    continue
                signal = self.signal_engine.evaluate(symbol, agg.snapshot(), quote)
                if signal is not None:
                    result.signals.append(signal)
                    self._update_positions(signal, agg.snapshot())
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"signal {symbol}: {exc}")
                log.exception("signal evaluation failed for %s", symbol)

        # --- 5. score + persist -------------------------------------------
        if result.signals:
            try:
                result.predictions = [
                    self.classifier.predict(s.features, symbol=s.symbol, side=s.side)
                    for s in result.signals
                ]
                self.store.save_signals(result.signals)
                self.store.save_predictions(
                    result.predictions, [s.timestamp for s in result.signals]
                )
                for pred in result.predictions:
                    log.info(
                        "  -> %s %s: %s (%.0f%%) %s",
                        pred.symbol, pred.side.value, pred.decision.value,
                        pred.probability * 100, pred.explanation,
                    )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"scoring/persist: {exc}")
                log.exception("scoring or persistence failed")

        result.duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(result.summary())
        return result

    # ------------------------------------------------------------------
    def _update_positions(self, signal: Signal, bars: pd.DataFrame) -> None:
        """
        Maintain live paper positions using the stop-and-reverse rule.

        The reverse crossover closes the open trade and opens the opposite one -
        the same logic the backtester applies, kept here so live and simulated
        results are directly comparable.
        """
        open_trade = self.open_trades.get(signal.symbol)

        if open_trade is not None and open_trade.side is not signal.side:
            self.simulator._close(
                open_trade, signal.timestamp, signal.ltp, "reverse_crossover", bars
            )
            self.closed_trades.append(open_trade)
            self.store.save_trades([open_trade])
            log.info(
                "CLOSED %s %s: %+.3f%% net over %.0f min",
                open_trade.symbol, open_trade.side.value,
                open_trade.net_pnl_pct, open_trade.duration_minutes,
            )
            self.open_trades.pop(signal.symbol, None)

        if signal.symbol not in self.open_trades:
            if signal.side.value == "SELL" and not self.simulator.cfg.allow_short:
                return
            self.open_trades[signal.symbol] = Trade(
                symbol=signal.symbol,
                side=signal.side,
                entry_time=signal.timestamp,
                entry_price=signal.ltp,
                features=dict(signal.features),
            )

    # ------------------------------------------------------------------
    def _score_and_save(self, signals: list[Signal]) -> int:
        """
        Score a batch of signals and persist the verdicts.

        Kept separate from the live path so a backfill produces exactly the same
        ACCEPT/AVOID rows the live loop would have written for those crossovers.
        An untrained model still writes rows - they come back UNKNOWN, which is
        information, not an error.
        """
        try:
            predictions = [
                self.classifier.predict(s.features, symbol=s.symbol, side=s.side)
                for s in signals
            ]
            return self.store.save_predictions(predictions, [s.timestamp for s in signals])
        except Exception as exc:  # noqa: BLE001
            log.error("scoring backfilled signals failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    def run(
        self,
        max_cycles: int | None = None,
        on_cycle: Callable[[CycleResult], None] | None = None,
    ) -> list[CycleResult]:
        """Run the loop until stopped or `max_cycles` is reached."""
        self.warm_up()
        results: list[CycleResult] = []
        interval = max(0.5, settings.data.poll_seconds)
        log.info("starting loop: %d symbols, %.1fs interval", len(self.symbols), interval)

        try:
            while not self._stop.is_set():
                if max_cycles is not None and self.cycle_count >= max_cycles:
                    break
                started = time.perf_counter()
                result = self.run_cycle()
                results.append(result)
                if on_cycle is not None:
                    try:
                        on_cycle(result)
                    except Exception:  # noqa: BLE001
                        log.exception("on_cycle callback failed")

                # Sleep the REMAINDER of the interval so cycle time does not
                # accumulate into ever-later polls.
                elapsed = time.perf_counter() - started
                if self._stop.wait(max(0.0, interval - elapsed)):
                    break
        except KeyboardInterrupt:
            log.info("interrupted - shutting down")
        finally:
            self.shutdown()
        return results

    def stop(self) -> None:
        self._stop.set()

    def shutdown(self) -> None:
        try:
            if self.open_trades:
                log.info("%d position(s) still open at shutdown", len(self.open_trades))
            self.provider.close()
        except Exception:  # noqa: BLE001
            log.exception("shutdown error")
        log.info("engine stopped after %d cycles", self.cycle_count)

    # ------------------------------------------------------------------
    def stats(self):
        return compute_stats(self.closed_trades)

    def backfill_training_data(self, min_trades: int = 200) -> pd.DataFrame:
        """
        Manufacture a training set by replaying every seeded symbol's history.

        The live loop produces a handful of crossovers an hour; a model needs
        hundreds of examples, so the initial dataset comes from history.
        """
        from .signals import extract_historical_signals

        self.warm_up()
        all_trades: list[Trade] = []
        all_signals: list[Signal] = []
        cfg = self.signal_engine.cfg

        for symbol, agg in self.ingestion.aggregators.items():
            try:
                bars = agg.snapshot()
                # Extract the crossovers here rather than letting run_symbol do
                # it internally, because the SIGNALS are worth keeping: they are
                # what the dashboard draws on the price chart and lists in the
                # Signals tab. Discarding them meant a backfill populated the
                # trade table while leaving every other view empty.
                signals = extract_historical_signals(bars, symbol, cfg)
                if not signals:
                    continue
                all_signals.extend(signals)
                trades = self.simulator.simulate(signals, compute_bar_features(bars, cfg))
                all_trades.extend(trades)
                log.debug("%s: %d crossovers -> %d simulated trades", symbol,
                          len(signals), len(trades))
            except Exception as exc:  # noqa: BLE001
                log.error("backfill failed for %s: %s", symbol, exc)

        if all_signals:
            self.store.save_signals(all_signals)
            self._score_and_save(all_signals)

        closed = [t for t in all_trades if not t.is_open]
        if closed:
            self.store.save_trades(closed)
        stats = compute_stats(closed)
        log.info("backfill complete: %s", stats.summary())
        if len(closed) < min_trades:
            log.warning(
                "only %d trades generated (wanted >= %d) - increase HISTORY_MINUTES "
                "or widen the symbol universe before training",
                len(closed), min_trades,
            )
        return trades_to_frame(closed)
