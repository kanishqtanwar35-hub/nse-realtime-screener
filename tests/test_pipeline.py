"""
Features, screening, signals, backtest, store, ingestion.

Grouped into one module because these layers are tested through each other -
a signal test needs bars, a backtest test needs signals.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nse_screener.backtest import TradeSimulator, compute_stats
from nse_screener.config import DataConfig, IndicatorConfig, ScreenerConfig, TradingConfig
from nse_screener.features import (FEATURE_COLUMNS, TickWindow,
                                   build_feature_snapshot, compute_bar_features,
                                   features_to_frame)
from nse_screener.ingestion import BarAggregator, clean_bars
from nse_screener.models import Decision, Side, Trade
from nse_screener.screener import screen_quotes
from nse_screener.signals import SignalEngine, extract_historical_signals


# ==========================================================================
class TestScreener:
    def test_price_band_is_inclusive(self, quote_factory):
        cfg = ScreenerConfig(min_ltp=30, max_ltp=500)
        assert len(screen_quotes([quote_factory(ltp=30.0)], cfg).passed) == 1
        assert len(screen_quotes([quote_factory(ltp=500.0)], cfg).passed) == 1
        assert len(screen_quotes([quote_factory(ltp=29.99)], cfg).passed) == 0
        assert len(screen_quotes([quote_factory(ltp=500.01)], cfg).passed) == 0

    def test_liquidity_requires_BOTH_sides(self, quote_factory):
        """A book deep on one side only is not tradeable in both directions."""
        cfg = ScreenerConfig(min_bid_qty=1_000_000, min_ask_qty=1_000_000)
        deep_bid_only = quote_factory(bid_qty=2_000_000, ask_qty=500)
        deep_ask_only = quote_factory(bid_qty=500, ask_qty=2_000_000)
        assert len(screen_quotes([deep_bid_only], cfg).passed) == 0
        assert len(screen_quotes([deep_ask_only], cfg).passed) == 0

    def test_rejection_reason_is_recorded(self, quote_factory):
        cfg = ScreenerConfig(min_ltp=30, max_ltp=500)
        result = screen_quotes([quote_factory(symbol="CHEAP", ltp=5.0)], cfg)
        assert "outside" in result.rejected["CHEAP"]
        assert result.stats["price_band"] == 1

    def test_malformed_quote_is_dropped(self, quote_factory):
        crossed = quote_factory()
        crossed.bid_price, crossed.ask_price = 101.0, 99.0   # crossed book
        assert len(screen_quotes([crossed], ScreenerConfig()).passed) == 0


# ==========================================================================
class TestTickWindow:
    def test_evicts_by_time_not_count(self):
        w = TickWindow("T", max_minutes=5)
        base = datetime(2024, 3, 1, 10, 0)
        from nse_screener.models import Quote

        for i in range(20):
            w.add(Quote("T", 100.0, 99.9, 100.1, 10, 10, ltq=100,
                        timestamp=base + timedelta(minutes=i)))
        # Only the last 5 minutes survive.
        assert len(w) <= 6

    def test_zero_ltq_is_not_recorded(self, quote_factory):
        """A book update with no trade must not drag the LTQ average to zero."""
        w = TickWindow("T")
        w.add(quote_factory(ltq=0))
        assert len(w) == 0

    def test_average_over_window(self, quote_factory):
        w = TickWindow("T")
        base = datetime(2024, 3, 1, 10, 0)
        for i, q in enumerate([100, 200, 300]):
            w.add(quote_factory(ltq=q, ts=base + timedelta(seconds=i * 10)))
        assert w.ltq_average(2, base + timedelta(seconds=30)) == pytest.approx(200.0)

    def test_empty_window_is_zero_not_an_error(self):
        assert TickWindow("T").ltq_average(5) == 0.0


# ==========================================================================
class TestFeatures:
    def test_snapshot_covers_the_whole_contract(self, crossing_bars, quote_factory):
        cfg = IndicatorConfig()
        featured = compute_bar_features(crossing_bars, cfg)
        snap = build_feature_snapshot(featured, quote_factory(), TickWindow("T"), cfg)
        assert set(snap) == set(FEATURE_COLUMNS)

    def test_every_feature_is_finite(self, crossing_bars, quote_factory):
        """The model must never receive a NaN it was not trained on."""
        cfg = IndicatorConfig()
        featured = compute_bar_features(crossing_bars, cfg)
        snap = build_feature_snapshot(featured, quote_factory(), TickWindow("T"), cfg)
        assert all(np.isfinite(v) for v in snap.values())

    def test_snapshot_survives_an_empty_bar_frame(self, quote_factory):
        cfg = IndicatorConfig()
        snap = build_feature_snapshot(pd.DataFrame(), quote_factory(), TickWindow("T"), cfg)
        assert set(snap) == set(FEATURE_COLUMNS)

    def test_imbalance_sign_and_range(self, quote_factory):
        cfg = IndicatorConfig()
        heavy_bid = quote_factory(bid_qty=900, ask_qty=100)
        snap = build_feature_snapshot(pd.DataFrame(), heavy_bid, TickWindow("T"), cfg)
        assert snap["order_book_imbalance"] == pytest.approx(0.8)
        assert -1.0 <= snap["order_book_imbalance"] <= 1.0

    def test_spike_ratio_defaults_to_neutral_without_a_baseline(self, quote_factory):
        cfg = IndicatorConfig()
        snap = build_feature_snapshot(pd.DataFrame(), quote_factory(), TickWindow("T"), cfg)
        assert snap["ltq_spike_2_20"] == 1.0

    def test_frame_conversion_fills_missing_columns(self):
        df = features_to_frame([{"smma_spread_pct": 1.0}])
        assert list(df.columns) == FEATURE_COLUMNS
        assert df.notna().all().all()

    def test_etq_windows_are_cumulative_sums(self, bars_factory, session_start):
        cfg = IndicatorConfig()
        bars = bars_factory([100.0] * 200, session_start, volume=10.0)
        out = compute_bar_features(bars, cfg)
        assert out["etq_5m"].iloc[-1] == pytest.approx(50.0)
        assert out["etq_20m"].iloc[-1] == pytest.approx(200.0)


# ==========================================================================
class TestSignalEngine:
    def test_emits_on_a_genuine_crossover(self, crossing_bars, quote_factory):
        engine = SignalEngine()
        sig = engine.evaluate("TEST", crossing_bars, quote_factory())
        # The engineered ramp crosses somewhere; evaluate() only sees the LAST
        # bar, so walk forward until we land on the crossing bar.
        featured = compute_bar_features(crossing_bars, IndicatorConfig())
        idx = featured.index[featured["crossover"] != 0]
        assert len(idx) >= 1, "fixture should contain a crossover"

        sig = engine.evaluate("TEST", crossing_bars.iloc[: idx[0] + 1], quote_factory())
        assert sig is not None
        assert sig.side is Side.BUY
        assert sig.features

    def test_never_emits_the_same_event_twice(self, crossing_bars, quote_factory):
        """The loop re-evaluates the same bar every poll - it must fire once."""
        featured = compute_bar_features(crossing_bars, IndicatorConfig())
        idx = featured.index[featured["crossover"] != 0][0]
        window = crossing_bars.iloc[: idx + 1]

        engine = SignalEngine()
        first = engine.evaluate("TEST", window, quote_factory())
        second = engine.evaluate("TEST", window, quote_factory())
        assert first is not None
        assert second is None

    def test_returns_none_without_enough_history(self, bars_factory, session_start, quote_factory):
        short = bars_factory([100.0] * 50, session_start)
        assert SignalEngine().evaluate("TEST", short, quote_factory()) is None

    def test_ignores_a_crossover_on_a_forward_filled_bar(self, crossing_bars, quote_factory):
        featured = compute_bar_features(crossing_bars, IndicatorConfig())
        idx = featured.index[featured["crossover"] != 0][0]
        window = crossing_bars.iloc[: idx + 1].copy()
        window["is_filled"] = False
        window.iloc[-1, window.columns.get_loc("is_filled")] = True

        assert SignalEngine(ignore_filled_bars=True).evaluate("T", window, quote_factory()) is None

    def test_historical_extraction_finds_every_crossover(self, crossing_bars):
        signals = extract_historical_signals(crossing_bars, "TEST")
        featured = compute_bar_features(crossing_bars, IndicatorConfig())
        assert len(signals) == int((featured["crossover"] != 0).sum())


# ==========================================================================
class TestTradeSimulator:
    @pytest.fixture
    def sim(self):
        return TradeSimulator(TradingConfig(cost_bps=10, slippage_bps=2, allow_short=True))

    def test_long_pnl_sign(self, sim, signal_factory):
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
            signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=30), ltp=110.0),
        ])
        assert len(trades) == 1
        assert trades[0].gross_pnl_pct == pytest.approx(10.0)

    def test_short_pnl_sign_is_inverted(self, sim, signal_factory):
        """A short that sees price FALL is a winner. Getting this backwards
        would mislabel half the training set."""
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.SELL, ts=t0, ltp=100.0),
            signal_factory(side=Side.BUY, ts=t0 + timedelta(minutes=30), ltp=90.0),
        ])
        assert trades[0].gross_pnl_pct == pytest.approx(10.0)

    def test_costs_are_charged_on_both_legs(self, sim, signal_factory):
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
            signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=10), ltp=110.0),
        ])
        # (10 bps + 2 bps) x 2 legs = 24 bps = 0.24%
        assert trades[0].gross_pnl_pct - trades[0].net_pnl_pct == pytest.approx(0.24)

    def test_a_small_winner_becomes_a_loser_after_costs(self, sim, signal_factory):
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
            signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=5), ltp=100.1),
        ])
        assert trades[0].gross_pnl_pct > 0
        assert trades[0].net_pnl_pct < 0
        assert not trades[0].is_profitable

    def test_duration_is_recorded(self, sim, signal_factory):
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
            signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=45), ltp=105.0),
        ])
        assert trades[0].duration_minutes == pytest.approx(45.0)

    def test_consecutive_same_side_signals_are_ignored(self, sim, signal_factory):
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
            signal_factory(side=Side.BUY, ts=t0 + timedelta(minutes=5), ltp=102.0),
            signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=10), ltp=105.0),
        ])
        assert len(trades) == 1
        assert trades[0].entry_price == 100.0

    def test_stop_and_reverse_chains_trades(self, sim, signal_factory):
        """
        BUY -> SELL -> BUY with shorts enabled produces TWO closed trades:
        the long closed by the SELL, and the short opened by that same SELL and
        closed by the final BUY. Every exit is also an entry.
        """
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
            signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=10), ltp=105.0),
            signal_factory(side=Side.BUY, ts=t0 + timedelta(minutes=20), ltp=102.0),
        ])
        assert len(trades) == 2
        assert [t.side for t in trades] == [Side.BUY, Side.SELL]
        assert all(t.exit_reason == "reverse_crossover" for t in trades)
        # The short sold at 105 and covered at 102 - a winner.
        assert trades[1].gross_pnl_pct == pytest.approx(100 * 3 / 105)

    def test_shorts_skipped_when_disabled(self, signal_factory):
        sim = TradeSimulator(TradingConfig(allow_short=False))
        t0 = datetime(2024, 3, 1, 10, 0)
        trades = sim.simulate([
            signal_factory(side=Side.SELL, ts=t0, ltp=100.0),
            signal_factory(side=Side.BUY, ts=t0 + timedelta(minutes=10), ltp=95.0),
        ])
        assert trades == []

    def test_excursions_are_direction_aware(self, sim, signal_factory, bars_factory, session_start):
        bars = bars_factory([100, 105, 95, 100], session_start)
        t0 = session_start
        trades = sim.simulate(
            [signal_factory(side=Side.BUY, ts=t0, ltp=100.0),
             signal_factory(side=Side.SELL, ts=t0 + timedelta(minutes=3), ltp=100.0)],
            bars,
        )
        trade = trades[0]
        assert trade.mfe_pct > 0      # saw +5%
        assert trade.mae_pct > 0      # saw -5%


class TestStats:
    def test_empty_is_safe(self):
        assert compute_stats([]).trades == 0

    def test_win_rate_and_totals(self):
        def closed(pnl):
            t = Trade("S", Side.BUY, datetime(2024, 3, 1, 10), 100.0)
            t.exit_time = datetime(2024, 3, 1, 11)
            t.net_pnl_pct = pnl
            t.duration_minutes = 60
            return t

        stats = compute_stats([closed(2.0), closed(-1.0), closed(3.0), closed(-4.0)])
        assert stats.trades == 4
        assert stats.wins == 2
        assert stats.win_rate == pytest.approx(50.0)
        assert stats.total_pnl_pct == pytest.approx(0.0)
        assert stats.profit_factor == pytest.approx(1.0)


# ==========================================================================
class TestIngestion:
    def test_gaps_are_filled_only_within_a_session(self):
        """
        The bug this guards: reindexing across an overnight break invents ~1,065
        phantom minutes and can carry one day's close into the next day's open.
        """
        cfg = DataConfig()
        day1 = [datetime(2024, 3, 1, 10, 0) + timedelta(minutes=i) for i in range(5)]
        day2 = [datetime(2024, 3, 4, 10, 0) + timedelta(minutes=i) for i in range(5)]
        df = pd.DataFrame({
            "timestamp": day1 + day2,
            "open": [100.0] * 10, "high": [100.0] * 10,
            "low": [100.0] * 10, "close": [100.0] * 10, "volume": [10.0] * 10,
        })
        out = clean_bars(df, cfg)
        assert len(out) == 10           # not thousands
        assert not out["is_filled"].any()

    def test_small_intra_session_gap_is_filled_and_flagged(self):
        cfg = DataConfig(max_forward_fill=3)
        stamps = [datetime(2024, 3, 1, 10, 0) + timedelta(minutes=i) for i in (0, 1, 4, 5)]
        df = pd.DataFrame({
            "timestamp": stamps,
            "open": [100.0] * 4, "high": [100.0] * 4,
            "low": [100.0] * 4, "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [10.0] * 4,
        })
        out = clean_bars(df, cfg)
        assert len(out) == 6
        assert out["is_filled"].sum() == 2
        # Synthetic bars carry no volume - they represent minutes with no trades.
        assert out.loc[out["is_filled"], "volume"].eq(0).all()

    def test_aggregator_closes_a_bar_on_the_minute_boundary(self, quote_factory):
        agg = BarAggregator("T")
        base = datetime(2024, 3, 1, 10, 0)
        assert agg.update(quote_factory(ts=base, ltp=100.0)) is False
        assert agg.update(quote_factory(ts=base + timedelta(seconds=30), ltp=101.0)) is False
        assert agg.update(quote_factory(ts=base + timedelta(minutes=1), ltp=102.0)) is True
        assert len(agg) == 1
        assert agg.bars.iloc[0]["high"] == 101.0

    def test_volume_uses_the_cumulative_delta(self, quote_factory):
        agg = BarAggregator("T")
        base = datetime(2024, 3, 1, 10, 0)
        agg.update(quote_factory(ts=base, ltq=100, volume=1000))
        agg.update(quote_factory(ts=base + timedelta(seconds=30), ltq=50, volume=1500))
        agg.update(quote_factory(ts=base + timedelta(minutes=1), ltq=10, volume=1510))
        assert agg.bars.iloc[0]["volume"] == pytest.approx(100 + 500)


# ==========================================================================
class TestStore:
    def test_saving_the_same_signal_twice_yields_one_row(self, tmp_store, signal_factory):
        """Idempotency: re-running a cycle must not duplicate history."""
        sig = signal_factory()
        tmp_store.save_signals([sig])
        tmp_store.save_signals([sig])
        assert tmp_store.counts()["signals"] == 1

    def test_trade_upsert_updates_rather_than_duplicates(self, tmp_store):
        t = Trade("S", Side.BUY, datetime(2024, 3, 1, 10), 100.0)
        tmp_store.save_trades([t])
        t.exit_time = datetime(2024, 3, 1, 11)
        t.exit_price = 110.0
        t.net_pnl_pct = 9.76
        tmp_store.save_trades([t])

        assert tmp_store.counts()["trades"] == 1
        assert tmp_store.load_trades().iloc[0]["net_pnl_pct"] == pytest.approx(9.76)

    def test_training_set_explodes_features(self, tmp_store):
        t = Trade("S", Side.BUY, datetime(2024, 3, 1, 10), 100.0,
                  features={"smma_spread_pct": 1.5, "volatility_pct": 0.3})
        t.exit_time = datetime(2024, 3, 1, 11)
        t.exit_price = 110.0
        t.net_pnl_pct = 9.76
        tmp_store.save_trades([t])

        df = tmp_store.load_training_set()
        assert "f_smma_spread_pct" in df.columns
        assert df.iloc[0]["f_smma_spread_pct"] == pytest.approx(1.5)

    def test_bars_round_trip_and_come_back_in_order(self, tmp_store, bars_factory,
                                                    session_start):
        """The dashboard charts whatever this returns, so order is not optional."""
        bars = bars_factory([100.0 + i for i in range(50)], session_start)
        bars["smma_fast"] = bars["close"].rolling(5).mean()
        bars["smma_slow"] = bars["close"].rolling(20).mean()
        tmp_store.save_bars("TEST", bars)

        out = tmp_store.load_bars("TEST", limit=30)
        assert len(out) == 30
        assert out["timestamp"].is_monotonic_increasing
        # `limit` takes the NEWEST rows, so the last bar must survive the cut.
        assert out["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1])
        assert tmp_store.bar_symbols() == ["TEST"]

    def test_replaying_a_session_overwrites_bars_rather_than_duplicating(
        self, tmp_store, bars_factory, session_start
    ):
        bars = bars_factory([100.0, 101.0, 102.0], session_start)
        tmp_store.save_bars("TEST", bars)
        bars["close"] = [200.0, 201.0, 202.0]
        tmp_store.save_bars("TEST", bars)

        out = tmp_store.load_bars("TEST")
        assert len(out) == 3
        assert out["close"].tolist() == [200.0, 201.0, 202.0]

    def test_tail_writes_only_the_newest_bars(self, tmp_store, bars_factory, session_start):
        """The live loop writes tail=3 every cycle; writing 1500 would be disk-bound."""
        bars = bars_factory([100.0 + i for i in range(40)], session_start)
        assert tmp_store.save_bars("TEST", bars, tail=3) == 3
        assert len(tmp_store.load_bars("TEST")) == 3

    def test_bars_with_unusable_timestamps_are_skipped_not_written_as_null(
        self, tmp_store, bars_factory, session_start
    ):
        bars = bars_factory([100.0, 101.0, 102.0], session_start)
        bars.loc[1, "timestamp"] = pd.NaT
        assert tmp_store.save_bars("TEST", bars) == 2


# ==========================================================================
class TestClassifier:
    def test_untrained_model_returns_UNKNOWN_not_a_crash(self):
        from nse_screener.ml.model import TradeClassifier

        pred = TradeClassifier().predict({c: 0.0 for c in FEATURE_COLUMNS}, symbol="X")
        assert pred.decision is Decision.UNKNOWN
        assert pred.probability == 0.5

    def test_train_predict_roundtrip(self, rng):
        from nse_screener.ml.model import TradeClassifier

        n = 200
        rows = []
        for i in range(n):
            spread = rng.normal(0, 1)
            rows.append({
                **{f"f_{c}": rng.normal(0, 1) for c in FEATURE_COLUMNS},
                "f_smma_spread_pct": spread,
                # A learnable relationship so the test asserts something real.
                "label": int(spread > 0),
                "entry_time": datetime(2024, 3, 1, 10) + timedelta(minutes=i),
            })
        df = pd.DataFrame(rows)

        clf = TradeClassifier()
        report = clf.train(df)
        assert report.n_samples == n
        assert clf.is_trained

        pred = clf.predict({**{c: 0.0 for c in FEATURE_COLUMNS}, "smma_spread_pct": 3.0})
        assert 0.0 <= pred.probability <= 1.0
        assert pred.decision in (Decision.ACCEPT, Decision.AVOID)
        assert pred.explanation

    def test_single_class_labels_raise_a_clear_error(self):
        from nse_screener.ml.model import TradeClassifier

        df = pd.DataFrame([{**{f"f_{c}": 0.0 for c in FEATURE_COLUMNS}, "label": 1}] * 80)
        with pytest.raises(ValueError, match="both winners and losers"):
            TradeClassifier().train(df)

    def test_save_and_load(self, tmp_path, rng):
        from nse_screener.ml.model import TradeClassifier

        df = pd.DataFrame([
            {**{f"f_{c}": rng.normal() for c in FEATURE_COLUMNS}, "label": i % 2}
            for i in range(100)
        ])
        clf = TradeClassifier()
        clf.train(df)
        path = clf.save(tmp_path / "m.joblib")

        reloaded = TradeClassifier.load(path)
        assert reloaded.is_trained
        assert reloaded.version == clf.version
