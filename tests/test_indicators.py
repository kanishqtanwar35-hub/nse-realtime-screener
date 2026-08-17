"""Indicator correctness. These are the tests that matter most - every signal
in the system is downstream of SMMA and crossover."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener.indicators import (add_indicators, crossover, momentum,
                                     rate_of_change, realised_volatility,
                                     smma, smma_ewm)


class TestSMMA:
    def test_matches_the_recursive_definition(self):
        """SMMA[i] = (SMMA[i-1]*(n-1) + price[i]) / n, seeded with SMA."""
        prices = pd.Series([10.0, 11, 12, 13, 14, 15, 16, 17])
        n = 3
        got = smma(prices, n)

        expected = [np.nan, np.nan, prices.iloc[:3].mean()]
        for i in range(3, len(prices)):
            expected.append((expected[-1] * (n - 1) + prices.iloc[i]) / n)

        np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-12)

    def test_recursive_and_vectorised_agree(self, rng):
        """
        The two implementations must be numerically identical - `smma_ewm` is
        used on the hot path precisely because it is a drop-in for `smma`.
        """
        prices = pd.Series(rng.normal(0, 1, 1000).cumsum() + 500)
        for period in (5, 20, 120):
            a, b = smma(prices, period), smma_ewm(prices, period)
            assert (a - b).abs().max() < 1e-9, f"divergence at period={period}"

    def test_seed_equals_sma(self):
        prices = pd.Series(np.arange(1.0, 51.0))
        out = smma(prices, 20)
        assert out.iloc[19] == pytest.approx(prices.iloc[:20].mean())

    def test_leading_values_are_nan_not_zero(self):
        """
        Fabricating a value before the seed would let a crossover fire on data
        that does not exist yet.
        """
        out = smma(pd.Series(np.arange(1.0, 31.0)), 20)
        assert out.iloc[:19].isna().all()
        assert out.iloc[19:].notna().all()

    def test_shorter_than_period_is_all_nan(self):
        assert smma(pd.Series([1.0, 2.0, 3.0]), 20).isna().all()

    def test_nan_gap_carries_level_forward(self):
        """A hole in prices must not reset or zero the average."""
        prices = pd.Series([10.0] * 5 + [np.nan] + [10.0] * 5)
        out = smma(prices, 3)
        assert out.notna().iloc[2:].all()
        assert out.iloc[5] == pytest.approx(out.iloc[4])

    def test_rejects_bad_period(self):
        with pytest.raises(ValueError):
            smma(pd.Series([1.0, 2.0]), 0)


class TestCrossover:
    def test_detects_both_directions(self):
        fast = pd.Series([1.0, 2, 3, 4, 3, 2, 1])
        slow = pd.Series([2.0, 2, 2, 2, 2, 2, 2])
        # spread: -1, 0, +1, +2, +1, 0, -1
        out = crossover(fast, slow)
        assert out.iloc[2] == 1      # first bar the spread is genuinely positive
        # index 5 has spread EXACTLY 0 - undecided, not a crossing. The signal
        # belongs on index 6, where the spread actually turns negative.
        assert out.iloc[5] == 0
        assert out.iloc[6] == -1

    def test_touch_and_retreat_is_not_a_crossover(self):
        """
        Fast rises to touch slow, then falls back without ever going above.
        The naive `prev >= 0` rule emits a phantom SELL here.
        """
        fast = pd.Series([1.0, 2.0, 1.0])
        slow = pd.Series([2.0, 2.0, 2.0])
        assert crossover(fast, slow).eq(0).all()

    def test_no_signal_while_either_series_is_nan(self):
        """
        The critical guard: SMMA(120) is NaN for 119 bars. A naive comparison
        would fire on the first valid bar and invent a signal.
        """
        fast = pd.Series([np.nan, np.nan, 3.0, 4.0])
        slow = pd.Series([np.nan, np.nan, 2.0, 2.0])
        out = crossover(fast, slow)
        assert out.iloc[:3].eq(0).all()

    def test_touching_without_crossing_is_not_a_signal(self):
        fast = pd.Series([1.0, 2.0, 1.0])
        slow = pd.Series([2.0, 2.0, 2.0])
        assert crossover(fast, slow).eq(0).all()

    def test_fires_once_per_event(self):
        fast = pd.Series([1.0, 3.0, 4.0, 5.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0])
        assert (crossover(fast, slow) == 1).sum() == 1

    def test_no_signal_from_an_undecided_start(self):
        """
        If the spread has never had a decided sign - two averages sitting exactly
        on top of each other - then moving off zero is a DIVERGENCE, not a
        crossing. Emitting there would manufacture a signal out of a period that
        carried no directional information at all.
        """
        fast = pd.Series([2.0, 2.0, 2.0, 3.0, 4.0])
        slow = pd.Series([2.0, 2.0, 2.0, 2.0, 2.0])
        assert crossover(fast, slow).eq(0).all()

    def test_zeros_between_two_same_signs_are_not_a_crossing(self):
        """Positive, flat through zero, positive again - no sign change."""
        fast = pd.Series([3.0, 2.0, 3.0])
        slow = pd.Series([2.0, 2.0, 2.0])
        assert crossover(fast, slow).eq(0).all()


class TestOtherIndicators:
    def test_rate_of_change(self):
        out = rate_of_change(pd.Series([100.0, 110.0, 121.0]), 1)
        assert out.iloc[1] == pytest.approx(10.0)
        assert out.iloc[2] == pytest.approx(10.0)

    def test_roc_survives_a_zero_base(self):
        assert not np.isinf(rate_of_change(pd.Series([0.0, 5.0]), 1)).any()

    def test_volatility_is_zero_for_a_flat_series(self):
        vol = realised_volatility(pd.Series([100.0] * 50), 20)
        assert vol.dropna().abs().max() == pytest.approx(0.0, abs=1e-12)

    def test_momentum_sign_follows_direction(self):
        rising = momentum(pd.Series(np.arange(100.0, 120.0)), 5)
        assert rising.dropna().gt(0).all()


class TestAddIndicators:
    def test_produces_the_expected_columns(self, crossing_bars):
        out = add_indicators(crossing_bars, 20, 120)
        for col in ("smma_fast", "smma_slow", "smma_spread",
                    "smma_spread_pct", "smma_spread_roc", "crossover"):
            assert col in out.columns

    def test_does_not_mutate_the_input(self, crossing_bars):
        before = list(crossing_bars.columns)
        add_indicators(crossing_bars, 20, 120)
        assert list(crossing_bars.columns) == before

    def test_finds_the_engineered_crossover(self, crossing_bars):
        out = add_indicators(crossing_bars, 20, 120)
        assert (out["crossover"] == 1).sum() >= 1

    def test_no_crossover_before_the_slow_average_exists(self, crossing_bars):
        out = add_indicators(crossing_bars, 20, 120)
        first_valid = out["smma_slow"].first_valid_index()
        assert (out["crossover"].iloc[: first_valid + 1] == 0).all()

    def test_missing_price_column_raises(self, crossing_bars):
        with pytest.raises(KeyError):
            add_indicators(crossing_bars, 20, 120, price_col="nope")
