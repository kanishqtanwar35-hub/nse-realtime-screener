"""
Tests for the analysis layer.

The bar for what is worth testing here is the same as everywhere else in this
project: things that fail SILENTLY. An analysis function that returns a
plausible-looking number from bad input is worse than one that raises, because
the number ends up in a report and gets believed. So the cases below are mostly
about direction, degeneracy and sample-size honesty rather than about arithmetic
that pandas already guarantees.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from nse_screener import analytics


# ==========================================================================
# Fixtures
# ==========================================================================
def make_trades(n: int = 200, seed: int = 7, noise: float = 0.15) -> pd.DataFrame:
    """
    A synthetic trade table with ONE genuinely predictive feature.

    `f_ltq_ratio_2_5` is planted: high values win, low values lose, with
    `noise` of the labels flipped so nothing is separable perfectly.
    `f_momentum_pct` is pure noise and `f_depth_ratio` is constant - between
    them they cover the three cases every ranking function has to handle.
    """
    rng = np.random.default_rng(seed)
    start = datetime(2024, 3, 1, 9, 15)

    signal = np.linspace(0.5, 1.5, n)
    rng.shuffle(signal)
    win = (signal > 1.0).astype(int)
    flip = rng.random(n) < noise
    win = np.where(flip, 1 - win, win)

    pnl = np.where(win == 1, 0.8 + rng.random(n) * 0.4, -(0.6 + rng.random(n) * 0.4))
    return pd.DataFrame({
        "symbol": [f"SYM{i % 4}" for i in range(n)],
        "side": ["BUY" if i % 2 else "SELL" for i in range(n)],
        # 20-minute spacing walks the entries across the whole session, so the
        # hour-of-day breakdown has something to group on.
        "entry_time": [start + timedelta(minutes=20 * i) for i in range(n)],
        "exit_time": [start + timedelta(minutes=20 * i + 30) for i in range(n)],
        "entry_price": 100.0 + rng.random(n) * 50,
        "exit_price": 100.0 + rng.random(n) * 50,
        "net_pnl_pct": pnl,
        "gross_pnl_pct": pnl + 0.24,
        "duration_minutes": rng.integers(5, 300, n).astype(float),
        "mae_pct": rng.random(n) * 1.2,
        "mfe_pct": rng.random(n) * 2.5,
        "label": win,
        "f_ltq_ratio_2_5": signal,
        "f_momentum_pct": rng.normal(0, 1, n),
        "f_depth_ratio": np.ones(n),
    })


@pytest.fixture
def trades() -> pd.DataFrame:
    return make_trades()


@pytest.fixture
def empty() -> pd.DataFrame:
    return pd.DataFrame()


# ==========================================================================
# Degenerate input
# ==========================================================================
@pytest.mark.parametrize(
    "func",
    [
        analytics.equity_curve,
        analytics.feature_separation,
        analytics.best_filters,
        analytics.time_of_day_performance,
        analytics.duration_performance,
        analytics.symbol_performance,
    ],
)
def test_every_table_survives_an_empty_frame(func, empty):
    """A dashboard tab opened before the first trade must not throw."""
    out = func(empty)
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_open_trades_are_excluded(trades):
    """A position still open has no realised P&L and cannot be analysed."""
    trades.loc[:9, "exit_time"] = None
    assert analytics.equity_curve(trades).shape[0] == len(trades) - 10


def test_win_label_is_recomputed_not_trusted(trades):
    """
    The stored label may predate a change in cost assumptions, so the analysis
    derives wins from the P&L column beside it rather than believing the flag.
    """
    trades["label"] = 0                       # deliberately wrong
    sep = analytics.feature_separation(trades)
    assert sep.iloc[0]["n_win"] > 0


# ==========================================================================
# Equity
# ==========================================================================
def test_equity_is_cumulative_and_drawdown_never_positive(trades):
    eq = analytics.equity_curve(trades)
    expected = trades.sort_values("entry_time")["net_pnl_pct"].cumsum().to_numpy()

    assert np.allclose(eq["equity"].to_numpy(), expected)
    assert (eq["drawdown"] <= 1e-12).all()
    # The peak is a running maximum, so it can never fall.
    assert (eq["peak"].diff().dropna() >= -1e-12).all()


# ==========================================================================
# Feature separation
# ==========================================================================
def test_the_planted_feature_ranks_first(trades):
    sep = analytics.feature_separation(trades)
    assert sep.iloc[0]["feature"] == "ltq_ratio_2_5"
    assert sep.iloc[0]["auc"] > 0.75
    assert sep.iloc[0]["median_win"] > sep.iloc[0]["median_loss"]


def test_noise_feature_sits_near_no_edge(trades):
    sep = analytics.feature_separation(trades).set_index("feature")
    assert abs(sep.loc["momentum_pct", "auc"] - 0.5) < 0.12


def test_constant_feature_is_flagged_not_dropped(trades):
    """
    Silently dropping a constant feature hides the train/serve gap: the tick
    features really are all-zero in history-derived training data, and that is
    a finding, not a nuisance.
    """
    sep = analytics.feature_separation(trades).set_index("feature")
    assert "depth_ratio" in sep.index
    assert sep.loc["depth_ratio", "auc"] == 0.5
    assert "constant" in sep.loc["depth_ratio", "note"]


def test_inverting_a_feature_mirrors_its_auc(trades):
    """
    A feature that predicts in the opposite direction is equally useful - you
    invert the filter - so `separation` must be symmetric about 0.5 while the
    AUC itself flips.
    """
    normal = analytics.feature_separation(trades).set_index("feature")
    trades["f_ltq_ratio_2_5"] = -trades["f_ltq_ratio_2_5"]
    flipped = analytics.feature_separation(trades).set_index("feature")

    assert flipped.loc["ltq_ratio_2_5", "auc"] == pytest.approx(
        1.0 - normal.loc["ltq_ratio_2_5", "auc"], abs=1e-9
    )
    assert flipped.loc["ltq_ratio_2_5", "separation"] == pytest.approx(
        normal.loc["ltq_ratio_2_5", "separation"], abs=1e-9
    )


def test_bare_feature_names_work_as_well_as_prefixed(trades):
    """The store emits f_-prefixed columns; the simulator emits bare ones."""
    bare = trades.rename(columns={c: c[2:] for c in trades.columns if c.startswith("f_")})
    assert (
        analytics.feature_separation(bare).iloc[0]["feature"]
        == analytics.feature_separation(trades).iloc[0]["feature"]
    )


# ==========================================================================
# Buckets
# ==========================================================================
def test_bucket_win_rate_rises_with_the_planted_feature(trades):
    table = analytics.bucket_performance(trades, "ltq_ratio_2_5", bins=5)
    assert len(table) == 5
    assert table["win_rate"].iloc[-1] > table["win_rate"].iloc[0]
    assert table["n"].sum() == len(trades)


def test_thin_buckets_are_dropped(trades):
    """A bucket too small to mean anything must not appear as a finding."""
    table = analytics.bucket_performance(trades, "ltq_ratio_2_5", bins=5, min_trades=1000)
    assert table.empty


def test_bucketing_a_constant_feature_returns_nothing(trades):
    assert analytics.bucket_performance(trades, "depth_ratio").empty


def test_best_filters_shortlists_the_planted_feature(trades):
    shortlist = analytics.best_filters(trades)
    assert shortlist.iloc[0]["feature"] == "ltq_ratio_2_5"
    assert shortlist.iloc[0]["spread"] > 10.0


# ==========================================================================
# Regime breakdowns
# ==========================================================================
def test_hour_and_duration_tables_respect_the_sample_floor(trades):
    hours = analytics.time_of_day_performance(trades, min_trades=5)
    assert not hours.empty
    assert (hours["n"] >= 5).all()

    durations = analytics.duration_performance(trades, min_trades=5)
    assert (durations["n"] >= 5).all()
    # Buckets are the fixed human-readable edges, not quantiles.
    assert set(durations["bucket"]) <= {"<15m", "15-30m", "30-60m", "1-2h", "2-4h", ">4h"}


def test_symbol_table_totals_match_the_trade_table(trades):
    by_symbol = analytics.symbol_performance(trades, min_trades=1)
    assert by_symbol["n"].sum() == len(trades)
    assert by_symbol["total_pnl_pct"].sum() == pytest.approx(trades["net_pnl_pct"].sum())


# ==========================================================================
# Excursions
# ==========================================================================
def test_suggested_stop_is_the_ninetieth_percentile_of_winner_mae(trades):
    report = analytics.excursion_report(trades)
    winners = trades[trades["net_pnl_pct"] > 0]["mae_pct"].abs()

    assert report.n == len(trades)
    assert report.stop_at_p90_winner_mae == pytest.approx(np.percentile(winners, 90))
    # It is quoted as "would leave 90% of winners alone", so it must.
    assert report.winners_stopped_at_that_level == pytest.approx(0.1, abs=0.03)


def test_excursion_report_is_empty_without_mae_columns(trades):
    assert analytics.excursion_report(trades.drop(columns=["mae_pct", "mfe_pct"])).n == 0


# ==========================================================================
# Model diagnostics
# ==========================================================================
def test_perfect_calibration_scores_near_zero_error():
    y = np.array([1] * 20 + [0] * 80 + [1] * 80 + [0] * 20)
    p = np.array([0.2] * 100 + [0.8] * 100)

    report = analytics.calibration(y, p, bins=10)
    assert report.expected_calibration_error < 0.01
    assert report.brier == pytest.approx(((p - y) ** 2).mean())
    assert report.n == 200


def test_calibration_catches_overconfidence():
    """A model that says 0.9 and wins half the time must be visible as such."""
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.9)
    assert analytics.calibration(y, p).expected_calibration_error == pytest.approx(0.4, abs=0.01)


def test_threshold_sweep_takes_fewer_trades_as_it_rises(trades):
    y = (trades["net_pnl_pct"] > 0).astype(int).to_numpy()
    p = np.linspace(0.1, 0.9, len(trades))
    sweep = analytics.threshold_sweep(y, p, trades["net_pnl_pct"].to_numpy())

    assert (sweep["n_taken"].diff().dropna() <= 0).all()
    assert (sweep["coverage"] <= 1.0).all()
    assert analytics.best_threshold(sweep)["total_pnl_pct"] == sweep[sweep["reliable"]][
        "total_pnl_pct"
    ].max()


def test_best_threshold_ignores_unreliable_rows():
    """
    The tail of the sweep is where a spurious optimum hides: one trade at 100%
    must never win the search.
    """
    sweep = pd.DataFrame({
        "threshold": [0.4, 0.5, 0.9],
        "n_taken": [50, 40, 1],
        "coverage": [1.0, 0.8, 0.02],
        "win_rate": [55.0, 58.0, 100.0],
        "avg_pnl_pct": [0.1, 0.12, 5.0],
        "total_pnl_pct": [5.0, 4.8, 5.0],
        "reliable": [True, True, False],
    })
    assert analytics.best_threshold(sweep)["threshold"] == 0.4


def test_strategy_comparison_includes_both_baselines(trades):
    y = (trades["net_pnl_pct"] > 0).astype(int).to_numpy()
    p = np.linspace(0.1, 0.9, len(trades))
    table = analytics.strategy_comparison(y, p, trades["net_pnl_pct"].to_numpy(), threshold=0.55)

    assert len(table) == 3
    assert table.iloc[0]["coverage"] == pytest.approx(1.0)
    assert table.iloc[0]["total_pnl_pct"] == pytest.approx(trades["net_pnl_pct"].sum())
    # Random selection takes exactly as many trades as the model did.
    assert table.iloc[2]["n"] == table.iloc[1]["n"]


# ==========================================================================
# Walk-forward
# ==========================================================================
def test_walk_forward_scores_are_out_of_sample(trades):
    folds, scored = analytics.walk_forward_scores(trades, n_folds=3)

    assert not folds.empty
    assert (folds["train_n"].diff().dropna() > 0).all()      # expanding window
    assert not scored.empty
    assert scored["oos_proba"].between(0.0, 1.0).all()
    # No trade may be scored twice, and none may come from the first block.
    assert scored["entry_time"].is_unique
    assert len(scored) < len(trades)


def test_walk_forward_finds_the_planted_edge(trades):
    folds = analytics.walk_forward(trades, n_folds=3)
    assert folds["auc"].dropna().mean() > 0.7


def test_walk_forward_refuses_a_tiny_dataset():
    folds, scored = analytics.walk_forward_scores(make_trades(n=20))
    assert folds.empty and scored.empty


def test_verdict_is_honest_about_a_coin_flip():
    folds = pd.DataFrame({"fold": [1, 2, 3], "auc": [0.50, 0.51, 0.49]})
    assert "no usable edge" in analytics.walk_forward_verdict(folds)

    unstable = pd.DataFrame({"fold": [1, 2, 3], "auc": [0.80, 0.45, 0.75]})
    assert "unstable" in analytics.walk_forward_verdict(unstable)

    assert "Not enough trades" in analytics.walk_forward_verdict(pd.DataFrame())


# ==========================================================================
# The whole battery
# ==========================================================================
def test_analyse_prefers_walk_forward_scores_over_supplied_ones(trades):
    bundle = analytics.analyse(trades, proba=np.full(len(trades), 0.99))

    assert bundle.n_trades == len(trades)
    assert bundle.score_source.startswith("walk-forward")
    assert not bundle.calibration.table.empty
    assert not bundle.comparison.empty
    assert bundle.verdict


def test_analyse_falls_back_when_there_is_no_walk_forward(trades):
    bundle = analytics.analyse(trades, proba=np.linspace(0.1, 0.9, len(trades)),
                               run_walk_forward=False)
    assert bundle.score_source.startswith("in-sample")
    assert bundle.folds.empty
    assert not bundle.sweep.empty


def test_analyse_works_with_no_model_at_all(trades):
    bundle = analytics.analyse(trades, run_walk_forward=False)
    assert bundle.n_trades == len(trades)
    assert not bundle.separation.empty
    assert bundle.score_source == "none"
    assert bundle.sweep.empty


def test_analyse_on_an_empty_store_returns_an_empty_bundle(empty):
    bundle = analytics.analyse(empty)
    assert bundle.n_trades == 0
    assert bundle.equity.empty
