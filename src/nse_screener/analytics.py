"""
Post-hoc analysis of the trade record.

The screener answers "what looks tradeable right now". This module answers the
harder question the brief actually asks in requirement 22 - *which filters
separate winners from losers* - and the question a reviewer will ask next:
*does the model add anything over taking every signal?*

Three principles run through everything here:

1. RANK-BASED, NOT MEAN-BASED. Trade P&L is heavy-tailed and small samples are
   the norm, so a difference in means is easy to manufacture from one outlier.
   Single-feature ROC AUC - the probability that a randomly chosen winner ranks
   above a randomly chosen loser on that feature alone - is scale-free, robust
   to outliers, and reads directly as "how well does this filter sort trades".

2. EVERY EDGE IS QUOTED WITH ITS SAMPLE SIZE. A 70% win rate over 7 trades is
   noise. Every table here carries `n`, and helpers refuse to report a bucket
   thinner than a floor rather than emitting a confident-looking number.

3. THE BASELINE IS ALWAYS SHOWN. A model that accepts 90% of signals and posts
   the same win rate as "take everything" has added nothing, however good its
   AUC looks. `strategy_comparison` puts the model next to take-everything and
   next to a random selector of the same size, which is the only comparison
   that can embarrass it.

Nothing here mutates its input, and every function degrades to an empty frame
rather than raising when the trade table is empty or missing a column.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, FEATURE_LABELS
from .utils import get_logger

log = get_logger(__name__)

# Below this many trades a bucket statistic is noise dressed as a finding.
MIN_BUCKET_TRADES = 5

PNL_COL = "net_pnl_pct"


# ==========================================================================
# Shared helpers
# ==========================================================================
def _prepare(trades: pd.DataFrame) -> pd.DataFrame:
    """Closed trades only, with a usable P&L column and a 0/1 win label."""
    if trades is None or trades.empty or PNL_COL not in trades.columns:
        return pd.DataFrame()

    df = trades.copy()
    if "exit_time" in df.columns:
        df = df[df["exit_time"].notna()]
    df = df[pd.to_numeric(df[PNL_COL], errors="coerce").notna()]
    if df.empty:
        return df

    df[PNL_COL] = df[PNL_COL].astype(float)
    # Recompute rather than trust a stored label: the label column may have been
    # written under a different cost assumption than the P&L beside it.
    df["win"] = (df[PNL_COL] > 0).astype(int)
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        df = df.sort_values("entry_time")
    return df.reset_index(drop=True)


def feature_column_map(df: pd.DataFrame) -> dict[str, str]:
    """
    Map canonical feature name -> the column that holds it in `df`.

    The store explodes the features JSON into `f_`-prefixed columns; a frame
    built straight from the simulator carries bare names. Accept either.
    """
    out: dict[str, str] = {}
    for name in FEATURE_COLUMNS:
        if f"f_{name}" in df.columns:
            out[name] = f"f_{name}"
        elif name in df.columns:
            out[name] = name
    return out


def _profit_factor(pnl: np.ndarray) -> float:
    gains = float(pnl[pnl > 0].sum())
    losses = float(abs(pnl[pnl <= 0].sum()))
    if losses > 0:
        return gains / losses
    return gains if gains else 0.0


def _group_stats(group: pd.DataFrame) -> dict[str, float]:
    pnl = group[PNL_COL].to_numpy(dtype=float)
    return {
        "n": int(len(group)),
        "win_rate": float(group["win"].mean() * 100.0),
        "avg_pnl_pct": float(pnl.mean()),
        "median_pnl_pct": float(np.median(pnl)),
        "total_pnl_pct": float(pnl.sum()),
        "profit_factor": _profit_factor(pnl),
    }


# ==========================================================================
# 1. Equity and drawdown
# ==========================================================================
def equity_curve(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative P&L with its running peak and the underwater (drawdown) series.

    Equity is a simple sum of percentage returns, not a compounded one: each
    trade is assumed to risk the same notional, which is what the flat
    stop-and-reverse rule in `backtest.py` actually describes. Compounding here
    would imply a position-sizing policy this system does not have.
    """
    df = _prepare(trades)
    if df.empty:
        return pd.DataFrame(columns=["entry_time", PNL_COL, "equity", "peak", "drawdown"])

    equity = df[PNL_COL].cumsum()
    peak = equity.cummax()
    out = pd.DataFrame(
        {
            "entry_time": df.get("entry_time", pd.Series(range(len(df)))),
            "symbol": df.get("symbol", ""),
            PNL_COL: df[PNL_COL],
            "equity": equity,
            "peak": peak,
            # Negative-or-zero, so it plots downward as an underwater curve.
            "drawdown": equity - peak,
        }
    )
    out["trade_no"] = np.arange(1, len(out) + 1)
    return out


# ==========================================================================
# 2. Which features separate winners from losers
# ==========================================================================
def _single_feature_auc(values: np.ndarray, wins: np.ndarray) -> float:
    """
    ROC AUC of one feature used alone as a score.

    Computed from ranks (Mann-Whitney U / n1n2), so ties are handled and the
    result is invariant to any monotone rescaling of the feature. 0.5 is no
    separation; below 0.5 means the feature separates in the opposite direction,
    which is just as usable as a filter - you invert it.
    """
    n_win = int(wins.sum())
    n_loss = int(len(wins) - n_win)
    if n_win == 0 or n_loss == 0:
        return 0.5
    ranks = pd.Series(values).rank().to_numpy()
    u = ranks[wins == 1].sum() - n_win * (n_win + 1) / 2.0
    return float(u / (n_win * n_loss))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided p-value; NaN when scipy is absent or the test is degenerate."""
    try:
        from scipy.stats import mannwhitneyu

        if len(a) < 3 or len(b) < 3:
            return float("nan")
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:  # noqa: BLE001 - scipy missing, or all values identical
        return float("nan")


def feature_separation(trades: pd.DataFrame, features: Sequence[str] | None = None) -> pd.DataFrame:
    """
    Rank every feature by how well it alone sorts winners from losers.

    Returns one row per feature with the winner/loser medians, the single-feature
    AUC, an effect size, and a p-value. `separation` is |AUC - 0.5| and is what
    the table sorts on: it treats a feature that is negatively predictive as
    exactly as interesting as a positively predictive one.

    The p-value is reported without a multiple-comparison correction and with 19
    features under test, so read it as a ranking aid, not as significance. The
    honest reading of a 0.03 here is "worth a second look", not "proven".
    """
    df = _prepare(trades)
    cols = feature_column_map(df) if not df.empty else {}
    if df.empty or not cols:
        return pd.DataFrame(
            columns=["feature", "label", "n_win", "n_loss", "median_win", "median_loss",
                     "auc", "separation", "cohens_d", "p_value"]
        )

    names = list(features) if features else list(cols)
    wins = df["win"].to_numpy()
    rows = []

    for name in names:
        col = cols.get(name)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if np.allclose(values, values[0]):
            # Constant column. On history-derived training data the tick
            # features are all zero here - see the train/serve note in the
            # README - and reporting an AUC of 0.5 for them is more honest than
            # dropping them silently.
            rows.append({
                "feature": name, "label": FEATURE_LABELS.get(name, name),
                "n_win": int(wins.sum()), "n_loss": int(len(wins) - wins.sum()),
                "median_win": float(values[0]), "median_loss": float(values[0]),
                "auc": 0.5, "separation": 0.0, "cohens_d": 0.0, "p_value": float("nan"),
                "note": "constant - no information in this dataset",
            })
            continue

        win_vals, loss_vals = values[wins == 1], values[wins == 0]
        auc = _single_feature_auc(values, wins)
        rows.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "n_win": int(len(win_vals)),
            "n_loss": int(len(loss_vals)),
            "median_win": float(np.median(win_vals)) if len(win_vals) else float("nan"),
            "median_loss": float(np.median(loss_vals)) if len(loss_vals) else float("nan"),
            "auc": auc,
            "separation": abs(auc - 0.5),
            "cohens_d": _cohens_d(win_vals, loss_vals),
            "p_value": _mann_whitney_p(win_vals, loss_vals),
            "note": "",
        })

    return (
        pd.DataFrame(rows)
        .sort_values("separation", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================================
# 3. Bucketed performance - the actual "filter" view
# ==========================================================================
def bucket_performance(
    trades: pd.DataFrame,
    feature: str,
    bins: int = 5,
    min_trades: int = MIN_BUCKET_TRADES,
) -> pd.DataFrame:
    """
    Win rate and expectancy across quantile buckets of one feature.

    Quantile buckets rather than equal-width ones: most of these features are
    ratios that pile up near 1.0, so equal-width bins would put 95% of trades in
    one bar and produce four buckets of noise. Buckets thinner than `min_trades`
    are dropped, with a log line naming them.

    A monotone win-rate gradient across buckets is the thing worth acting on -
    that is a tradeable filter. A single strong bucket surrounded by noise is
    usually an artefact of where the quantile edges happened to fall.
    """
    df = _prepare(trades)
    cols = feature_column_map(df) if not df.empty else {}
    col = cols.get(feature)
    empty = pd.DataFrame(columns=["bucket", "n", "win_rate", "avg_pnl_pct",
                                  "median_pnl_pct", "total_pnl_pct", "profit_factor"])
    if df.empty or col is None:
        return empty

    values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if values.nunique() < 2:
        return empty

    try:
        df["bucket"] = pd.qcut(values, q=bins, duplicates="drop", precision=3)
    except ValueError:
        return empty

    rows = []
    for bucket, group in df.groupby("bucket", observed=True, sort=True):
        stats = _group_stats(group)
        if stats["n"] < min_trades:
            log.debug("bucket %s of %s dropped: only %d trades", bucket, feature, stats["n"])
            continue
        rows.append({"bucket": str(bucket), "low": float(bucket.left),
                     "high": float(bucket.right), **stats})

    return pd.DataFrame(rows).reset_index(drop=True)


def best_filters(trades: pd.DataFrame, bins: int = 4, min_trades: int = MIN_BUCKET_TRADES) -> pd.DataFrame:
    """
    Sweep every feature and report the win-rate spread its buckets produce.

    `spread` is the best bucket's win rate minus the worst one's. It is a blunt
    instrument - it does not care whether the gradient is monotone - so it is a
    shortlist generator, not a conclusion. Inspect the winners with
    `bucket_performance` before believing any of them.
    """
    df = _prepare(trades)
    if df.empty:
        return pd.DataFrame(columns=["feature", "label", "spread", "best_bucket",
                                     "best_win_rate", "worst_bucket", "worst_win_rate", "n_buckets"])

    baseline = float(df["win"].mean() * 100.0)
    rows = []
    for name in feature_column_map(df):
        table = bucket_performance(df, name, bins=bins, min_trades=min_trades)
        if len(table) < 2:
            continue
        best = table.loc[table["win_rate"].idxmax()]
        worst = table.loc[table["win_rate"].idxmin()]
        rows.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "spread": float(best["win_rate"] - worst["win_rate"]),
            "best_bucket": best["bucket"],
            "best_win_rate": float(best["win_rate"]),
            "best_n": int(best["n"]),
            "worst_bucket": worst["bucket"],
            "worst_win_rate": float(worst["win_rate"]),
            "worst_n": int(worst["n"]),
            "baseline_win_rate": baseline,
            "n_buckets": int(len(table)),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spread", ascending=False).reset_index(drop=True)


# ==========================================================================
# 4. When and where the edge lives
# ==========================================================================
def time_of_day_performance(trades: pd.DataFrame, min_trades: int = MIN_BUCKET_TRADES) -> pd.DataFrame:
    """
    Performance by hour of the trading session.

    Intraday mean reversion behaves differently at 09:20 than at 14:30 - the
    opening auction hangover and the pre-close position squaring are genuinely
    different regimes - so an hour-of-day breakdown is the cheapest regime split
    available.
    """
    df = _prepare(trades)
    if df.empty or "entry_time" not in df.columns:
        return pd.DataFrame(columns=["hour", "n", "win_rate", "avg_pnl_pct", "total_pnl_pct"])

    df = df[df["entry_time"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["hour", "n", "win_rate", "avg_pnl_pct", "total_pnl_pct"])

    df["hour"] = df["entry_time"].dt.hour
    rows = [
        {"hour": int(hour), "label": f"{int(hour):02d}:00", **_group_stats(group)}
        for hour, group in df.groupby("hour", sort=True)
        if len(group) >= min_trades
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def duration_performance(trades: pd.DataFrame, min_trades: int = MIN_BUCKET_TRADES) -> pd.DataFrame:
    """
    Performance by how long the trade was held.

    Fixed, human-meaningful edges rather than quantiles: "under 15 minutes" is a
    category a trader can act on, whereas "the second quintile of hold time" is
    not. A stop-and-reverse system that only makes money on multi-hour holds is
    telling you the fast line is too fast.
    """
    df = _prepare(trades)
    if df.empty or "duration_minutes" not in df.columns:
        return pd.DataFrame(columns=["bucket", "n", "win_rate", "avg_pnl_pct", "total_pnl_pct"])

    edges = [0, 15, 30, 60, 120, 240, np.inf]
    labels = ["<15m", "15-30m", "30-60m", "1-2h", "2-4h", ">4h"]
    df = df[pd.to_numeric(df["duration_minutes"], errors="coerce").notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["bucket", "n", "win_rate", "avg_pnl_pct", "total_pnl_pct"])

    df["bucket"] = pd.cut(df["duration_minutes"].astype(float), bins=edges,
                          labels=labels, right=False)
    rows = [
        {"bucket": str(bucket), **_group_stats(group)}
        for bucket, group in df.groupby("bucket", observed=True, sort=True)
        if len(group) >= min_trades
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def symbol_performance(trades: pd.DataFrame, min_trades: int = MIN_BUCKET_TRADES) -> pd.DataFrame:
    """Per-symbol breakdown, so one runaway name cannot masquerade as an edge."""
    df = _prepare(trades)
    if df.empty or "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "n", "win_rate", "avg_pnl_pct", "total_pnl_pct"])

    rows = [
        {"symbol": str(symbol), **_group_stats(group)}
        for symbol, group in df.groupby("symbol", sort=False)
        if len(group) >= min_trades
    ]
    if not rows:
        return pd.DataFrame(columns=["symbol", "n", "win_rate", "avg_pnl_pct", "total_pnl_pct"])
    return pd.DataFrame(rows).sort_values("total_pnl_pct", ascending=False).reset_index(drop=True)


# ==========================================================================
# 5. Excursions - what the exit rule is costing
# ==========================================================================
@dataclass(frozen=True)
class ExcursionReport:
    """
    How much of the available move each trade actually captured.

    `capture_ratio` is realised P&L divided by the best unrealised gain the
    trade ever showed. Well below 1.0 on winners means the exit rule - wait for
    the reverse crossover - is systematically giving profit back, which is an
    argument for a trailing exit rather than for a better entry filter.
    """

    n: int
    avg_mae_pct: float
    avg_mfe_pct: float
    avg_mae_winners: float
    avg_mae_losers: float
    capture_ratio: float
    edge_ratio: float
    stop_at_p90_winner_mae: float
    winners_stopped_at_that_level: float
    losers_stopped_at_that_level: float

    def summary(self) -> str:
        return (
            f"{self.n} trades | avg MAE {self.avg_mae_pct:.2f}% | avg MFE {self.avg_mfe_pct:.2f}% | "
            f"capture {self.capture_ratio:.1%} | edge ratio {self.edge_ratio:.2f} | "
            f"a {self.stop_at_p90_winner_mae:.2f}% stop would cut "
            f"{self.losers_stopped_at_that_level:.0%} of losers and "
            f"{self.winners_stopped_at_that_level:.0%} of winners"
        )

    @classmethod
    def empty(cls) -> "ExcursionReport":
        return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def excursion_report(trades: pd.DataFrame) -> ExcursionReport:
    """
    Turn the MAE/MFE columns into an exit-rule critique and a stop suggestion.

    The suggested stop is the 90th percentile of winners' MAE: a level that
    would have left 90% of the winning trades alone. Quoting how many losers it
    would also have cut makes the trade-off explicit instead of presenting a
    stop level as free money.
    """
    df = _prepare(trades)
    if df.empty or not {"mae_pct", "mfe_pct"} <= set(df.columns):
        return ExcursionReport.empty()

    mae = pd.to_numeric(df["mae_pct"], errors="coerce").fillna(0.0).abs()
    mfe = pd.to_numeric(df["mfe_pct"], errors="coerce").fillna(0.0).abs()
    wins = df["win"] == 1
    if not wins.any() or mfe.sum() == 0:
        return ExcursionReport.empty()

    winner_mae = mae[wins]
    stop = float(np.percentile(winner_mae, 90)) if len(winner_mae) else 0.0
    return ExcursionReport(
        n=int(len(df)),
        avg_mae_pct=float(mae.mean()),
        avg_mfe_pct=float(mfe.mean()),
        avg_mae_winners=float(winner_mae.mean()) if len(winner_mae) else 0.0,
        avg_mae_losers=float(mae[~wins].mean()) if (~wins).any() else 0.0,
        # Winners only: dividing a loss by an unrealised gain is not a "capture".
        capture_ratio=float(df.loc[wins, PNL_COL].sum() / mfe[wins].sum()) if mfe[wins].sum() else 0.0,
        edge_ratio=float(mfe.mean() / mae.mean()) if mae.mean() else 0.0,
        stop_at_p90_winner_mae=stop,
        winners_stopped_at_that_level=float((winner_mae > stop).mean()) if len(winner_mae) else 0.0,
        losers_stopped_at_that_level=float((mae[~wins] > stop).mean()) if (~wins).any() else 0.0,
    )


# ==========================================================================
# 6. Is the model any good, and is 0.55 the right threshold
# ==========================================================================
@dataclass
class CalibrationReport:
    """Predicted probability against realised frequency."""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    brier: float = float("nan")
    expected_calibration_error: float = float("nan")
    n: int = 0

    def summary(self) -> str:
        if not self.n:
            return "no scored trades"
        return (f"n={self.n} | Brier {self.brier:.4f} | "
                f"calibration error {self.expected_calibration_error:.3f}")


def calibration(y_true: Sequence[int], proba: Sequence[float], bins: int = 10) -> CalibrationReport:
    """
    Does a 0.70 from this model actually win 70% of the time?

    ROC AUC only measures ranking, so a model can rank perfectly and still be
    numerically useless: if every score sits between 0.49 and 0.52, an ACCEPT
    threshold of 0.55 never fires. Calibration is what makes the threshold - and
    any expected-value sizing built on it - meaningful, which is why it belongs
    next to the AUC rather than buried.

    Brier score is the mean squared error of the probabilities (lower is better;
    0.25 is what you get by always saying 0.5). ECE is the average gap between
    predicted and realised frequency, weighted by bin population.
    """
    y = pd.Series(list(y_true)).astype(float)
    p = pd.Series(list(proba)).astype(float)
    mask = y.notna() & p.notna()
    y, p = y[mask].reset_index(drop=True), p[mask].reset_index(drop=True)
    if y.empty:
        return CalibrationReport()

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        rows.append({
            "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
            "bin_mid": float((edges[b] + edges[b + 1]) / 2),
            "n": int(sel.sum()),
            "predicted": float(p[sel].mean()),
            "actual": float(y[sel].mean()),
        })

    table = pd.DataFrame(rows)
    ece = (
        float((table["n"] / len(y) * (table["predicted"] - table["actual"]).abs()).sum())
        if not table.empty else float("nan")
    )
    return CalibrationReport(
        table=table,
        brier=float(((p - y) ** 2).mean()),
        expected_calibration_error=ece,
        n=int(len(y)),
    )


def threshold_sweep(
    y_true: Sequence[int],
    proba: Sequence[float],
    pnl: Sequence[float],
    thresholds: Sequence[float] | None = None,
    min_taken: int = MIN_BUCKET_TRADES,
) -> pd.DataFrame:
    """
    What every possible ACCEPT threshold would have earned.

    `ACCEPT_THRESHOLD` ships at 0.55, which is a convention, not a finding. This
    sweeps the whole range and reports coverage, win rate and total P&L at each
    level, so the default can be defended or replaced with evidence.

    Total P&L is the column to optimise, not win rate: a threshold that takes
    three trades at 100% is worse than one that takes eighty at 55%. Rows with
    fewer than `min_taken` accepted trades are kept but flagged, because the
    tail of the sweep is exactly where a spurious optimum likes to hide.
    """
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(proba), dtype=float)
    r = np.asarray(list(pnl), dtype=float)
    if not len(y) or not (len(y) == len(p) == len(r)):
        return pd.DataFrame(columns=["threshold", "n_taken", "coverage", "win_rate",
                                     "avg_pnl_pct", "total_pnl_pct", "reliable"])

    grid = list(thresholds) if thresholds is not None else np.round(np.arange(0.30, 0.81, 0.025), 3)
    rows = []
    for t in grid:
        sel = p >= t
        n = int(sel.sum())
        rows.append({
            "threshold": float(t),
            "n_taken": n,
            "coverage": float(n / len(y)),
            "win_rate": float(y[sel].mean() * 100.0) if n else 0.0,
            "avg_pnl_pct": float(r[sel].mean()) if n else 0.0,
            "total_pnl_pct": float(r[sel].sum()) if n else 0.0,
            "reliable": bool(n >= min_taken),
        })
    return pd.DataFrame(rows)


def best_threshold(sweep: pd.DataFrame) -> dict[str, float]:
    """Pick the total-P&L-maximising threshold among the rows with enough trades."""
    if sweep is None or sweep.empty:
        return {}
    usable = sweep[sweep["reliable"]] if "reliable" in sweep.columns else sweep
    if usable.empty:
        return {}
    row = usable.loc[usable["total_pnl_pct"].idxmax()]
    return {k: float(row[k]) for k in
            ("threshold", "n_taken", "coverage", "win_rate", "avg_pnl_pct", "total_pnl_pct")}


def strategy_comparison(
    y_true: Sequence[int],
    proba: Sequence[float],
    pnl: Sequence[float],
    threshold: float = 0.55,
    seed: int = 42,
) -> pd.DataFrame:
    """
    The model against the two baselines that can embarrass it.

    * "Take every signal" - what the crossover rule alone earns. If the model
      cannot beat this, the ML layer is decoration.
    * "Random, same count" - accepts the same NUMBER of trades at random, so the
      comparison is not confounded by simply trading less. Averaged over 200
      draws to keep one lucky sample from deciding the verdict.

    Same trade population, same costs, three selection rules.
    """
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(proba), dtype=float)
    r = np.asarray(list(pnl), dtype=float)
    if not len(y) or not (len(y) == len(p) == len(r)):
        return pd.DataFrame()

    def row(name: str, sel: np.ndarray) -> dict[str, float]:
        n = int(sel.sum())
        taken = r[sel]
        return {
            "strategy": name,
            "n": n,
            "coverage": float(n / len(y)),
            "win_rate": float(y[sel].mean() * 100.0) if n else 0.0,
            "avg_pnl_pct": float(taken.mean()) if n else 0.0,
            "total_pnl_pct": float(taken.sum()) if n else 0.0,
            "profit_factor": _profit_factor(taken) if n else 0.0,
        }

    accepted = p >= threshold
    n_accepted = int(accepted.sum())

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(200):
        pick = np.zeros(len(y), dtype=bool)
        if n_accepted:
            pick[rng.choice(len(y), size=n_accepted, replace=False)] = True
        draws.append(row("random", pick))
    random_avg = {
        "strategy": f"Random ({n_accepted} trades, 200-draw mean)",
        "n": n_accepted,
        "coverage": float(n_accepted / len(y)),
        **{k: float(np.mean([d[k] for d in draws]))
           for k in ("win_rate", "avg_pnl_pct", "total_pnl_pct", "profit_factor")},
    }

    return pd.DataFrame([
        row("Take every signal", np.ones(len(y), dtype=bool)),
        row(f"Model ACCEPT (p >= {threshold:.2f})", accepted),
        random_avg,
    ])


# ==========================================================================
# 7. Walk-forward validation
# ==========================================================================
def walk_forward_scores(
    dataset: pd.DataFrame, n_folds: int = 5, label_col: str = "label"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Expanding-window validation across time, keeping the out-of-sample scores.

    A single chronological split - what `TradeClassifier.train` reports - gives
    one number that depends entirely on which regime happened to land in the
    tail. Walk-forward refits on everything before each fold and tests on the
    fold, so you see whether the edge is stable or whether one lucky window is
    carrying the headline AUC. The spread across folds is the interesting part;
    a mean of 0.60 made of 0.75 and 0.45 is not a 0.60 model.

    Returns `(folds, scored)`. `scored` matters as much as the fold table: every
    trade in it carries a probability produced by a model that never saw it,
    which is the only defensible input to a calibration curve or a threshold
    sweep. Scoring the training set with the shipped model instead would put an
    optimistic bend in both, and the threshold picked from it would not survive
    live data.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    from .ml.model import TradeClassifier

    cols = ["fold", "train_n", "test_n", "test_positive_rate", "auc", "accuracy"]
    if dataset is None or dataset.empty or label_col not in dataset.columns:
        return pd.DataFrame(columns=cols), pd.DataFrame()

    df = dataset.copy()
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        df = df.sort_values("entry_time")
    df = df.reset_index(drop=True)

    n = len(df)
    if n < (n_folds + 1) * 10:
        n_folds = max(2, n // 20)
    if n_folds < 2 or n < 40:
        log.warning("walk-forward needs ~40+ trades; got %d", n)
        return pd.DataFrame(columns=cols), pd.DataFrame()

    # Fold k trains on everything before it and tests on the next slice. The
    # first block is training-only, so there are n_folds test windows.
    edges = np.linspace(0, n, n_folds + 2, dtype=int)
    rows: list[dict] = []
    scored: list[pd.DataFrame] = []

    for k in range(1, len(edges) - 1):
        train = df.iloc[: edges[k]]
        test = df.iloc[edges[k]: edges[k + 1]]
        if test.empty or train[label_col].nunique() < 2:
            continue

        clf = TradeClassifier()
        try:
            X_train = clf._extract_features(train)
            X_test = clf._extract_features(test)
            model = clf._build_estimator()
            model.fit(X_train, train[label_col].astype(int).to_numpy())
            y_test = test[label_col].astype(int).to_numpy()
            proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan")
            acc = float(accuracy_score(y_test, (proba >= 0.5).astype(int)))
        except Exception as exc:  # noqa: BLE001
            log.error("walk-forward fold %d failed: %s", k, exc)
            continue

        block = test.copy()
        block["oos_proba"] = proba
        block["fold"] = k
        scored.append(block)

        rows.append({
            "fold": k,
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "test_positive_rate": float(y_test.mean()),
            "auc": auc,
            "accuracy": acc,
        })

    folds = pd.DataFrame(rows, columns=cols)
    out = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    return folds, out


def walk_forward(dataset: pd.DataFrame, n_folds: int = 5, label_col: str = "label") -> pd.DataFrame:
    """Fold-by-fold table only. See `walk_forward_scores` for the scores too."""
    return walk_forward_scores(dataset, n_folds=n_folds, label_col=label_col)[0]


def walk_forward_verdict(folds: pd.DataFrame) -> str:
    """One sentence a reviewer can read without opening the table."""
    if folds is None or folds.empty or folds["auc"].notna().sum() == 0:
        return "Not enough trades to walk forward - treat the single-split AUC as provisional."

    auc = folds["auc"].dropna()
    mean, spread = float(auc.mean()), float(auc.std(ddof=0))
    if mean < 0.53:
        verdict = "no usable edge - the model is not separating winners from losers out of sample"
    elif spread > 0.10:
        verdict = "unstable - the mean hides fold-to-fold swings, so one regime is carrying it"
    elif mean < 0.60:
        verdict = "modest but consistent - usable as a filter, not as a green light"
    else:
        verdict = "consistent discrimination across folds"
    return (f"Walk-forward AUC {mean:.3f} +/- {spread:.3f} over {len(auc)} folds "
            f"(range {auc.min():.3f}-{auc.max():.3f}): {verdict}.")


# ==========================================================================
# 8. One call that runs everything
# ==========================================================================
@dataclass
class AnalysisBundle:
    """Everything the dashboard tab and the HTML report both need."""

    n_trades: int = 0
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    separation: pd.DataFrame = field(default_factory=pd.DataFrame)
    filters: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_hour: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_duration: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)
    excursions: ExcursionReport = field(default_factory=ExcursionReport.empty)
    calibration: CalibrationReport = field(default_factory=CalibrationReport)
    sweep: pd.DataFrame = field(default_factory=pd.DataFrame)
    best: dict[str, float] = field(default_factory=dict)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)
    folds: pd.DataFrame = field(default_factory=pd.DataFrame)
    verdict: str = ""
    # Where the probabilities behind calibration/sweep/comparison came from.
    score_source: str = "none"


def analyse(
    trades: pd.DataFrame,
    proba: Sequence[float] | None = None,
    threshold: float = 0.55,
    run_walk_forward: bool = True,
) -> AnalysisBundle:
    """
    Run the whole battery over one trade table.

    Model diagnostics prefer WALK-FORWARD scores over any `proba` passed in.
    A supplied probability almost always comes from the shipped model scoring
    trades it was fitted on, and an in-sample calibration curve looks better
    than the model is - which would make the threshold sweep recommend a
    threshold that cannot be reproduced live. `proba` is used only as a fallback
    when there are too few trades to walk forward, and `score_source` records
    which of the two the numbers came from so the caller can label them.
    """
    df = _prepare(trades)
    bundle = AnalysisBundle(n_trades=int(len(df)))
    if df.empty:
        return bundle

    bundle.equity = equity_curve(df)
    bundle.separation = feature_separation(df)
    bundle.filters = best_filters(df)
    bundle.by_hour = time_of_day_performance(df)
    bundle.by_duration = duration_performance(df)
    bundle.by_symbol = symbol_performance(df)
    bundle.excursions = excursion_report(df)

    scored = pd.DataFrame()
    if run_walk_forward and "label" in df.columns:
        bundle.folds, scored = walk_forward_scores(df)
        bundle.verdict = walk_forward_verdict(bundle.folds)

    if not scored.empty:
        y = (scored[PNL_COL] > 0).astype(int).to_numpy()
        p = scored["oos_proba"].to_numpy()
        pnl = scored[PNL_COL].to_numpy()
        bundle.score_source = "walk-forward (out of sample)"
    elif proba is not None and len(proba) == len(df):
        y = df["win"].to_numpy()
        p = np.asarray(list(proba), dtype=float)
        pnl = df[PNL_COL].to_numpy()
        bundle.score_source = "in-sample (too few trades to walk forward)"
    else:
        return bundle

    bundle.calibration = calibration(y, p)
    bundle.sweep = threshold_sweep(y, p, pnl)
    bundle.best = best_threshold(bundle.sweep)
    bundle.comparison = strategy_comparison(y, p, pnl, threshold=threshold)
    return bundle
