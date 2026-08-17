"""
Chart construction for the dashboard.

Built on Altair rather than Plotly or Matplotlib, for three reasons that all
point the same way:

  * Altair ships with Streamlit already. Adding Plotly would mean a new runtime
    dependency, a bigger frozen bundle, and another PyInstaller hidden-import
    to chase - for charts a declarative grammar draws just as well.
  * Layered, declarative specs. A price chart here is literally "close line +
    two SMMA lines + crossover points", composed with `+`, which is far easier
    to review than imperative axis plumbing.
  * The Vega-Lite output is interactive by default: pan, zoom and hover come
    free, and a reviewer scrubbing along a crossover to read the exact values
    is the whole point of putting a chart on the screen.

Every function here is pure - frame in, chart out - so the panels stay thin and
the charts can be rendered from a script as easily as from Streamlit.

Colour is used consistently across the whole app and never as the only signal:
green/red carry win/loss, but every chart that uses them also labels the axis.
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

# One palette, used everywhere, so "green" means the same thing on every tab.
UP = "#1d6b3d"
DOWN = "#a52222"
NEUTRAL = "#6b7684"
ACCENT = "#2b6cb0"
FAST = "#e08a1e"
SLOW = "#7048a8"

_EMPTY = pd.DataFrame({"msg": ["no data yet"]})


def _placeholder(message: str) -> alt.Chart:
    """A chart-shaped object for the empty case, so callers never branch."""
    return (
        alt.Chart(pd.DataFrame({"msg": [message]}))
        .mark_text(size=13, color=NEUTRAL)
        .encode(text="msg:N")
        .properties(height=120)
    )


# ==========================================================================
# Price, indicators and crossovers
# ==========================================================================
def price_chart(
    bars: pd.DataFrame,
    signals: pd.DataFrame | None = None,
    symbol: str = "",
    height: int = 380,
) -> alt.Chart:
    """
    Close price with both SMMA lines and every crossover marked.

    This is the chart the whole assignment is about: the fast line crossing the
    slow one IS the signal, and seeing them together is the only way to judge
    whether a crossover happened in a trend or in chop. BUY and SELL markers sit
    on the price line at the exact bar the signal fired, so a reviewer can check
    that the marker really is at a crossing and not one bar late - the classic
    repainting bug this system is built to avoid.
    """
    if bars is None or bars.empty:
        return _placeholder("No bars stored yet - run the engine or backfill history.")

    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return _placeholder("No usable bar timestamps.")

    series = {"close": "Close", "smma_fast": "SMMA 20", "smma_slow": "SMMA 120"}
    present = [c for c in series if c in df.columns]
    long = (
        df[["timestamp", *present]]
        .melt("timestamp", var_name="series", value_name="price")
        .dropna(subset=["price"])
    )
    long["series"] = long["series"].map(series)

    base = alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b %H:%M")),
        y=alt.Y("price:Q", title="Price (Rs)", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "series:N",
            title=None,
            scale=alt.Scale(
                domain=["Close", "SMMA 20", "SMMA 120"],
                range=[NEUTRAL, FAST, SLOW],
            ),
            legend=alt.Legend(orient="top"),
        ),
        strokeDash=alt.condition(
            alt.datum.series == "Close", alt.value([1, 0]), alt.value([1, 0])
        ),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Time", format="%d %b %H:%M"),
            alt.Tooltip("series:N", title="Series"),
            alt.Tooltip("price:Q", title="Value", format=".2f"),
        ],
    )

    layers = [base]

    if signals is not None and not signals.empty:
        sig = signals.copy()
        if symbol and "symbol" in sig.columns:
            sig = sig[sig["symbol"] == symbol]
        time_col = "timestamp" if "timestamp" in sig.columns else "signal_time"
        if not sig.empty and time_col in sig.columns:
            sig[time_col] = pd.to_datetime(sig[time_col], errors="coerce")
            sig = sig.dropna(subset=[time_col])
            # Only mark signals inside the plotted window, otherwise a stale
            # signal from an earlier session stretches the x-axis flat.
            sig = sig[(sig[time_col] >= df["timestamp"].min())
                      & (sig[time_col] <= df["timestamp"].max())]
        if not sig.empty:
            markers = (
                alt.Chart(sig)
                .mark_point(size=110, filled=True, opacity=0.95)
                .encode(
                    x=alt.X(f"{time_col}:T"),
                    y=alt.Y("ltp:Q"),
                    shape=alt.Shape(
                        "side:N",
                        scale=alt.Scale(domain=["BUY", "SELL"],
                                        range=["triangle-up", "triangle-down"]),
                        legend=alt.Legend(orient="top", title=None),
                    ),
                    color=alt.Color(
                        "side:N",
                        scale=alt.Scale(domain=["BUY", "SELL"], range=[UP, DOWN]),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip(f"{time_col}:T", title="Signal", format="%d %b %H:%M"),
                        alt.Tooltip("side:N", title="Side"),
                        alt.Tooltip("ltp:Q", title="LTP", format=".2f"),
                    ],
                )
            )
            layers.append(markers)

    title = f"{symbol} - price, SMMA 20/120 and crossovers" if symbol else "Price and SMMA"
    return (
        alt.layer(*layers)
        .properties(height=height, title=title)
        .interactive(bind_y=False)
    )


def spread_chart(bars: pd.DataFrame, height: int = 150) -> alt.Chart:
    """
    SMMA20 minus SMMA120, the quantity whose sign change IS the crossover.

    Plotted underneath the price chart because a crossover is much easier to see
    as a zero crossing than as two lines touching, and the depth of the excursion
    either side shows how convincing the trend was.
    """
    if bars is None or bars.empty or not {"smma_fast", "smma_slow"} <= set(bars.columns):
        return _placeholder("SMMA columns not stored for this symbol.")

    df = bars.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["spread"] = df["smma_fast"] - df["smma_slow"]
    df = df.dropna(subset=["timestamp", "spread"])
    if df.empty:
        return _placeholder("Not enough bars for an SMMA spread yet.")

    area = alt.Chart(df).mark_area(opacity=0.65).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b %H:%M")),
        y=alt.Y("spread:Q", title="SMMA20 - SMMA120"),
        color=alt.condition(alt.datum.spread > 0, alt.value(UP), alt.value(DOWN)),
        tooltip=[alt.Tooltip("timestamp:T", format="%d %b %H:%M"),
                 alt.Tooltip("spread:Q", format=".3f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=NEUTRAL, strokeDash=[4, 3]).encode(y="y:Q")
    return (area + zero).properties(height=height)


# ==========================================================================
# Performance
# ==========================================================================
def equity_chart(equity: pd.DataFrame, height: int = 260) -> alt.Chart:
    """Cumulative P&L with the peak it is measured against."""
    if equity is None or equity.empty:
        return _placeholder("No closed trades yet.")

    df = equity.copy()
    x = alt.X("entry_time:T", title=None, axis=alt.Axis(format="%d %b %H:%M"))

    band = alt.Chart(df).mark_area(opacity=0.18, color=DOWN).encode(
        x=x, y=alt.Y("equity:Q", title="Cumulative net P&L (%)"), y2="peak:Q",
    )
    line = alt.Chart(df).mark_line(strokeWidth=2, color=ACCENT).encode(
        x=x,
        y=alt.Y("equity:Q", title="Cumulative net P&L (%)"),
        tooltip=[alt.Tooltip("entry_time:T", title="Entry", format="%d %b %H:%M"),
                 alt.Tooltip("symbol:N", title="Symbol"),
                 alt.Tooltip("net_pnl_pct:Q", title="Trade", format="+.3f"),
                 alt.Tooltip("equity:Q", title="Equity", format="+.2f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(
        color=NEUTRAL, strokeDash=[4, 3]).encode(y="y:Q")
    # The shaded band between equity and its running peak IS the drawdown, so
    # depth and duration underwater are both visible without a second axis.
    return (band + line + zero).properties(height=height, title="Equity curve and drawdown")


def drawdown_chart(equity: pd.DataFrame, height: int = 140) -> alt.Chart:
    if equity is None or equity.empty:
        return _placeholder("No closed trades yet.")
    return (
        alt.Chart(equity)
        .mark_area(color=DOWN, opacity=0.6)
        .encode(
            x=alt.X("entry_time:T", title=None, axis=alt.Axis(format="%d %b %H:%M")),
            y=alt.Y("drawdown:Q", title="Underwater (%)"),
            tooltip=[alt.Tooltip("entry_time:T", format="%d %b %H:%M"),
                     alt.Tooltip("drawdown:Q", format=".2f")],
        )
        .properties(height=height)
    )


def pnl_distribution_chart(trades: pd.DataFrame, height: int = 220) -> alt.Chart:
    """
    Histogram of net P&L per trade.

    Worth its space because the shape decides how to read the average: a
    positive mean built from one huge winner and forty small losers is a
    different system from one with a symmetric spread, and no headline statistic
    distinguishes them.
    """
    if trades is None or trades.empty or "net_pnl_pct" not in trades.columns:
        return _placeholder("No closed trades yet.")

    df = trades.dropna(subset=["net_pnl_pct"]).copy()
    if df.empty:
        return _placeholder("No closed trades yet.")
    df["outcome"] = np.where(df["net_pnl_pct"] > 0, "Win", "Loss")

    hist = alt.Chart(df).mark_bar(opacity=0.85).encode(
        x=alt.X("net_pnl_pct:Q", bin=alt.Bin(maxbins=40), title="Net P&L per trade (%)"),
        y=alt.Y("count():Q", title="Trades"),
        color=alt.Color("outcome:N", title=None,
                        scale=alt.Scale(domain=["Win", "Loss"], range=[UP, DOWN]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("count():Q", title="Trades")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        color=NEUTRAL, strokeDash=[4, 3]).encode(x="x:Q")
    return (hist + zero).properties(height=height, title="Distribution of trade outcomes")


def mae_mfe_chart(trades: pd.DataFrame, height: int = 300) -> alt.Chart:
    """
    Every trade as (worst excursion, best excursion), coloured by outcome.

    Points far right and low are trades that showed a good profit and were still
    closed for a loss - each one is evidence for a trailing exit. Points hugging
    the left edge never went against the position at all, and a stop placed
    beyond that cluster would be free.
    """
    need = {"mae_pct", "mfe_pct", "net_pnl_pct"}
    if trades is None or trades.empty or not need <= set(trades.columns):
        return _placeholder("MAE/MFE not recorded for these trades.")

    df = trades.dropna(subset=list(need)).copy()
    if df.empty:
        return _placeholder("MAE/MFE not recorded for these trades.")
    df["outcome"] = np.where(df["net_pnl_pct"] > 0, "Win", "Loss")
    df["mae_pct"] = df["mae_pct"].abs()
    df["mfe_pct"] = df["mfe_pct"].abs()

    points = alt.Chart(df).mark_circle(size=70, opacity=0.65).encode(
        x=alt.X("mae_pct:Q", title="Maximum adverse excursion (%)"),
        y=alt.Y("mfe_pct:Q", title="Maximum favourable excursion (%)"),
        color=alt.Color("outcome:N", title=None,
                        scale=alt.Scale(domain=["Win", "Loss"], range=[UP, DOWN]),
                        legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("symbol:N"), alt.Tooltip("side:N"),
                 alt.Tooltip("mae_pct:Q", title="MAE %", format=".2f"),
                 alt.Tooltip("mfe_pct:Q", title="MFE %", format=".2f"),
                 alt.Tooltip("net_pnl_pct:Q", title="Net %", format="+.3f")],
    )
    limit = float(max(df["mae_pct"].max(), df["mfe_pct"].max()))
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0.0, limit], "y": [0.0, limit]}))
        .mark_line(color=NEUTRAL, strokeDash=[4, 3])
        .encode(x="x:Q", y="y:Q")
    )
    return (points + diagonal).properties(
        height=height, title="Excursions: what each trade risked against what it offered"
    )


# ==========================================================================
# Feature analysis
# ==========================================================================
def separation_chart(separation: pd.DataFrame, top: int = 12, height: int = 340) -> alt.Chart:
    """
    Ranked single-feature separation, drawn as distance from the 0.5 no-edge line.

    Bars are signed on purpose: a feature with an AUC of 0.40 is exactly as
    useful as one at 0.60 - you just invert the filter - and a chart of absolute
    values would hide the direction you need in order to act on it.
    """
    if separation is None or separation.empty:
        return _placeholder("Not enough trades to measure feature separation.")

    df = separation.head(top).copy()
    df["edge"] = df["auc"] - 0.5
    df["direction"] = np.where(df["edge"] >= 0, "Higher favours winners", "Lower favours winners")

    bars = alt.Chart(df).mark_bar().encode(
        x=alt.X("edge:Q", title="Single-feature AUC minus 0.5 (0 = no separation)"),
        y=alt.Y("label:N", sort="-x", title=None),
        color=alt.Color(
            "direction:N", title=None,
            scale=alt.Scale(domain=["Higher favours winners", "Lower favours winners"],
                            range=[UP, DOWN]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[alt.Tooltip("label:N", title="Feature"),
                 alt.Tooltip("auc:Q", title="AUC", format=".3f"),
                 alt.Tooltip("median_win:Q", title="Median (winners)", format=".4f"),
                 alt.Tooltip("median_loss:Q", title="Median (losers)", format=".4f"),
                 alt.Tooltip("p_value:Q", title="p", format=".3f"),
                 alt.Tooltip("n_win:Q", title="Winners"),
                 alt.Tooltip("n_loss:Q", title="Losers")],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=NEUTRAL).encode(x="x:Q")
    return (bars + zero).properties(
        height=height, title="Which features separate winners from losers"
    )


def bucket_chart(buckets: pd.DataFrame, label: str, baseline: float | None = None,
                 height: int = 280) -> alt.Chart:
    """Win rate across quantile buckets of one feature, against the overall rate."""
    if buckets is None or buckets.empty:
        return _placeholder("Not enough trades in each bucket to compare.")

    bars = alt.Chart(buckets).mark_bar(color=ACCENT, opacity=0.85).encode(
        x=alt.X("bucket:N", sort=list(buckets["bucket"]), title=f"{label} (quantile buckets)"),
        y=alt.Y("win_rate:Q", title="Win rate (%)"),
        tooltip=[alt.Tooltip("bucket:N", title="Range"),
                 alt.Tooltip("n:Q", title="Trades"),
                 alt.Tooltip("win_rate:Q", title="Win rate %", format=".1f"),
                 alt.Tooltip("avg_pnl_pct:Q", title="Avg net %", format="+.3f"),
                 alt.Tooltip("profit_factor:Q", title="Profit factor", format=".2f")],
    )
    labels = alt.Chart(buckets).mark_text(dy=-8, size=11, color=NEUTRAL).encode(
        x=alt.X("bucket:N", sort=list(buckets["bucket"])),
        y="win_rate:Q",
        text=alt.Text("n:Q", title="n"),
    )
    layers = [bars, labels]
    if baseline is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [baseline]}))
            .mark_rule(color=DOWN, strokeDash=[5, 3])
            .encode(y="y:Q")
        )
    # The number above each bar is its trade count: a tall bar over n=6 is noise,
    # and the chart should say so without the reader opening the table.
    return alt.layer(*layers).properties(height=height)


def hour_chart(by_hour: pd.DataFrame, height: int = 240) -> alt.Chart:
    """Win rate and total P&L by hour of session."""
    if by_hour is None or by_hour.empty:
        return _placeholder("Not enough trades per hour to break the session down.")

    bars = alt.Chart(by_hour).mark_bar().encode(
        x=alt.X("label:N", title="Hour of session"),
        y=alt.Y("total_pnl_pct:Q", title="Total net P&L (%)"),
        color=alt.condition(alt.datum.total_pnl_pct > 0, alt.value(UP), alt.value(DOWN)),
        tooltip=[alt.Tooltip("label:N", title="Hour"),
                 alt.Tooltip("n:Q", title="Trades"),
                 alt.Tooltip("win_rate:Q", title="Win rate %", format=".1f"),
                 alt.Tooltip("total_pnl_pct:Q", title="Total %", format="+.2f")],
    )
    return bars.properties(height=height, title="When the edge shows up")


# ==========================================================================
# Model diagnostics
# ==========================================================================
def calibration_chart(table: pd.DataFrame, height: int = 280) -> alt.Chart:
    """
    Predicted probability against realised win rate, with the ideal diagonal.

    Points below the diagonal mean the model is overconfident: a bucket it calls
    70% actually wins less often. That is the failure that matters most here,
    because the ACCEPT rule is a threshold on exactly these numbers.
    """
    if table is None or table.empty:
        return _placeholder("No scored trades to calibrate against.")

    ideal = (
        alt.Chart(pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}))
        .mark_line(color=NEUTRAL, strokeDash=[4, 3])
        .encode(x=alt.X("x:Q", title="Predicted win probability"),
                y=alt.Y("y:Q", title="Actual win rate"))
    )
    points = alt.Chart(table).mark_circle(color=ACCENT).encode(
        x=alt.X("predicted:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("actual:Q", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("n:Q", title="Trades", scale=alt.Scale(range=[40, 400])),
        tooltip=[alt.Tooltip("bin:N", title="Probability bin"),
                 alt.Tooltip("n:Q", title="Trades"),
                 alt.Tooltip("predicted:Q", title="Predicted", format=".3f"),
                 alt.Tooltip("actual:Q", title="Actual", format=".3f")],
    )
    line = alt.Chart(table).mark_line(color=ACCENT, strokeWidth=1.5).encode(
        x="predicted:Q", y="actual:Q")
    return (ideal + line + points).properties(
        height=height, title="Calibration: is a 70% really a 70%?"
    )


def threshold_chart(sweep: pd.DataFrame, current: float | None = None,
                    best: float | None = None, height: int = 280) -> alt.Chart:
    """
    Total P&L as a function of the ACCEPT threshold.

    Shows what the shipped 0.55 default costs or earns against every other
    choice. Unreliable points - too few accepted trades to mean anything - are
    drawn hollow rather than dropped, because where the curve stops being
    trustworthy is itself the finding.
    """
    if sweep is None or sweep.empty:
        return _placeholder("No scored trades to sweep.")

    df = sweep.copy()
    df["confidence"] = np.where(df.get("reliable", True), "Enough trades", "Too few trades")

    line = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=2).encode(
        x=alt.X("threshold:Q", title="ACCEPT threshold"),
        y=alt.Y("total_pnl_pct:Q", title="Total net P&L (%)"),
    )
    points = alt.Chart(df).mark_point(size=55, filled=True).encode(
        x="threshold:Q", y="total_pnl_pct:Q",
        color=alt.Color("confidence:N", title=None,
                        scale=alt.Scale(domain=["Enough trades", "Too few trades"],
                                        range=[ACCENT, NEUTRAL]),
                        legend=alt.Legend(orient="top")),
        opacity=alt.condition(alt.datum.reliable, alt.value(0.95), alt.value(0.35)),
        tooltip=[alt.Tooltip("threshold:Q", format=".3f"),
                 alt.Tooltip("n_taken:Q", title="Trades taken"),
                 alt.Tooltip("coverage:Q", title="Coverage", format=".0%"),
                 alt.Tooltip("win_rate:Q", title="Win rate %", format=".1f"),
                 alt.Tooltip("total_pnl_pct:Q", title="Total %", format="+.2f")],
    )
    layers = [line, points]
    for value, colour, dash in ((current, DOWN, [5, 3]), (best, UP, [2, 2])):
        if value is not None:
            layers.append(
                alt.Chart(pd.DataFrame({"x": [float(value)]}))
                .mark_rule(color=colour, strokeDash=dash)
                .encode(x="x:Q")
            )
    return alt.layer(*layers).properties(
        height=height, title="What every ACCEPT threshold would have earned"
    )


def comparison_chart(comparison: pd.DataFrame, height: int = 240) -> alt.Chart:
    """The model beside take-everything and beside random selection."""
    if comparison is None or comparison.empty:
        return _placeholder("No scored trades to compare against.")

    return (
        alt.Chart(comparison)
        .mark_bar()
        .encode(
            x=alt.X("total_pnl_pct:Q", title="Total net P&L (%)"),
            y=alt.Y("strategy:N", sort="-x", title=None),
            color=alt.condition(alt.datum.total_pnl_pct > 0, alt.value(UP), alt.value(DOWN)),
            tooltip=[alt.Tooltip("strategy:N", title="Selection rule"),
                     alt.Tooltip("n:Q", title="Trades"),
                     alt.Tooltip("coverage:Q", title="Coverage", format=".0%"),
                     alt.Tooltip("win_rate:Q", title="Win rate %", format=".1f"),
                     alt.Tooltip("total_pnl_pct:Q", title="Total %", format="+.2f"),
                     alt.Tooltip("profit_factor:Q", title="Profit factor", format=".2f")],
        )
        .properties(height=height, title="Does the model beat taking every signal?")
    )


def walk_forward_chart(folds: pd.DataFrame, height: int = 240) -> alt.Chart:
    """Per-fold out-of-sample AUC against the 0.5 coin-flip line."""
    if folds is None or folds.empty:
        return _placeholder("Not enough trades to walk forward.")

    bars = alt.Chart(folds).mark_bar(color=ACCENT, opacity=0.85).encode(
        x=alt.X("fold:O", title="Fold (chronological)"),
        y=alt.Y("auc:Q", title="Out-of-sample ROC AUC", scale=alt.Scale(domain=[0.0, 1.0])),
        tooltip=[alt.Tooltip("fold:O", title="Fold"),
                 alt.Tooltip("train_n:Q", title="Trained on"),
                 alt.Tooltip("test_n:Q", title="Tested on"),
                 alt.Tooltip("auc:Q", title="AUC", format=".3f"),
                 alt.Tooltip("accuracy:Q", title="Accuracy", format=".3f")],
    )
    chance = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(
        color=DOWN, strokeDash=[5, 3]).encode(y="y:Q")
    # The dashed line is a coin flip. Bars that straddle it are the honest
    # summary of this model on simulated data.
    return (bars + chance).properties(height=height, title="Walk-forward stability")


def importance_chart(importances: dict[str, float], labels: dict[str, str] | None = None,
                     height: int = 380) -> alt.Chart:
    """Global feature importance, sorted."""
    if not importances:
        return _placeholder("Model exposes no feature importances.")

    labels = labels or {}
    df = pd.DataFrame(
        [{"feature": labels.get(k, k), "importance": float(v)} for k, v in importances.items()]
    ).sort_values("importance", ascending=False)

    return (
        alt.Chart(df)
        .mark_bar(color=SLOW, opacity=0.85)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=[alt.Tooltip("feature:N"), alt.Tooltip("importance:Q", format=".4f")],
        )
        .properties(height=height, title="What the model leans on")
    )
