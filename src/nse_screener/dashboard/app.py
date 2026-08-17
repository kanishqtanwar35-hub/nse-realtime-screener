"""
Streamlit dashboard.

Runs as a SEPARATE process from the engine and talks to it only through the
SQLite store. That separation is deliberate:

  * A dashboard rerun (every interaction, every auto-refresh) must not restart
    the ingestion loop or re-seed history.
  * Streamlit reruns the whole script top to bottom on every refresh, so any
    state held here would be rebuilt constantly.
  * If the dashboard crashes, the engine keeps trading. If the engine stops,
    the dashboard still shows the last known state.

There is also a self-contained demo mode (--demo / sidebar toggle) that runs the
engine in-process for a few cycles, so the dashboard can be shown working
without a second terminal.

    streamlit run src/nse_screener/dashboard/app.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Streamlit executes this file as a SCRIPT, not as part of the package, so
# `nse_screener` has to be importable from wherever the script happens to live.
#
# When FROZEN, do nothing: the package is already inside the PyInstaller archive
# and the bootloader has set sys.path correctly. Adding the bundle directory
# here would put a data-only `nse_screener/` folder ahead of the real frozen
# package, Python would treat it as a namespace package, and every submodule
# import would fail with ModuleNotFoundError.
if not getattr(sys, "frozen", False):
    _SRC = Path(__file__).resolve().parents[2]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from nse_screener import analytics
from nse_screener.backtest import compute_stats
from nse_screener.config import resolve_universe, settings
from nse_screener.dashboard import charts
from nse_screener.features import FEATURE_LABELS
from nse_screener.ml.model import TradeClassifier
from nse_screener.models import Side, Trade
from nse_screener.store import Store

st.set_page_config(
    page_title="NSE Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container { padding-top: 2.2rem; padding-bottom: 2rem; }
  [data-testid="stMetricValue"] { font-size: 1.5rem; }
  .pill { padding: 2px 10px; border-radius: 10px; font-size: 0.78rem; font-weight: 600; }
  .ok   { background:#e6f4ea; color:#1d6b3d; }
  .bad  { background:#fdecea; color:#a52222; }
  .warn { background:#fff6e0; color:#8a6410; }
  .muted{ color:#6b7684; font-size:0.82rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Data access - cached so a rerun does not hammer SQLite
# ==========================================================================
@st.cache_resource
def get_store() -> Store:
    return Store()


@st.cache_resource
def get_classifier() -> TradeClassifier:
    return TradeClassifier.load()


@st.cache_data(ttl=5)
def load_screen() -> pd.DataFrame:
    return get_store().load_screen()


@st.cache_data(ttl=5)
def load_metrics() -> pd.DataFrame:
    return get_store().load_live_metrics()


@st.cache_data(ttl=5)
def load_signals(limit: int = 200) -> pd.DataFrame:
    return get_store().load_signals(limit)


@st.cache_data(ttl=5)
def load_predictions(limit: int = 200) -> pd.DataFrame:
    return get_store().load_predictions(limit)


@st.cache_data(ttl=5)
def load_trades() -> pd.DataFrame:
    return get_store().load_trades()


@st.cache_data(ttl=20)
def load_bars(symbol: str, limit: int = 600) -> pd.DataFrame:
    return get_store().load_bars(symbol, limit)


@st.cache_data(ttl=60)
def bar_symbols() -> list[str]:
    return get_store().bar_symbols()


@st.cache_data(ttl=60)
def load_training_set() -> pd.DataFrame:
    return get_store().load_training_set()


@st.cache_data(ttl=900, show_spinner="Refitting walk-forward folds ...")
def run_analysis(n_trades: int, threshold: float) -> analytics.AnalysisBundle:
    """
    The full analysis battery, cached hard.

    `n_trades` is not used inside - it is the cache key. Walk-forward refits the
    model once per fold, which is far too slow to redo on a 15-second dashboard
    refresh, and the result cannot change unless the trade count does. This
    cache is deliberately NOT cleared by `clear_caches()`: the live tables go
    stale every few seconds, the analysis does not.
    """
    return analytics.analyse(load_training_set(), threshold=threshold)


def clear_caches() -> None:
    """Drop the live views only. The analysis cache is keyed on trade count."""
    load_screen.clear()
    load_metrics.clear()
    load_signals.clear()
    load_predictions.clear()
    load_trades.clear()
    load_bars.clear()


# ==========================================================================
# Sidebar
# ==========================================================================
def sidebar() -> dict:
    st.sidebar.title("NSE Screener")
    st.sidebar.caption("SMMA crossover signals with ML trade validation")

    st.sidebar.subheader("Refresh")
    auto = st.sidebar.toggle("Auto-refresh", value=True)
    every = st.sidebar.slider(
        "Every (seconds)", 5, 120, settings.dashboard_refresh_seconds, step=5, disabled=not auto
    )
    if st.sidebar.button("Refresh now", width="stretch"):
        clear_caches()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Configuration")
    st.sidebar.markdown(
        f"""
        <span class="muted">
        provider &nbsp;<b>{settings.provider}</b><br>
        price band &nbsp;<b>Rs {settings.screener.min_ltp:.0f} - {settings.screener.max_ltp:.0f}</b><br>
        depth floor &nbsp;<b>{settings.screener.min_bid_qty:,}</b> both sides<br>
        SMMA &nbsp;<b>{settings.indicators.smma_fast} / {settings.indicators.smma_slow}</b><br>
        accept at &nbsp;<b>&ge; {settings.model.accept_threshold:.0%}</b>
        </span>
        """,
        unsafe_allow_html=True,
    )

    if settings.screener.min_bid_qty >= 1_000_000:
        st.sidebar.warning(
            "Depth floor is 1,000,000 on both sides - far above typical NSE cash "
            "top-of-book depth. Expect an empty screen against a live feed.",
            icon="⚠️",
        )
    if settings.provider == "simulated":
        st.sidebar.info("Simulated data. Not tradeable.", icon="🧪")

    st.sidebar.divider()
    st.sidebar.subheader("Demo")
    st.sidebar.caption("Runs the engine in-process to populate this dashboard.")
    cycles = st.sidebar.number_input("Cycles", 1, 50, 5)
    run_demo = st.sidebar.button("Run demo cycles", width="stretch")
    backfill = st.sidebar.button("Backfill history", width="stretch",
                                 help="Replay history to generate trades for the tables below")

    return {"auto": auto, "every": every, "cycles": int(cycles),
            "run_demo": run_demo, "backfill": backfill}


# ==========================================================================
# Panels
# ==========================================================================
def header(store: Store, clf: TradeClassifier) -> None:
    counts = store.counts()
    trades = load_trades()
    stats = compute_stats(_frame_to_trades(trades))

    c = st.columns(6)
    c[0].metric("Symbols screened", counts.get("screen_snapshots", 0))
    c[1].metric("Signals", counts.get("signals", 0))
    c[2].metric("Closed trades", stats.trades)
    c[3].metric("Win rate", f"{stats.win_rate:.1f}%" if stats.trades else "-")
    c[4].metric("Total P&L", f"{stats.total_pnl_pct:+.2f}%" if stats.trades else "-")
    c[5].metric("Model", "loaded" if clf.is_trained else "none")

    if not clf.is_trained:
        st.warning(
            "No trained model - every signal will show as UNKNOWN. "
            "Run `python scripts/train_model.py` to fit one.",
            icon="🤖",
        )


def panel_screen() -> None:
    df = load_screen()
    if df.empty:
        st.info("No screen data yet. Start the engine, or use the Demo button in the sidebar.")
        return

    passed = df[df["passed"] == 1] if "passed" in df.columns else df
    failed = df[df["passed"] == 0] if "passed" in df.columns else pd.DataFrame()

    a, b = st.columns([3, 1])
    a.subheader(f"Passing the screen ({len(passed)})")
    b.metric("Pass rate", f"{len(passed) / max(len(df), 1):.0%}")

    if passed.empty:
        st.warning(
            "Nothing passed. On a live feed this is almost always the 1,000,000 "
            "two-sided depth floor - real NSE top-of-book depth in the Rs 30-500 "
            "band is hundreds to low thousands. Lower MIN_BID_QTY / MIN_ASK_QTY.",
            icon="🔍",
        )
    else:
        # The assignment's main deliverable: ONE ROW PER STOCK carrying LTP,
        # SMMA20/120, ETQ 5/20/60m, average LTP 20/60m and full market depth.
        metrics = load_metrics()
        if metrics.empty:
            st.info("Indicators are still warming up - they appear once bars are seeded.")
            view = passed[["symbol", "ltp", "bid_price", "ask_price",
                           "bid_qty", "ask_qty", "ltq"]].copy()
        else:
            view = metrics[metrics["symbol"].isin(passed["symbol"])].copy()
            # DataFrame.pop() has no default argument - unlike dict.pop() - so
            # drop(errors="ignore") is the safe way to remove an optional column.
            view = view.drop(columns=["captured_at"], errors="ignore")

        st.dataframe(
            view,
            width="stretch", hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="small"),
                "ltp": st.column_config.NumberColumn("LTP", format="%.2f"),
                "smma_20": st.column_config.NumberColumn("SMMA 20", format="%.2f"),
                "smma_120": st.column_config.NumberColumn("SMMA 120", format="%.2f"),
                "smma_signal": st.column_config.TextColumn("Trend", width="small"),
                "etq_5m": st.column_config.NumberColumn("ETQ 5m", format="%d"),
                "etq_20m": st.column_config.NumberColumn("ETQ 20m", format="%d"),
                "etq_60m": st.column_config.NumberColumn("ETQ 60m", format="%d"),
                "avg_ltp_20m": st.column_config.NumberColumn("Avg LTP 20m", format="%.2f"),
                "avg_ltp_60m": st.column_config.NumberColumn("Avg LTP 60m", format="%.2f"),
                "bid_price": st.column_config.NumberColumn("Bid", format="%.2f"),
                "bid_qty": st.column_config.NumberColumn("Bid Qty", format="%d"),
                "ask_price": st.column_config.NumberColumn("Ask", format="%.2f"),
                "ask_qty": st.column_config.NumberColumn("Ask Qty", format="%d"),
                "ltq": st.column_config.NumberColumn("LTQ", format="%d"),
                "ltq_avg_2m": st.column_config.NumberColumn("LTQ avg 2m", format="%.0f"),
                "ltq_avg_5m": st.column_config.NumberColumn("LTQ avg 5m", format="%.0f"),
                "ltq_ratio_2_5": st.column_config.NumberColumn("LTQ 2m/5m", format="%.2f"),
                "spread_pct": st.column_config.NumberColumn("Spread %", format="%.3f"),
                "imbalance": st.column_config.ProgressColumn(
                    "Book imbalance", min_value=-1.0, max_value=1.0, format="%.2f"
                ),
            },
        )
        st.caption(
            "SMMA 20/120 - ETQ over 5/20/60 min - average LTP over 20/60 min - "
            "live market depth. LTQ 2m/5m above 1.0 means trade size is stepping in."
        )

    if not failed.empty:
        with st.expander(f"Rejected ({len(failed)})"):
            st.dataframe(failed[["symbol", "ltp", "bid_qty", "ask_qty"]],
                         width="stretch", hide_index=True)


def panel_charts() -> None:
    """
    Price, both SMMA lines, and every crossover marked on the bar it fired.

    A table of SMMA values proves the numbers were computed. Only the chart
    proves they were computed CORRECTLY - a marker sitting one bar past the
    crossing is a repainting bug you can see in a second and would never find in
    a column of floats.
    """
    symbols = bar_symbols()
    if not symbols:
        st.info(
            "No bars stored yet. Use **Backfill history** or **Run demo cycles** in the "
            "sidebar - the engine writes bars to the store as it seeds and as they close."
        )
        return

    metrics = load_metrics()
    default = 0
    if not metrics.empty:
        # Default to a symbol that is actually trending, so the first chart a
        # reviewer sees has a crossover on it rather than a flat line.
        live = metrics[metrics["symbol"].isin(symbols)]
        if not live.empty and "smma_20" in live.columns:
            spread = (live["smma_20"] - live["smma_120"]).abs()
            default = symbols.index(live.loc[spread.idxmax(), "symbol"])

    c1, c2 = st.columns([2, 1])
    symbol = c1.selectbox("Symbol", symbols, index=min(default, len(symbols) - 1))
    window = c2.select_slider("Bars shown", options=[120, 240, 480, 900, 1500], value=480)

    bars = load_bars(symbol, int(window))
    if bars.empty:
        st.warning(f"No bars stored for {symbol}.")
        return

    signals = load_signals(500)
    st.altair_chart(charts.price_chart(bars, signals, symbol), width="stretch")
    st.altair_chart(charts.spread_chart(bars), width="stretch")
    st.caption(
        "The lower panel is SMMA20 minus SMMA120. Every zero crossing there is a signal "
        "above - crossovers are evaluated on closed bars only, so a marker never moves "
        "once it is drawn."
    )

    sym_signals = signals[signals["symbol"] == symbol] if not signals.empty else pd.DataFrame()
    trades = load_trades()
    sym_trades = trades[trades["symbol"] == symbol] if not trades.empty else pd.DataFrame()

    a, b, c = st.columns(3)
    a.metric("Bars stored", len(bars))
    b.metric("Crossovers", len(sym_signals))
    c.metric("Round trips", len(sym_trades))

    if not sym_trades.empty:
        with st.expander(f"Trades on {symbol} ({len(sym_trades)})"):
            st.dataframe(
                sym_trades[[c for c in ("side", "entry_time", "entry_price", "exit_time",
                                        "exit_price", "net_pnl_pct", "duration_minutes")
                            if c in sym_trades.columns]],
                width="stretch", hide_index=True,
            )


def panel_analytics() -> None:
    """
    The question the brief asks last and the hardest: which filters separate
    winners from losers.
    """
    dataset = load_training_set()
    if dataset.empty:
        st.info("No closed trades yet. Use **Backfill history** in the sidebar.")
        return

    bundle = run_analysis(len(dataset), settings.model.accept_threshold)
    if not bundle.n_trades:
        st.info("No closed trades to analyse yet.")
        return

    st.caption(f"{bundle.n_trades} closed trades analysed.")

    # --- equity ----------------------------------------------------------
    st.altair_chart(charts.equity_chart(bundle.equity), width="stretch")
    a, b = st.columns(2)
    with a:
        st.altair_chart(charts.drawdown_chart(bundle.equity), width="stretch")
    with b:
        st.altair_chart(charts.pnl_distribution_chart(dataset), width="stretch")

    st.divider()

    # --- what separates winners from losers ------------------------------
    st.subheader("Which features separate winners from losers")
    st.caption(
        "Each bar is that feature used ALONE as a score, measured as ROC AUC minus 0.5. "
        "It is rank-based, so one outsized trade cannot manufacture an edge, and the sign "
        "tells you which direction to filter in."
    )
    st.altair_chart(charts.separation_chart(bundle.separation), width="stretch")

    if not bundle.separation.empty:
        with st.expander("Full separation table"):
            st.dataframe(
                bundle.separation[["label", "n_win", "n_loss", "median_win", "median_loss",
                                   "auc", "cohens_d", "p_value"]],
                width="stretch", hide_index=True,
                column_config={
                    "label": st.column_config.TextColumn("Feature"),
                    "n_win": st.column_config.NumberColumn("Winners", format="%d"),
                    "n_loss": st.column_config.NumberColumn("Losers", format="%d"),
                    "median_win": st.column_config.NumberColumn("Median (win)", format="%.4f"),
                    "median_loss": st.column_config.NumberColumn("Median (loss)", format="%.4f"),
                    "auc": st.column_config.NumberColumn("AUC", format="%.3f"),
                    "cohens_d": st.column_config.NumberColumn("Effect size", format="%.3f"),
                    "p_value": st.column_config.NumberColumn("p", format="%.3f"),
                },
            )
            st.caption(
                "p-values carry no multiple-comparison correction and 19 features are under "
                "test, so treat them as a ranking aid rather than as significance."
            )

    # --- drill into one feature ------------------------------------------
    st.subheader("Build a filter")
    available = list(analytics.feature_column_map(dataset))
    if available:
        # Pre-select the feature the brief singles out, if it carries information.
        preferred = "ltq_ratio_2_5" if "ltq_ratio_2_5" in available else available[0]
        if not bundle.separation.empty:
            best_feature = bundle.separation.iloc[0]["feature"]
            if bundle.separation.iloc[0]["separation"] > 0.05:
                preferred = best_feature

        col1, col2 = st.columns([3, 1])
        feature = col1.selectbox(
            "Feature", available, index=available.index(preferred),
            format_func=lambda f: FEATURE_LABELS.get(f, f),
        )
        bins = col2.select_slider("Buckets", options=[3, 4, 5, 6], value=5)

        table = analytics.bucket_performance(dataset, feature, bins=int(bins))
        baseline = float((dataset["net_pnl_pct"] > 0).mean() * 100.0)
        st.altair_chart(
            charts.bucket_chart(table, FEATURE_LABELS.get(feature, feature), baseline),
            width="stretch",
        )
        st.caption(
            f"Dashed line is the {baseline:.1f}% win rate across all trades. The number over "
            "each bar is how many trades it holds - a tall bar over a small count is noise. "
            "A monotone gradient across buckets is the only pattern worth trading."
        )
        if not table.empty:
            st.dataframe(table[["bucket", "n", "win_rate", "avg_pnl_pct",
                                "total_pnl_pct", "profit_factor"]],
                         width="stretch", hide_index=True)

    if not bundle.filters.empty:
        with st.expander("Shortlist: every feature ranked by win-rate spread"):
            st.dataframe(
                bundle.filters[["label", "spread", "best_bucket", "best_win_rate", "best_n",
                                "worst_bucket", "worst_win_rate", "worst_n"]],
                width="stretch", hide_index=True,
            )
            st.caption(
                "Spread is best bucket minus worst. It ignores whether the gradient is "
                "monotone, so this is a shortlist to inspect above, not a conclusion."
            )

    st.divider()

    # --- regime and exit analysis ----------------------------------------
    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.hour_chart(bundle.by_hour), width="stretch")
        if not bundle.by_duration.empty:
            st.markdown("**By holding time**")
            st.dataframe(bundle.by_duration[["bucket", "n", "win_rate", "avg_pnl_pct",
                                             "total_pnl_pct"]],
                         width="stretch", hide_index=True)
    with right:
        st.altair_chart(charts.mae_mfe_chart(dataset), width="stretch")
        ex = bundle.excursions
        if ex.n:
            st.markdown(f"**Exit quality** — {ex.summary()}")
            if ex.capture_ratio < 0.5:
                st.warning(
                    f"Winners realise only {ex.capture_ratio:.0%} of the move they showed. "
                    "Waiting for the reverse crossover is giving profit back - a trailing "
                    "exit is the change with the most upside here, more than a better entry "
                    "filter.",
                    icon="📉",
                )

    if not bundle.by_symbol.empty:
        with st.expander(f"By symbol ({len(bundle.by_symbol)})"):
            st.dataframe(bundle.by_symbol, width="stretch", hide_index=True)
            st.caption("If one symbol carries the total, the edge is a story about that "
                       "symbol, not about the strategy.")


def panel_signals() -> None:
    signals = load_signals()
    preds = load_predictions()

    if signals.empty:
        st.info(
            "No signals yet. SMMA20/SMMA120 crossovers are rare - a few per symbol "
            "per day - so a short run may produce none. Use **Backfill history** in "
            "the sidebar to harvest historical crossovers."
        )
        return

    # Join each signal to its model verdict.
    merged = signals.copy()
    if not preds.empty:
        merged = signals.merge(
            preds[["symbol", "side", "signal_time", "probability", "decision", "explanation"]],
            left_on=["symbol", "side", "timestamp"],
            right_on=["symbol", "side", "signal_time"],
            how="left",
        )

    cols = ["timestamp", "symbol", "side", "ltp"]
    for extra in ("probability", "decision", "reason", "explanation"):
        if extra in merged.columns:
            cols.append(extra)

    st.subheader(f"Signals ({len(merged)})")
    st.dataframe(
        merged[cols],
        width="stretch", hide_index=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Time", format="DD MMM HH:mm"),
            "ltp": st.column_config.NumberColumn("Entry LTP", format="%.2f"),
            "probability": st.column_config.ProgressColumn(
                "Win probability", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
            "side": st.column_config.TextColumn("Side", width="small"),
            "decision": st.column_config.TextColumn("Decision", width="small"),
        },
    )

    if "decision" in merged.columns and merged["decision"].notna().any():
        counts = merged["decision"].value_counts()
        c = st.columns(3)
        c[0].metric("ACCEPT", int(counts.get("ACCEPT", 0)))
        c[1].metric("AVOID", int(counts.get("AVOID", 0)))
        c[2].metric("UNKNOWN", int(counts.get("UNKNOWN", 0)))


def panel_trades() -> None:
    df = load_trades()
    if df.empty:
        st.info("No trades yet. Use **Backfill history** in the sidebar.")
        return

    trades = _frame_to_trades(df)
    stats = compute_stats(trades)

    c = st.columns(6)
    c[0].metric("Trades", stats.trades)
    c[1].metric("Wins", stats.wins)
    c[2].metric("Win rate", f"{stats.win_rate:.1f}%")
    c[3].metric("Avg P&L", f"{stats.avg_pnl_pct:+.3f}%")
    c[4].metric("Profit factor", f"{stats.profit_factor:.2f}")
    c[5].metric("Max drawdown", f"{stats.max_drawdown_pct:.2f}%")

    equity = analytics.equity_curve(df)
    if not equity.empty:
        st.altair_chart(charts.equity_chart(equity), width="stretch")
        st.caption(
            "The shaded band is the distance below the running peak - depth AND duration "
            "of every drawdown, which a bare cumulative line hides."
        )

    st.subheader("Trade log")
    view_cols = ["symbol", "side", "entry_time", "entry_price", "exit_time",
                 "exit_price", "pnl_points", "net_pnl_pct", "duration_minutes",
                 "mae_pct", "mfe_pct", "exit_reason"]
    st.dataframe(
        df[[c for c in view_cols if c in df.columns]],
        width="stretch", hide_index=True,
        column_config={
            "entry_price": st.column_config.NumberColumn("Entry", format="%.2f"),
            "exit_price": st.column_config.NumberColumn("Exit", format="%.2f"),
            "pnl_points": st.column_config.NumberColumn("P&L (pts)", format="%.2f"),
            "net_pnl_pct": st.column_config.NumberColumn("Net P&L %", format="%.3f"),
            "duration_minutes": st.column_config.NumberColumn("Mins", format="%.0f"),
            "mae_pct": st.column_config.NumberColumn("MAE %", format="%.2f"),
            "mfe_pct": st.column_config.NumberColumn("MFE %", format="%.2f"),
        },
    )


def panel_model(clf: TradeClassifier) -> None:
    if not clf.is_trained:
        st.warning("No model loaded. Run `python scripts/train_model.py`.", icon="🤖")
        return

    st.subheader("Model")
    st.caption(f"version `{clf.version}`")

    report_path = Path(settings.paths.model).with_suffix(".report.json")
    if report_path.exists():
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        c = st.columns(5)
        c[0].metric("Samples", report.get("n_samples", 0))
        c[1].metric("Test accuracy", f"{report.get('test_accuracy', 0):.3f}")
        c[2].metric("Precision", f"{report.get('precision', 0):.3f}")
        c[3].metric("Recall", f"{report.get('recall', 0):.3f}")
        c[4].metric("ROC AUC", f"{report.get('roc_auc', 0):.3f}")

        auc = report.get("roc_auc", 0.5)
        if auc < 0.55:
            st.error(
                f"ROC AUC {auc:.3f} is at or near chance. The model is not "
                "discriminating between winning and losing crossovers. Treat every "
                "ACCEPT as unvalidated.",
                icon="📉",
            )
        elif auc < 0.65:
            st.warning(f"ROC AUC {auc:.3f} - modest discrimination. Use as a filter, "
                       "not as a green light.", icon="📊")
        else:
            st.success(f"ROC AUC {auc:.3f}", icon="📈")

        for w in report.get("warnings", []):
            st.warning(w, icon="⚠️")

    if clf.importances:
        st.altair_chart(
            charts.importance_chart(clf.importances, FEATURE_LABELS), width="stretch"
        )
        st.caption(
            "Global importance says what the model leans on overall. It does not say the "
            "feature is predictive - a tree will happily split hard on noise. The "
            "**Analytics** tab measures each feature against outcomes directly."
        )

    # ------------------------------------------------------------------
    # Out-of-sample diagnostics. Everything below is scored by models that
    # never saw the trade being scored - see analytics.walk_forward_scores.
    # ------------------------------------------------------------------
    dataset = load_training_set()
    if dataset.empty:
        return

    bundle = run_analysis(len(dataset), settings.model.accept_threshold)
    if bundle.folds.empty and bundle.calibration.n == 0:
        st.info(
            "Not enough closed trades for out-of-sample diagnostics. Backfill more history "
            "and they appear here.",
            icon="🔬",
        )
        return

    st.divider()
    st.subheader("Does it hold up out of sample?")
    if bundle.verdict:
        st.markdown(f"**{bundle.verdict}**")
    st.caption(f"Probabilities below come from {bundle.score_source}.")

    if not bundle.folds.empty:
        st.altair_chart(charts.walk_forward_chart(bundle.folds), width="stretch")
        st.caption(
            "Each fold refits on everything before it and is tested on what follows. "
            "A single train/test split can be lucky; five consecutive ones cannot all be."
        )

    left, right = st.columns(2)
    with left:
        st.altair_chart(charts.calibration_chart(bundle.calibration.table), width="stretch")
        if bundle.calibration.n:
            st.caption(
                f"{bundle.calibration.summary()}. Brier 0.25 is what you score by always "
                "predicting 0.5, so anything at or above that is worse than a shrug."
            )
    with right:
        st.altair_chart(charts.comparison_chart(bundle.comparison), width="stretch")
        if not bundle.comparison.empty:
            st.dataframe(
                bundle.comparison[["strategy", "n", "win_rate", "avg_pnl_pct",
                                   "total_pnl_pct", "profit_factor"]],
                width="stretch", hide_index=True,
                column_config={
                    "strategy": st.column_config.TextColumn("Selection rule"),
                    "win_rate": st.column_config.NumberColumn("Win rate %", format="%.1f"),
                    "avg_pnl_pct": st.column_config.NumberColumn("Avg %", format="%+.3f"),
                    "total_pnl_pct": st.column_config.NumberColumn("Total %", format="%+.2f"),
                    "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
                },
            )

    # --- threshold ------------------------------------------------------
    st.subheader("Is 0.55 the right threshold?")
    current = settings.model.accept_threshold
    best = bundle.best.get("threshold") if bundle.best else None
    st.altair_chart(charts.threshold_chart(bundle.sweep, current, best), width="stretch")

    if bundle.best:
        gap = bundle.sweep.loc[
            (bundle.sweep["threshold"] - current).abs().idxmin()
        ] if not bundle.sweep.empty else None
        cols = st.columns(4)
        cols[0].metric("Shipped default", f"{current:.2f}")
        cols[1].metric("Best on this data", f"{bundle.best['threshold']:.3f}")
        cols[2].metric("Trades it takes", int(bundle.best["n_taken"]))
        cols[3].metric("Total P&L", f"{bundle.best['total_pnl_pct']:+.2f}%")
        if gap is not None:
            st.caption(
                f"At the shipped {current:.2f} the model takes {int(gap['n_taken'])} trades "
                f"for {gap['total_pnl_pct']:+.2f}%. The optimum above is fitted to this "
                "sample - it is a diagnostic, not a setting to copy blindly. Move "
                "ACCEPT_THRESHOLD only if the whole curve, not one point, favours it."
            )


# ==========================================================================
def _frame_to_trades(df: pd.DataFrame) -> list[Trade]:
    """Rebuild Trade objects so the shared stats function can be reused."""
    if df is None or df.empty:
        return []
    out: list[Trade] = []
    for row in df.itertuples(index=False):
        if pd.isna(getattr(row, "exit_time", None)):
            continue
        t = Trade(
            symbol=row.symbol,
            side=Side(row.side),
            entry_time=pd.to_datetime(row.entry_time).to_pydatetime(),
            entry_price=float(row.entry_price),
        )
        t.exit_time = pd.to_datetime(row.exit_time).to_pydatetime()
        t.net_pnl_pct = float(getattr(row, "net_pnl_pct", 0.0) or 0.0)
        t.duration_minutes = float(getattr(row, "duration_minutes", 0.0) or 0.0)
        out.append(t)
    return out


def run_demo(cycles: int, backfill: bool) -> None:
    """In-process engine run so the dashboard can be demonstrated standalone."""
    from nse_screener.engine import ScreenerEngine

    with st.spinner("Seeding history and running the engine ..."):
        engine = ScreenerEngine(symbols=resolve_universe())
        if backfill:
            engine.backfill_training_data(min_trades=0)
            st.success("Backfill complete - trades and signals written to the store.")
        else:
            results = engine.run(max_cycles=cycles)
            n = sum(len(r.signals) for r in results)
            st.success(f"Ran {len(results)} cycles, produced {n} signal(s).")
    clear_caches()


def main() -> None:
    opts = sidebar()
    store = get_store()
    clf = get_classifier()

    st.title("NSE Real-Time Screener")
    st.caption(
        f"SMMA({settings.indicators.smma_fast})/SMMA({settings.indicators.smma_slow}) "
        f"crossovers · ML trade validation · last refreshed "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )

    if opts["run_demo"]:
        run_demo(opts["cycles"], backfill=False)
    if opts["backfill"]:
        run_demo(0, backfill=True)

    # Auto-refresh via a FRAGMENT rather than sleep+rerun.
    #
    # The obvious implementation - time.sleep(n) then st.rerun() - blocks the
    # script thread for the whole interval, so the sidebar stops responding and
    # the page cannot even finish painting before it is torn down. A fragment
    # reruns only its own subtree on a timer, leaving the rest of the app live.
    interval = f"{opts['every']}s" if opts["auto"] else None

    @st.fragment(run_every=interval)
    def live_panels() -> None:
        clear_caches()
        header(store, clf)
        st.caption(f"updated {datetime.now().strftime('%H:%M:%S')}")
        st.divider()

        tabs = st.tabs(["Screen", "Charts", "Signals & Decisions", "Trades",
                        "Analytics", "Model"])
        with tabs[0]:
            panel_screen()
        with tabs[1]:
            panel_charts()
        with tabs[2]:
            panel_signals()
        with tabs[3]:
            panel_trades()
        with tabs[4]:
            panel_analytics()
        with tabs[5]:
            panel_model(clf)

    live_panels()


if __name__ == "__main__":
    main()
