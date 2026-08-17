#!/usr/bin/env python
"""
Offline analysis of the trade record.

Runs the full battery from `nse_screener.analytics` over whatever is in the
store and writes a self-contained HTML report plus a plain-text summary.

Why this exists alongside the dashboard: the dashboard is for watching, this is
for concluding. A reviewer, or you the next morning, wants one artefact that
says what the data supports - which features separate winners from losers, where
the edge lives in the session, whether the exit rule is leaking profit, and
whether the model beats taking every signal. Scrolling a live UI is a bad way to
answer any of those.

    python scripts/run_analysis.py
    python scripts/run_analysis.py --output reports/analysis.html --open

Exit codes: 0 analysis written, 1 nothing to analyse.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from nse_screener import analytics  # noqa: E402
from nse_screener.config import settings  # noqa: E402
from nse_screener.dashboard import charts  # noqa: E402
from nse_screener.features import FEATURE_LABELS  # noqa: E402
from nse_screener.store import Store  # noqa: E402
from nse_screener.utils import get_logger  # noqa: E402

log = get_logger("analysis")


# ==========================================================================
# Findings - the part a human actually reads
# ==========================================================================
def findings(bundle: analytics.AnalysisBundle, dataset: pd.DataFrame) -> list[str]:
    """
    Turn the tables into plain sentences, hedged in proportion to the evidence.

    Every claim here is generated from the numbers, never asserted: if the data
    does not support a finding, the sentence says so rather than being omitted.
    An empty findings list would read as "nothing to see"; "no feature separates
    better than chance" is a result.
    """
    out: list[str] = []
    n = bundle.n_trades
    win_rate = float((dataset["net_pnl_pct"] > 0).mean() * 100.0)
    total = float(dataset["net_pnl_pct"].sum())
    out.append(
        f"{n} closed round trips, {win_rate:.1f}% of them profitable net of costs, "
        f"{total:+.2f}% cumulative. Costs and slippage are charged on both legs, so "
        "this is what the strategy actually keeps."
    )

    # --- feature separation ---------------------------------------------
    sep = bundle.separation
    if not sep.empty:
        top = sep.iloc[0]
        if top["separation"] < 0.05:
            out.append(
                f"No single feature separates winners from losers meaningfully - the best, "
                f"{top['label']}, reaches an AUC of only {top['auc']:.3f} against 0.5 for a "
                "coin flip. On simulated data that is the correct result and not a bug: the "
                "price process has no learnable relationship between entry state and outcome."
            )
        else:
            direction = "higher" if top["auc"] > 0.5 else "lower"
            out.append(
                f"The strongest single filter is {top['label']}: {direction} values favour "
                f"winners, single-feature AUC {top['auc']:.3f} "
                f"(median {top['median_win']:.4f} on winners against {top['median_loss']:.4f} "
                f"on losers, p={top['p_value']:.3f}). With 19 features under test and no "
                "multiple-comparison correction, treat that p-value as a ranking aid."
            )

        constant = sep[sep.get("note", "").astype(str).str.startswith("constant")]
        if len(constant):
            out.append(
                f"{len(constant)} feature(s) are constant across this dataset "
                f"({', '.join(constant['label'].head(4))}...). These are the tick-clock "
                "features: historical bars carry no tick data, so they are zero in training "
                "but populated live. That train/serve gap is documented in the README and is "
                "the single biggest correctness issue in the ML layer."
            )

    # --- best bucketed filter -------------------------------------------
    filters = bundle.filters
    if not filters.empty:
        best = filters.iloc[0]
        if best["spread"] >= 10.0 and min(best["best_n"], best["worst_n"]) >= 10:
            out.append(
                f"Bucketing by {best['label']} splits the population usefully: the "
                f"{best['best_bucket']} bucket wins {best['best_win_rate']:.1f}% of the time "
                f"over {int(best['best_n'])} trades, against {best['worst_win_rate']:.1f}% in "
                f"{best['worst_bucket']} over {int(best['worst_n'])}. Check the gradient is "
                "monotone before trading it - a single strong bucket is usually where the "
                "quantile edges happened to fall."
            )
        else:
            out.append(
                f"No feature produces a bucket split worth trading; the widest win-rate "
                f"spread is {best['spread']:.1f} points ({best['label']}), which at these "
                "sample sizes is inside the noise."
            )

    # --- session timing --------------------------------------------------
    by_hour = bundle.by_hour
    if len(by_hour) >= 2:
        best_h = by_hour.loc[by_hour["total_pnl_pct"].idxmax()]
        worst_h = by_hour.loc[by_hour["total_pnl_pct"].idxmin()]
        verb = "gives back" if worst_h["total_pnl_pct"] < 0 else "adds only"
        out.append(
            f"By session hour, {best_h['label']} contributes {best_h['total_pnl_pct']:+.2f}% "
            f"over {int(best_h['n'])} trades while {worst_h['label']} {verb} "
            f"{worst_h['total_pnl_pct']:+.2f}% over {int(worst_h['n'])}. A time-of-day filter "
            "is the cheapest regime control available and needs no new data."
        )

    # --- exits ------------------------------------------------------------
    ex = bundle.excursions
    if ex.n:
        out.append(
            f"Exit quality: winners capture {ex.capture_ratio:.0%} of the move they showed "
            f"(average MFE {ex.avg_mfe_pct:.2f}%, average MAE {ex.avg_mae_pct:.2f}%). "
            f"A stop at {ex.stop_at_p90_winner_mae:.2f}% would have cut "
            f"{ex.losers_stopped_at_that_level:.0%} of losers at the cost of "
            f"{ex.winners_stopped_at_that_level:.0%} of winners."
        )
        if ex.capture_ratio < 0.5:
            out.append(
                "That capture ratio is the most actionable number in this report: waiting "
                "for the reverse crossover systematically gives profit back, so a trailing "
                "exit has more upside here than any improvement to the entry filter."
            )

    # --- model ------------------------------------------------------------
    if bundle.verdict:
        out.append(bundle.verdict)

    comp = bundle.comparison
    if not comp.empty and len(comp) >= 2:
        everything = comp.iloc[0]
        model = comp.iloc[1]
        delta = float(model["total_pnl_pct"] - everything["total_pnl_pct"])
        verb = "beats" if delta > 0 else "trails"
        out.append(
            f"Against the only baseline that matters, the model {verb} taking every signal by "
            f"{abs(delta):.2f} percentage points ({model['total_pnl_pct']:+.2f}% over "
            f"{int(model['n'])} trades against {everything['total_pnl_pct']:+.2f}% over "
            f"{int(everything['n'])}), scored out of sample. "
            + ("Random selection of the same size is shown alongside, so the difference is "
               "not just an artefact of trading less."
               if delta > 0 else
               "On this data the ML layer is not earning its place; the honest read is that "
               "it needs live tick features, not a bigger forest.")
        )

    if bundle.best:
        out.append(
            f"Threshold sweep: total P&L peaks at {bundle.best['threshold']:.3f} "
            f"({int(bundle.best['n_taken'])} trades, {bundle.best['win_rate']:.1f}% win rate) "
            f"against the shipped default of {settings.model.accept_threshold:.2f}. That "
            "optimum is fitted to this sample and should not be copied into config without "
            "seeing the same shape on fresh data."
        )

    return out


# ==========================================================================
# HTML rendering
# ==========================================================================
CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 15px/1.65 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 2.5rem 1.5rem 4rem; background: #ffffff; color: #1a1f2b; }
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.7rem; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 2.6rem 0 .6rem; padding-bottom: .35rem;
     border-bottom: 1px solid #e3e7ee; }
h3 { font-size: .95rem; margin: 1.6rem 0 .4rem; color: #46506a; }
.sub { color: #6b7684; font-size: .88rem; margin-bottom: 2rem; }
.cards { display: flex; flex-wrap: wrap; gap: .75rem; margin: 1.2rem 0 .5rem; }
.card { flex: 1 1 150px; border: 1px solid #e3e7ee; border-radius: 10px; padding: .7rem .9rem; }
.card .k { font-size: .74rem; text-transform: uppercase; letter-spacing: .04em; color: #6b7684; }
.card .v { font-size: 1.35rem; font-weight: 600; margin-top: .15rem; }
ol.findings { padding-left: 1.2rem; }
ol.findings li { margin-bottom: .8rem; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .6rem 0 1.2rem; }
th, td { border-bottom: 1px solid #eceff4; padding: .4rem .55rem; text-align: right; }
th { background: #f7f9fc; font-weight: 600; text-align: right; color: #46506a;
     position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; }
tr:hover td { background: #f9fbfd; }
.wrap { overflow-x: auto; }
.chart { margin: 1rem 0 1.6rem; }
.note { background: #fff8e6; border-left: 3px solid #e0a52a; padding: .7rem .9rem;
        border-radius: 0 6px 6px 0; font-size: .87rem; margin: 1rem 0; }
footer { margin-top: 3rem; color: #6b7684; font-size: .8rem;
         border-top: 1px solid #e3e7ee; padding-top: 1rem; }
@media (prefers-color-scheme: dark) {
  body { background: #12151c; color: #e6e9ef; }
  h2 { border-color: #262b36; } h3 { color: #97a1b8; }
  .card, footer { border-color: #262b36; }
  th { background: #1a1f2a; color: #97a1b8; }
  th, td { border-color: #222733; }
  tr:hover td { background: #171c25; }
  .note { background: #2a2314; border-color: #8a6410; }
}
"""


def _table(df: pd.DataFrame, columns: list[str] | None = None, floats: str = "%.3f") -> str:
    if df is None or df.empty:
        return "<p class='sub'>Not enough data for this table.</p>"
    view = df[[c for c in (columns or df.columns) if c in df.columns]]
    return f"<div class='wrap'>{view.to_html(index=False, float_format=lambda v: floats % v, border=0)}</div>"


def _chart_block(chart, name: str) -> tuple[str, str]:
    """Return (div, script) for one Vega-Lite chart."""
    try:
        spec = chart.to_json()
    except Exception as exc:  # noqa: BLE001
        log.warning("chart %s could not be serialised: %s", name, exc)
        return "", ""
    return (
        f"<div class='chart' id='{name}'></div>",
        f"vegaEmbed('#{name}', {spec}, {{actions: false}});",
    )


def render_html(bundle: analytics.AnalysisBundle, dataset: pd.DataFrame, notes: list[str]) -> str:
    win_rate = float((dataset["net_pnl_pct"] > 0).mean() * 100.0)
    total = float(dataset["net_pnl_pct"].sum())
    avg = float(dataset["net_pnl_pct"].mean())
    max_dd = float((bundle.equity["drawdown"].min()) if not bundle.equity.empty else 0.0)

    blocks, scripts = [], []

    def add(chart, name: str) -> None:
        div, script = _chart_block(chart, name)
        if div:
            blocks.append((name, div))
            scripts.append(script)

    add(charts.equity_chart(bundle.equity), "equity")
    add(charts.pnl_distribution_chart(dataset), "pnl")
    add(charts.separation_chart(bundle.separation), "separation")
    if not bundle.filters.empty:
        top_feature = bundle.filters.iloc[0]["feature"]
        add(
            charts.bucket_chart(
                analytics.bucket_performance(dataset, top_feature),
                FEATURE_LABELS.get(top_feature, top_feature),
                win_rate,
            ),
            "buckets",
        )
    add(charts.hour_chart(bundle.by_hour), "hours")
    add(charts.mae_mfe_chart(dataset), "excursions")
    if not bundle.folds.empty:
        add(charts.walk_forward_chart(bundle.folds), "walkforward")
    if bundle.calibration.n:
        add(charts.calibration_chart(bundle.calibration.table), "calibration")
        add(charts.threshold_chart(bundle.sweep, settings.model.accept_threshold,
                                   bundle.best.get("threshold")), "threshold")
    if not bundle.comparison.empty:
        add(charts.comparison_chart(bundle.comparison), "comparison")

    by_name = dict(blocks)
    findings_html = "\n".join(f"<li>{f}</li>" for f in notes)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NSE Screener - trade analysis</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
</head><body><main>

<h1>NSE Screener - trade analysis</h1>
<div class="sub">
  SMMA({settings.indicators.smma_fast})/SMMA({settings.indicators.smma_slow}) crossovers ·
  {settings.provider} data · costs {settings.trading.cost_bps} bps + slippage
  {settings.trading.slippage_bps} bps on both legs ·
  generated {datetime.now().strftime('%d %b %Y %H:%M')}
</div>

<div class="cards">
  <div class="card"><div class="k">Closed trades</div><div class="v">{bundle.n_trades}</div></div>
  <div class="card"><div class="k">Win rate</div><div class="v">{win_rate:.1f}%</div></div>
  <div class="card"><div class="k">Total net P&amp;L</div><div class="v">{total:+.2f}%</div></div>
  <div class="card"><div class="k">Average trade</div><div class="v">{avg:+.3f}%</div></div>
  <div class="card"><div class="k">Max drawdown</div><div class="v">{max_dd:.2f}%</div></div>
</div>

<h2>Findings</h2>
<ol class="findings">{findings_html}</ol>

<h2>Performance</h2>
{by_name.get('equity', '')}
{by_name.get('pnl', '')}

<h2>What separates winners from losers</h2>
<p class="sub">Each feature used alone as a score, measured as ROC AUC. Rank-based, so a
single outsized trade cannot manufacture an edge; the sign says which way to filter.</p>
{by_name.get('separation', '')}
{_table(bundle.separation, ['label', 'n_win', 'n_loss', 'median_win', 'median_loss', 'auc', 'cohens_d', 'p_value'], '%.4f')}

<h3>Best bucketed filter</h3>
{by_name.get('buckets', '')}
{_table(bundle.filters, ['label', 'spread', 'best_bucket', 'best_win_rate', 'best_n', 'worst_bucket', 'worst_win_rate', 'worst_n'], '%.2f')}

<h2>Where the edge lives</h2>
{by_name.get('hours', '')}
{_table(bundle.by_hour, ['label', 'n', 'win_rate', 'avg_pnl_pct', 'total_pnl_pct', 'profit_factor'], '%.2f')}
<h3>By holding time</h3>
{_table(bundle.by_duration, ['bucket', 'n', 'win_rate', 'avg_pnl_pct', 'total_pnl_pct', 'profit_factor'], '%.2f')}
<h3>By symbol</h3>
{_table(bundle.by_symbol, ['symbol', 'n', 'win_rate', 'avg_pnl_pct', 'total_pnl_pct', 'profit_factor'], '%.2f')}

<h2>Exit quality</h2>
<p class="sub">{bundle.excursions.summary()}</p>
{by_name.get('excursions', '')}

<h2>Model, out of sample</h2>
<p class="sub">Probabilities from {bundle.score_source}. Every trade below was scored by a
model fitted only on trades that came before it.</p>
{by_name.get('walkforward', '')}
{_table(bundle.folds, ['fold', 'train_n', 'test_n', 'test_positive_rate', 'auc', 'accuracy'], '%.3f')}
{by_name.get('calibration', '')}
{by_name.get('threshold', '')}
{by_name.get('comparison', '')}
{_table(bundle.comparison, ['strategy', 'n', 'coverage', 'win_rate', 'avg_pnl_pct', 'total_pnl_pct', 'profit_factor'], '%.3f')}

<div class="note">
Charts are rendered by Vega-Lite from a CDN, so they need an internet connection.
Every table above is plain HTML and renders offline.
</div>

<footer>
Simulated or historical data unless a broker feed was configured. No orders are placed
anywhere in this system - all trades are simulated round trips.
</footer>
</main>
<script>{' '.join(scripts)}</script>
</body></html>"""


# ==========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse the stored trade record.")
    parser.add_argument("--output", default="reports/analysis.html",
                        help="HTML report path (default: reports/analysis.html)")
    parser.add_argument("--threshold", type=float, default=settings.model.accept_threshold,
                        help="ACCEPT threshold used for the strategy comparison")
    parser.add_argument("--no-walk-forward", action="store_true",
                        help="Skip walk-forward refitting (much faster, no OOS diagnostics)")
    parser.add_argument("--json", metavar="PATH", help="Also write the findings as JSON")
    parser.add_argument("--open", action="store_true", help="Open the report when done")
    args = parser.parse_args()

    store = Store()
    dataset = store.load_training_set()
    if dataset is None or dataset.empty:
        print(
            "No closed trades in the store.\n"
            "  python scripts/train_model.py --history-minutes 4000\n"
            "generates a trade history to analyse.",
            file=sys.stderr,
        )
        return 1

    print(f"analysing {len(dataset)} closed trades ...")
    bundle = analytics.analyse(
        dataset, threshold=args.threshold, run_walk_forward=not args.no_walk_forward
    )
    notes = findings(bundle, dataset)

    print("\n" + "=" * 78)
    print("FINDINGS")
    print("=" * 78)
    for i, note in enumerate(notes, 1):
        print(f"\n{i}. {note}")

    if not bundle.separation.empty:
        print("\n" + "-" * 78)
        print("TOP FEATURES BY SEPARATION (single-feature AUC, 0.5 = no edge)")
        print("-" * 78)
        for row in bundle.separation.head(8).itertuples(index=False):
            print(f"  {row.label:<34} AUC {row.auc:.3f}   p={row.p_value:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(bundle, dataset, notes), encoding="utf-8")
    print(f"\nreport written to {output.resolve()}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "n_trades": bundle.n_trades,
                    "score_source": bundle.score_source,
                    "verdict": bundle.verdict,
                    "findings": notes,
                    "best_threshold": bundle.best,
                    "separation": bundle.separation.to_dict("records"),
                },
                indent=2, default=str,
            ),
            encoding="utf-8",
        )
        print(f"findings written to {Path(args.json).resolve()}")

    if args.open:
        webbrowser.open(output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
