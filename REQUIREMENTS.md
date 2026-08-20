# Requirement traceability

Every line of the assignment, mapped to where it is implemented and how it was
verified. Status is deliberately blunt — anything not fully done says so.

Legend: **✅ done & verified** · **⚠️ done, caveat** · **❌ not done**

---

## Program requirements

| # | Requirement | Status | Where | Verified by |
|---|---|---|---|---|
| 1 | Scan all NSE stocks, LTP ₹30–₹500 | ⚠️ | `screener.py`, `universe.py` | Price band unit-tested (inclusive bounds). **Universe defaults to 28 symbols** — see note 1 |
| 2 | Bid Qty > 10,00,000 **and** Ask Qty > 10,00,000 | ✅ | `screener.py:screen_quotes` | Test asserts one-sided depth **fails** the screen |
| 3 | Display SMMA(20), SMMA(120) | ✅ | `indicators.py:smma`, dashboard cols `smma_20`/`smma_120` | Recursive vs vectorised agree to 1e-9; seed == SMA; rendered in main table |
| 4 | ETQ for last 5 / 20 / 60 min | ✅ | `features.py:compute_bar_features` | Unit test on known volumes; cols `etq_5m/20m/60m` |
| 5 | Average LTP for last 20 / 60 min | ✅ | `features.py`, cols `avg_ltp_20m`/`avg_ltp_60m` | Present in `live_metrics`; rendered |
| 6 | Market depth: bid/ask price and qty | ✅ | `models.py:Quote`, dashboard | Rendered in main table |
| 7 | Live tabular dashboard, **one row per stock**, auto-refresh | ✅ | `dashboard/app.py`, `store.live_metrics` | `AppTest`: 20 columns, 6 tabs, 27 metrics, 0 exceptions. Refresh via `st.fragment(run_every=)` |

### Beyond the brief

Added because a screener that only tabulates numbers cannot be checked, and a model
nobody validates out of sample cannot be trusted.

| Addition | Where | Why it earns its place |
|---|---|---|
| **Price / SMMA / crossover chart** | `dashboard/charts.py`, **Charts** tab, `store.bars` | A table of SMMA values proves the numbers were computed; only the chart proves they were computed *correctly*. A marker one bar past the crossing is a repainting bug you see in a second and would never find in a column of floats |
| **Analysis layer** | `analytics.py`, **Analytics** tab, `scripts/run_analysis.py` | Turns requirement 22 from "here is feature importance" into a measured answer, with sample sizes attached |
| **Walk-forward validation** | `analytics.walk_forward_scores` | The shipped single split reports AUC 0.573; walk-forward reports 0.529 ± 0.099. Without it the submission would quote the optimistic number |
| **Calibration + threshold sweep** | `analytics.calibration`, `threshold_sweep` | `ACCEPT_THRESHOLD=0.55` was a convention. The sweep tests it against every alternative on out-of-sample scores |
| **Baseline comparison** | `analytics.strategy_comparison` | The model is measured against *take every signal* and *random selection of the same size*. On this data it trails the first — reported, not hidden |
| **Standalone HTML report** | `scripts/run_analysis.py` | One artefact a reviewer can open without launching Streamlit, with findings written as sentences generated from the numbers |

### AI/ML analysis

| # | Requirement | Status | Where |
|---|---|---|---|
| 8 | Detect **every** SMMA crossover | ✅ | `signals.py`, `extract_historical_signals` |
| 9 | Predict whether a crossover is profitable | ✅ | `ml/model.py:TradeClassifier.predict` |
| 10 | Identify crossovers to avoid | ✅ | `Decision.AVOID` below `ACCEPT_THRESHOLD` |
| 11 | Display probability / confidence | ✅ | `probability` column, progress bar |
| 12 | Explain **why** accept / reject | ✅ | Per-instance attribution — SHAP if installed, else importance × deviation |

### Crossover & trading logic

| # | Requirement | Status | Where |
|---|---|---|---|
| 13 | Buy = SMMA20 crosses **above** SMMA120 | ✅ | `indicators.py:crossover` |
| 14 | Sell = SMMA20 crosses **below** SMMA120 | ✅ | same |
| 15 | Record entry LTP, exit on reverse crossover, record exit LTP | ✅ | `backtest.py:TradeSimulator.simulate` |
| 16 | P/L = Sell LTP − Buy LTP | ✅ | `Trade.pnl_points` (absolute) **and** `net_pnl_pct` |
| 17 | Evaluate profitability for both Buy and Sell trades | ✅ | Direction-aware; test asserts a short profits when price falls |

### ML requirement

| # | Requirement | Status | Where |
|---|---|---|---|
| 18 | LTQ-based features | ✅ | `ltq_avg_2m/5m/20m`, `ltq_spike_2_20`, `ltq_spike_5_20` |
| 19 | **Compare avg LTQ 2 min vs 5 min** | ✅ | `ltq_ratio_2_5` — named explicitly in the brief, shown in the table and fed to the model |
| 20 | Additional quantitative features | ✅ | 19 features total — SMMA spread + RoC, price distance, momentum, volatility, ETQ ratios, spread %, order-book imbalance, depth ratio |
| 21 | Any suitable ML algorithm | ✅ | RandomForest default, XGBoost via `ML_ALGORITHM` |
| 22 | Identify filters separating winners from losers | ✅ | `analytics.py` — single-feature AUC, quantile bucket win rates, and a ranked shortlist. Surfaced in the **Analytics** tab and `scripts/run_analysis.py`. Feature *importance* alone was not enough: a tree splits hard on noise, so importance says what the model leans on, not what predicts |

---

## Deliverables

| Deliverable | Status | Note |
|---|---|---|
| Python source code | ✅ | 29 modules, 104 tests |
| **Executable (.exe)** | ✅ | Built **and run**: `dist5/nse-screener/nse-screener.exe` + `-dashboard.exe`. Four packaging bugs found and fixed — see `packaging/BUILD_EXE.md` |
| **Screen recording** | ❌ | Not included in this repository |
| Credentials removed/masked | ✅ | Env-only; `.env` gitignored; `redacted()` logs presence, never values |

---

## The three things you must do before submitting

### 1. Get a Fyers or Angel One account and test against it
The assignment lists this as a **prerequisite**. My broker adapters are written
against documented API shapes but have **never executed against a live account**.
Expect to fix at least one field-name mismatch. Budget half a day.

```bash
# after filling .env
python scripts/run_live.py --provider angelone --relax-liquidity --cycles 5
```

### 2. Load the full NSE universe
Requirement 1 says *all* NSE stocks. The app ships with 28. Download
`EQUITY_L.csv` from nseindia.com, save it as `data/nse_universe.csv`, then raise
`MAX_SYMBOLS`.

**Do not jump straight to 1,800.** Warm-up is one history call per symbol; at
~8 req/s that is ~4 minutes before the first signal, and quote polling becomes
request-bound. Step up: 50 → 200 → 500, measuring cycle time each step.

### 3. Decide what to say about the liquidity filter
Both-sides > 10,00,000 will return **an empty screen** on real NSE cash data in
the ₹30–500 band. It is implemented exactly as specified. In your submission,
say so explicitly and show both: the filter as specified, and a relaxed run that
demonstrates the rest of the pipeline. That reads as understanding the market,
not as failing the spec.

---

## Known gaps and honest caveats

**Note 1 — universe size.** `universe.py` resolves from a local CSV → broker
instrument master → 28-symbol fallback. The plumbing for all of NSE is there; the
data file is not shipped because NSE's endpoint needs a browser session.

**Note 2 — train/serve skew.** The model trains on trades harvested from
historical bars, which carry no tick data. So `ltq_*`, `bid_ask_spread_pct`,
`order_book_imbalance` and `depth_ratio` are **zero in training but populated
live**. Documented in `signals.extract_historical_signals` and the README. The
clean fix is to collect live snapshots for a few sessions and retrain — which
also makes the LTQ hypothesis in the brief genuinely testable.

**Note 3 — model quality is data-limited, and the analysis proves it.** The
single chronological split reports ROC AUC 0.573. Walk-forward across five
expanding windows reports **0.529 ± 0.099** (range 0.347–0.642), and the strategy
comparison shows the model **trailing** take-every-signal on out-of-sample scores.
That is the **correct** result: the simulator contains no learnable relationship
between entry features and outcome. Real broker data is where the LTQ hypothesis
gets its actual test. Both `train_model.py` and `run_analysis.py` print this
interpretation rather than presenting a number as a success.

**Note 4 — polling, not streaming.** Quotes are polled, so ticks between polls
are missed. Bar volume uses the cumulative-volume delta rather than summing LTQ,
which corrects the totals — but a websocket feed is strictly better and is the
obvious Assignment 2 upgrade.

---

## What was actually verified

| Check | Result |
|---|---|
| `pytest` | 104 passed |
| Fresh end-to-end train | 429 trades, 451 crossovers, 112,000 bars persisted, model fitted and saved |
| Live loop (simulated) | 3 cycles, 27 symbols in `live_metrics`, 0 errors |
| Dashboard | `AppTest`: 6 tabs, 27 metrics, 9 tables, **0 exceptions** |
| Analysis report | `run_analysis.py` writes `reports/analysis.html` + `findings.json`, 8 generated findings |
| Broker adapters | **Not verified — no live account** |
| `.exe` CLI | Runs a full screening session, writes its DB beside the exe |
| `.exe` dashboard | Serves, health `ok`, renders, loads the model, no errors |
| `.exe` on a machine **without Python** | **Not verified — test this before submitting** |

**Eight real bugs were found and fixed during verification:**

Application (4): cross-session gap filling producing 5,008 phantom bars; a phantom
SELL on touch-and-retreat; a blocking `sleep` in the dashboard refresh;
`DataFrame.pop()` misuse.

Packaging (4): the bundled app script shadowing the real package;
`collect_submodules` failing **silently** because `src/` was not on `sys.path` at
spec-parse time; the dashboard entry point never importing the package it needs;
and a `UnicodeEncodeError` from Streamlit's own code on a cp1252 console.

The packaging bugs were only findable by running the exe — every broken build
still returned HTTP 200. Details in `packaging/BUILD_EXE.md`.

**Note on the build folder:** builds land in a numbered `distN/` because Windows
keeps a handle on the previous `dist/` after the frozen dashboard has run once,
and even `Rename-Item` is refused until the handle is released. It does not reach
anyone: the shipped archive is built with `nse-screener/` as its root, so the
build folder's name never appears outside this machine. `.gitignore` covers
`dist[0-9]*/` for the same reason.
