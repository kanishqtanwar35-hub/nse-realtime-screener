# NSE Real-Time Screener

Real-time NSE stock screening, SMMA crossover signal generation, trade simulation,
ML-based trade validation, and a post-hoc analysis layer that says which filters
actually separate winners from losers — with an interactive Streamlit dashboard.

Runs with **zero configuration** on simulated data, or against **Fyers** / **Angel One**
when credentials are supplied through the environment.

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐
│ PROVIDER │──▶│ INGESTION │──▶│ SCREENER │──▶│ SIGNALS │──▶│ BACKTEST │──▶│ ML MODEL  │
│ fyers /  │   │ bars +    │   │ price +  │   │  SMMA   │   │  trades  │   │ accept /  │
│ angel /  │   │ cleaning  │   │liquidity │   │crossover│   │  + P&L   │   │  avoid    │
│ simulated│   └───────────┘   └──────────┘   └─────────┘   └──────────┘   └───────────┘
└──────────┘          │              │             │              │              │
                      └──────────────┴─────────────┴──────────────┴──────────────┘
                                              │
                                    ┌─────────▼─────────┐      ┌──────────────┐
                                    │  SQLite store     │─────▶│  Streamlit   │
                                    │  (WAL, idempotent)│      │  dashboard   │
                                    └─────────┬─────────┘      └──────────────┘
                                              │
                                    ┌─────────▼─────────┐      ┌──────────────┐
                                    │  ANALYTICS        │─────▶│  HTML report │
                                    │  separation ·     │      │  + Analytics │
                                    │  buckets · walk-  │      │  dashboard   │
                                    │  forward · sweep  │      │  tab         │
                                    └───────────────────┘      └──────────────┘
```

---

> ### Documentation map
>
> | Document | Read it for |
> |---|---|
> | **[REQUIREMENTS.md](REQUIREMENTS.md)** | Every assignment requirement → implementation → verification status |
> | **[BUILD_JOURNAL.md](BUILD_JOURNAL.md)** | How it was built: decisions, alternatives rejected, and the 8 bugs found |
> | **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** | System internals, the quant domain, and the roadmap of improvements |
> | **[packaging/BUILD_EXE.md](packaging/BUILD_EXE.md)** | Freezing to .exe, and the 4 packaging bugs that had to be solved |

## Quick start

```bash
cd nse-screener
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python scripts/train_model.py --history-minutes 4000    # build a dataset + fit a model
python scripts/run_analysis.py --open                    # what the data actually supports
python scripts/run_live.py --relax-liquidity --cycles 10
python scripts/run_dashboard.py                          # http://localhost:8501
```

No `.env` needed — it defaults to the simulated provider. To go live, copy
`.env.example` to `.env` and fill in your broker credentials.

---

## ⚠️ Read this before running against a live feed

**Three things in the specification will not behave as you might expect. All are
implemented as asked, and all are configurable.**

### 1. The liquidity filter will match almost nothing

The brief requires **both** bid and ask quantity above **1,000,000**. Real NSE
cash-segment top-of-book depth for a ₹30–500 stock is typically **hundreds to low
thousands**. Against a live feed this screen will return an empty set nearly always.

Implemented exactly as specified (`MIN_BID_QTY` / `MIN_ASK_QTY`, default 1,000,000),
with a `--relax-liquidity` flag that drops both to 5,000 so you can exercise the
full pipeline. The screener logs a loud warning when it rejects ≥80% of the universe
on depth, so an empty dashboard is never a mystery.

### 2. "At least 60 minutes of history" is not enough for SMMA(120)

SMMA(120) on 1-minute bars needs **120 bars minimum** before it produces a value at
all, and realistically several hundred before it has settled. With 60 minutes of
history the slow line is `NaN` and **no signal can ever fire**.

`HISTORY_MINUTES` therefore defaults to **1500**, and `MIN_BARS_REQUIRED` (default
150) excludes any symbol that cannot supply enough bars, with a log line naming it.

### 3. Train/serve skew in the ML layer

The model is trained on trades harvested by replaying **historical bars**. Historical
bars contain no tick data, so `ltq_avg_*`, `ltq_spike_*`, `bid_ask_spread_pct`,
`order_book_imbalance` and `depth_ratio` are **zero in training** but **populated
live**. The model sees richer inputs at serving time than it was fitted on.

Three ways to resolve it, in increasing order of effort:

| Option | Trade-off |
|---|---|
| Train on bar features only (drop the 6 tick features) | Honest, immediately correct, loses microstructure signal |
| Persist live feature snapshots and retrain on those | Correct; needs weeks of live collection |
| Reconstruct tick features from bar proxies | Approximate but usable now |

The shipped default trains on what history can supply and **documents the skew rather
than hiding it**. `signals.extract_historical_signals` carries the same note.

---

## Configuration

Everything is environment-driven. **No credential is ever hardcoded, logged, or
written to disk.** See `.env.example` for the full list.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_PROVIDER` | `simulated` | `simulated` \| `fyers` \| `angelone` |
| `SYMBOLS` | *(built-in list)* | Comma-separated universe |
| `MIN_LTP` / `MAX_LTP` | `30` / `500` | Price band |
| `MIN_BID_QTY` / `MIN_ASK_QTY` | `1000000` | Two-sided depth floor |
| `SMMA_FAST` / `SMMA_SLOW` | `20` / `120` | Crossover periods |
| `HISTORY_MINUTES` | `1500` | Bars to seed per symbol |
| `POLL_SECONDS` | `5` | Cycle interval |
| `COST_BPS` / `SLIPPAGE_BPS` | `10` / `2` | Charged on **both** legs |
| `ACCEPT_THRESHOLD` | `0.55` | Probability at or above which a trade is ACCEPTed |
| `ML_ALGORITHM` | `random_forest` | or `xgboost` (falls back if not installed) |

### Broker credentials

**Fyers** uses a browser OAuth redirect that cannot run headless. Generate an access
token once and export it:

```bash
python -m nse_screener.providers.fyers     # prints the authorisation URL
set FYERS_ACCESS_TOKEN=<token>
```

**Angel One** logs in with TOTP. `ANGEL_TOTP_SECRET` is the **base32 seed** from
authenticator enrolment, not the six-digit code — the code is derived at login. The
adapter downloads and caches the public scrip master (~10 MB, cached 24 h) because
SmartAPI addresses instruments by numeric token.

> **Both broker adapters are written against documented API shapes but have never
> been executed against a live account.** Response parsing is defensive throughout —
> a schema difference degrades to a dropped symbol plus a warning rather than a crash.
> Validate against your own account before trusting either with capital.

If a broker cannot be reached, `get_provider()` logs the failure loudly and **falls
back to simulated data** so the dashboard stays up. Pass `--no-fallback` to make a
live feed a hard requirement instead.

---

## What it computes

**Per bar** — SMMA(20), SMMA(120), their spread and spread rate-of-change, price
distance from each SMMA, short-term momentum, realised volatility, ETQ totals over
5/20/60 minutes, and volume concentration.

**Per tick** — LTQ rolling averages over 2/5/20 minutes, LTQ spike ratios (2m vs 20m,
5m vs 20m), bid-ask spread %, order book imbalance, and bid/ask depth ratio.

Two clocks, deliberately kept separate: LTQ is a *per-trade* quantity with no
meaningful per-minute equivalent, so it is tracked in a time-bounded tick window
rather than derived from bars. `features.FEATURE_COLUMNS` is the single contract
between the feature layer, the model, and the dashboard.

### Signals

`BUY` when SMMA20 crosses **above** SMMA120, `SELL` when it crosses **below**. Two
guarantees:

- **No repainting.** Crossovers are evaluated on **closed bars only**. An indicator
  computed on the in-progress minute changes as ticks arrive, so a signal taken from
  it can appear and vanish within the same minute.
- **No duplicates.** Per-symbol state remembers the last emitted crossover, so one
  event fires exactly once even though the same bar is re-evaluated every poll.

A crossover is a change in the **sign** of the spread measured against the last time
the sign was *decided*. Equality is undecided, not "above" — so a fast line that rises
to touch the slow line and falls away again does **not** emit a phantom SELL.

### Trade simulation

Enter at the crossover LTP, exit at the reverse crossover — always-in-the-market,
stop-and-reverse. Costs and slippage are charged on **both legs**, because labelling
trades profitable on a gross basis trains the model to accept trades that lose money
after friction. MAE/MFE are direction-aware, so shorts are measured correctly.

---

## The analysis layer

The screener says what looks tradeable now. `analytics.py` answers the harder
question — **which filters actually separate winners from losers**, and **does the
model beat simply taking every signal**. It drives the dashboard's *Analytics* tab and
a standalone report:

```bash
python scripts/run_analysis.py --open              # writes reports/analysis.html
python scripts/run_analysis.py --json findings.json --no-walk-forward
```

| What it computes | Why it is built this way |
|---|---|
| **Single-feature AUC** — the probability a random winner outranks a random loser on that feature alone | Rank-based, so one outsized trade cannot manufacture an edge, and scale-free across ratios and percentages. A difference-of-means table is noise at these sample sizes |
| **Quantile bucket performance** — win rate and expectancy across buckets of any feature | These features are ratios that pile up near 1.0; equal-width bins would put most trades in one bar. Buckets thinner than 5 trades are dropped, not reported |
| **Walk-forward validation** — expanding-window refits, fold by fold | A single chronological split gives one number that depends on which regime landed in the tail. Here the single split reports **AUC 0.573** and walk-forward reports **0.529 ± 0.099** across five folds — that gap *is* the finding |
| **Calibration** — predicted probability against realised frequency, with Brier and ECE | ROC AUC only measures ranking. A model can rank well and still never cross a 0.55 threshold. Calibration is what makes any threshold mean something |
| **Threshold sweep** — total P&L at every possible `ACCEPT_THRESHOLD` | 0.55 is a convention, not a finding. The sweep either defends it or replaces it, and flags the thin tail where a spurious optimum hides |
| **Strategy comparison** — the model against *take every signal* and against *random selection of the same size* | The only comparison that can embarrass the model. Random is averaged over 200 draws, so a win is not just an artefact of trading less |
| **Excursion analysis** — MAE/MFE, capture ratio, and a stop level quoted with its cost | Reports "this stop cuts 93% of losers and 10% of winners" rather than presenting a stop as free money |
| **Time-of-day and holding-time breakdowns** | The cheapest regime split available, and it needs no new data |

Three rules run through all of it: **every edge is quoted with its sample size**, **the
baseline is always shown**, and model diagnostics use **walk-forward out-of-sample
scores** rather than the shipped model scoring trades it was fitted on — an in-sample
calibration curve would recommend a threshold that cannot be reproduced live.

### What it reports on the shipped simulated data

> No single feature separates winners from losers meaningfully — the best reaches an
> AUC of 0.529 against 0.5 for a coin flip. Walk-forward AUC is 0.529 ± 0.099 across
> five folds, and the model **trails** taking every signal. Winners capture only 53% of
> the move they showed, so a trailing exit has more upside than any entry filter.

That is the correct result for a simulator whose price process contains no learnable
relationship between entry state and outcome. The report says exactly that, in those
words, instead of presenting a number it cannot defend.

---

## Project layout

```
src/nse_screener/
  config.py          all settings, env-driven, no secrets in code
  models.py          Quote / Signal / Trade / Prediction
  utils.py           logging, retry+jitter, rate limiter
  indicators.py      SMMA (recursive + vectorised), crossover
  features.py        FEATURE_COLUMNS contract, TickWindow
  providers/         base ABC, simulated, fyers, angelone, registry
  ingestion.py       session-scoped gap filling, bar aggregation
  screener.py        price band + two-sided liquidity, with reasons
  signals.py         crossover engine
  backtest.py        trade simulator, MAE/MFE, stats
  analytics.py       separation, buckets, calibration, sweep, walk-forward
  ml/model.py        train / predict / explain / persist
  store.py           SQLite (WAL), idempotent upserts, bars for charting
  engine.py          the orchestration loop
  dashboard/app.py   Streamlit UI
  dashboard/charts.py  Altair chart specs, one palette, pure functions
scripts/             run_live.py, train_model.py, run_analysis.py, run_dashboard.py
tests/               104 tests
```

---

## Dashboard

```bash
python scripts/run_dashboard.py          # or: streamlit run src/nse_screener/dashboard/app.py
```

Six tabs:

| Tab | What it shows |
|---|---|
| **Screen** | One row per stock — LTP, SMMA 20/120, ETQ 5/20/60m, average LTP 20/60m, full depth, LTQ 2m/5m ratio, and the rejection reason for everything that failed |
| **Charts** | Price with both SMMA lines and **every crossover marked on the bar it fired**, over an SMMA-spread panel whose zero crossings are those same signals |
| **Signals & Decisions** | Each crossover joined to its model verdict, probability and per-instance explanation |
| **Trades** | Equity curve with the drawdown band, win rate, profit factor, MAE/MFE trade log |
| **Analytics** | Feature separation, an interactive bucket explorer for building a filter, session-hour and holding-time breakdowns, the MAE/MFE excursion scatter |
| **Model** | Metrics and importance, plus walk-forward folds, the calibration curve, the threshold sweep, and the model against both baselines |

Charts are Altair — already a Streamlit dependency, so no new runtime requirement and
nothing extra to chase through PyInstaller — and interactive by default, which matters
because a reviewer scrubbing along a crossover to read the exact values is the point of
putting a chart on screen.

The dashboard runs as a **separate process** and talks to the engine only through
SQLite. A dashboard rerun must not restart ingestion; if the dashboard crashes the
engine keeps running, and vice versa. Auto-refresh uses `st.fragment(run_every=...)`
rather than `sleep`+`rerun`, so the UI stays responsive between refreshes. The analysis
cache is deliberately *not* cleared on refresh — the live tables go stale every few
seconds, five refitted walk-forward models do not.

**No second terminal needed:** the sidebar has **Run demo cycles** and **Backfill
history** buttons that drive the engine in-process. Backfill populates bars, signals,
model verdicts and trades in one click, so every tab has data.

---

## Testing

```bash
pytest                # 104 tests, ~36 s
```

Coverage focuses on the things that fail silently: SMMA correctness (recursive vs
vectorised agreement to 1e-9, seed equals SMA, no fabricated values before the seed),
crossover edge cases (NaN guards, touch-and-retreat, undecided starts), the two-sided
liquidity rule, P&L sign for **both** long and short, costs charged on both legs,
session-scoped gap filling, and store idempotency.

The analysis layer is tested the same way — for direction, degeneracy and sample-size
honesty rather than for arithmetic pandas already guarantees. A planted predictive
feature must rank first; a noise feature must land near 0.5; inverting a feature must
mirror its AUC exactly; a constant feature must be *flagged*, not silently dropped;
thin buckets must vanish; walk-forward scores must be genuinely out of sample and
never score a trade twice; and a one-trade 100% win rate must never win the threshold
search. An analysis function that returns a plausible number from bad input is worse
than one that raises, because the number ends up in a report and gets believed.

---

## Known limitations

1. **Broker adapters unverified against live accounts** (see above).
2. **Train/serve skew** in the ML features (see above).
3. **Polling, not streaming.** Quotes are polled on an interval, so ticks between
   polls are missed. Bar volume uses the broker's cumulative day volume delta rather
   than summing LTQ, which corrects for this — but a websocket feed would be strictly
   better and is the natural next step.
4. **Model quality is data-limited.** On simulated data walk-forward AUC sits near
   chance, which is the correct result: the simulator's price process contains no
   learnable relationship between entry features and outcome. `train_model.py` and
   `run_analysis.py` both say so plainly rather than presenting a meaningless number
   as a success — and the analysis explicitly reports that the model currently
   **trails** taking every signal.
5. **No live order placement.** Trades are simulated only. There is no broker order
   API call anywhere in this codebase.
6. **The analysis is descriptive, not causal.** Bucket win rates and single-feature
   AUCs are associations found in one sample. p-values carry no multiple-comparison
   correction with 19 features under test, and the threshold sweep's optimum is fitted
   to the same data it is measured on. Everything here is a hypothesis to test on
   fresh data, and the report is written to say so.

---

## Packaging

See [`packaging/BUILD_EXE.md`](packaging/BUILD_EXE.md). Short version: the CLI
freezes cleanly with PyInstaller; the Streamlit dashboard does **not** freeze into a
plain one-file exe and needs a launcher approach, which that document explains.
