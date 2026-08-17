# Build Journal — how this system was built

A record of the development process: the order things were built in, the
decisions made and the alternatives rejected, and — most usefully — the bugs
that were found and what each one taught.

Written to be defensible. Every claim here is something you can be questioned on
and answer from the code.

---

## The approach: architecture before features

The temptation with a brief like this is to open a notebook, pull some quotes,
compute an SMMA, and grow outward. That produces a demo that works once on the
author's machine.

I started from the **data flow** instead:

```
provider → ingestion → screener → indicators → signals → backtest → ML → dashboard
```

Each stage takes a defined input and returns a defined output, and each is a
separate module. Three concrete payoffs, all of which came due later:

1. **The simulated provider was written before any broker code.** That meant the
   entire pipeline could be developed and tested without a market being open,
   without credentials, and without rate limits. It stopped being a fallback
   feature and became the primary development environment.
2. **The ML layer never knows where features come from.** `FEATURE_COLUMNS` is a
   list of names; the model takes a dict. Live features and historical features
   flow through the same code path.
3. **The dashboard reads from SQLite, not from the engine.** A dashboard rerun
   cannot restart the ingestion loop, and either process can crash without
   taking the other down.

**The rule I applied throughout:** if a module needs to import three others to do
its job, the boundary is in the wrong place.

---

## Build order, and why

### 1. Config and models first

Before any logic: `config.py` (every setting, env-driven) and `models.py`
(`Quote`, `Signal`, `Trade`, `Prediction`).

Doing this first forced an early decision — **no credential is ever read outside
`config.py`**. There is no `os.getenv()` anywhere else in the codebase. That is
what makes "credentials removed or masked" a structural property rather than a
thing to remember before submitting.

`BrokerCredentials.redacted()` returns `{"fyers_client_id": "<set>"}` — presence,
never value. It is what gets logged.

### 2. Indicators, with the definition written out

SMMA is defined recursively:

```
SMMA[n-1] = SMA(price, n)                     ← seed
SMMA[i]   = (SMMA[i-1] × (n-1) + price[i]) / n
```

I implemented it **twice**: once as an explicit numpy loop (`smma`, unambiguous,
easy to verify against the formula) and once vectorised through
`ewm(alpha=1/n, adjust=False)` (`smma_ewm`, ~50× faster).

Then wrote the test that matters:

```python
assert (smma(prices, period) - smma_ewm(prices, period)).abs().max() < 1e-9
```

Two implementations that must agree is a far stronger correctness check than one
implementation and a hand-picked expected value. If either drifts, the test
fails.

**Deliberate choice: leading values are `NaN`, not zero or forward-filled.**
SMMA(120) genuinely has no value for its first 119 bars. Filling that would let a
crossover fire on data that does not exist.

### 3. The simulated provider — treated as a real component

The naive version is a random walk. That was useless: a pure random walk produces
almost no SMMA crossovers, so there was nothing to build a signal engine against.

The simulator therefore has:

- **Regime-switching drift** — periods of directional bias, so the two averages
  actually separate and cross at a realistic rate
- **Mean reversion to an anchor**, so prices stay inside the ₹30–500 band
- **An intraday volume U-curve** — heavy at open and close, thin at midday
- **Lognormal LTQ**, giving the heavy right tail real trade sizes have, so the
  spike-ratio features vary instead of sitting at 1.0
- **A liquid/illiquid split** (~45/55), so the liquidity filter actually
  discriminates rather than passing or failing everyone
- **Tick-size snapping** — real LTPs are multiples of ₹0.05, never arbitrary floats

**Why this mattered:** it is what let me test the screener, the signal engine and
the backtest end to end before writing a line of broker code.

### 4. Ingestion — where the first real bug lived

Then the screener, signal engine, backtest, ML layer, store, and orchestration
loop. At that point the system ran end to end on simulated data.

That is where verification started finding things.

---

## The bugs — the most useful part of this document

Eight real defects, four in the application and four in packaging. Each one is a
story worth being able to tell.

### Bug 1 — 5,008 phantom bars per symbol

**Symptom:** the log filled with
`5008 bar(s) left unfilled - gap longer than max_forward_fill=3`, repeated per
symbol.

**Cause:** `clean_bars` reindexed onto a continuous 1-minute grid spanning the
whole timestamp range. But market data has an **overnight gap of ~1,065 minutes**
and a weekend gap of ~3,000. The code was manufacturing a row for every minute
between Friday's close and Monday's open.

**The worse consequence:** a forward fill could carry Friday's closing price into
Monday's open. Silent, and it would corrupt every indicator downstream.

**Fix:** group by trading date, reindex **within each session**, concatenate. A
gap can now only be filled inside one trading day.

> **The lesson:** time-series code that ignores session boundaries is wrong even
> when it looks right. The bug announced itself as noise in the logs; the actual
> danger was silent data corruption.

### Bug 2 — a phantom SELL signal

**Symptom:** a unit test I wrote for touch-and-retreat failed.

**Cause:** the obvious crossover implementation is

```python
down = (prev >= 0) & (now < 0)
```

That treats *"the spread was exactly zero"* as *"the spread was above"*. A fast
average that rises to **touch** the slow one and falls away again emits a SELL —
even though it never went above.

**Fix:** a crossover is a change in the **sign** of the spread measured against
the last time the sign was *decided*. Zeros are skipped, not counted as either
side.

```python
sign = np.sign(diff)
decided = sign.replace(0.0, np.nan).ffill().shift(1)   # last DECIDED sign
up   = (sign > 0) & (decided < 0)
down = (sign < 0) & (decided > 0)
```

**This fix broke a test fixture**, which turned out to be the more interesting
finding. The fixture was flat-then-ramp: two SMMAs over a constant price are
*exactly equal*, so the spread is 0 — undecided, never negative — and the
subsequent rise is a **divergence from equality, not a crossing**. The fixture
had to be rebuilt as down-then-up-then-down to produce genuine sign changes.

> **The lesson:** floating-point equality is rare in real data and common in
> synthetic data. A test fixture that cannot occur in production is not a test.

### Bug 3 — the dashboard froze between refreshes

**Symptom:** the sidebar stopped responding; screenshots caught a blank skeleton.

**Cause:** auto-refresh implemented as `time.sleep(n)` then `st.rerun()`. That
blocks Streamlit's script thread for the entire interval — the page cannot even
finish painting before being torn down.

**Fix:** `@st.fragment(run_every="15s")`, which reruns only that subtree on a
timer and leaves the rest of the app live.

> **The lesson:** in an event-loop framework, `sleep` is almost always the wrong
> tool. The symptom looked like a rendering bug; the cause was a threading model
> misunderstanding.

### Bug 4 — `DataFrame.pop()` has no default

```python
view.pop("captured_at", None)     # TypeError: takes 2 positional args but 3 given
```

`dict.pop(key, default)` works. `DataFrame.pop(key)` does not take a default.
Fixed with `drop(columns=[...], errors="ignore")`.

> **The lesson:** API similarity is not API compatibility. Caught only because I
> ran the dashboard through Streamlit's `AppTest` harness rather than eyeballing it.

### Bugs 5–8 — packaging, and why they were the hardest

All four produced errors that pointed away from the cause, and **every broken
build still returned HTTP 200**. The server started fine and died on first render.

**Bug 5 — the bundled script shadowed the package.** Streamlit execs `app.py` by
path, so it ships as a data file. Bundling it to `nse_screener/dashboard/app.py`
created `_internal/nse_screener/` containing *only* that file. Python found it
first, treated it as an implicit namespace package, and every submodule import
failed — while `import nse_screener` appeared to succeed.

**Bug 6 — `collect_submodules` failed silently.** Adding it made things *worse*:
the error changed from `No module named 'nse_screener.backtest'` to
`No module named 'nse_screener'`. Cause: `collect_submodules()` **imports** the
package to walk it, and runs at spec-parse time — before `Analysis(pathex=...)`
applies. It could not import the package, returned almost nothing, and raised **no
warning at all**.

Found by comparing the build TOCs:

```
Analysis-00.toc (CLI)        43 nse_screener entries
Analysis-01.toc (dashboard)   3 nse_screener entries   ← there it is
```

**That one command located the bug after two failed hypotheses.** When runtime
symptoms mislead, inspect the build artefact.

**Bug 7 — the entry point never imported the package.** `dashboard_entry.py`
imports only `streamlit`. Streamlit finds `app.py` **by path at runtime**, so
static analysis has no way to know the package is needed. Fixed by importing it
explicitly in the entry point even though nothing there calls it.

**Bug 8 — `UnicodeEncodeError` from Streamlit's own code.** Streamlit 1.61 prints
a `U+26A0` through `click` into a cp1252 Windows console and dies with a full
traceback. Harmless, but it looks like a crash. Fixed by forcing UTF-8 stdio
before streamlit is imported.

> **The lesson that generalises:** *a frozen app that starts is not a frozen app
> that works.* All three broken builds passed a health check. Freezing must be
> verified by exercising a real code path, and PyInstaller cannot see any
> dependency reached through a runtime path lookup — Streamlit apps, plugin
> loaders, `importlib` by string.

---

## Design decisions, and what was rejected

### Costs charged on both legs

`net_pnl = gross_pnl − (cost_bps + slippage_bps) × 2`, at 24 bps round trip.

**Rejected:** labelling on gross P&L. That would train the model to accept trades
that lose money after friction — the single most expensive mistake available
here, because it produces a model that looks good and loses money. A test asserts
that a +0.1% gross trade is labelled a **loser**.

### Time-ordered train/test split

**Rejected:** `train_test_split(shuffle=True)`. Trades are a time series. A random
split lets the model learn from trades that happen *after* the ones it is
evaluated on — future information leaking backwards, producing a validation score
that will not survive contact with live data.

### Two clocks, kept separate

Bar features come from 1-minute OHLCV. Tick features (LTQ averages, spread,
imbalance) come from a **time-bounded** in-memory tick window.

**Rejected:** a fixed-length deque. A liquid stock produces far more ticks per
minute than a thin one, so a fixed length would silently mean a different
lookback per symbol — the features would not be comparable across the universe.

### Signals only on closed bars

**Rejected:** evaluating the in-progress minute. An indicator computed on a
partial candle changes as ticks arrive, so the signal *repaints* — it can appear
and vanish within the same minute. Per-symbol state also makes emission
idempotent: the same bar is re-evaluated every poll and fires exactly once.

### Volume from the cumulative delta, not summed LTQ

Polling misses ticks between calls, so summing LTQ **undercounts**. Using the
delta of the broker's cumulative day volume is correct regardless of poll rate,
with an LTQ fallback and a guard for the counter resetting at session start.

### Binaries in a Volume, metadata in a table

Not applicable here, but the same principle drove **SQLite with WAL**: the engine
writes while the dashboard reads. Without WAL the dashboard blocks on a locked
database mid-write.

Every write is an **upsert on a natural key** (`symbol + timestamp + side`), so
re-running a cycle updates rather than duplicates. Running it twice is safe.

---

## What I would do differently

**Write the packaging spec earlier.** I left it to the end and it produced four of
the eight bugs. Freezing a Streamlit app is genuinely hard, and finding that out
on the last day is bad sequencing. A trivial "hello world" freeze on day one would
have surfaced the whole class of problem.

**Design the feature contract before the ML layer.** `FEATURE_COLUMNS` emerged
after the features existed. Defining it first would have exposed the train/serve
skew — historical bars have no tick data, so six features are zero in training and
populated live — as a **design decision** rather than something discovered during
verification.

**Test the dashboard from the start.** Streamlit's `AppTest` harness runs the
script and asserts on rendered elements. I found it late; it caught two bugs
immediately and would have caught them sooner.

**Push harder on the "all NSE stocks" requirement early.** Scaling from 28 to
~1,800 symbols is not a config change — warm-up is one history call per symbol,
so it is a *minutes-long* operation, and quote polling becomes request-bound. That
should have shaped the ingestion design from the start rather than being a
documented caveat.

---

## The second pass: making the system answerable

The first version could screen, signal, simulate and predict. It could not
**answer questions about itself**, and two gaps made that obvious.

**Gap 1 — nothing could be checked visually.** The dashboard showed SMMA 20 and
SMMA 120 as columns of floats. That proves the numbers were computed; it does not
prove they were computed *correctly*. The specific failure it could not catch is
repainting: an indicator evaluated on the in-progress minute produces a signal
that appears and vanishes within that minute, and in a table it looks identical
to a correct one. On a chart, a marker sitting one bar past the crossing is
visible in a second.

That required bars in the store. The engine held them in memory and the dashboard
is a separate process, so the price chart was not merely missing — it was
*impossible*. Adding a `bars` table keyed on `(symbol, timestamp)` fixed it, with
the live loop writing only `tail=3` on the bars that just closed. Writing a
1,500-row history every five-second poll would have made the cycle disk-bound to
service a chart.

**Gap 2 — requirement 22 was answered with the wrong tool.** "Identify filters
separating winners from losers" was satisfied with a feature-importance bar chart.
Importance says what the model *leans on*; a tree splits hard on noise and will
happily rank a useless feature first. It says nothing about whether the feature
predicts outcomes.

`analytics.py` answers it directly: each feature used *alone* as a score, measured
as single-feature ROC AUC from ranks. Rank-based because trade P&L is heavy-tailed
and a difference of means is one outlier from a finding.

**The decision that changed a number.** Calibration curves and threshold sweeps
need probabilities. The easy path is to score the training set with the shipped
model — and it is wrong, because those trades are exactly what it was fitted on.
So `walk_forward_scores` returns the out-of-sample probabilities alongside the
fold table, and every model diagnostic runs on those instead.

It immediately paid for itself. The single chronological split reports **AUC
0.573**. Five expanding windows report **0.529 ± 0.099**, ranging 0.347–0.642.
The strategy comparison then showed the model **trailing** take-every-signal.
Without walk-forward this submission would have quoted 0.573 and believed it.

**Charts: Altair, not Plotly.** Plotly is the reflex choice and it was the wrong
one here. Altair already ships with Streamlit, so it adds no runtime dependency,
nothing to the frozen bundle, and no new PyInstaller hidden-import to chase —
and the packaging section of this journal is a long argument for not adding any
of those three. Layered declarative specs also review better than imperative axis
plumbing.

**One thing I got wrong on the first attempt.** The analysis was initially wired
into the same `clear_caches()` the auto-refresh calls, so every 15-second refresh
refit five models. The dashboard was unusable. The fix is that the analysis cache
is keyed on trade count and never cleared by the refresh: the live tables go stale
every few seconds, five refitted models do not.

---

## What is verified, and what is not

**Verified:** 104 tests pass. The full pipeline runs on simulated data. 429 trades
and 451 crossovers simulated, 112,000 bars persisted, a model trained and saved.
The dashboard renders 6 tabs, 27 metrics and 9 tables with 0 exceptions. The
analysis report generates 8 findings from the data. Both executables run.

**Not verified — and this is the honest boundary:**

- **The broker adapters have never executed against a live account.** They are
  written against documented request/response shapes with defensive parsing (every
  field access has a fallback, so a schema difference drops one symbol with a
  warning rather than crashing). But *documented* and *actual* are different
  things, and I expect at least one field name to be wrong.
- **The executables have not been run on a machine with no Python installed** —
  which is the entire point of freezing.
- **The ML model has never seen real market data.** On simulated data walk-forward
  AUC is 0.529 ± 0.099, and the model trails taking every signal. That is the
  *correct* result: the simulator contains no learnable relationship between entry
  features and outcome. Whether the LTQ hypothesis holds is an **open empirical
  question** that only real data answers.
- **The walk-forward folds are not purged.** Trades whose holding periods straddle
  a fold boundary leak slightly across it, and with a mean hold of ~21 hours on
  this dataset that is not a rounding error. A purge-and-embargo scheme is the
  first thing to add before trusting any fold number — noted in HOW_IT_WORKS §9
  rather than quietly ignored.

I would rather state that boundary precisely than present a number I cannot
defend.
