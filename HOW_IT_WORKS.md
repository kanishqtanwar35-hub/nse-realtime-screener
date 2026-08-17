# How It Works — system, domain, and where to go next

Three parts:

1. **The system** — what happens, in order, when you run it
2. **The domain** — why SMMA crossovers behave the way they do, and what the LTQ
   hypothesis is really asking
3. **The edge** — concrete upgrades, the ML traps specific to trading, and the
   things that separate a competent submission from a strong one

Part 3 is where the value is. Parts 1 and 2 are what you need to defend it.

---

# PART 1 — The system

## One cycle, end to end

```
     ┌────────────────────────────────────────────────────────────┐
     │ 1. FETCH      provider.fetch_quotes(symbols)               │
     │               batched, rate-limited, retried               │
     ├────────────────────────────────────────────────────────────┤
     │ 2. SCREEN     ₹30 ≤ LTP ≤ ₹500                             │
     │               bid_qty > 10L AND ask_qty > 10L              │
     │               rejections keep their REASON                 │
     ├────────────────────────────────────────────────────────────┤
     │ 3. AGGREGATE  every quote → tick window (LTQ clock)        │
     │               every quote → bar aggregator (1-min clock)   │
     │               returns the set of symbols whose bar CLOSED  │
     ├────────────────────────────────────────────────────────────┤
     │ 3b. DISPLAY   one row per screened stock →  live_metrics   │
     │               LTP, SMMA20/120, ETQ 5/20/60, avg LTP 20/60, │
     │               full depth, LTQ averages + 2m/5m ratio       │
     ├────────────────────────────────────────────────────────────┤
     │ 4. SIGNAL     only symbols that passed AND closed a bar     │
     │               → crossover? → feature snapshot → Signal      │
     ├────────────────────────────────────────────────────────────┤
     │ 5. SCORE      model.predict(features)                       │
     │               → probability, ACCEPT/AVOID, explanation      │
     ├────────────────────────────────────────────────────────────┤
     │ 6. PERSIST    idempotent upserts → SQLite (WAL)             │
     └────────────────────────────────────────────────────────────┘
                              ↓ dashboard reads, separate process
```

**Why step 4 is filtered twice.** Only symbols that *passed the screen* **and**
*just closed a bar* are evaluated. Re-evaluating an unchanged bar burns CPU and
cannot produce anything new — the crossover state is identical.

**Why step 3 feeds tick windows for *every* quote, not just screened ones.** The
LTQ baseline needs continuous observation. If you only record ticks for symbols
currently passing the liquidity filter, the 20-minute average is full of holes
whenever a stock drifts in and out of the screen.

## The two clocks

This is the design idea most worth being able to explain.

| | Bar clock | Tick clock |
|---|---|---|
| Source | 1-minute OHLCV | raw quote stream in memory |
| Holds | SMMA, ETQ, momentum, volatility, avg LTP | LTQ averages, spread, imbalance |
| Bounded by | bar count | **wall-clock time** |
| Why | indicators need regular intervals | LTQ is *per trade*, not per minute |

LTQ has no meaningful per-minute equivalent — it is the size of the most recent
trade, and a minute may contain one trade or four hundred. The tick window is
bounded by **time, not count**, because a fixed-length buffer would mean a
30-second lookback for a liquid stock and a two-hour lookback for a thin one, and
the features would not be comparable across the universe.

## SMMA, precisely

```
SMMA[n-1] = SMA(price, n)                     ← seed
SMMA[i]   = (SMMA[i-1] × (n-1) + price[i]) / n
```

Equivalent to an exponential MA with `alpha = 1/n` — so SMMA(20) has a smoothing
factor of 0.05, and SMMA(120) of 0.0083. **SMMA(120) is extremely slow**: its
centre of mass sits roughly 120 bars back. On 1-minute bars that is two hours of
lag, which is the single most important fact about this strategy's behaviour.

Implemented twice — explicit recursion and vectorised `ewm` — and a test asserts
they agree to 1e-9.

**Leading values are `NaN`.** SMMA(120) genuinely has no value for 119 bars.
Which is why:

> The brief says "at least the last 60 minutes of historical data." That is **not
> enough**. With 60 bars, SMMA(120) is `NaN` and **no signal can ever fire**. The
> default lookback is 1,500 minutes and `MIN_BARS_REQUIRED` excludes any symbol
> that cannot supply enough. Being able to say this shows you understand the
> indicator rather than just implementing it.

## Crossover detection

A crossover is a change in the **sign** of `SMMA20 − SMMA120`, compared against
the last time the sign was *decided*. Equality is undecided — not "above".

Two guarantees:

- **No repainting.** Evaluated on closed bars only. An indicator on a partial
  candle changes as ticks arrive, so a signal taken from it can appear and vanish
  within the same minute.
- **No duplicates.** Per-symbol state remembers the last emitted event, so one
  crossover fires exactly once across many polls of the same bar.

## Trade simulation

Enter at the crossover LTP, exit at the reverse — always in the market,
stop-and-reverse. Every exit is also an entry.

- **P&L both ways:** `pnl_points` (the brief's `Sell LTP − Buy LTP`) and
  `net_pnl_pct`. Points are what the spec asks for; percentage is what the model
  and statistics use, because 2 points on a ₹35 stock and 2 points on a ₹480 stock
  are not comparable.
- **Direction-aware.** A short profits as price falls; `Side.sign` makes P&L,
  MAE and MFE work for both directions from one code path.
- **Costs on both legs**, so a small gross winner is correctly labelled a loser.
- **MAE/MFE** — worst and best unrealised excursion during the trade. They answer
  "would a stop have been hit?" and "did we give back a winner?", which P&L cannot.

## The ML layer

`FEATURE_COLUMNS` — 19 features, fixed order — is the contract between the
feature layer, the model and the dashboard. The model receives a dict; it never
knows whether the source was a live tick or a replayed bar.

Every feature is **guaranteed finite**. Missing values become `0.0`, which for
these features means "neutral / no information" — never `NaN`, because a model
must not receive a value it was never trained on.

**Explanations** are per-instance, not just global importance. Global importance
tells you what the model relies on overall; it says nothing about why *this*
trade scored 0.71. Uses SHAP when installed, otherwise ranks by
`importance × |z-score vs the training median|` — "features the model cares
about, on which this trade is unusual". That heuristic is documented as an
approximation, not a causal attribution.

## The analysis layer

`analytics.py` exists because the model layer answers "what does the classifier
think" and nothing answers "**is the classifier worth having**". Four design
decisions carry it:

**1. Rank statistics, not means.** Trade P&L is heavy-tailed and samples are small,
so a difference in means is one outlier away from a finding. Single-feature ROC AUC
— computed from ranks as Mann-Whitney U / n₁n₂ — is scale-free, outlier-resistant,
and reads directly as "how well does this filter sort trades". `separation` is
`|AUC − 0.5|`, so a feature predicting in the *opposite* direction ranks as highly as
one predicting forward: you just invert the filter.

**2. Walk-forward scores, not in-sample ones.** Calibration curves and threshold
sweeps must be fed probabilities from a model that never saw the trade. Scoring the
training set with the shipped model bends both optimistically and would recommend a
threshold that cannot be reproduced live. `walk_forward_scores` therefore returns the
out-of-sample probabilities alongside the fold table, and `AnalysisBundle.score_source`
records which the numbers came from so the UI can label them.

**3. Sample size travels with every claim.** Every table carries `n`; buckets thinner
than five trades are dropped with a log line; `best_threshold` refuses rows flagged
unreliable, because the tail of a sweep is exactly where a spurious optimum hides.
The bucket chart prints the trade count above each bar so a tall bar over `n=6`
announces itself.

**4. The baseline is always on screen.** `strategy_comparison` puts the model beside
*take every signal* and beside *random selection of the same size*, averaged over 200
draws. A model that accepts 90% of signals and matches take-everything has added
nothing, however good its AUC looks — and on the shipped simulated data that is
precisely what the comparison reports.

The same functions feed the **Analytics** dashboard tab and `scripts/run_analysis.py`,
which writes a standalone HTML report whose findings are *generated from the numbers*.
If the data supports nothing, the report says "no single feature separates winners
from losers meaningfully" rather than quietly omitting the section.

---

# PART 2 — The domain

## Why moving-average crossovers usually lose money

Worth understanding, because the assignment is fundamentally asking you to fix it.

**1. They are lagging by construction.** A crossover confirms a move that has
already happened. With SMMA(120) on 1-minute bars, confirmation arrives hours in.

**2. They whipsaw in ranging markets.** When price oscillates, the averages cross
repeatedly and every crossing is a loss. Markets range far more than they trend —
this is where the losses concentrate.

**3. Stop-and-reverse caps the winners.** Crossover systems are *low win rate,
high payoff* — maybe 35–45% winners, but the winners must be large. Exiting only
at the reverse crossover means giving back a large part of every trend before
exiting, which compresses exactly the payoff the system depends on.

**4. Costs are brutal at this frequency.** 24 bps round trip against a per-trade
edge of a few tens of bps is a very large tax.

**So the brief's real question is not "does the crossover work?" — it is "can you
tell in advance which crossovers are worth taking?"** That reframing is the whole
assignment, and stating it explicitly signals that you understand the problem.

## What the LTQ hypothesis is actually claiming

> *"We believe profitable trades are often accompanied by a sudden increase in LTQ
> in the direction of the trade."*

That is an **order-flow** hypothesis. The intuition: informed participants trade
in size, so a burst of large trades signals conviction, and a crossover confirmed
by large trades is more likely to follow through than one on thin volume.

This is a real and well-studied idea. Related concepts you can name:

- **Order flow imbalance (OFI)** — net signed volume, the difference between
  buyer- and seller-initiated trades
- **Trade size distribution** — institutions historically traded larger clips;
  algorithmic slicing has muddied this, which is itself a caveat worth raising
- **VPIN** (Volume-Synchronized Probability of Informed Trading) — a measure of
  order-flow toxicity built from signed volume buckets
- **Kyle's lambda** — price impact per unit of order flow

### ⚠️ The critical gap in the hypothesis as stated

**LTQ alone is directionless.** A 50,000-share trade tells you size was involved.
It does **not** tell you whether it was a buy or a sell.

The phrase *"in the direction of the trade"* is doing enormous work in that
sentence, and raw LTQ cannot deliver it. What you need is **signed volume** —
classify each trade as buyer- or seller-initiated:

| Method | Rule | Notes |
|---|---|---|
| **Quote rule** | LTP > mid → buy; LTP < mid → sell | Simple, needs top-of-book, ~85% accurate |
| **Tick rule** | LTP > previous LTP → buy | Works without quotes; less accurate |
| **Lee–Ready** | Quote rule, tick rule as tiebreak at mid | The standard reference |

Then the feature becomes:

```
signed_ltq  = ltq × (+1 if buyer-initiated else −1)
ofi_2m      = Σ signed_ltq over 2 minutes
ofi_ratio   = ofi_2m / Σ|signed_ltq| over 20 minutes    ∈ [−1, +1]
```

Now you can test the hypothesis as literally stated: *is a BUY crossover with
strongly positive OFI more profitable than one with negative OFI?*

**This is the single highest-value upgrade available, and identifying it is the
strongest thing you can say about this assignment.** It shows you read the
hypothesis critically rather than implementing it verbatim.

The building blocks are already in place: `Quote` has `bid_price`, `ask_price`
and `mid`, so the quote rule is a few lines in `TickWindow.add`.

## Microstructure, briefly

- **Bid-ask spread** — the round-trip cost of immediacy. A wide spread means a
  thin book, and thin books gap.
- **Order book imbalance** — `(bid_qty − ask_qty) / (bid_qty + ask_qty)`. Weak
  short-horizon predictive power for the next tick; noisy, and easily spoofed.
- **Depth ratio** — same information, unbounded scale.

The honest caveat: **top-of-book depth is the least reliable data in the feed.**
It updates thousands of times per second, is routinely spoofed, and a polling
client sees an arbitrary snapshot. Features built on it are noisy by nature — which
is worth saying out loud rather than presenting them as strong signals.

## And the liquidity filter

The brief requires bid **and** ask quantity above 10,00,000. Real NSE cash
top-of-book depth in the ₹30–500 band is typically **hundreds to low thousands**.
As specified, this filter returns an empty set.

Implemented exactly as asked, with `--relax-liquidity` to demonstrate the rest of
the pipeline. **Show both and explain why** — that reads as understanding the
market, not as failing the spec.

---

# PART 3 — The edge

Ordered by impact per unit of effort.

## 1. Meta-labeling — name this concept

Your architecture already *is* meta-labeling, from López de Prado's
*Advances in Financial Machine Learning*. Knowing the term is a genuine edge.

**The idea:** split the problem in two.

- A **primary model** decides *direction* — here, the SMMA crossover
- A **secondary (meta) model** decides *whether to act* — a binary
  take/skip on the primary model's signal

Why it works better than one model predicting direction:

- The secondary model solves an **easier problem**. "Is this signal good?" has far
  more learnable structure than "which way will price go?"
- You can tune **precision on the ACCEPT class** independently, which is what
  actually matters — you only lose money on trades you take
- The primary model stays interpretable and rule-based

> *"The architecture is meta-labeling: the SMMA crossover is the primary model
> determining direction, and the ML layer is a secondary model deciding whether to
> act on each signal. That's why the label is 'was this trade profitable' rather
> than 'which way did price go'."*

That sentence is worth a lot in an interview.

## 2. Signed order flow (see Part 2)

The highest-value feature upgrade. Turns a directionless LTQ into a directional
one, and makes the brief's hypothesis actually testable.

## 3. Triple-barrier labeling

Current labelling: profitable at the reverse crossover, or not. That conflates
"a good entry" with "a good exit rule".

**Triple barrier** sets three exits and labels by whichever is hit first:

```
  upper barrier   +2σ   → label +1   (profit target)
  lower barrier   −1σ   → label −1   (stop loss)
  vertical barrier  T   → label  0   (time out)
```

Barriers scaled by **realised volatility**, so a 1% move in a quiet stock and a 1%
move in a volatile one are not treated as equivalent.

Why it is better: it labels the **quality of the entry** rather than the outcome
of one particular exit rule, and it directly encodes a risk/reward ratio.

## 4. Purged cross-validation with an embargo

Time-ordered splitting (already implemented) is necessary but **not sufficient**.

Trades **overlap in time**. A trade opened at 10:00 and closed at 14:00 shares
information with one opened at 13:00. Put one in train and one in test and you
have leakage even with a chronological split.

**The fixes:**
- **Purging** — drop training samples whose label period overlaps the test set
- **Embargo** — additionally drop a small window after the test set, because serial
  correlation leaks backwards too

Naming purging and embargo is a strong signal of real quant-ML exposure.

## 5. Stop measuring accuracy — **partly built**

Accuracy is the wrong metric here, for two reasons.

**You only act on ACCEPTs.** A model that says AVOID to everything can score 60%
accuracy and be worthless. **Precision on the ACCEPT class** is what matters.

**Better still — expected value:**

```
EV = P(win) × avg_win − P(loss) × avg_loss − costs
```

A 40% win rate with a 3:1 payoff beats a 60% win rate with a 1:2 payoff. Report:

| Metric | Why |
|---|---|
| Precision @ ACCEPT | You only lose on trades you take |
| EV per trade, net of costs | The only number that pays rent |
| Profit factor | Gross wins / gross losses |
| Sharpe / Sortino | Return per unit of risk; Sortino ignores upside vol |
| Max drawdown | What you must survive |
| **Deflated Sharpe** | Corrects for having tried many configurations |

**Deflated Sharpe Ratio** deserves a mention: if you test 50 variants and pick the
best, its Sharpe is inflated by selection. DSR corrects for the number of trials.
Almost nobody at this level mentions multiple-testing bias.

**What is built:** profit factor, max drawdown, expectancy per trade and per bucket,
and a **threshold sweep** that optimises total net P&L rather than win rate — because
three trades at 100% is worse than eighty at 55%. `analytics.calibration` adds Brier
score and expected calibration error, which matter more than AUC once you are
thresholding on the probability itself.

**What is not:** Sharpe/Sortino (they need a return series with a defined holding
period, and this system is always-in-the-market with no flat time) and the Deflated
Sharpe correction. The multiple-testing problem is real here and is *disclosed* rather
than corrected: 19 features are ranked with no correction to their p-values, and the
report and README both say to read them as a ranking aid, not as significance.

## 6. Alternative bars — the clock is a choice

Time bars are the default and they are statistically poor: markets are not
uniformly active, so 1-minute bars oversample lunchtime and undersample the open.

| Bar type | Closes after | Property |
|---|---|---|
| Time | 1 minute | Familiar; poor statistics |
| **Tick** | N trades | Better-behaved returns |
| **Volume** | N shares | Closer to information arrival |
| **Dollar** | ₹N traded | Robust to price level changes |
| **Imbalance** | order flow imbalance threshold | Samples when *informed* activity happens |

Volume or dollar bars produce returns closer to IID and normally distributed,
which is what most statistical machinery assumes. Given the assignment's focus on
traded quantity, **volume bars are a natural fit** — and proposing them shows you
understand that the sampling clock is a modelling decision, not a given.

## 7. A regime filter

Crossovers fail in ranging markets. Add a regime classifier and only take signals
in trending conditions:

- **ADX > 25** — the classic, simple and defensible
- **Efficiency ratio** — net move / sum of absolute moves over N bars
- **Hurst exponent** — >0.5 trending, <0.5 mean-reverting
- **Volatility regime** — realised vol percentile vs its own history

Cheap to add, and it attacks the largest single source of losses.

## 8. Fix the train/serve skew

Currently documented, not solved: historical bars carry no tick data, so six
features are **zero in training and populated live**.

Three options, in increasing order of correctness:

1. Train on bar features only — honest, immediately correct, loses microstructure
2. Persist live feature snapshots and retrain on those — correct, needs weeks
3. Reconstruct tick features from bar proxies — approximate but available now

**Option 2 is the right answer**, and it is also what makes the LTQ hypothesis
testable at all. The infrastructure exists: signals already persist their full
feature snapshot to SQLite. Run it for a few weeks and retrain on real live
snapshots.

## 9. Walk-forward validation — **built**

A single train/test split tests one moment. **Walk-forward** retrains on a rolling
window and tests on the next period, repeatedly — which tests whether the edge
*persists*, and surfaces regime dependence a single split hides.

Implemented in `analytics.walk_forward_scores` and shown on the **Model** tab. It
immediately earned its keep: the single chronological split reports AUC **0.573**,
five expanding windows report **0.529 ± 0.099** with a range of 0.347–0.642. The
mean is barely above a coin flip and the spread says one window was carrying the
headline number. Without this, the submission would have quoted 0.573 and believed it.

The remaining upgrade is a **purge and embargo** between train and test (see §4):
trades whose holding periods straddle a fold boundary still leak slightly. With a
mean hold time of ~21 hours on this dataset that is not a rounding error, and it is
the first thing to fix before trusting any fold number.

## 10. Position sizing

The system currently treats every accepted trade identically. It shouldn't.

- Size proportional to model confidence
- **Kelly fraction** (in practice, half-Kelly — full Kelly is too aggressive for
  estimated edges)
- Volatility targeting — smaller size in volatile names so each position
  contributes equal risk

Turning a probability into a *size* rather than a binary is the natural next step,
and it is where meta-labeling pays off most.

---

## The traps — what goes wrong in trading ML

| Trap | What it looks like | Guard |
|---|---|---|
| **Look-ahead bias** | Suspiciously good backtest | Only use data available at decision time. Check every `shift()` |
| **Survivorship bias** | Strategy works on today's index | Use point-in-time constituents |
| **Random split on time series** | Test ≈ train accuracy | Time-ordered split ✅ *(implemented)* |
| **Overlapping labels** | Leakage despite a chronological split | Purging + embargo |
| **Overfitting a small sample** | Train 0.86, test 0.45 | More data; fewer features; simpler model |
| **Multiple testing** | "Best of 50 variants" | Deflated Sharpe; hold out a final untouched set |
| **Non-stationarity** | Worked last year, not now | Walk-forward; regime features; retrain schedule |
| **Ignoring costs** | Profitable gross, negative net | Charge both legs ✅ *(implemented)* |
| **Train/serve skew** | Live ≠ backtest | One feature path ⚠️ *(documented, not solved)* |
| **Class imbalance** | High accuracy, never accepts | `class_weight="balanced"` ✅ *(implemented)* |

---

## If you get asked "what would you do with another month?"

Answer in priority order — the ordering is itself the signal:

1. **Collect live data.** Nothing else matters without it. Run the pipeline every
   session, persisting feature snapshots. Everything below depends on this.
2. **Add signed order flow.** Makes the brief's own hypothesis testable.
3. **Switch to triple-barrier labels** with volatility-scaled barriers.
4. **Purged CV with embargo**, and report precision-on-ACCEPT and EV rather than
   accuracy.
5. **Add a regime filter** and measure how much of the loss it removes.
6. **Walk-forward validation** to test whether the edge persists.
7. **Only then** consider a more complex model. Going from RandomForest to a
   neural network with 400 samples and leaky labels is the wrong lever — and
   saying so is a stronger answer than proposing an LSTM.

> The point to make: **the data and the labels are the bottleneck, not the
> algorithm.** Most candidates will reach for a bigger model. The gains here are
> in signed order flow, honest labelling, and leak-free validation.

---

## The three sentences worth memorising

**On the architecture:**
> "It's meta-labeling — the SMMA crossover is the primary model determining
> direction, and the ML layer is a secondary model deciding whether to act. That's
> why the label is 'was this trade profitable' rather than 'which way did price
> go'."

**On the LTQ hypothesis:**
> "Raw LTQ is directionless — a large trade tells you size was involved, not
> whether it was a buy or a sell. To test 'a spike in the direction of the trade'
> you need signed volume, classifying each trade as buyer- or seller-initiated
> with the quote rule. That's the first thing I'd add."

**On the model's honest quality:**
> "ROC AUC is about 0.58 on simulated data, which is barely above chance — and
> that's the correct result, because the simulator has no learnable relationship
> between entry features and outcome. Whether the hypothesis holds is an open
> empirical question that only real market data answers. I'd rather show that
> number and explain it than present one I can't defend."
