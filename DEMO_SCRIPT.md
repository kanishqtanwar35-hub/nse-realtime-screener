# Screen recording script

The assignment asks for a recording showing **program execution, live stock
screening, the real-time dashboard, and AI/ML analysis and predictions.**

Target **8 minutes**. Anything longer and the reviewer skips. If you have to cut,
cut the executable section — not the Analytics or Model tabs.

**Record with:** OBS Studio (free) or Windows `Win+G`. 1080p, capture the whole
screen, keep your face off camera. **Say the numbers out loud** — a silent
screen recording of a table is much weaker than one where you explain what the
reviewer is looking at.

---

## Before you hit record

```bash
# 1. clean slate so the run is reproducible on camera
rm -rf data/screener.db data/trade_model.joblib logs/*

# 2. pre-train, so you are not filming a 2-minute wait
python scripts/train_model.py --history-minutes 5000

# 3. generate the report you will show at the end
python scripts/run_analysis.py

# 4. confirm the dashboard opens, and click through the Analytics and Model
#    tabs once so the walk-forward cache is warm before you record
python scripts/run_dashboard.py
```

**Check your screen for secrets before recording.** Close `.env`, clear your
terminal history, and if a broker token is in your shell scrollback, open a new
terminal. The brief explicitly requires credentials to be masked.

---

## The script

### 0:00–0:40 — What it is
Show the repo tree. One sentence per layer:

> "Data providers, ingestion, screening, indicators, signals, backtest, ML, and
> the dashboard. Every layer is a separate module and there are 65 tests."

Run `pytest` on camera. It takes 8 seconds and it makes the point that the code
is tested better than any claim you could make about it.

### 0:40–1:40 — Screening (requirements 1 & 2)
```bash
python scripts/run_live.py --provider angelone --cycles 3
```
Point at the log line: `screen: N passed / M`.

**Then say the thing that shows judgement:**

> "The spec asks for bid and ask quantity both above ten lakh. On real NSE cash
> data in the ₹30–500 band, top-of-book depth is usually hundreds to a few
> thousand — so this filter as written returns almost nothing. I've implemented
> it exactly as specified, and I've also made it configurable so the rest of the
> pipeline can be demonstrated."

Re-run with `--relax-liquidity`. Now symbols pass. **That contrast is the single
most valuable 30 seconds in your recording.**

### 1:40–3:10 — The dashboard (requirements 3–7)
```bash
python scripts/run_dashboard.py
```
Walk the **Screen** tab left to right, naming each requirement as you point:

- `ltp` — the price band filter
- `smma_20`, `smma_120`, `smma_signal` — requirement 3
- `etq_5m / 20m / 60m` — requirement 4
- `avg_ltp_20m / 60m` — requirement 5
- `bid_price / bid_qty / ask_price / ask_qty` — requirement 6
- `ltq`, `ltq_avg_2m`, `ltq_avg_5m`, **`ltq_ratio_2_5`** — the ML hypothesis

> "One row per stock, auto-refreshing. The last column is the 2-minute versus
> 5-minute LTQ ratio — the comparison the brief singles out. Above 1.0 means
> trade size is stepping in right now."

Let it auto-refresh once on camera so the reviewer sees the numbers change.

### 3:10–4:00 — The Charts tab

Open **Charts**. Pick a symbol and let the price chart load.

> "Close price with SMMA 20 and SMMA 120, and every crossover marked on the bar
> it fired. The panel underneath is the spread between the two averages — every
> zero crossing down there is a marker up here. This is how you check the
> signals are right: crossovers are evaluated on **closed bars only**, so if the
> indicator repainted, the marker would sit a bar past the crossing. It doesn't."

Hover a marker so the tooltip shows on camera. Zoom in on one crossover.

**This is the fastest correctness proof in the whole video** — a table of SMMA
values proves the numbers exist; the chart proves they're right.

### 4:00–5:00 — Signals and ML (requirements 8–12)
Open **Signals & Decisions**. For one row, read the whole line aloud:

> "SMMA20 crossed above SMMA120 on IDEA. The model gives it a 63% win
> probability, so ACCEPT. The explanation names the drivers: trade-size spike
> supports it, wide bid-ask spread weighs against."

Then **Trades**: equity curve, win rate, profit factor, and the log showing
`P&L (pts)` **and** `Net P&L %`, MAE and MFE.

> "P&L in points is exactly the spec's Sell LTP minus Buy LTP. I also track
> percentage, because points aren't comparable across a ₹35 stock and a ₹480 one,
> and costs are charged on both legs — a trade that looks green gross can be red
> net."

### 5:00–6:10 — Analytics tab (requirement 22, done properly)

This is the section that separates the submission from everyone else's.

Open **Analytics**. Point at the separation chart:

> "This is every feature used *alone* as a score, measured as ROC AUC minus 0.5.
> It's rank-based, so a single outsized trade can't manufacture an edge. The
> answer on this data is that nothing separates winners from losers — the best
> feature reaches 0.53 against 0.5 for a coin flip."

Change the feature in the bucket explorer so the chart redraws live:

> "This is the filter builder. Quantile buckets, because these features are
> ratios that pile up near 1.0 and equal-width bins would put everything in one
> bar. The number over each bar is the trade count — a tall bar over six trades
> is noise, and the chart says so."

Then the excursion scatter:

> "Every trade as worst excursion against best excursion. Winners capture only
> 53% of the move they showed, which means waiting for the reverse crossover is
> giving profit back. A trailing exit has more upside here than any entry
> filter — that's the most actionable thing in the whole analysis."

### 6:10–7:10 — Model tab, and the honest number

Feature importance first, then the out-of-sample section:

> "The single train/test split says AUC 0.573. Walk-forward across five
> expanding windows says 0.529 plus or minus 0.099, ranging from 0.35 to 0.64.
> The mean is barely above a coin flip and the spread says one window was
> carrying it. I'd rather show you that than the 0.573."

Then the comparison chart — **do not skip this**:

> "And here's the model against the only baseline that matters. It *trails*
> taking every signal. On simulated data that's the correct result: the price
> process has no learnable relationship between entry state and outcome. There's
> also a real train/serve gap — historical bars carry no tick data, so the LTQ
> features are zero in training but populated live. Collecting live snapshots for
> a few sessions and retraining is what makes the LTQ hypothesis testable."

**This is the most senior thing you can say in the whole video.** Most
submissions will show a suspiciously high accuracy and no awareness of why.

Finish on the threshold sweep:

> "0.55 was a convention, not a finding, so I swept every threshold on
> out-of-sample scores. That's how you'd defend the setting or replace it."

### 7:10–7:40 — The report and the executable

```bash
python scripts/run_analysis.py --open
```

> "Same analysis as a standalone HTML report, so a reviewer doesn't have to
> launch Streamlit. The findings are sentences generated from the numbers — if
> the data supports nothing, it says so."

Then run `dist\nse-screener\nse-screener.exe --relax-liquidity --cycles 3` and
the dashboard exe. Show it working with no Python installed.

### 7:40–8:00 — Close
> "Modular, 104 tests, no hardcoded credentials, and it falls back to simulated
> data if the broker is unreachable so the dashboard never dies mid-session.
> The documented gaps are in REQUIREMENTS.md."

---

## Do not

- Do not show `.env`, tokens, or your client ID at any point.
- Do not claim the ML model is accurate. Show the number and explain it.
- Do not hide the empty screen at the strict liquidity threshold — **explain**
  it. A reviewer who knows NSE will test exactly that, and an explanation beats
  a silently empty table.
- Do not speed up or cut the auto-refresh. Seeing it tick is the proof it's live.
- Do not skip the strategy comparison because the model loses. A candidate who
  measures their model against a baseline and reports the loss is worth more than
  one whose model "wins" against nothing.

---

## One timing warning

The first load of the **Analytics** or **Model** tab refits five walk-forward
models and takes about **12 seconds**, with a spinner. It is cached for 15
minutes afterwards. Open those tabs once before you start recording so the
cache is warm, or narrate the wait — it's five models being fitted, not a hang.
