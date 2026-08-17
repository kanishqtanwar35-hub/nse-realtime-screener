"""
Technical indicators.

SMMA (Smoothed Moving Average, a.k.a. Wilder's / RMA) is defined recursively:

    SMMA[0]  = SMA(price, n)                       <- seed at index n-1
    SMMA[i]  = (SMMA[i-1] * (n - 1) + price[i]) / n

That recursion is algebraically identical to an EWM with alpha = 1/n and
adjust=False, seeded with the SMA. We implement it with numpy directly because
the explicit form is unambiguous and easy to test - and at O(n) over a few
thousand bars it is not the bottleneck. `smma_ewm` proves the equivalence and
is the fast path for wide frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smma(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothed moving average.

    Returns NaN for the first `period - 1` observations - there is genuinely no
    value there, and forward-filling it would fabricate signal.
    """
    if period <= 0:
        raise ValueError("period must be positive")

    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = values.size
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(out, index=series.index, name=f"smma_{period}")

    # Seed with the simple average of the first window.
    seed = np.nanmean(values[:period])
    if not np.isfinite(seed):
        return pd.Series(out, index=series.index, name=f"smma_{period}")
    out[period - 1] = seed

    inv = 1.0 / period
    prev = seed
    for i in range(period, n):
        price = values[i]
        if not np.isfinite(price):
            # A gap must not reset the average: carry the level forward.
            out[i] = prev
            continue
        prev = (prev * (period - 1) + price) * inv
        out[i] = prev

    return pd.Series(out, index=series.index, name=f"smma_{period}")


def smma_ewm(series: pd.Series, period: int) -> pd.Series:
    """
    Vectorised SMMA via pandas ewm. Same numbers as `smma`, ~50x faster on long
    series. Used when computing over many symbols at once.
    """
    if period <= 0:
        raise ValueError("period must be positive")
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if len(s) < period:
        return pd.Series(np.nan, index=series.index, name=f"smma_{period}")

    seeded = s.copy()
    seeded.iloc[: period - 1] = np.nan
    seeded.iloc[period - 1] = s.iloc[:period].mean()
    out = seeded.ewm(alpha=1.0 / period, adjust=False, ignore_na=True).mean()
    out.iloc[: period - 1] = np.nan
    return out.rename(f"smma_{period}")


def rate_of_change(series: pd.Series, window: int) -> pd.Series:
    """Percentage change over `window` bars. Guarded against a zero base."""
    prev = series.shift(window)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (series - prev) / prev.abs() * 100.0
    return out.replace([np.inf, -np.inf], np.nan).rename(f"roc_{window}")


def realised_volatility(close: pd.Series, window: int, annualise: bool = False) -> pd.Series:
    """
    Standard deviation of log returns over `window` bars, in percent.

    On 1-minute bars, annualising means x sqrt(375 * 252) - useful for comparing
    across timeframes, misleading if you forget it is a 1-minute estimate.
    """
    log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
    vol = log_ret.rolling(window, min_periods=max(2, window // 2)).std() * 100.0
    if annualise:
        vol = vol * np.sqrt(375 * 252)
    return vol.rename(f"volatility_{window}")


def momentum(close: pd.Series, window: int) -> pd.Series:
    """Simple price momentum: percent change over the window."""
    return rate_of_change(close, window).rename(f"momentum_{window}")


def rolling_sum(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.rolling(window, min_periods=min_periods or 1).sum()


def rolling_mean(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    return series.rolling(window, min_periods=min_periods or 1).mean()


def crossover(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    +1 where `fast` crosses ABOVE `slow`, -1 where it crosses BELOW, else 0.

    Semantics, and why they are not the naive `prev <= 0 and now > 0`:

      * A crossover is a change in the SIGN of (fast - slow), compared against
        the last time the sign was DECIDED. Equality (spread exactly 0) is
        undecided, not "above" - so the comparison walks back past any run of
        zeros to the last non-zero sign.

        This matters: with the naive rule, a fast line that rises to touch the
        slow line and then falls away again emits a phantom SELL, because the
        touch counts as "was >= 0". Touch-and-retreat is not a crossover, and on
        flat SMMA pairs it happens often enough to pollute the signal stream.

      * Bars where either series is NaN produce 0. SMMA(120) is NaN for its
        first 119 bars, and a naive comparison would fire on the first valid
        bar - a signal generated from data that does not exist yet.
    """
    diff = fast - slow
    sign = pd.Series(np.sign(diff.to_numpy(dtype=float)), index=diff.index)
    sign[diff.isna()] = np.nan

    # Last DECIDED sign strictly before each bar: drop zeros, carry forward.
    decided = sign.replace(0.0, np.nan).ffill().shift(1)

    up = (sign > 0) & (decided < 0)
    down = (sign < 0) & (decided > 0)

    out = pd.Series(0, index=fast.index, dtype=int)
    out[up.fillna(False)] = 1
    out[down.fillna(False)] = -1
    return out.rename("crossover")


def add_indicators(
    bars: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    *,
    price_col: str = "close",
    fast: bool = True,
) -> pd.DataFrame:
    """
    Attach SMMA fast/slow, their spread, spread RoC, and the crossover marker.

    Returns a copy; the input frame is never mutated.
    """
    if price_col not in bars.columns:
        raise KeyError(f"price column {price_col!r} not in {list(bars.columns)}")

    df = bars.copy()
    fn = smma_ewm if fast else smma
    df["smma_fast"] = fn(df[price_col], fast_period)
    df["smma_slow"] = fn(df[price_col], slow_period)

    df["smma_spread"] = df["smma_fast"] - df["smma_slow"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["smma_spread_pct"] = (df["smma_spread"] / df["smma_slow"].abs()) * 100.0
    df["smma_spread_pct"] = df["smma_spread_pct"].replace([np.inf, -np.inf], np.nan)

    # Rate of change of the SPREAD, not of price: how fast the lines converge or
    # diverge is what tells you a crossover is imminent or losing conviction.
    df["smma_spread_roc"] = df["smma_spread"].diff()
    df["crossover"] = crossover(df["smma_fast"], df["smma_slow"])
    return df
