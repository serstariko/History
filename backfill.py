"""
Backfill missing prefix of a time series while preserving historical distribution.

Method:
  1. Infer frequency and build a regular calendar from the requested start date.
  2. Decompose the observed series (STL when enough data; seasonal means otherwise).
  3. Extrapolate trend backward to the requested start.
  4. Reuse the seasonal profile from history.
  5. Resample residuals with a seasonal block bootstrap to keep distribution & ACF.
  6. Blend the seam at the junction with the first observed value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

ModelType = Literal["additive", "multiplicative"]


@dataclass
class BackfillResult:
    """Full series after backfill plus diagnostics."""

    series: pd.Series
    generated: pd.Series
    observed: pd.Series
    residuals_pool: pd.Series
    season_period: int
    freq: str
    model: ModelType
    n_generated: int


def _infer_freq(index: pd.DatetimeIndex) -> str:
    if len(index) < 2:
        raise ValueError("Need at least 2 observations to infer frequency.")

    inferred = pd.infer_freq(index)
    if inferred:
        return inferred

    deltas = pd.Series(index[1:]).diff().dropna().dt.total_seconds()
    median_sec = float(deltas.median())
    day = 24 * 3600
    if abs(median_sec - day) < 3600:
        return "D"
    if abs(median_sec - 7 * day) < 12 * 3600:
        return "W"
    if 27 * day <= median_sec <= 32 * day:
        return "MS"
    if 89 * day <= median_sec <= 93 * day:
        return "QS"
    if 360 * day <= median_sec <= 370 * day:
        return "YS"
    if abs(median_sec - 3600) < 120:
        return "h"
    raise ValueError(
        f"Could not infer a regular frequency (median step ≈ {median_sec / day:.2f} days). "
        "Please provide a regularly spaced series."
    )


def _season_period_for_freq(freq: str, index: pd.DatetimeIndex) -> int:
    f = freq.upper()
    if f.startswith("H"):
        return 24
    if f.startswith("D"):
        # Prefer annual seasonality when the history is long enough for STL.
        if len(index) >= 2 * 365:
            return 365
        return 7
    if f.startswith("W"):
        return 52
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    if f.startswith("Y") or f.startswith("A"):
        return 1
    if len(index) >= 14:
        return 7
    return max(2, min(12, len(index) // 3))


def _make_full_index(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
) -> pd.DatetimeIndex:
    idx = pd.date_range(start=start, end=end, freq=freq)
    if len(idx) == 0:
        raise ValueError("Empty date range for the requested start/end.")
    return idx


def _seasonal_means(values: pd.Series, period: int) -> pd.Series:
    positions = np.arange(len(values)) % period
    means = pd.Series(values.to_numpy(), index=positions).groupby(level=0).mean()
    seasonal = pd.Series(
        [float(means.loc[p]) for p in positions],
        index=values.index,
        name="seasonal",
    )
    return seasonal - seasonal.mean()


def _decompose_working(
    y_work: pd.Series,
    period: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose series already in working space (levels or logs)."""
    if period < 2 or len(y_work) < 2 * period:
        seasonal = _seasonal_means(y_work, period)
        trend = y_work.rolling(
            window=max(3, period | 1), center=True, min_periods=1
        ).mean()
        resid = y_work - trend - seasonal
        return trend, seasonal, resid

    res = STL(y_work, period=period, robust=True).fit()
    return res.trend, res.seasonal, res.resid


def _extrapolate_trend_backward(
    trend: pd.Series,
    target_index: pd.DatetimeIndex,
    n_back: int,
) -> pd.Series:
    if n_back <= 0:
        return trend.reindex(target_index)

    y = trend.dropna()
    if len(y) < 2:
        fill = float(y.iloc[0]) if len(y) else 0.0
        out = trend.reindex(target_index)
        out.iloc[:n_back] = fill
        return out

    window = min(len(y), max(8, len(y) // 5))
    head = y.iloc[:window]
    x = np.arange(len(head), dtype=float)
    slope, intercept = np.polyfit(x, head.to_numpy(dtype=float), 1)
    back_x = np.arange(-n_back, 0, dtype=float)
    back_vals = intercept + slope * back_x

    out = trend.reindex(target_index)
    out.iloc[:n_back] = back_vals
    return out


def _seasonal_pattern(seasonal: pd.Series, period: int) -> np.ndarray:
    vals = seasonal.dropna().to_numpy(dtype=float)
    if len(vals) < period:
        tiled = np.resize(vals, period)
        return tiled - tiled.mean()

    n = (len(vals) // period) * period
    mat = vals[:n].reshape(-1, period)
    pattern = mat.mean(axis=0)
    return pattern - pattern.mean()


def _block_bootstrap_residuals(
    residuals: np.ndarray,
    n_needed: int,
    period: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    resid = residuals[np.isfinite(residuals)]
    if len(resid) == 0:
        return np.zeros(n_needed)

    block_size = max(1, min(block_size, len(resid)))
    out = np.empty(n_needed)
    pos = 0
    while pos < n_needed:
        phase = pos % period
        candidate_starts = [
            i
            for i in range(0, len(resid) - block_size + 1)
            if i % period == phase
        ]
        if not candidate_starts:
            candidate_starts = list(range(0, max(1, len(resid) - block_size + 1)))
            if not candidate_starts:
                candidate_starts = [0]

        start = int(rng.choice(candidate_starts))
        block = resid[start : start + block_size]
        take = min(len(block), n_needed - pos)
        out[pos : pos + take] = block[:take]
        pos += take

    return out


def _blend_seam(
    generated: np.ndarray,
    first_observed: float,
    blend_len: int,
) -> np.ndarray:
    if blend_len <= 0 or len(generated) == 0:
        return generated

    blend_len = min(blend_len, len(generated))
    gap = first_observed - generated[-1]
    weights = np.linspace(0.0, 1.0, blend_len + 1)[1:]
    generated = generated.copy()
    generated[-blend_len:] = generated[-blend_len:] + weights * gap
    return generated


def backfill_series(
    series: pd.Series,
    desired_start: pd.Timestamp | str,
    *,
    model: ModelType = "additive",
    season_period: Optional[int] = None,
    block_size: Optional[int] = None,
    blend_len: Optional[int] = None,
    freq: Optional[str] = None,
    random_state: Optional[int] = 42,
) -> BackfillResult:
    """
    Generate the missing prefix of `series` so the result starts at `desired_start`.

    Parameters
    ----------
    series :
        Observed time series with a DatetimeIndex.
    desired_start :
        Date from which the completed series must begin.
    model :
        ``additive`` (default) or ``multiplicative`` (requires positive values).
    season_period :
        Seasonal period for STL / bootstrap. Auto-detected if omitted.
    block_size :
        Bootstrap block length. Defaults to ``season_period``.
    blend_len :
        Points used to blend the seam. Defaults to ``max(1, season_period // 4)``.
    freq :
        Pandas offset alias. Inferred if omitted.
    random_state :
        RNG seed for reproducibility.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series.index must be a DatetimeIndex.")

    y = series.sort_index().astype(float).dropna()
    if y.index.has_duplicates:
        y = y[~y.index.duplicated(keep="first")]

    if len(y) < 4:
        raise ValueError("Need at least 4 observed points.")

    desired_start = pd.Timestamp(desired_start)
    observed_start = pd.Timestamp(y.index[0])
    observed_end = pd.Timestamp(y.index[-1])

    if desired_start >= observed_start:
        raise ValueError(
            f"desired_start ({desired_start.date()}) must be before the first "
            f"observed date ({observed_start.date()})."
        )

    if model == "multiplicative" and (y <= 0).any():
        raise ValueError("Multiplicative model requires strictly positive values.")

    freq = freq or _infer_freq(y.index)
    period = season_period or _season_period_for_freq(freq, y.index)
    period = max(2, int(period))
    block_size = int(block_size) if block_size else period
    blend_len = int(blend_len) if blend_len is not None else max(1, period // 4)
    rng = np.random.default_rng(random_state)

    full_index = _make_full_index(desired_start, observed_end, freq)
    y_grid = y.reindex(full_index)
    n_back = int(full_index.get_indexer([observed_start])[0])
    if n_back <= 0:
        raise ValueError(
            "No missing prefix on the inferred calendar. "
            "Check desired_start and series frequency."
        )

    observed_on_grid = y_grid.iloc[n_back:].dropna()
    if len(observed_on_grid) < 4:
        raise ValueError("Too few observations after aligning to the calendar grid.")

    # Working space: levels (additive) or logs (multiplicative)
    if model == "multiplicative":
        y_work = np.log(observed_on_grid)
    else:
        y_work = observed_on_grid.copy()

    trend_w, seasonal_w, resid_w = _decompose_working(y_work, period)

    full_trend_w = _extrapolate_trend_backward(trend_w, full_index, n_back)
    pattern = _seasonal_pattern(seasonal_w, period)

    full_seasonal_w = pd.Series(index=full_index, dtype=float)
    for i in range(len(full_index)):
        full_seasonal_w.iloc[i] = pattern[(i - n_back) % period]

    boot = _block_bootstrap_residuals(
        resid_w.to_numpy(dtype=float),
        n_needed=n_back,
        period=period,
        block_size=block_size,
        rng=rng,
    )

    generated_work = (
        full_trend_w.iloc[:n_back].to_numpy()
        + full_seasonal_w.iloc[:n_back].to_numpy()
        + boot
    )

    if model == "multiplicative":
        target_seam = float(np.log(observed_on_grid.iloc[0]))
        generated_work = _blend_seam(generated_work, target_seam, blend_len)
        generated_vals = np.exp(generated_work)
    else:
        target_seam = float(observed_on_grid.iloc[0])
        generated_vals = _blend_seam(generated_work, target_seam, blend_len)

    generated = pd.Series(
        generated_vals, index=full_index[:n_back], name=y.name or "value"
    )
    completed = pd.concat([generated, observed_on_grid])
    completed = completed[~completed.index.duplicated(keep="last")].sort_index()
    completed.name = y.name or "value"

    return BackfillResult(
        series=completed,
        generated=generated,
        observed=observed_on_grid,
        residuals_pool=resid_w,
        season_period=period,
        freq=freq,
        model=model,
        n_generated=n_back,
    )


def load_series_from_frame(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
) -> pd.Series:
    """Parse a dataframe into a clean DatetimeIndex series."""
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(f"Columns must include '{date_col}' and '{value_col}'.")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    out = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates), name=value_col)
    out = out[out.index.notna()].dropna().sort_index()
    if out.index.has_duplicates:
        out = out.groupby(level=0).mean()
    return out
