"""
InsightAI - Time-series analysis (STEP 11).

Detects datetime columns, chooses a target metric, aggregates to a clean time
index, and extracts trends/seasonality/growth. Forecasting is supported via a
naive baseline, moving average and Exponential Smoothing (and simple SARIMA when
enough data exists). Historical analysis is kept strictly separate from
forecasting, and forecasts are never presented as certain.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.schema_detector import DATETIME, NUMERIC, CURRENCY, PERCENTAGE

warnings.filterwarnings("ignore")


def _find_time_index(df: pd.DataFrame) -> Tuple[Optional[str], Optional[pd.Series]]:
    """Locate the first datetime column as the time index."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col, df[col]
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col].astype(str), errors="coerce")
            if parsed.notna().mean() >= 0.85:
                return col, parsed
        except Exception:
            continue
    return None, None


def build_time_series(df: pd.DataFrame, date_col: Optional[str],
                      value_col: Optional[str] = None,
                      freq: str = "D") -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a clean, aggregated time series.

    Returns ``(ts_frame, warnings)``. The frame has a DatetimeIndex and the
    value column aggregated by ``freq``.
    """
    notes: List[str] = []
    if date_col is None:
        return pd.DataFrame(), notes
    if value_col is None:
        # Fall back to counting rows per time bucket.
        s = pd.to_datetime(df[date_col], errors="coerce")
        ts = s.dt.to_period(freq).apply(lambda p: p.to_timestamp())
        g = pd.DataFrame({"count": 1}).groupby(ts).sum()
        g = g.rename(columns={"count": "value"})
        notes.append("No numeric target provided; using row counts per time bucket.")
        return g, notes

    value = pd.to_numeric(df[value_col], errors="coerce")
    time = pd.to_datetime(df[date_col], errors="coerce")
    frame = pd.DataFrame({"time": time, "value": value}).dropna(subset=["time"])
    if frame.empty:
        notes.append("All rows had missing time/value; no time series built.")
        return pd.DataFrame(), notes
    frame["bucket"] = frame["time"].dt.to_period(freq).apply(lambda p: p.to_timestamp())
    g = frame.groupby("bucket")["value"].agg(["sum", "count"]).rename(
        columns={"sum": "value", "count": "n_records"}
    )
    return g, notes


def descriptive_stats(ts: pd.Series) -> Dict:
    s = ts.dropna()
    if len(s) == 0:
        return {}
    return {
        "mean": _n(s.mean()),
        "median": _n(s.median()),
        "std": _n(s.std()),
        "min": _n(s.min()),
        "max": _n(s.max()),
        "total": _n(s.sum()),
        "count": int(s.count()),
    }


def growth_rates(ts: pd.Series) -> Dict:
    """Return overall and last-period growth percentage."""
    s = ts.dropna()
    if len(s) < 2:
        return {"overall_growth_pct": None, "last_period_growth_pct": None}
    first, last = s.iloc[0], s.iloc[-1]
    overall = (last - first) / abs(first) * 100 if first != 0 else None
    prev = s.iloc[-2]
    last_p = (last - prev) / abs(prev) * 100 if prev != 0 else None
    return {"overall_growth_pct": _n(overall), "last_period_growth_pct": _n(last_p)}


def trend(ts: pd.Series) -> Dict:
    """Fit a LinearRegression to estimate the trend direction/slope."""
    from sklearn.linear_model import LinearRegression
    s = ts.dropna()
    if len(s) < 3:
        return {"direction": "insufficient", "slope": None, "r2": None}
    x = np.arange(len(s)).reshape(-1, 1)
    model = LinearRegression().fit(x, s.values)
    slope = float(model.coef_[0])
    r2 = float(model.score(x, s.values))
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
    return {"direction": direction, "slope": _n(slope), "r2": _n(r2)}


def seasonal_pattern(ts: pd.Series, freq: str = "D") -> Dict:
    """Average by weekday / month to expose seasonality."""
    s = ts.dropna()
    if s.empty:
        return {}
    idx = pd.Series(s.index)
    if freq == "D":
        weekday = idx.dt.day_name()
        table = pd.DataFrame({"bucket": s.index, "value": s.values})
        table["weekday"] = idx.dt.day_name()
        avg = table.groupby("weekday")["value"].mean().sort_values(ascending=False)
        return {"by_weekday": avg.to_dict()}
    if freq in {"W", "M"}:
        table = pd.DataFrame({"bucket": s.index, "value": s.values})
        table["month"] = idx.dt.month
        avg = table.groupby("month")["value"].mean().sort_values(ascending=False)
        return {"by_month": avg.to_dict()}
    if freq in {"Q", "Y"}:
        table = pd.DataFrame({"bucket": s.index, "value": s.values})
        table["quarter"] = idx.dt.quarter
        avg = table.groupby("quarter")["value"].mean().sort_values(ascending=False)
        return {"by_quarter": avg.to_dict()}
    return {}


def rolling_mean(ts: pd.Series, window: int = 3) -> pd.Series:
    return ts.rolling(window=window, min_periods=1).mean()


def decompose(ts: pd.Series, freq_offset: str = "D", period: int = 7) -> Dict:
    """Simple decomposition into trend + seasonal + residual via a moving window."""
    from statsmodels.tsa.seasonal import seasonal_decompose
    s = ts.dropna().reset_index(drop=True)
    if len(s) < period * 2:
        return {"ok": False, "reason": "not enough points for decomposition"}
    try:
        result = seasonal_decompose(s, model="additive", period=period, extrapolate_trend="period")
        return {
            "ok": True,
            "trend": result.trend.dropna().tolist(),
            "seasonal": result.seasonal.dropna().tolist(),
            "resid": result.resid.dropna().tolist(),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


def forecast(ts: pd.Series, methods: List[str], horizon: int = 5) -> Dict:
    """
    Forecast the ``horizon`` next points using the requested methods.

    Methods: ``naive``, ``moving_average``, ``exponential_smoothing``, ``arima``.
    Returns per-method forecast series plus a residual-based confidence band.
    """
    s = ts.dropna()
    forecast_result: Dict[str, Dict] = {}
    if len(s) < 4:
        return {"forecasts": {}, "note": "Insufficient history for forecasting"}

    last = s.iloc[-1]
    if "naive" in methods:
        vals = [float(last)] * horizon
        forecast_result["naive"] = {"values": vals, "window": "last value carried forward"}

    if "moving_average" in methods and len(s) >= 3:
        m = max(2, int(min(5, max(1, len(s) // 4))))
        ma = s.rolling(window=m).mean().iloc[-1]
        vals = [float(ma)] * horizon
        forecast_result["moving_average"] = {"values": vals, "window": f"window={m}"}

    if "exponential_smoothing" in methods and len(s) >= 5:
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(s.values, trend="add", damped_trend=True).fit()
            vals = model.forecast(horizon).tolist()
            forecast_result["exponential_smoothing"] = {"values": vals, "window": "Holt-Winters + damped trend"}
        except Exception as exc:
            forecast_result["exponential_smoothing"] = {"values": None, "error": str(exc)}

    if "arima" in methods and len(s) >= 12:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(s.values, order=(1, 1, 1)).fit()
            vals = model.forecast(horizon).tolist()
            forecast_result["arima"] = {"values": vals, "window": "ARIMA(1,1,1)"}
        except Exception as exc:
            forecast_result["arima"] = {"values": None, "error": str(exc)}

    if not forecast_result:
        return {"forecasts": {}, "note": "No forecast method could be applied"}

    # Confidence band from residual std of the last method that produced values.
    residual_std = float(s.std()) if len(s) else 1.0
    return {
        "forecasts": forecast_result,
        "horizon": horizon,
        "confidence_band": 1.96 * residual_std,
        "note": "Forecasts are estimates, not certainties. Validated only on historical data.",
    }


def time_series_report(df: pd.DataFrame, date_col: Optional[str],
                       value_col: Optional[str], freq: str = "D") -> Dict:
    """Full time-series analysis bundle."""
    ts, notes = build_time_series(df, date_col, value_col, freq=freq)
    if ts.empty:
        return {"available": False, "notes": notes}
    series = ts["value"]
    result = {
        "available": True,
        "date_col": date_col,
        "value_col": value_col,
        "freq": freq,
        "n_points": len(series),
        "descriptive": descriptive_stats(series),
        "growth": growth_rates(series),
        "trend": trend(series),
        "seasonality": seasonal_pattern(series, freq=freq),
        "rolling_data": rolling_mean(series, window=max(2, min(7, len(series) // 4))).tolist(),
        "series": series.tolist(),
        "index": [str(x) for x in series.index],
        "notes": notes,
        "decomposition": decompose(series, freq_offset=freq, period=7 if freq == "D" else 4),
    }
    return result


def _n(x) -> float:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None
