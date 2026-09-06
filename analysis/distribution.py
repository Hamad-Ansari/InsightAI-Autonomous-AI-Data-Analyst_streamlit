"""
InsightAI - Distribution analysis (STEP 7 / Dimension B).

Analyzes the spread of numeric variables: central tendency, dispersion,
skewness, kurtosis, percentiles and outlier summaries. Visualization is handled
by the charts module; this module only computes numbers.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE


def distribution_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return per-column distribution statistics for plotting/summaries."""
    rows: List[Dict] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        q1, q2, q3 = s.quantile(0.25), s.quantile(0.5), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (s < lo) | (s > hi)
        skew = stats.skew(s, bias=True)
        kurt = stats.kurtosis(s, fisher=True, bias=True)
        rows.append({
            "column": col,
            "count": int(s.count()),
            "mean": _n(s.mean()),
            "median": _n(s.median()),
            "std": _n(s.std()),
            "min": _n(s.min()),
            "p5": _n(s.quantile(0.05)),
            "p25": _n(q1),
            "p50": _n(q2),
            "p75": _n(q3),
            "p95": _n(s.quantile(0.95)),
            "max": _n(s.max()),
            "range": _n(s.max() - s.min()),
            "iqr": _n(iqr),
            "skewness": _n(skew),
            "kurtosis": _n(kurt),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_pct": round(100.0 * outlier_mask.sum() / len(s), 2),
            "shape": "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "symmetric",
            "tails": "heavy" if abs(kurt) > 1.5 else "light" if abs(kurt) < 0.5 else "normal",
        })
    return pd.DataFrame(rows)


def ecdf_values(s: pd.Series, n: int = 500) -> Dict[str, List]:
    """Sample empirical CDF points for ECDF plotting."""
    s = pd.to_numeric(s, errors="coerce").dropna().sort_values()
    if len(s) == 0:
        return {"x": [], "y": []}
    if len(s) > n:
        idx = np.linspace(0, len(s) - 1, n).astype(int)
        s = s.iloc[idx]
    y = np.arange(1, len(s) + 1) / len(s)
    return {"x": s.tolist(), "y": y.tolist()}


def distribution_report(df: pd.DataFrame, schema_types: Dict[str, str]) -> pd.DataFrame:
    numeric_cols = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    return distribution_summary(df, numeric_cols)


def _n(x) -> float:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None
