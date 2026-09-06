"""
InsightAI - Outlier detection & classification.

Implements IQR, Z-score and (optionally) Isolation Forest. Outliers are never
silently deleted: they are classified as Normal / Potential / Extreme and the
user chooses whether to keep, remove, or winsorise them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def iqr_bounds(series: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    """Return ``(lower, upper)`` Tukey fences for a numeric series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def zscore_flags(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return Z-scores for a numeric series (NaN safe)."""
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return (series - series.mean()) / std


def classify_outliers(
    series: pd.Series,
    method: str = "iqr",
    extreme_factor: float = 3.0,
) -> Dict[str, pd.Series]:
    """
    Classify values into normal / potential / extreme outlier.

    Method ``iqr`` (default) uses Tukey fences with ``extreme_factor`` as the
    extreme multiplier. Method ``zscore`` uses |z| > 3 (potential) and > 4.5
    (extreme). Returns dict of boolean Series labelled 'normal', 'potential',
    'extreme'.
    """
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 4:
        return {
            "normal": pd.Series(True, index=series.index),
            "potential": pd.Series(False, index=series.index),
            "extreme": pd.Series(False, index=series.index),
            "scores": pd.Series(0.0, index=series.index),
        }

    if method == "zscore":
        z = zscore_flags(s)
        extreme = z.abs() > 4.5
        potential = (z.abs() > 3.0) & ~extreme
        scores = z
    else:  # iqr
        lo, hi = iqr_bounds(s, k=1.5)
        lo_e, hi_e = iqr_bounds(s, k=extreme_factor)
        potential = ((s < lo) | (s > hi)) & ~((s < lo_e) | (s > hi_e))
        extreme = (s < lo_e) | (s > hi_e)
        scores = (s - lo) if ((hi - lo) != 0) else pd.Series(0.0, index=series.index)

    normal = ~(potential | extreme)
    return {
        "normal": normal,
        "potential": potential.fillna(False),
        "extreme": extreme.fillna(False),
        "scores": scores,
    }


def outlier_report(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = "iqr",
) -> pd.DataFrame:
    """Return a per-column aggregate outlier report."""
    rows: List[Dict] = []
    for col in numeric_cols:
        res = classify_outliers(df[col], method=method)
        n_pot = int(res["potential"].sum())
        n_ext = int(res["extreme"].sum())
        total = n_pot + n_ext
        n = int(df[col].notna().sum())
        rows.append({
            "column": col,
            "total_numeric": n,
            "potential_outliers": n_pot,
            "extreme_outliers": n_ext,
            "outlier_count": total,
            "outlier_pct": round(100.0 * total / n, 2) if n else 0.0,
            "classification": "High" if (n and total / n > 0.05) else ("Moderate" if total else "Low"),
        })
    return pd.DataFrame(rows)


def apply_outlier_strategy(
    df: pd.DataFrame,
    numeric_cols: List[str],
    strategy: str = "keep",  # keep | remove | winsorize
    method: str = "iqr",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Apply the chosen outlier strategy. Returns new df + log."""
    out = df.copy()
    log: List[Dict] = []
    if strategy == "keep":
        for col in numeric_cols:
            res = classify_outliers(out[col], method=method)
            total = int((res["potential"] | res["extreme"]).sum())
            if total:
                log.append({"column": col, "operation": "keep_outliers",
                            "rows_affected": total, "reason": "outliers preserved (user choice)"})
        return out, log

    if strategy == "remove":
        n_removed = 0
        for col in numeric_cols:
            res = classify_outliers(out[col], method=method)
            mask = ~(res["potential"] | res["extreme"])
            before = len(out)
            out = out.loc[mask].copy()
            n_removed += before - len(out)
        if n_removed:
            log.append({"column": "ALL_ROWS", "operation": "remove_outliers",
                        "rows_affected": n_removed,
                        "reason": f"dropped {n_removed} rows containing outliers"})
        return out, log

    if strategy == "winsorize":
        for col in numeric_cols:
            res = classify_outliers(out[col], method=method)
            s = pd.to_numeric(out[col], errors="coerce")
            lo, hi = iqr_bounds(s, k=1.5)
            s = s.clip(lower=lo, upper=hi)
            out[col] = s
            total = int((res["potential"] | res["extreme"]).sum())
            if total:
                log.append({"column": col, "operation": "winsorize",
                            "rows_affected": total,
                            "reason": f"capped outliers to [{lo:.4g}, {hi:.4g}]"})
        return out, log

    # Unknown strategy -> no-op
    return out, log
