"""
InsightAI - Anomaly & risk analysis (STEP 13).

Flags unusual records using IQR, Z-score and (optionally) Isolation Forest,
building an anomaly table and a per-column anomaly report. Anomalies are
reported, never deleted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE, DATETIME


def isolation_forest_flags(df: pd.DataFrame, cols: List[str], contamination: float = 0.02) -> Optional[pd.Series]:
    """Flag anomalous rows with IsolationForest; returns None if it fails."""
    try:
        from sklearn.ensemble import IsolationForest
        data = df[cols].apply(pd.to_numeric, errors="coerce").fillna(df[cols].median(numeric_only=True))
        # Guard against constant / NaN columns.
        valid = [c for c in data.columns if data[c].std() > 0]
        if not valid:
            return None
        data = data[valid]
        model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        preds = model.fit_predict(data)
        return pd.Series(preds == -1, index=df.index, name="anomaly")
    except Exception:
        return None


def zscore_anomaly_mask(df: pd.DataFrame, cols: List[str], threshold: float = 3.0) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        z = ((s - s.mean()) / std).abs() > threshold
        mask = mask | z
    return mask


def anomaly_table(df: pd.DataFrame, cols: List[str], method: str = "iqr",
                  use_isolation_forest: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Build an anomaly record table. Returns ``(table, notes)`` where the table
    lists the flagged rows with their index and the offending column value.
    """
    notes: List[Dict] = []
    df = df.copy()
    if method == "zscore":
        mask = zscore_anomaly_mask(df, cols)
        notes.append({"method": "zscore", "threshold": 3.0})
    elif method == "isolation_forest" or use_isolation_forest:
        mask = isolation_forest_flags(df, cols)
        if mask is not None:
            notes.append({"method": "isolation_forest", "contamination": 0.02})
        else:
            mask = zscore_anomaly_mask(df, cols, threshold=3.0)
            notes.append({"method": "zscore(fallback)", "threshold": 3.0})
    else:
        # IQR-based row-level anomaly
        mask = pd.Series(False, index=df.index)
        for col in cols:
            s = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = mask | ((s < lo) | (s > hi))
        notes.append({"method": "iqr", "k": 1.5})

    df["is_anomaly"] = mask.values
    rows = df.loc[mask].copy()
    # Identify the column(s) that most likely drove the anomaly (max |z|).
    zsign = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            continue
        zsign.append((col, ((s - s.mean()) / std).abs()))
    if zsign:
        driver_info = []
        for idx in rows.index:
            best = max(zsign, key=lambda zc: zc[1].get(idx, 0))
            zval = float(best[1].get(idx, 0))
            driver_info.append({"column": best[0], "z_score": round(abs(zval), 2)})
        rows["driver"] = [d["column"] for d in driver_info]
        rows["z_score"] = [d["z_score"] for d in driver_info]

    return rows, notes


def anomaly_report(df: pd.DataFrame, cols: List[str], method: str = "iqr") -> Dict:
    """Return counts/percentages and a per-column anomaly summary."""
    per_col: List[Dict] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            per_col.append({"column": col, "anomaly_count": 0, "anomaly_pct": 0.0, "method": "iqr"})
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_anom = int(((s < lo) | (s > hi)).sum())
        per_col.append({
            "column": col,
            "anomaly_count": n_anom,
            "anomaly_pct": round(100.0 * n_anom / len(s), 2),
            "method": "iqr",
        })
    table, notes = anomaly_table(df, cols, method=method)
    return {
        "per_column": per_col,
        "total_anomalies": int(table.shape[0]),
        "anomaly_pct": round(100.0 * table.shape[0] / max(len(df), 1), 2),
        "notes": notes,
        "method": method,
    }
