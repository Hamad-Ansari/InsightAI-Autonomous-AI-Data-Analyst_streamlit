"""
InsightAI - Dataset profiling (STEP 2).

Produces a structural fingerprint of the dataset: dimensions, memory usage,
per-column dtype/type, cardinality, uniqueness and missingness.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ingestion.schema_detector import ColumnProfile, detect_schema, schema_summary
from utils.helpers import human_bytes


def profile_dataset(df: pd.DataFrame) -> Dict:
    """Return a full dataset profile as a nested dict."""
    n_rows, n_cols = df.shape
    mem = df.memory_usage(deep=True).sum()
    profiles: List[ColumnProfile] = detect_schema(df)
    summary = schema_summary(profiles)

    per_col = []
    for p in profiles:
        col = df[p.name]
        stats = {"dtype": p.dtype, "semantic_type": p.semantic_type}
        if np.issubdtype(col.dtype, np.number):
            s = pd.to_numeric(col, errors="coerce")
            stats["min"] = _num(s.min())
            stats["max"] = _num(s.max())
            stats["mean"] = _num(s.mean())
            stats["median"] = _num(s.median())
            stats["std"] = _num(s.std())
        per_col.append({**p.to_dict(), **{k: v for k, v in stats.items() if k != "dtype"}})

    return {
        "rows": n_rows,
        "columns": n_cols,
        "memory": mem,
        "memory_human": human_bytes(mem),
        "schema_summary": summary,
        "site_columns": profiles,
        "columns_detail": per_col,
        "duplicates": int(df.duplicated().sum()),
        "missing_total": int(df.isna().sum().sum()),
        "missing_total_pct": round(float(df.isna().mean().mean() * 100), 2),
    }


def _num(x) -> float:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None


def table_of_schema(profiles: List[ColumnProfile]) -> pd.DataFrame:
    """Return a human-friendly schema table."""
    return pd.DataFrame([p.to_dict() for p in profiles])
