"""
InsightAI - Univariate analysis (STEP 5).

Computes descriptive statistics for numeric, categorical and text columns.
``describe_*`` functions return plain dicts / frames that Python computed; the
LLM never recalculates these numbers.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, TEXT, DATETIME, BOOLEAN


def describe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a tidy numeric-descriptives table."""
    rows: List[Dict] = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        rows.append({
            "column": col,
            "count": int(s.count()),
            "mean": _n(s.mean()),
            "median": _n(s.median()),
            "mode": _n(s.mode().iloc[0] if not s.mode().empty else None),
            "std": _n(s.std()),
            "variance": _n(s.var()),
            "min": _n(s.min()),
            "max": _n(s.max()),
            "q25": _n(s.quantile(0.25)),
            "q50": _n(s.quantile(0.50)),
            "q75": _n(s.quantile(0.75)),
            "skewness": _n(stats.skew(s, bias=True)),
            "kurtosis": _n(stats.kurtosis(s, fisher=True, bias=True)),
            "iqr": _n(s.quantile(0.75) - s.quantile(0.25)),
        })
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame, cols: List[str], top_n: int = 10) -> pd.DataFrame:
    """Return value-frequency summary per categorical column."""
    rows: List[Dict] = []
    for col in cols:
        vc = df[col].value_counts(dropna=True)
        for value, count in vc.head(top_n).items():
            rows.append({
                "column": col,
                "value": str(value),
                "count": int(count),
                "pct": round(100.0 * count / max(len(df[col].dropna()), 1), 2),
            })
    return pd.DataFrame(rows)


def describe_text(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return text-column summaries: length, word count, missingness."""
    rows: List[Dict] = []
    for col in cols:
        s = df[col].astype(str).dropna()
        if len(s) == 0:
            continue
        lengths = s.str.len()
        words = s.str.split(r"\s+").apply(len)
        rows.append({
            "column": col,
            "count": int(s.count()),
            "avg_length": _n(lengths.mean()),
            "max_length": int(lengths.max()),
            "avg_word_count": _n(words.mean()),
            "max_word_count": int(words.max()),
            "missing": int(df[col].isna().sum()),
            "missing_pct": round(float(df[col].isna().mean() * 100), 2),
        })
    return pd.DataFrame(rows)


def univariate_report(df: pd.DataFrame, schema_types: Dict[str, str]) -> Dict:
    """Build the univariate summary grouped by semantic type."""
    numeric_cols = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    cat_cols = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    text_cols = [c for c, t in schema_types.items() if t == TEXT]
    return {
        "numeric": describe_numeric(df, numeric_cols),
        "categorical": describe_categorical(df, cat_cols),
        "text": describe_text(df, text_cols),
    }


def _n(x) -> float:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None
