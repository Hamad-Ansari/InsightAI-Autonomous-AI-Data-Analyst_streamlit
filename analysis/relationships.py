"""
InsightAI - Relationship analysis (STEP 9 / Dimension D).

Computes Pearson & Spearman correlation, covariance and identifies the strongest
numeric relationships. Also computes Cramér's V for categorical associations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, BOOLEAN


def correlation_matrix(df: pd.DataFrame, cols: List[str], method: str = "pearson") -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    return sub.corr(method=method)


def covariance_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    return sub.cov()


def strongest_relationships(corr: pd.DataFrame, top_n: int = 10) -> List[Dict]:
    """Return the strongest |correlation| pairs excluding self-correlation."""
    pairs: List[Dict] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            val = corr.loc[a, b]
            if np.isnan(val):
                continue
            pairs.append({"a": a, "b": b, "corr": round(float(val), 4), "strength": abs(float(val))})
    pairs.sort(key=lambda x: x["strength"], reverse=True)
    return pairs[:top_n]


def cramers_v(table: pd.DataFrame) -> float:
    """Compute Cramér's V for two categorical variables from a contingency table."""
    chi2 = _chi2(table)
    n = table.to_numpy().sum()
    r, c = table.shape
    if n == 0 or min(r, c) < 2 or chi2 == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def _chi2(observed: pd.DataFrame) -> float:
    obs = observed.to_numpy(dtype=float)
    row = obs.sum(axis=1, keepdims=True)
    col = obs.sum(axis=0, keepdims=True)
    total = obs.sum()
    if total == 0:
        return 0.0
    expected = (row @ col) / total
    expected = np.where(expected == 0, 1e-9, expected)
    return float(np.sum((obs - expected) ** 2 / expected))


def categorical_associations(df: pd.DataFrame, cols: List[str], top_n: int = 10) -> List[Dict]:
    """Compute Cramér's V across pairs of categorical columns."""
    pairs: List[Dict] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                table = pd.crosstab(df[a].astype(str), df[b].astype(str))
                v = cramers_v(table)
                pairs.append({"a": a, "b": b, "cramers_v": round(v, 4)})
            except Exception:
                continue
    pairs.sort(key=lambda x: x["cramers_v"], reverse=True)
    return pairs[:top_n]


def relationship_report(df: pd.DataFrame, schema_types: Dict[str, str]) -> Dict:
    numeric_cols = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    cat_cols = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    result: Dict = {"pearson": {}, "spearman": {}, "covariance": {}, "categorical": []}
    if len(numeric_cols) >= 2:
        corr_p = correlation_matrix(df, numeric_cols)
        corr_s = correlation_matrix(df, numeric_cols, method="spearman")
        result["pearson"] = {"matrix": corr_p.to_dict(), "strongest": strongest_relationships(corr_p)}
        result["spearman"] = {"matrix": corr_s.to_dict(), "strongest": strongest_relationships(corr_s)}
        result["covariance"] = covariance_matrix(df, numeric_cols).to_dict()
    if len(cat_cols) >= 2:
        result["categorical"] = categorical_associations(df, cat_cols)
    result["numeric_cols"] = numeric_cols
    result["categorical_cols"] = cat_cols
    return result


def for_insights(pearson: pd.DataFrame, top_n: int = 5) -> List[Dict]:
    """Short relationship notes for the LLM."""
    return strongest_relationships(pearson, top_n=top_n)
