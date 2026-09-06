"""
InsightAI - Composition analysis (STEP 6 / Dimension A).

Answers: "What is this dataset made of?" by computing category/segment share
tables and, where a numeric metric column is given, the metric share by group.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, BOOLEAN, TEXT


def composition_by_category(df: pd.DataFrame, cat_col: str,
                            value_col: Optional[str] = None, top_n: int = 15) -> pd.DataFrame:
    """
    Compute share table for a categorical column.

    - If ``value_col`` given: sum metric per group + share of that metric.
    - Else: count of rows per group + row share.
    """
    sub = df[[cat_col] + ([value_col] if value_col else [])].copy()
    sub[cat_col] = sub[cat_col].astype(str)
    if value_col and value_col in sub.columns:
        sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
        grouped = sub.groupby(cat_col, dropna=False)[value_col].sum(min_count=1).dropna().sort_values(ascending=False)
        total = float(grouped.sum()) if grouped.sum() else 0.0
        out = pd.DataFrame({
            "category": grouped.index,
            "value": grouped.values,
            "share_pct": [round(100.0 * v / total, 2) if total else 0.0 for v in grouped.values],
        })
    else:
        counts = sub[cat_col].value_counts(dropna=True)
        total = int(counts.sum()) if counts.sum() else 1
        out = pd.DataFrame({
            "category": counts.index,
            "count": counts.values,
            "share_pct": [round(100.0 * v / total, 2) for v in counts.values],
        })
    # Optional grouping of the long tail.
    if len(out) > top_n:
        head = out.head(top_n).copy()
        tail_sum_value = out["value"].sum() if value_col else out["count"].sum()
        tail_share = round(100.0 * (out.iloc[top_n:]["share_pct"].sum()), 2) if value_col else round(
            100.0 * (out.iloc[top_n:]["count"].sum() / total), 2
        )
        head = head.reset_index(drop=True)
        head.loc[len(head)] = {
            "category": "Other",
            "value": float(out.iloc[top_n:]["value"].sum()) if value_col else float(out.iloc[top_n:]["count"].sum()),
            "share_pct": tail_share,
        }
        out = head
    return out.reset_index(drop=True)


def composition_report(df: pd.DataFrame, schema_types: Dict[str, str]) -> Dict:
    """Return composition tables for the most relevant categorical columns."""
    cat_cols = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    numeric_cols = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    results: Dict[str, Dict] = {}
    for col in cat_cols:
        table = composition_by_category(df, col, top_n=15)
        results[col] = table.to_dict(orient="records")
        if numeric_cols:
            # Optionally metric-weighted composition for the first numeric col.
            table_val = composition_by_category(df, col, value_col=numeric_cols[0], top_n=15)
            results[col + "|value:" + numeric_cols[0]] = table_val.to_dict(orient="records")
    return {"categories": cat_cols, "tables": results}


def for_insights(df: pd.DataFrame, schema_types: Dict[str, str], top_n: int = 5) -> List[Dict]:
    """Short, insight-ready slice summarising top categories (for the LLM)."""
    insights: List[Dict] = []
    cat_cols = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    for col in cat_cols[:4]:
        counts = df[col].astype(str).value_counts(dropna=True)
        total = int(counts.sum()) if counts.sum() else 1
        top = counts.head(top_n)
        insights.append({
            "dimension": "composition",
            "column": col,
            "top_categories": {str(k): round(100.0 * v / total, 2) for k, v in top.items()},
        })
    return insights
