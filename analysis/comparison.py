"""
InsightAI - Comparison analysis (STEP 8 / Dimension C).

Compares a numeric metric across categorical groups (e.g. revenue by region),
producing aggregated comparison tables plus top/bottom rankings.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, BOOLEAN


def compare_groups(df: pd.DataFrame, group_col: str, value_col: str,
                   agg: str = "mean", top_n: int = 15) -> pd.DataFrame:
    """Aggregate ``value_col`` by ``group_col`` using ``agg``."""
    sub = df[[group_col, value_col]].copy()
    sub[group_col] = sub[group_col].astype(str)
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    agg_func = {"mean": "mean", "sum": "sum", "median": "median", "min": "min", "max": "max",
                "count": "count", "std": "std"}.get(agg, "mean")
    grouped = sub.groupby(group_col, dropna=False)[value_col].agg(agg_func).dropna().sort_values(ascending=False)
    out = pd.DataFrame({"group": grouped.index, "value": grouped.values}).reset_index(drop=True)
    total = float(out["value"].sum()) if agg == "sum" else float(out["value"].mean())
    out["summary"] = round(100.0 * out["value"] / total, 2) if total else 0.0
    if len(out) > top_n:
        out = out.head(top_n)
    return out


def comparison_report(df: pd.DataFrame, schema_types: Dict[str, str]) -> Dict:
    """Return comparison tables for categorical x numeric pairs."""
    cat_cols = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    numeric_cols = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    tables: Dict[str, Dict] = {}
    for gcol in cat_cols[:8]:
        for vcol in numeric_cols[:4]:
            key = f"{gcol} ~ {vcol}"
            try:
                table = compare_groups(df, gcol, vcol, agg="mean", top_n=15)
                tables[key] = {
                    "group_col": gcol,
                    "value_col": vcol,
                    "rows": table.to_dict(orient="records"),
                    "top": table.head(5).to_dict(orient="records"),
                    "bottom": table.tail(3).to_dict(orient="records"),
                }
            except Exception:
                continue
    return tables


def for_insights(df: pd.DataFrame, schema_types: Dict[str, str], top_n: int = 5) -> List[Dict]:
    """Short comparison notes for the LLM."""
    insights: List[Dict] = []
    cat_cols = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    numeric_cols = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    for gcol in cat_cols[:3]:
        if not numeric_cols:
            break
        vcol = numeric_cols[0]
        try:
            table = compare_groups(df, gcol, vcol, agg="mean", top_n=top_n)
            insights.append({
                "dimension": "comparison",
                "group": gcol,
                "metric": vcol,
                "top_groups": {r["group"]: r["value"] for r in table.head(top_n).to_dict(orient="records")},
                "bottom_group": {"group": str(table.iloc[-1]["group"]), "value": float(table.iloc[-1]["value"])},
            })
        except Exception:
            continue
    return insights
