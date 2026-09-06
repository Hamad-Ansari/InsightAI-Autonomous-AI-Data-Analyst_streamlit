"""
InsightAI - Automatic chart selection engine.

Recommends an appropriate chart for a pair of columns based on their semantic
types, following the rule set:

    numeric + numeric   -> scatter
    categorical + numeric -> bar
    numeric             -> histogram
    datetime + numeric  -> line
    categorical + categorical -> grouped/stacked bar
    correlation matrix  -> heatmap
    composition         -> donut / treemap / stacked bar
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ingestion.schema_detector import (
    NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, DATETIME, TEXT, BOOLEAN, ID,
)


def recommend_for_pair(x_type: str, y_type: str) -> Dict:
    """Recommend a chart type and rationale for a given (x, y) type pair."""
    x, y = x_type, y_type
    numeric = {NUMERIC, CURRENCY, PERCENTAGE}
    categorical = {CATEGORICAL, BOOLEAN, ID}

    if x in numeric and y in numeric:
        return {"chart": "scatter", "reason": "Both variables are numeric -> scatter plot"}
    if x in categorical and y in numeric:
        return {"chart": "bar", "reason": "Categorical x numeric -> bar chart"}
    if x in numeric and y in categorical:
        return {"chart": "bar", "reason": "Numeric x categorical -> bar chart (swapped)"}
    if x == DATETIME and y in numeric:
        return {"chart": "line", "reason": "Datetime x numeric -> line chart"}
    if x in numeric and y == DATETIME:
        return {"chart": "line", "reason": "Numeric x datetime -> line chart (swapped)"}
    if x in categorical and y in categorical:
        return {"chart": "grouped_bar", "reason": "Two categoricals -> grouped/stacked bar"}
    if x == DATETIME and y in categorical:
        return {"chart": "line", "reason": "Datetime x categorical -> line/trend chart"}
    if x == TEXT:
        return {"chart": "bar", "reason": "Text column -> frequency bar chart"}
    return {"chart": "table", "reason": "No strong chart recommendation; showing table"}


def recommend_single(x_type: str) -> Dict:
    """Recommend a chart for a single column."""
    numeric = {NUMERIC, CURRENCY, PERCENTAGE}
    categorical = {CATEGORICAL, BOOLEAN, ID}
    if x_type in numeric:
        return {"chart": "histogram", "reason": "Numeric single variable -> histogram/KDE"}
    if x_type in categorical:
        return {"chart": "bar", "reason": "Categorical single variable -> frequency bar"}
    if x_type == DATETIME:
        return {"chart": "line", "reason": "Datetime single variable -> line over time"}
    if x_type == TEXT:
        return {"chart": "word_freq", "reason": "Text column -> top terms bar"}
    return {"chart": "table", "reason": "default table"}


def four_dimension_charts(schema: Dict[str, str]) -> Dict:
    """Return the recommended charts for the four analysis dimensions."""
    numeric = [c for c, t in schema.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    categorical = [c for c, t in schema.items() if t in {CATEGORICAL, BOOLEAN}]
    datetime_cols = [c for c, t in schema.items() if t == DATETIME]

    rec: Dict[str, Dict] = {"composition": {}, "distribution": {}, "comparison": {}, "relationship": {}}
    if categorical:
        rec["composition"]["default"] = {"chart": "donut", "column": categorical[0]}
        rec["comparison"]["default"] = {"chart": "bar", "group_col": categorical[0],
                                        "value_col": numeric[0] if numeric else None}
    if numeric:
        rec["distribution"]["default"] = {"chart": "histogram", "column": numeric[0]}
        rec["relationship"]["default"] = {"chart": "heatmap"}
        rec["relationship"]["scatter"] = {"chart": "scatter", "x": numeric[0],
                                          "y": numeric[-1] if len(numeric) > 1 else numeric[0]}
    if datetime_cols and numeric:
        rec["comparison"]["time"] = {"chart": "line", "date_col": datetime_cols[0],
                                     "value_col": numeric[0]}
    return rec
