"""
InsightAI - Missing value handling.

Provides conservative strategies for imputing missing values. The module is
explicit: it never mutates data silently and always returns a log of the
operations performed. Numeric columns are filled with median (default) or mean;
categorical with the mode; datetime columns are left untouched by default.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ingestion.schema_detector import (
    NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, TEXT, DATETIME, BOOLEAN, ID,
)


def fill_missing_values(
    df: pd.DataFrame,
    schema_types: Dict[str, str],
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    datetime_strategy: str = "none",
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Fill missing values per column based on its semantic type.

    Returns a new DataFrame plus an ordered list of cleaning-log records:
    ``{"column", "operation", "rows_affected", "reason"}``.
    """
    out = df.copy()
    log: List[Dict] = []
    numeric_types = {NUMERIC, CURRENCY, PERCENTAGE}

    for col in df.columns:
        stype = schema_types.get(str(col), TEXT)
        missing = int(df[col].isna().sum())
        if missing == 0:
            continue

        if stype in numeric_types:
            series_num = pd.to_numeric(out[col], errors="coerce")
            if series_num.notna().sum() == 0:
                out[col] = out[col].fillna("")
                log.append(_entry(col, "fill_empty", missing, "no valid numeric values; cleared"))
                continue
            if numeric_strategy == "mean":
                fill_value = series_num.mean()
                op = "fill_mean"
            else:
                fill_value = series_num.median()
                op = "fill_median"
            filled = series_num.fillna(fill_value)
            # Preserve original dtype when possible.
            if pd.api.types.is_integer_dtype(df[col]) and np.isclose(fill_value, round(fill_value)):
                try:
                    filled = filled.round().astype("int64")
                except Exception:
                    pass
            out[col] = filled
            log.append(_entry(col, op, missing, f"imputed with {fill_value:.4g}"))

        elif stype == DATETIME:
            if datetime_strategy in {"ffill", "bfill", "none"}:
                if datetime_strategy == "ffill":
                    out[col] = out[col].ffill()
                elif datetime_strategy == "bfill":
                    out[col] = out[col].bfill()
                log.append(_entry(col, f"datetime_{datetime_strategy}", missing,
                                  "datetime not imputed blindly"))
            # 'none' => leave as is (safe default)

        elif stype in {CATEGORICAL, TEXT, BOOLEAN, ID, NUMERIC} or stype == "":
            if categorical_strategy == "mode":
                try:
                    mode_val = out[col].mode(dropna=True)
                    fill_value = mode_val.iloc[0] if not mode_val.empty else ""
                except Exception:
                    fill_value = ""
                out[col] = out[col].fillna(fill_value)
                log.append(_entry(col, "fill_mode", missing, f"imputed with mode '{fill_value}'"))
            else:
                out[col] = out[col].fillna("")
                log.append(_entry(col, "fill_empty", missing, "filled with empty string"))
        else:
            # Fallback for UNKNOWN / anything else.
            out[col] = out[col].fillna("")
            log.append(_entry(col, "fill_empty", missing, "left empty"))

    return out, log


def _entry(column: str, operation: str, rows_affected: int, reason: str) -> Dict:
    return {
        "column": str(column),
        "operation": operation,
        "rows_affected": int(rows_affected),
        "reason": reason,
    }
