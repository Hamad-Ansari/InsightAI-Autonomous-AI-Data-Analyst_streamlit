"""
InsightAI - Orchestrated data-cleaning pipeline.

Runs a sequence of *safe* automatic operations: string trimming, category-case
normalisation, date parsing, numeric-string coercion, exact-duplicate removal,
missing-value imputation, and dropping empty/constant columns. Every change is
recorded in a cleaning log and the original frame is never modified.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from cleaning.missing_values import fill_missing_values
from cleaning.duplicates import remove_duplicates
from ingestion.schema_detector import (
    NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, TEXT, DATETIME, BOOLEAN, ID,
    detect_schema,
)


def _trim_and_case(df: pd.DataFrame, schema_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Trim whitespace and normalise obvious category casing."""
    out = df.copy()
    log: List[Dict] = []
    for col in out.columns:
        if not pd.api.types.is_string_dtype(out[col]) and out[col].dtype != object:
            continue
        series = out[col]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            stripped = series.astype(object).apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            n_changed = int((stripped != series).sum())
            if n_changed:
                out[col] = stripped
                log.append({"column": col, "operation": "trim_whitespace",
                            "rows_affected": n_changed, "reason": "trimmed surrounding whitespace"})

            stype = schema_types.get(str(col), "")
            if stype == CATEGORICAL:
                lowered = out[col].astype(object).apply(
                    lambda v: v.strip() if isinstance(v, str) else v
                )
                # Normalise obvious case difference: map to title/lower consistently.
                # Use a casefold map over the unique values.
                uniq = out[col].dropna().astype(str)
                if len(uniq) > 0:
                    mapping = {v: v.strip().title() for v in uniq.unique() if isinstance(v, str)}
                    mapped = out[col].astype(object).map(
                        lambda v: mapping.get(v, v) if isinstance(v, str) else v
                    )
                    n_case = int((mapped != out[col]).sum())
                    if n_case:
                        out[col] = mapped
                        log.append({"column": col, "operation": "normalize_category_case",
                                    "rows_affected": n_case,
                                    "reason": "normalised inconsistent capitalisation"})
    return out, log


def _coerce_dates(df: pd.DataFrame, schema_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Convert recognised date-like string columns to datetimes."""
    out = df.copy()
    log: List[Dict] = []
    for col in out.columns:
        stype = schema_types.get(str(col), "")
        if stype != DATETIME:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            continue
        if not pd.api.types.is_object_dtype(out[col]):
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        n_ok = int(parsed.notna().sum())
        if n_ok:
            out[col] = parsed
            log.append({"column": col, "operation": "parse_datetime",
                        "rows_affected": n_ok, "reason": f"parsed {n_ok} values to datetime"})
    return out, log


def _coerce_numeric_strings(df: pd.DataFrame, schema_types: Dict[str, str]) -> Tuple[pd.DataFrame, List[Dict]]:
    """Coerce string columns that look numeric (e.g. '1,234.5') to float."""
    out = df.copy()
    log: List[Dict] = []
    for col in out.columns:
        stype = schema_types.get(str(col), "")
        if stype not in {NUMERIC, CURRENCY, PERCENTAGE}:
            continue
        series = out[col]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        cleaned = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        n_ok = int(coerced.notna().sum())
        if n_ok and n_ok > 0 and np.issubdtype(coerced.dtype, np.number):
            out[col] = coerced
            log.append({"column": col, "operation": "coerce_numeric",
                        "rows_affected": n_ok, "reason": "converted numeric strings to numbers"})
    return out, log


def _drop_empty_constant(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully-empty and constant columns; return new frame."""
    out = df.copy()
    all_na = [c for c in out.columns if out[c].isna().all()]
    if all_na:
        out = out.drop(columns=all_na)
    constant = [c for c in out.columns if out[c].nunique(dropna=True) <= 1]
    if constant:
        out = out.drop(columns=constant)
    return out


def clean_dataset(
    df: pd.DataFrame,
    *,
    schema_types: Optional[Dict[str, str]] = None,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    drop_duplicates: bool = True,
    normalize_case: bool = True,
) -> Tuple[pd.DataFrame, List[Dict], Dict[str, Dict]]:
    """
    Run the automatic cleaning pipeline.

    Returns ``(cleaned_df, cleaning_log, before_after)`` where ``cleaning_log``
    is a list of records and ``before_after`` contains summary stats.
    """
    before = _profile_summary(df)
    if schema_types is None:
        schema_types = {p.name: p.semantic_type for p in detect_schema(df)}

    working = df.copy()
    log: List[Dict] = []

    # 1. Trim + case normalisation.
    if normalize_case:
        working, logs = _trim_and_case(working, schema_types)
        log.extend(logs)

    # 2. Date parsing.
    working, logs = _coerce_dates(working, schema_types)
    log.extend(logs)

    # 3. Numeric-string coercion.
    working, logs = _coerce_numeric_strings(working, schema_types)
    log.extend(logs)

    # 4. Drop duplicates.
    if drop_duplicates:
        working, logs = remove_duplicates(working)
        log.extend(logs)

    # 5. Missing values.
    working, logs = fill_missing_values(working, schema_types,
                                        numeric_strategy=numeric_strategy,
                                        categorical_strategy=categorical_strategy)
    log.extend(logs)

    # 6. Drop empty / constant columns.
    dropped = len(working.columns) - len(_drop_empty_constant(working).columns)
    working = _drop_empty_constant(working)
    if dropped:
        log.append({"column": "ALL_COLUMNS", "operation": "drop_empty_or_constant",
                    "rows_affected": dropped,
                    "reason": f"dropped {dropped} empty/constant column(s)"})

    after = _profile_summary(working)
    return working, log, {"before": before, "after": after}


def _profile_summary(df: pd.DataFrame) -> Dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing": int(df.isna().sum().sum()),
        "missing_pct": round(float(df.isna().mean().mean() * 100), 2),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": int(df.select_dtypes(include=np.number).shape[1]),
    }
