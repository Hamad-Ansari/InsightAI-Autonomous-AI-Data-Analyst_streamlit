"""
InsightAI - Schema detection engine.

Classifies every column of a DataFrame into one of a small set of semantic
types using robust heuristics, so downstream modules (composition, comparison,
sentiment, time-series) can decide what is relevant without hard-coding the
domain (e-commerce, retail, banking, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from utils.helpers import (
    is_probably_boolean,
    is_probably_currency,
    is_probably_id,
    is_probably_percentage,
)

# Semantic types
NUMERIC = "NUMERIC"
CATEGORICAL = "CATEGORICAL"
DATETIME = "DATETIME"
TEXT = "TEXT"
BOOLEAN = "BOOLEAN"
ID = "ID"
CURRENCY = "CURRENCY"
PERCENTAGE = "PERCENTAGE"
UNKNOWN = "UNKNOWN"

_DATETIME_STR_RE = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|^\d{4}-\d{2}-\d{2}[T ]"
)
_TEXT_PATTERN_HEURISTIC = re.compile(
    r"(review|comment|feedback|message|description|text|content|note|transcript|query|address)", re.I
)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    semantic_type: str
    cardinality: int
    missing: int
    missing_pct: float
    unique: int

    def to_dict(self) -> Dict:
        return {
            "column": self.name,
            "dtype": self.dtype,
            "semantic_type": self.semantic_type,
            "cardinality": self.cardinality,
            "missing": self.missing,
            "missing_pct": round(self.missing_pct, 4),
            "unique": self.unique,
        }


def _coerce_numeric_errors(series: pd.Series) -> int:
    """Count values that look numeric but fail numeric coercion."""
    if np.issubdtype(series.dtype, np.number):
        return 0
    nonnull = series.dropna()
    if nonnull.empty:
        return 0
    coerced = pd.to_numeric(nonnull, errors="coerce")
    return int(coerced.isna().sum())


def detect_datetime(series: pd.Series, name: str) -> bool:
    """Return True if a series parses as dates."""
    if np.issubdtype(series.dtype, np.datetime64):
        return True
    if np.issubdtype(series.dtype, np.number):
        return False
    samples = series.dropna().astype(str).str.strip()
    if samples.empty:
        return False
    n = min(len(samples), 200)
    try:
        try:
            parsed = pd.to_datetime(samples.head(n), errors="coerce")
        except Exception:
            return False
        return parsed.notna().mean() >= 0.85
    except Exception:
        return False


def classify_series(name: str, series: pd.Series, n_rows: int) -> str:
    """Return the semantic type for a single column."""
    if series.dtype == bool:
        return BOOLEAN

    unique = series.nunique(dropna=True)
    missing = int(series.isna().sum())

    # Already-typed numeric columns.
    if np.issubdtype(series.dtype, np.number):
        if is_probably_boolean(name, series):
            return BOOLEAN
        if is_probably_id(name, series) and unique >= n_rows * 0.8:
            return ID
        if is_probably_currency(name):
            return CURRENCY
        if is_probably_percentage(name, series):
            return PERCENTAGE
        return NUMERIC

    if np.issubdtype(series.dtype, np.datetime64):
        return DATETIME

    if series.dtype == object:
        # Try datetime detection for string columns.
        if detect_datetime(series, name):
            return DATETIME

        # Try numeric coercion for object columns (e.g. "1,234" or "12.5").
        numeric_errors = _coerce_numeric_errors(series)
        if numeric_errors == 0 and unique <= n_rows:
            return NUMERIC

        if is_probably_id(name, series):
            return ID
        if is_probably_boolean(name, series):
            return BOOLEAN

        # Decide TEXT vs CATEGORICAL by cardinality / name heuristics.
        if _TEXT_PATTERN_HEURISTIC.search(name) or unique > 60:
            return TEXT
        if unique <= 60:
            return CATEGORICAL
        return TEXT

    return UNKNOWN


def detect_schema(df: pd.DataFrame) -> List[ColumnProfile]:
    """Detect the semantic schema for every column."""
    n = len(df)
    profiles: List[ColumnProfile] = []
    for col in df.columns:
        series = df[col]
        stype = classify_series(str(col), series, n)
        profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                semantic_type=stype,
                cardinality=int(series.nunique(dropna=True)),
                missing=int(series.isna().sum()),
                missing_pct=float(series.isna().mean()),
                unique=int(series.nunique(dropna=True)),
            )
        )
    return profiles


def schema_summary(profiles: List[ColumnProfile]) -> Dict:
    """Aggregate counts by semantic type."""
    counts: Dict[str, int] = {}
    for p in profiles:
        counts[p.semantic_type] = counts.get(p.semantic_type, 0) + 1
    return {
        "total_columns": len(profiles),
        "by_type": counts,
        "numeric": sum(1 for p in profiles if p.semantic_type in {NUMERIC, CURRENCY, PERCENTAGE}),
        "categorical": sum(1 for p in profiles if p.semantic_type == CATEGORICAL),
        "datetime": sum(1 for p in profiles if p.semantic_type == DATETIME),
        "text": sum(1 for p in profiles if p.semantic_type == TEXT),
        "boolean": sum(1 for p in profiles if p.semantic_type == BOOLEAN),
        "id": sum(1 for p in profiles if p.semantic_type == ID),
    }


def columns_by_type(profiles: List[ColumnProfile]) -> Dict[str, List[str]]:
    """Return column name lists grouped by semantic type."""
    mapping: Dict[str, List[str]] = {}
    for p in profiles:
        mapping.setdefault(p.semantic_type, []).append(p.name)
    return mapping
