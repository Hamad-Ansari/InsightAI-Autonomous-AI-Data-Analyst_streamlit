"""
InsightAI - Small pure-Python helpers shared across modules.

Kept dependency-free so every module can import them cheaply.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- Column heuristics

# Words that commonly appear in identifier / key columns.
_ID_WORDS = re.compile(
    r"(^|_|.)(id|uuid|guid|key|code|number|ref|reference)(_|$)", re.IGNORECASE
)
# Words that hint at date-ish columns.
_DATE_WORDS = re.compile(
    r"(date|time|dt|timestamp|created|updated|year|month|day|week|quarter|period)",
    re.IGNORECASE,
)
_BOOL_WORDS = re.compile(r"(^|_)(is|has|active|enabled|flag|verified|deleted)(_|$)", re.IGNORECASE)
_CURRENCY_WORDS = re.compile(r"(price|amount|revenue|cost|salary|fee|total|sales|profit|income|cash)")


def is_probably_id(name: str, sample: pd.Series) -> bool:
    """Heuristic to detect identifier / key columns."""
    name_l = str(name).strip().lower()
    if _ID_WORDS.search(name_l):
        return True
    # All-unique, object/string column of integers is very likely an ID.
    if sample.dtype == object and sample.nunique(dropna=True) == sample.notna().sum():
        try:
            pd.to_numeric(sample, errors="raise")
            return True
        except Exception:
            return False
    return False


def is_probably_boolean(name: str, series: pd.Series) -> Optional[bool]:
    """Return True/False if column is likely boolean, else None."""
    name_l = str(name).strip().lower()
    if _BOOL_WORDS.search(name_l):
        return True
    uniques = series.dropna().astype(str).str.strip().str.lower().unique()
    if set(uniques) <= {"true", "false", "t", "f", "yes", "no", "0", "1", "y", "n"}:
        return True
    return None


def is_probably_currency(name: str) -> bool:
    return bool(_CURRENCY_WORDS.search(str(name).strip().lower()))


def is_probably_percentage(name: str, series: pd.Series) -> bool:
    """Column with percentage-like name or bounded [0,1] decimals."""
    name_l = str(name).strip().lower()
    if "%" in name_l or "percent" in name_l or name_l.endswith("_rate") or name_l.endswith("_ratio"):
        return True
    if np.issubdtype(series.dtype, np.number):
        s = series.dropna()
        if len(s) > 0 and (s.min() >= 0) and (s.max() <= 1):
            return True
    return False


def is_datetimeish(name: str, series: pd.Series) -> bool:
    """True if column name hints at a date/time field."""
    return bool(_DATE_WORDS.search(str(name).strip().lower()))


# --------------------------------------------------------------------------- Misc data utils


def human_bytes(n: int) -> str:
    """Format a byte count into a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def safe_json(data: Any) -> str:
    """Serialize an object to compact JSON, tolerating non-serialisable values."""
    try:
        return json.dumps(data, default=str, indent=2)
    except TypeError:
        return json.dumps({"error": "non-serialisable"}, default=str)


def chunk_list(items: Iterable[str], size: int) -> List[List[str]]:
    """Split a list into chunks of at most ``size`` items."""
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def round2(x: Any) -> Any:
    """Round a number to 2 decimals returning ``None`` for nulls."""
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return round(float(x), 2)
    except (TypeError, ValueError):
        return x


def series_to_list(series: pd.Series, limit: int = 30) -> List[Any]:
    """Return a list representation of a series truncated to ``limit`` values."""
    return series.head(limit).tolist()
