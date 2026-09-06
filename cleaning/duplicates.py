"""
InsightAI - Duplicate record handling.

Detects exact duplicate rows and offers to remove them. A log entry records how
many rows were dropped. Full-duplicate removal is the safe default; subset
duplicates (based on a subset of columns) can be enabled by the caller.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd


def detect_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.Series:
    """Return a boolean mask of duplicate rows (excluding the first occurrence)."""
    return df.duplicated(subset=subset, keep="first")


def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None, inplace_log: bool = True
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Remove exact duplicate rows; return the clean frame and a log entry."""
    mask = detect_duplicates(df, subset=subset)
    n_dup = int(mask.sum())
    out = df.loc[~mask].copy()
    log: List[Dict] = []
    if n_dup > 0:
        log.append({
            "column": "ALL_ROWS",
            "operation": "remove_exact_duplicates",
            "rows_affected": n_dup,
            "reason": f"removed {n_dup} exact duplicate row(s)"
            + (f" judged by {subset}" if subset else ""),
        })
    return out, log
