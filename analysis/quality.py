"""
InsightAI - Data quality assessment (STEP 3).

Calculates a 0-100 Data Quality Score from missingness, duplicate rate,
invalid values and inconsistent categorical values, plus a breakdown of the
individual sub-scores and the dimensions that drag the score down.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ingestion.schema_detector import NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, TEXT, DATETIME
from utils.helpers import is_probably_boolean


def assess_quality(df: pd.DataFrame, schema_types: Dict[str, str]) -> Dict:
    """Return data-quality metrics, sub-scores and an overall 0-100 score."""
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        return {"score": 0, "sub_scores": {}, "issues": [], "overall_grade": "F"}

    # 1. Missingness sub-score.
    missing_pct = float(df.isna().mean().mean())
    missing_score = _clamp(100 - (missing_pct * 100))

    # 2. Duplicate sub-score.
    dup_rate = float(df.duplicated().sum() / n_rows)
    duplicate_score = _clamp(100 - (dup_rate * 100))

    # 3. Invalid value sub-score (numeric coercion failures in numeric columns).
    invalid_count = 0
    invalid_total = 0
    for col in df.columns:
        stype = schema_types.get(str(col), "")
        series = df[col]
        invalid_total += int(series.notna().sum())
        if stype in {NUMERIC, CURRENCY, PERCENTAGE} and not pd.api.types.is_numeric_dtype(series):
            invalid_count += int(pd.to_numeric(series, errors="coerce").isna().sum())
    invalid_rate = invalid_count / invalid_total if invalid_total else 0.0
    invalid_score = _clamp(100 - (invalid_rate * 100))

    # 4. Consistency sub-score (categorical with many near-duplicate 'Noise' values).
    consistency_penalty = 0.0
    consistency_samples = 0
    for col in df.columns:
        stype = schema_types.get(str(col), "")
        if stype == CATEGORICAL:
            strs = df[col].dropna().astype(str).str.strip().str.lower()
            if len(strs) > 0:
                # Fraction of values that differ only by case/whitespace.
                normalized = strs.str.title()
                dup_norm = int(normalized.duplicated().sum())
                consistency_penalty += dup_norm / len(strs)
                consistency_samples += 1
    consistency_score = _clamp(
        100 - (100.0 * consistency_penalty / consistency_samples) if consistency_samples else 100
    )

    # 5. Empty/constant column penalty.
    empties = sum(1 for c in df.columns if df[c].isna().all() or df[c].nunique(dropna=True) <= 1)
    empty_score = _clamp(100 - (100.0 * empties / n_cols))

    # Weighted overall score.
    weights = {"missing": 0.30, "duplicates": 0.20, "invalid": 0.20, "consistency": 0.15, "emptiness": 0.15}
    score = round(
        missing_score * weights["missing"]
        + duplicate_score * weights["duplicates"]
        + invalid_score * weights["invalid"]
        + consistency_score * weights["consistency"]
        + empty_score * weights["emptiness"],
        1,
    )

    sub_scores = {
        "missing": missing_score,
        "duplicates": duplicate_score,
        "invalid_values": invalid_score,
        "consistency": consistency_score,
        "emptiness": empty_score,
    }

    issues: List[Dict] = []
    if missing_pct > 0.05:
        issues.append({"type": "missing", "detail": f"{missing_pct * 100:.1f}% of cells are missing"})
    if dup_rate > 0.02:
        issues.append({"type": "duplicates", "detail": f"{dup_rate * 100:.1f}% duplicate rows"})
    if invalid_rate > 0.02:
        issues.append({"type": "invalid", "detail": f"{invalid_rate * 100:.1f}% invalid numeric values"})
    if empties:
        issues.append({"type": "emptiness", "detail": f"{empties} empty/constant column(s)"})

    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 45 else "F"
    return {
        "score": score,
        "grade": grade,
        "sub_scores": sub_scores,
        "issues": issues,
        "missing_pct": round(missing_pct * 100, 2),
        "duplicate_rate": round(dup_rate * 100, 2),
        "invalid_rate": round(invalid_rate * 100, 2),
    }


def _clamp(x: float) -> float:
    return max(0.0, min(100.0, float(x)))
