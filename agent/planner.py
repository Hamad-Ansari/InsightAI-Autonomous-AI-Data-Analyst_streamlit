"""
InsightAI - Analysis planner.

Decides *which* steps of the 15-step workflow are relevant based on the dataset
characteristics (e.g. skip time-series when no datetime column, skip sentiment
when no text column, skip categorical composition when no categoricals). This is
the "agent decides which tools are required" layer.
"""

from __future__ import annotations

from typing import Dict, List

from ingestion.schema_detector import (
    NUMERIC, CURRENCY, PERCENTAGE, CATEGORICAL, DATETIME, TEXT, BOOLEAN,
)


def build_plan(schema_types: Dict[str, str], n_rows: int = 0) -> List[Dict]:
    """Return an ordered plan of analysis steps with rationale."""
    numeric = [c for c, t in schema_types.items() if t in {NUMERIC, CURRENCY, PERCENTAGE}]
    categorical = [c for c, t in schema_types.items() if t in {CATEGORICAL, BOOLEAN}]
    datetime_cols = [c for c, t in schema_types.items() if t == DATETIME]
    text_cols = [c for c, t in schema_types.items() if t == TEXT]

    plan: List[Dict] = [
        {"step": 1, "tool": "profile_dataset", "run": True, "reason": "always used"},
        {"step": 2, "tool": "detect_data_types", "run": True, "reason": "always used"},
        {"step": 3, "tool": "assess_quality", "run": True, "reason": "always used"},
        {"step": 4, "tool": "clean_dataset", "run": True, "reason": "always used"},
    ]

    plan.append({"step": 5, "tool": "analyze_missing_data", "run": True,
                 "reason": "always used"})
    plan.append({"step": 6, "tool": "detect_outliers", "run": bool(numeric),
                 "reason": f"{len(numeric)} numeric column(s)"})

    plan.append({"step": 7, "tool": "analyze_distribution", "run": bool(numeric),
                 "reason": f"{len(numeric)} numeric column(s)"})
    plan.append({"step": 8, "tool": "analyze_composition", "run": bool(categorical),
                 "reason": f"{len(categorical)} categorical column(s)"})
    plan.append({"step": 9, "tool": "analyze_comparison", "run": bool(categorical and numeric),
                 "reason": f"{len(categorical)} categorical x {len(numeric)} numeric"})
    plan.append({"step": 10, "tool": "analyze_relationships", "run": bool(numeric and len(numeric) >= 2),
                 "reason": f"{len(numeric)} numeric column(s) for correlation"})

    plan.append({"step": 11, "tool": "analyze_time_series", "run": bool(datetime_cols),
                 "reason": f"{len(datetime_cols)} datetime column(s)"})
    plan.append({"step": 12, "tool": "analyze_sentiment", "run": bool(text_cols),
                 "reason": f"{len(text_cols)} text column(s)"})
    plan.append({"step": 13, "tool": "detect_anomalies", "run": bool(numeric),
                 "reason": f"{len(numeric)} numeric column(s)"})

    # Steps 14-15 are always relevant; they are handled by the LLM + reporting.
    plan.append({"step": 14, "tool": "generate_business_insights", "run": True,
                 "reason": "ollama reasoning over computed results"})
    plan.append({"step": 15, "tool": "generate_report", "run": True,
                 "reason": "always used"})

    relevant = [p for p in plan if p["run"]]
    skipped = [p for p in plan if not p["run"]]
    return {"plan": relevant, "executed": [p["tool"] for p in relevant],
            "skipped": [p["tool"] for p in skipped]}
