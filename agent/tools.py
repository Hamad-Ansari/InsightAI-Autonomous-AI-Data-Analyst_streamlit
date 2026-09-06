"""
InsightAI - Agent tools.

Each tool is a thin, testable function that operates on an
:class:`AnalysisContext`. Tools never call the LLM for calculations - they only
compute with Python/pandas/sklearn/statsmodels and hand structured results to
the reasoning layer. A simple registry exposes the tool catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ingestion.loader import load_dataset
from ingestion.schema_detector import detect_schema, schema_summary, columns_by_type
from analysis import (profiler, quality, univariate, composition, distribution,
                      comparison, relationships, sentiment, timeseries, anomaly)
from cleaning.cleaner import clean_dataset
from cleaning.outliers import outlier_report


@dataclass
class AnalysisContext:
    """Carries data + computed results through the pipeline."""
    df: pd.DataFrame
    cleaned_df: Optional[pd.DataFrame] = None
    schema: Optional[list] = None
    schema_types: Dict[str, str] = field(default_factory=dict)
    profile: Dict = field(default_factory=dict)
    quality: Dict = field(default_factory=dict)
    cleaning_log: List[Dict] = field(default_factory=list)
    univariate: Dict = field(default_factory=dict)
    composition_results: Dict = field(default_factory=dict)
    distribution_results: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison_results: Dict = field(default_factory=dict)
    relationship_results: Dict = field(default_factory=dict)
    sentiment_results: Dict = field(default_factory=dict)
    timeseries_results: Dict = field(default_factory=dict)
    anomaly_results: Dict = field(default_factory=dict)


# ---------------------------------------------------------------- Registration


def register(name: str, prereqs: List[str], func: Callable) -> Dict:
    return {"name": name, "prereqs": prereqs, "func": func}


def tool(name: str, func: Callable, prereqs: Optional[List[str]] = None) -> Dict:
    return register(name, prereqs or [], func)


TOOLS: Dict[str, Dict] = {}


def register_tool(t: Dict) -> None:
    TOOLS[t["name"]] = t


# ---------------------------------------------------------------- Tool implementations


def t_load_dataset(ctx: AnalysisContext, path: str, **kw) -> AnalysisContext:
    ctx.df = load_dataset(path, **kw)
    return ctx


def t_profile_dataset(ctx: AnalysisContext) -> AnalysisContext:
    ctx.profile = profiler.profile_dataset(ctx.df)
    return ctx


def t_detect_data_types(ctx: AnalysisContext) -> AnalysisContext:
    ctx.schema = detect_schema(ctx.df)
    ctx.schema_types = {p.name: p.semantic_type for p in ctx.schema}
    return ctx


def t_quality(ctx: AnalysisContext) -> AnalysisContext:
    ctx.quality = quality.assess_quality(ctx.df, ctx.schema_types)
    return ctx


def t_clean_dataset(ctx: AnalysisContext, **kw) -> AnalysisContext:
    ctx.cleaned_df, ctx.cleaning_log, _ = clean_dataset(ctx.df, schema_types=ctx.schema_types, **kw)
    return ctx


def t_detect_outliers(ctx: AnalysisContext, method: str = "iqr") -> AnalysisContext:
    numeric = [c for c, t in ctx.schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    table = outlier_report(ctx.cleaned_df, numeric, method=method)
    ctx.__dict__["outlier_frame"] = table
    return ctx


def t_analyze_composition(ctx: AnalysisContext) -> AnalysisContext:
    ctx.composition_results = composition.composition_report(ctx.cleaned_df, ctx.schema_types)
    return ctx


def t_analyze_distribution(ctx: AnalysisContext) -> AnalysisContext:
    ctx.distribution_results = distribution.distribution_report(ctx.cleaned_df, ctx.schema_types)
    return ctx


def t_analyze_comparison(ctx: AnalysisContext) -> AnalysisContext:
    ctx.comparison_results = comparison.comparison_report(ctx.cleaned_df, ctx.schema_types)
    return ctx


def t_analyze_relationships(ctx: AnalysisContext) -> AnalysisContext:
    ctx.relationship_results = relationships.relationship_report(ctx.cleaned_df, ctx.schema_types)
    return ctx


def t_analyze_missing_data(ctx: AnalysisContext) -> AnalysisContext:
    numeric = [c for c, t in ctx.schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    ctx.univariate = univariate.univariate_report(ctx.cleaned_df, ctx.schema_types)
    ctx.__dict__["missing_report"] = {
        "per_column": ctx.cleaned_df.isna().sum().sort_values(ascending=False).to_dict(),
        "total": int(ctx.cleaned_df.isna().sum().sum()),
    }
    return ctx


def t_analyze_sentiment(ctx: AnalysisContext, use_vader: bool = True) -> AnalysisContext:
    text_cols = [c for c, t in ctx.schema_types.items() if t == "TEXT"]
    ctx.sentiment_results = sentiment.analyze_sentiment(ctx.cleaned_df, text_cols, use_vader=use_vader)
    return ctx


def t_analyze_time_series(ctx: AnalysisContext, freq: str = "D") -> AnalysisContext:
    date_cols = [c for c, t in ctx.schema_types.items() if t == "DATETIME"]
    numeric = [c for c, t in ctx.schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    date_col = date_cols[0] if date_cols else None
    value_col = numeric[0] if numeric else None
    ctx.timeseries_results = timeseries.time_series_report(ctx.cleaned_df, date_col, value_col, freq=freq)
    return ctx


def t_detect_anomalies(ctx: AnalysisContext, method: str = "iqr") -> AnalysisContext:
    numeric = [c for c, t in ctx.schema_types.items() if t in {"NUMERIC", "CURRENCY", "PERCENTAGE"}]
    ctx.anomaly_results = anomaly.anomaly_report(ctx.cleaned_df, numeric, method=method)
    return ctx


# ---------------------------------------------------------------- Registry population
for _t in [
    tool("load_dataset", t_load_dataset, ["df"]),
    tool("profile_dataset", t_profile_dataset, ["df"]),
    tool("detect_data_types", t_detect_data_types, ["df"]),
    tool("assess_quality", t_quality, ["df", "schema_types"]),
    tool("clean_dataset", t_clean_dataset, ["df", "schema_types"]),
    tool("detect_outliers", t_detect_outliers, ["cleaned_df", "schema_types"]),
    tool("analyze_composition", t_analyze_composition, ["cleaned_df", "schema_types"]),
    tool("analyze_distribution", t_analyze_distribution, ["cleaned_df", "schema_types"]),
    tool("analyze_comparison", t_analyze_comparison, ["cleaned_df", "schema_types"]),
    tool("analyze_relationships", t_analyze_relationships, ["cleaned_df", "schema_types"]),
    tool("analyze_missing_data", t_analyze_missing_data, ["cleaned_df", "schema_types"]),
    tool("analyze_sentiment", t_analyze_sentiment, ["cleaned_df", "schema_types"]),
    tool("analyze_time_series", t_analyze_time_series, ["cleaned_df", "schema_types"]),
    tool("detect_anomalies", t_detect_anomalies, ["cleaned_df", "schema_types"]),
]:
    register_tool(_t)
