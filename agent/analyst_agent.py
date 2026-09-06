"""
InsightAI - Autonomous analyst agent.

Orchestrates the full 15-step workflow: it takes an uploaded dataset, decides
which tools to run via the planner, executes each tool in order, applies the
RAG retriever, and has Ollama generate business insights. All deterministic
analysis is done in Python; the agent is the conductor, not the calculator.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from agent import tools
from agent.planner import build_plan
from agent.tools import AnalysisContext
from ingestion.schema_detector import detect_schema, columns_by_type
from llm.insight_generator import generate_executive_insights
from llm.ollama_client import OllamaClient
from rag.retriever import Retriever


class AnalystAgent:
    """Runs the full autonomous analysis pipeline."""

    def __init__(self, ollama: Optional[OllamaClient] = None,
                 retriever: Optional[Retriever] = None):
        self.ollama = ollama or OllamaClient()
        self.retriever = retriever

    def run(self, df: pd.DataFrame, **clean_kwargs) -> Dict:
        """Execute the pipeline and return a comprehensive results dict."""
        ctx = AnalysisContext(df=df)
        ctx = tools.TOOLS["profile_dataset"]["func"](ctx)
        ctx = tools.TOOLS["detect_data_types"]["func"](ctx)
        ctx = tools.TOOLS["assess_quality"]["func"](ctx)
        ctx = tools.TOOLS["clean_dataset"]["func"](ctx, **clean_kwargs)

        # Regenerate schema on the cleaned frame (types may have changed).
        schema_clean = detect_schema(ctx.cleaned_df)
        ctx.schema_types = {p.name: p.semantic_type for p in schema_clean}

        plan = build_plan(ctx.schema_types)

        runnable = {
            "analyze_missing_data": tools.TOOLS["analyze_missing_data"]["func"],
            "detect_outliers": tools.TOOLS["detect_outliers"]["func"],
            "analyze_distribution": tools.TOOLS["analyze_distribution"]["func"],
            "analyze_composition": tools.TOOLS["analyze_composition"]["func"],
            "analyze_comparison": tools.TOOLS["analyze_comparison"]["func"],
            "analyze_relationships": tools.TOOLS["analyze_relationships"]["func"],
            "analyze_time_series": tools.TOOLS["analyze_time_series"]["func"],
            "analyze_sentiment": tools.TOOLS["analyze_sentiment"]["func"],
            "detect_anomalies": tools.TOOLS["detect_anomalies"]["func"],
        }
        for p in plan["plan"]:
            func = runnable.get(p["tool"])
            if func:
                try:
                    ctx = func(ctx)
                except Exception:
                    continue

        # Build the LLM-facing analytics report.
        from llm.insight_generator import build_analytics_report
        insight_slices = {
            "composition": composition_for_insights(ctx),
            "distribution": ctx.distribution_results.to_dict(orient="records") if not ctx.distribution_results.empty else [],
            "comparison": comparison_for_insights(ctx),
        }
        report = build_analytics_report(
            profile=ctx.profile,
            quality=ctx.quality,
            cleaning_log=ctx.cleaning_log,
            univariate=ctx.univariate,
            composition=insight_slices["composition"],
            distribution=insight_slices["distribution"],
            comparison=insight_slices["comparison"],
            relationships=ctx.relationship_results,
            sentiment=ctx.sentiment_results or None,
            timeseries=ctx.timeseries_results or None,
            anomaly=ctx.anomaly_results or None,
        )
        insights = generate_executive_insights(report, retriever=self.retriever,
                                               ollama=self.ollama)

        by_type = columns_by_type(schema_clean)
        return {
            "profile": ctx.profile,
            "quality": ctx.quality,
            "cleaned_df": ctx.cleaned_df,
            "cleaning_log": ctx.cleaning_log,
            "schema_types": ctx.schema_types,
            "columns_by_type": by_type,
            "plan": plan,
            "univariate": ctx.univariate,
            "distribution": ctx.distribution_results,
            "composition": ctx.composition_results,
            "comparison": ctx.comparison_results,
            "relationships": ctx.relationship_results,
            "sentiment": ctx.sentiment_results,
            "timeseries": ctx.timeseries_results,
            "anomaly": ctx.anomaly_results,
            "outlier_frame": ctx.__dict__.get("outlier_frame"),
            "missing_report": ctx.__dict__.get("missing_report"),
            "analytics_report": report,
            "insights": insights,
        }


def composition_for_insights(ctx: AnalysisContext):
    from analysis.composition import for_insights
    try:
        return for_insights(ctx.cleaned_df, ctx.schema_types)
    except Exception:
        return []


def comparison_for_insights(ctx: AnalysisContext):
    from analysis.comparison import for_insights
    try:
        return for_insights(ctx.cleaned_df, ctx.schema_types)
    except Exception:
        return []
