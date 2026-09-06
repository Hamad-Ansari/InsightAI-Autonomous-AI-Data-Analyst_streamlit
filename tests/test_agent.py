"""Tests for the agent tool registry, planner and the full agent run."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.tools import TOOLS, AnalysisContext
from agent.planner import build_plan
from agent.analyst_agent import AnalystAgent
from llm.insight_generator import deterministic_insights, _parse_sections
from reporting.report_generator import generate_all


def test_tools_registered():
    expected = {"load_dataset", "profile_dataset", "detect_data_types", "clean_dataset",
                "detect_outliers", "analyze_composition", "analyze_distribution",
                "analyze_comparison", "analyze_relationships", "analyze_missing_data",
                "analyze_sentiment", "analyze_time_series", "detect_anomalies"}
    assert expected.issubset(TOOLS.keys())


def test_planner_skips_sentiment_without_text():
    schema = {"a": "NUMERIC", "b": "CATEGORICAL", "c": "DATETIME"}
    plan = build_plan(schema)
    executed = plan["executed"]
    assert "analyze_sentiment" not in executed
    assert "analyze_time_series" in executed


def test_planner_skips_timeseries_without_date():
    schema = {"a": "NUMERIC", "b": "CATEGORICAL"}
    plan = build_plan(schema)
    assert "analyze_time_series" not in plan["executed"]


def test_agent_run_deterministic(small_df):
    agent = AnalystAgent()
    r = agent.run(small_df)
    assert r["profile"]["rows"] == len(small_df)
    assert "quality" in r
    assert r["insights"]["source"] in {"ollama", "deterministic"}
    assert "data_quality" in r["analytics_report"]


def test_deterministic_insights_structure(small_df):
    agent = AnalystAgent()
    r = agent.run(small_df)
    txt = r["insights"]["text"]
    assert "Executive Summary" in txt
    assert "Data Quality" in txt
    sections = r["insights"]["structured"]
    assert "Executive Summary" in sections


def test_report_generation(small_df):
    agent = AnalystAgent()
    r = agent.run(small_df)
    exports = generate_all(r, "test", charts={})
    assert "html" in exports
    assert "json" in exports
    assert exports["json"]
    # PDF optional (ReportLab may be missing in minimal envs).
    if "pdf" in exports:
        assert exports["pdf"]
