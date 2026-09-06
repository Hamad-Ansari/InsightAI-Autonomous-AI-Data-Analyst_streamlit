"""Tests for the chart selector and Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import pytest

import visualization.charts as charts
from visualization.chart_selector import (
    recommend_for_pair,
    recommend_single,
    four_dimension_charts,
)
from ingestion.schema_detector import NUMERIC, CATEGORICAL, DATETIME, TEXT


def test_recommend_pair():
    assert recommend_for_pair(NUMERIC, NUMERIC)["chart"] == "scatter"
    assert recommend_for_pair(CATEGORICAL, NUMERIC)["chart"] == "bar"
    assert recommend_for_pair(DATETIME, NUMERIC)["chart"] == "line"


def test_recommend_single():
    assert recommend_single(NUMERIC)["chart"] == "histogram"
    assert recommend_single(CATEGORICAL)["chart"] == "bar"


def test_four_dimensions():
    schema = {"a": NUMERIC, "b": CATEGORICAL, "c": DATETIME}
    rec = four_dimension_charts(schema)
    assert "composition" in rec and "distribution" in rec


def test_distribution_chart():
    fig = charts.distribution_chart(pd.Series([1, 2, 3, 4, 5, 100]), kind="histogram")
    assert fig is not None and len(fig.data) >= 1


def test_composition_chart():
    table = pd.DataFrame({"category": ["A", "B"], "value": [70, 30], "share_pct": [70, 30]})
    fig = charts.composition_chart(table, kind="donut")
    assert fig is not None


def test_correlation_heatmap():
    corr = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=["a", "b"], index=["a", "b"])
    fig = charts.correlation_heatmap(corr)
    assert fig is not None


def test_bar_and_scatter():
    fig = charts.bar_chart(pd.Series({"A": 10, "B": 20}))
    assert fig is not None
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    fig2 = charts.scatter_chart(df, "x", "y")
    assert fig2 is not None
