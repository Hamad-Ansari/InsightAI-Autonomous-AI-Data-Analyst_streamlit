"""Tests for the analysis modules (profiler, quality, composition, etc.)."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.profiler import profile_dataset
from analysis.quality import assess_quality
from analysis.composition import composition_by_category
from analysis.distribution import distribution_summary
from analysis.comparison import compare_groups
from analysis.relationships import correlation_matrix, strongest_relationships
from analysis.sentiment import analyze_sentiment
from analysis.timeseries import build_time_series, time_series_report
from analysis.anomaly import isolation_forest_flags, anomaly_report
from ingestion.schema_detector import detect_schema, classify_series, NUMERIC, CATEGORICAL, DATETIME
from cleaning.cleaner import clean_dataset


def _schema(df):
    return {p.name: p.semantic_type for p in detect_schema(df)}


def test_profile(small_df):
    p = profile_dataset(small_df)
    assert p["rows"] == 60
    assert p["columns"] == 6
    assert p["memory_human"]


def test_schema_detection(small_df):
    st = _schema(small_df)
    assert st["revenue"] == "CURRENCY"  # revenue -> currency by name heuristic
    assert st["order_date"] == "DATETIME"
    assert st["region"] in {CATEGORICAL}


def test_quality_score(small_df):
    q = assess_quality(small_df, _schema(small_df))
    assert 0 <= q["score"] <= 100


def test_composition(small_df):
    table = composition_by_category(small_df, "region")
    assert table["share_pct"].sum() == pytest.approx(100.0, abs=1.5)


def test_distribution(small_df):
    d = distribution_summary(small_df, ["revenue"])
    assert len(d) == 1
    assert "skewness" in d.columns


def test_comparison(small_df):
    t = compare_groups(small_df, "region", "revenue", agg="mean")
    assert len(t) <= 4


def test_relationships(small_df):
    corr = correlation_matrix(small_df, ["revenue", "quantity"])
    pairs = strongest_relationships(corr)
    assert isinstance(pairs, list)


def test_sentiment(text_df):
    res = analyze_sentiment(text_df, ["review"], use_vader=True)
    assert res["available"] is True
    s = res["summary"]["review"]
    assert s["positive_pct"] + s["neutral_pct"] + s["negative_pct"] == pytest.approx(100.0, abs=0.01)


def test_time_series(small_df):
    ts, notes = build_time_series(small_df, "order_date", "revenue", freq="D")
    assert len(ts) > 0
    r = time_series_report(small_df, "order_date", "revenue", freq="D")
    assert r["available"] is True


def test_anomaly(small_df):
    flags = isolation_forest_flags(small_df, ["revenue", "quantity"])
    # It may be None if sklearn unavailable, but here it should produce a series.
    assert flags is None or len(flags) == len(small_df)
    rep = anomaly_report(small_df, ["revenue", "quantity"], method="iqr")
    assert "total_anomalies" in rep
