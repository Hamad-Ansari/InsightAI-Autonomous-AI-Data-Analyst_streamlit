"""Tests for the data cleaning engine, missing values, duplicates & outliers."""

from __future__ import annotations

import pandas as pd
import pytest

from cleaning.cleaner import clean_dataset
from cleaning.duplicates import remove_duplicates, detect_duplicates
from cleaning.missing_values import fill_missing_values
from cleaning.outliers import classify_outliers, apply_outlier_strategy
from ingestion.schema_detector import detect_schema


def test_remove_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    mask = detect_duplicates(df)
    assert int(mask.sum()) == 1
    clean, log = remove_duplicates(df)
    assert len(clean) == 2
    assert log and log[0]["rows_affected"] == 1


def test_fill_missing_values_median():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0], "cat": ["a", "b", None, "a"]})
    schema = {"x": "NUMERIC", "cat": "CATEGORICAL"}
    out, log = fill_missing_values(df, schema, numeric_strategy="median", categorical_strategy="mode")
    assert out["x"].isna().sum() == 0
    assert out["cat"].isna().sum() == 0
    assert len(log) == 2


def test_clean_dataset_preserves_original():
    df = pd.DataFrame({
        "id": [" 1 ", "1", "2"],
        "name": [" Adele ", "Adele", "Bob"],
        "price": ["$10", "$20", "$30"],
    })
    schema = {p.name: p.semantic_type for p in detect_schema(df)}
    clean, log, ba = clean_dataset(df, schema_types=schema)
    # Original untouched.
    assert df.loc[0, "id"] == " 1 "
    assert clean is not None
    assert "before" in ba and "after" in ba


def test_outlier_classification():
    s = pd.Series([1, 2, 2, 3, 3, 3, 100, 4, 3, 2])
    res = classify_outliers(s, method="iqr")
    assert int(res["extreme"].sum()) >= 1


def test_apply_outlier_strategy_winsorize():
    s = pd.DataFrame({"v": [1, 2, 3, 4, 5, 200]})
    out, log = apply_outlier_strategy(s, ["v"], strategy="winsorize")
    assert out["v"].max() < 200
    assert any(x["operation"] == "winsorize" for x in log)
