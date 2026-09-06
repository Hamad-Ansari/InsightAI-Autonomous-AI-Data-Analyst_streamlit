"""Tests for the data ingestion engine."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from ingestion.loader import load_dataset
from utils.validators import UnsupportedFormatError, EmptyDatasetError, sanitize_filename
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_csv(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    df = load_dataset(p)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_tsv(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("x\ty\n1\t2\n3\t4\n", encoding="utf-8")
    df = load_dataset(p)
    assert df.shape == (2, 2)


def test_load_unsupported_extension(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_dataset(p)


def test_sanitize_filename():
    assert "/" not in sanitize_filename("../evil.csv")
    assert "\\" not in sanitize_filename("..\\evil.csv")
    assert sanitize_filename("normal.csv") == "normal.csv"
    # Unsafe path chars are replaced but spacess remain (spaces are legal).
    assert ":" not in sanitize_filename("a:b.csv")


def test_oversize_rejected(tmp_path):
    from utils.validators import OversizeError
    p = tmp_path / "big.csv"
    p.write_text("a\n1\n" * 2000, encoding="utf-8")
    # 0 MB limit means any non-zero file size exceeds it.
    with pytest.raises(OversizeError):
        load_dataset(p, max_mb=0)
