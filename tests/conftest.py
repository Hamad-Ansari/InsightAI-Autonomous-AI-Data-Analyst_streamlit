"""
InsightAI - Pytest shared fixtures and path setup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def small_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "order_id": [f"ORD-{i}" for i in range(60)],
        "order_date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "region": rng.choice(["North", "South", "East", "West"], 60),
        "product": rng.choice(["A", "B", "C"], 60),
        "revenue": rng.uniform(10, 500, 60).round(2),
        "quantity": rng.poisson(2, 60) + 1,
    })


@pytest.fixture
def text_df() -> pd.DataFrame:
    return pd.DataFrame({
        "review": [
            "excellent product great value", "terrible quality very bad",
            "it is okay", "love it highly recommend", "worst experience ever",
            "good but slow", "average", "amazing fast delivery",
        ],
        "rating": [5, 1, 3, 5, 1, 3, 3, 5],
        "comment_date": pd.date_range("2024-05-01", periods=8, freq="D"),
    })
