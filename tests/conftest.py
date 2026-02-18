"""Shared fixtures for the MCP Data Science test suite."""
import asyncio
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


# ---------------------------------------------------------------------------
# Helper: synchronous MCP tool caller
# ---------------------------------------------------------------------------

class ToolCaller:
    """Wraps a FastMCP tool registry for synchronous test invocation."""

    def __init__(self, mcp_instance: FastMCP):
        self._tools = mcp_instance._tool_manager._tools

    def __call__(self, tool_name: str, **kwargs):
        return asyncio.run(self._tools[tool_name].run(kwargs))

    def has(self, tool_name: str) -> bool:
        return tool_name in self._tools


# ---------------------------------------------------------------------------
# Fixture: sample DataFrame (mirrors sample_data.csv, 12 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Name":     ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank",
                     "Grace", "Alice", "Bob", "Henry", "Ivy", "Jack"],
        "Age":      [30, 25, 35, 28, 30, None, 42, 30, 25, 38, 22, 31],
        "Revenue":  [1500.50, 2300.00, 800.75, None, 1500.50, 3200.00,
                     950.00, 1500.50, 2300.00, 4100.00, 600.00, None],
        "City":     ["Paris", "Lyon", "Paris", "Marseille", "Paris", "Lyon",
                     None, "Paris", "Lyon", "Marseille", "Paris", "Lyon"],
        "Category": ["A", "B", "A", "C", "A", "B", "A", "A", "B", "C", "B", "A"],
        "Score":    [85.2, 92.1, 78.5, 88.0, 85.2, None, 71.3, 85.2, 92.1, 95.7, 68.9, 82.4],
    })


# ---------------------------------------------------------------------------
# Fixture: numeric-only dataset for modeling / feature_selection (100 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def numeric_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.normal(50, 10, n)
    x2 = rng.normal(100, 20, n)
    x3 = rng.uniform(0, 1, n)
    noise = rng.normal(0, 5, n)
    target = 3 * x1 + 0.5 * x2 + noise
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})


# ---------------------------------------------------------------------------
# Fixture: datetime dataset (30 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def datetime_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "date": dates,
        "start": dates,
        "end": dates + pd.Timedelta(days=31),
        "value": range(30),
    })


# ---------------------------------------------------------------------------
# Fixture: classification dataset (100 rows)
# ---------------------------------------------------------------------------

@pytest.fixture
def classification_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = (x1 + x2 > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "target": target})


# ---------------------------------------------------------------------------
# Convenience paths
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
SAMPLE_CSV = str(TESTS_DIR / "sample_data.csv")
