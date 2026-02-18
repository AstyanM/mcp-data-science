"""Tests for encoding tools (encoding.py)."""
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import encoding
from conftest import ToolCaller


@pytest.fixture
def env():
    """Clean DataFrame without NaN for encoding tests."""
    mcp = FastMCP("test")
    store = DataStore()
    df = pd.DataFrame({
        "Name":     ["Alice", "Bob", "Charlie", "Diana", "Eve",
                     "Frank", "Grace", "Henry", "Ivy", "Jack"],
        "City":     ["Paris", "Lyon", "Paris", "Marseille", "Paris",
                     "Lyon", "Paris", "Marseille", "Paris", "Lyon"],
        "Category": ["A", "B", "A", "C", "A", "B", "A", "C", "B", "A"],
        "Revenue":  [1500, 2300, 800, 1200, 1500, 3200, 950, 4100, 600, 1800],
        "Score":    [85.2, 92.1, 78.5, 88.0, 85.2, 91.0, 71.3, 95.7, 68.9, 82.4],
    })
    store.add("test", df)
    encoding.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestOneHotEncode:
    def test_drop_first(self, env):
        store, call = env
        result = call("one_hot_encode", columns=["Category"])
        df = store.get("test")
        # A, B, C → 2 columns with drop_first=True
        cat_cols = [c for c in df.columns if c.startswith("Category_")]
        assert len(cat_cols) == 2
        assert "Category" not in df.columns

    def test_no_drop(self, env):
        store, call = env
        result = call("one_hot_encode", columns=["Category"], drop_first=False)
        df = store.get("test")
        cat_cols = [c for c in df.columns if c.startswith("Category_")]
        assert len(cat_cols) == 3

    def test_missing_column(self, env):
        _, call = env
        result = call("one_hot_encode", columns=["ZZZ"])
        assert "Error" in str(result) or "error" in str(result).lower()


class TestTargetEncode:
    def test_happy(self, env):
        store, call = env
        result = call("target_encode", columns=["City"],
                       target_column="Revenue")
        df = store.get("test")
        assert df["City"].dtype in ("float64", "float32")

    def test_missing_target(self, env):
        _, call = env
        result = call("target_encode", columns=["City"],
                       target_column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestLabelEncode:
    def test_happy(self, env):
        store, call = env
        result = call("label_encode", columns=["Category"])
        df = store.get("test")
        assert pd.api.types.is_integer_dtype(df["Category"])

    def test_missing_column(self, env):
        _, call = env
        result = call("label_encode", columns=["ZZZ"])
        assert "Error" in str(result) or "error" in str(result).lower()


class TestFrequencyEncode:
    def test_happy(self, env):
        store, call = env
        result = call("frequency_encode", columns=["City"])
        df = store.get("test")
        assert df["City"].dtype == "float64"
        # Frequencies should sum to ~1 per unique value
        assert df["City"].max() <= 1.0

    def test_missing_column(self, env):
        _, call = env
        result = call("frequency_encode", columns=["ZZZ"])
        assert "Error" in str(result) or "error" in str(result).lower()
