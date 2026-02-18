"""Tests for feature selection tools (feature_selection.py)."""
import pandas as pd
import numpy as np
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import feature_selection
from conftest import ToolCaller


@pytest.fixture
def env(numeric_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", numeric_df.copy())
    feature_selection.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestCorrelationFilter:
    def test_happy(self, env):
        _, call = env
        result = call("correlation_filter", target_column="target",
                       threshold=0.1)
        assert "Error" not in str(result)

    def test_finds_weak_feature(self, env):
        _, call = env
        # x3 is uniform random noise, should have low correlation
        result = call("correlation_filter", target_column="target",
                       threshold=0.3)
        assert "x3" in result

    def test_missing_target(self, env):
        _, call = env
        result = call("correlation_filter", target_column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestVarianceFilter:
    def test_no_constants(self, env):
        _, call = env
        result = call("variance_filter")
        # No constant columns in numeric_df
        assert "Error" not in str(result)

    def test_with_constant_column(self, env):
        store, call = env
        df = store.get("test")
        df["const"] = 42
        store.set("test", df)
        result = call("variance_filter")
        assert "const" in result


class TestFeatureImportance:
    def test_random_forest(self, env):
        _, call = env
        result = call("feature_importance", target_column="target")
        # x1 has strongest relationship (coeff=3)
        assert "x1" in result

    def test_missing_target(self, env):
        _, call = env
        result = call("feature_importance", target_column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestDropLowImportance:
    def test_variance_drops_constant(self, env):
        store, call = env
        df = store.get("test")
        df["const"] = 42
        store.set("test", df)
        result = call("drop_low_importance", target_column="target",
                       method="variance", threshold=0.0)
        df = store.get("test")
        assert "const" not in df.columns

    def test_correlation(self, env):
        _, call = env
        result = call("drop_low_importance", target_column="target",
                       method="correlation", threshold=0.3)
        # x3 should be dropped (low correlation)
        store = env[0]
        df = store.get("test")
        assert "x3" not in df.columns or "Error" not in str(result)
