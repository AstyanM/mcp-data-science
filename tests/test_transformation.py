"""Tests for transformation tools (transformation.py)."""
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import transformation
from conftest import ToolCaller


@pytest.fixture
def env(sample_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", sample_df.copy())
    transformation.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestCreateColumn:
    def test_happy(self, env):
        store, call = env
        result = call("create_column", new_column="AgeScore",
                       expression="Age * Score")
        df = store.get("test")
        assert "AgeScore" in df.columns

    def test_bad_expression(self, env):
        _, call = env
        result = call("create_column", new_column="bad",
                       expression="NONEXISTENT + 1")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestLogTransform:
    def test_log1p(self, env):
        store, call = env
        result = call("log_transform", columns=["Revenue"])
        df = store.get("test")
        assert "Log_Revenue" in df.columns

    def test_log10(self, env):
        store, call = env
        result = call("log_transform", columns=["Revenue"], method="log10")
        df = store.get("test")
        assert "Log_Revenue" in df.columns

    def test_missing_column(self, env):
        _, call = env
        result = call("log_transform", columns=["ZZZ"])
        assert "Error" in str(result) or "error" in str(result).lower()


class TestNormalize:
    def test_minmax(self, env):
        store, call = env
        result = call("normalize", columns=["Score"])
        df = store.get("test")
        scores = df["Score"].dropna()
        assert scores.min() >= -0.01
        assert scores.max() <= 1.01

    def test_standard(self, env):
        store, call = env
        result = call("normalize", columns=["Score"], method="standard")
        df = store.get("test")
        scores = df["Score"].dropna()
        assert abs(scores.mean()) < 0.5  # roughly zero-centered

    def test_robust(self, env):
        store, call = env
        result = call("normalize", columns=["Score"], method="robust")
        assert "Error" not in str(result)


class TestApplyMapping:
    def test_happy(self, env):
        store, call = env
        result = call("apply_mapping", column="Category",
                       mapping={"A": "Alpha", "B": "Beta"})
        df = store.get("test")
        assert "Alpha" in df["Category"].values


class TestConvertDtype:
    def test_to_str(self, env):
        store, call = env
        result = call("convert_dtype", columns=["Age"], dtype="str")
        df = store.get("test")
        assert pd.api.types.is_string_dtype(df["Age"])

    def test_bad_dtype(self, env):
        _, call = env
        result = call("convert_dtype", columns=["Age"], dtype="xyz")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestReplaceValues:
    def test_string(self, env):
        store, call = env
        result = call("replace_values", column="City",
                       old_value="Paris", new_value="Paris-FR")
        df = store.get("test")
        assert "Paris-FR" in df["City"].values
        assert "Paris" not in df["City"].dropna().values

    def test_missing_column(self, env):
        _, call = env
        result = call("replace_values", column="ZZZ",
                       old_value="x", new_value="y")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestSelectDtypes:
    def test_include_number(self, env):
        store, call = env
        result = call("select_dtypes", include=["number"])
        df = store.get("test")
        assert "Name" not in df.columns
        assert "Age" in df.columns or "Revenue" in df.columns

    def test_exclude_object(self, env):
        store, call = env
        result = call("select_dtypes", exclude=["object"])
        df = store.get("test")
        assert "Name" not in df.columns


class TestStringClean:
    def test_strip_lower(self, env):
        store, call = env
        result = call("string_clean", column="City",
                       operations=["strip", "lower"])
        df = store.get("test")
        non_null = df["City"].dropna()
        assert all(v == v.lower() for v in non_null)

    def test_upper(self, env):
        store, call = env
        result = call("string_clean", column="City", operations=["upper"])
        df = store.get("test")
        non_null = df["City"].dropna()
        assert all(v == v.upper() for v in non_null)

    def test_replace(self, env):
        store, call = env
        result = call("string_clean", column="City",
                       operations=["replace"], replace_old="a", replace_new="@")
        df = store.get("test")
        non_null = df["City"].dropna()
        assert all("a" not in v for v in non_null)
