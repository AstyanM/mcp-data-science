"""Tests for cleaning tools (cleaning.py)."""
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import cleaning
from conftest import ToolCaller


@pytest.fixture
def env(sample_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", sample_df.copy())
    cleaning.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestDropDuplicates:
    def test_default(self, env):
        store, call = env
        result = call("drop_duplicates")
        df = store.get("test")
        assert len(df) == 10  # 12 - 2 duplicates

    def test_subset(self, env):
        store, call = env
        result = call("drop_duplicates", subset=["Name"])
        df = store.get("test")
        assert df["Name"].is_unique

    def test_keep_none(self, env):
        store, call = env
        result = call("drop_duplicates", keep="none")
        df = store.get("test")
        # Alice (3 occurrences) and Bob (2 occurrences) all removed
        assert "Alice" not in df["Name"].values
        assert "Bob" not in df["Name"].values


class TestDropColumns:
    def test_happy(self, env):
        store, call = env
        result = call("drop_columns", columns=["Score"])
        df = store.get("test")
        assert "Score" not in df.columns

    def test_missing_column(self, env):
        _, call = env
        result = call("drop_columns", columns=["ZZZ"])
        assert "Error" in str(result) or "error" in str(result).lower()


class TestDropMissing:
    def test_any(self, env):
        store, call = env
        result = call("drop_missing")
        df = store.get("test")
        assert df.isnull().sum().sum() == 0
        assert len(df) < 12

    def test_subset(self, env):
        store, call = env
        result = call("drop_missing", subset=["Revenue"])
        df = store.get("test")
        assert df["Revenue"].isnull().sum() == 0

    def test_all(self, env):
        store, call = env
        result = call("drop_missing", how="all")
        df = store.get("test")
        assert len(df) == 12  # no all-null rows


class TestFillMissing:
    def test_value(self, env):
        store, call = env
        result = call("fill_missing", columns=["Revenue"], strategy="value", value="0")
        df = store.get("test")
        assert df["Revenue"].isnull().sum() == 0

    def test_median(self, env):
        store, call = env
        result = call("fill_missing", columns=["Revenue"], strategy="median")
        df = store.get("test")
        assert df["Revenue"].isnull().sum() == 0

    def test_mode(self, env):
        store, call = env
        result = call("fill_missing", columns=["City"], strategy="mode")
        df = store.get("test")
        assert df["City"].isnull().sum() == 0

    def test_ffill(self, env):
        store, call = env
        result = call("fill_missing", columns=["Revenue"], strategy="ffill")
        df = store.get("test")
        assert df["Revenue"].isnull().sum() == 0


class TestFilterRows:
    def test_gt(self, env):
        store, call = env
        result = call("filter_rows", column="Revenue", operator=">", value="2000")
        df = store.get("test")
        assert all(df["Revenue"] > 2000)

    def test_eq(self, env):
        store, call = env
        result = call("filter_rows", column="City", operator="==", value="Paris")
        df = store.get("test")
        assert all(df["City"] == "Paris")

    def test_contains(self, env):
        store, call = env
        result = call("filter_rows", column="Name", operator="contains", value="a")
        df = store.get("test")
        assert all(df["Name"].str.contains("a"))

    def test_isin(self, env):
        store, call = env
        result = call("filter_rows", column="Category", operator="isin", value="A,B")
        df = store.get("test")
        assert set(df["Category"].unique()).issubset({"A", "B"})

    def test_missing_column(self, env):
        _, call = env
        result = call("filter_rows", column="ZZZ", operator="==", value="x")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestRenameColumns:
    def test_happy(self, env):
        store, call = env
        result = call("rename_columns", mapping={"Name": "FullName"})
        df = store.get("test")
        assert "FullName" in df.columns
        assert "Name" not in df.columns


class TestClipOutliers:
    def test_iqr(self, env):
        store, call = env
        result = call("clip_outliers", column="Revenue")
        assert "Error" not in str(result)

    def test_quantile(self, env):
        store, call = env
        result = call("clip_outliers", column="Revenue", method="quantile")
        assert "Error" not in str(result)


class TestSortValues:
    def test_ascending(self, env):
        store, call = env
        result = call("sort_values", columns=["Age"])
        df = store.get("test")
        ages = df["Age"].dropna().tolist()
        assert ages == sorted(ages)

    def test_descending(self, env):
        store, call = env
        result = call("sort_values", columns=["Age"], ascending=False)
        df = store.get("test")
        ages = df["Age"].dropna().tolist()
        assert ages == sorted(ages, reverse=True)

    def test_missing_column(self, env):
        _, call = env
        result = call("sort_values", columns=["ZZZ"])
        assert "Error" in str(result) or "error" in str(result).lower()
