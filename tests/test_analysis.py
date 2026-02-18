"""Tests for analysis tools (analysis.py)."""
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import analysis
from conftest import ToolCaller


@pytest.fixture
def env(sample_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", sample_df.copy())
    analysis.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestGetCorrelation:
    def test_pearson(self, env):
        _, call = env
        result = call("get_correlation", col_a="Age", col_b="Revenue")
        assert "correlation" in result.lower() or "Correlation" in result

    def test_spearman(self, env):
        _, call = env
        result = call("get_correlation", col_a="Age", col_b="Revenue",
                       method="spearman")
        assert "Error" not in str(result)

    def test_missing_column(self, env):
        _, call = env
        result = call("get_correlation", col_a="ZZZ", col_b="Revenue")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestGetValueCounts:
    def test_happy(self, env):
        _, call = env
        result = call("get_value_counts", column="Category")
        assert "A" in result and "B" in result

    def test_missing_column(self, env):
        _, call = env
        result = call("get_value_counts", column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestDetectOutliers:
    def test_iqr(self, env):
        _, call = env
        result = call("detect_outliers", column="Revenue")
        assert "outlier" in result.lower() or "Outlier" in result or "bound" in result.lower()

    def test_zscore(self, env):
        _, call = env
        result = call("detect_outliers", column="Revenue", method="zscore")
        assert "Error" not in str(result)

    def test_not_numeric(self, env):
        _, call = env
        result = call("detect_outliers", column="Name")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestGroupAggregate:
    def test_mean(self, env):
        _, call = env
        result = call("group_aggregate", group_by=["City"],
                       agg_column="Revenue")
        assert "Paris" in result or "Lyon" in result

    def test_sum(self, env):
        _, call = env
        result = call("group_aggregate", group_by=["City"],
                       agg_column="Revenue", agg_func="sum")
        assert "Error" not in str(result)

    def test_missing_column(self, env):
        _, call = env
        result = call("group_aggregate", group_by=["ZZZ"],
                       agg_column="Revenue")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestCrosstab:
    def test_happy(self, env):
        _, call = env
        result = call("crosstab", index_col="City", columns_col="Category")
        assert "Paris" in result or "A" in result

    def test_normalize(self, env):
        _, call = env
        result = call("crosstab", index_col="City", columns_col="Category",
                       normalize=True)
        assert "Error" not in str(result)

    def test_missing_column(self, env):
        _, call = env
        result = call("crosstab", index_col="ZZZ", columns_col="Category")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestAddRowIndex:
    def test_default(self, env):
        store, call = env
        result = call("add_row_index")
        df = store.get("test")
        assert "index" in df.columns
        assert df["index"].iloc[0] == 0

    def test_custom(self, env):
        store, call = env
        result = call("add_row_index", column_name="row_id", start=1)
        df = store.get("test")
        assert "row_id" in df.columns
        assert df["row_id"].iloc[0] == 1
