"""Tests for inspection tools (inspection.py)."""
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import inspection
from conftest import ToolCaller


@pytest.fixture
def env(sample_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", sample_df.copy())
    inspection.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestGetHead:
    def test_default(self, env):
        _, call = env
        result = call("get_head")
        assert "Alice" in result

    def test_custom_n(self, env):
        _, call = env
        result = call("get_head", n=2)
        assert "Alice" in result and "Bob" in result


class TestGetTail:
    def test_default(self, env):
        _, call = env
        result = call("get_tail")
        assert "Jack" in result

    def test_custom_n(self, env):
        _, call = env
        result = call("get_tail", n=2)
        assert "Jack" in result


class TestGetInfo:
    def test_happy(self, env):
        _, call = env
        result = call("get_info")
        assert "Name" in result
        assert "non-null" in result.lower() or "Non-Null" in result


class TestGetStatistics:
    def test_happy(self, env):
        _, call = env
        result = call("get_statistics")
        assert "mean" in result.lower() or "count" in result.lower()


class TestGetShape:
    def test_happy(self, env):
        _, call = env
        result = call("get_shape")
        assert "12" in result and "6" in result


class TestQualityReport:
    def test_happy(self, env):
        _, call = env
        result = call("quality_report")
        assert "Missing" in result or "missing" in result
        assert "Duplicate" in result or "duplicate" in result


class TestGetUniqueValues:
    def test_happy(self, env):
        _, call = env
        result = call("get_unique_values", column="Category")
        assert "A" in result and "B" in result and "C" in result

    def test_missing_column(self, env):
        _, call = env
        result = call("get_unique_values", column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestGetColumnProfile:
    def test_numeric(self, env):
        _, call = env
        result = call("get_column_profile", column="Revenue")
        assert "mean" in result.lower() or "min" in result.lower()

    def test_string(self, env):
        _, call = env
        result = call("get_column_profile", column="City")
        assert "Paris" in result or "unique" in result.lower()

    def test_missing_column(self, env):
        _, call = env
        result = call("get_column_profile", column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestSampleData:
    def test_default(self, env):
        _, call = env
        result = call("sample_data", n=4)
        assert isinstance(result, str) and len(result) > 10
