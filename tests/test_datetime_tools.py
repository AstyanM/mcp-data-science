"""Tests for datetime tools (datetime_tools.py)."""
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import datetime_tools
from conftest import ToolCaller


@pytest.fixture
def env(datetime_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", datetime_df.copy())
    datetime_tools.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestExtractDatetimeParts:
    def test_default(self, env):
        store, call = env
        result = call("extract_datetime_parts", column="date")
        df = store.get("test")
        assert "date_year" in df.columns or "date_month" in df.columns

    def test_custom_parts(self, env):
        store, call = env
        result = call("extract_datetime_parts", column="date",
                       parts=["year", "month", "is_weekend"])
        df = store.get("test")
        assert "date_year" in df.columns
        assert "date_is_weekend" in df.columns

    def test_missing_column(self, env):
        _, call = env
        result = call("extract_datetime_parts", column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()

    def test_auto_convert_string_dates(self, env):
        store, call = env
        # Replace with string dates to test auto-conversion
        df = store.get("test")
        df["str_date"] = df["date"].astype(str)
        store.set("test", df)
        result = call("extract_datetime_parts", column="str_date")
        df = store.get("test")
        assert "str_date_year" in df.columns


class TestDatetimeDiff:
    def test_days(self, env):
        store, call = env
        result = call("datetime_diff", column_a="end", column_b="start",
                       new_column="duration")
        df = store.get("test")
        assert "duration" in df.columns
        assert df["duration"].iloc[0] == pytest.approx(31, abs=1)

    def test_hours(self, env):
        store, call = env
        result = call("datetime_diff", column_a="end", column_b="start",
                       new_column="hours", unit="hours")
        df = store.get("test")
        assert "hours" in df.columns
        assert df["hours"].iloc[0] == pytest.approx(31 * 24, abs=24)

    def test_missing_column(self, env):
        _, call = env
        result = call("datetime_diff", column_a="ZZZ", column_b="start",
                       new_column="dur")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestDatetimeFilter:
    def test_both_bounds(self, env):
        store, call = env
        result = call("datetime_filter", column="date",
                       start="2024-01-05", end="2024-01-15")
        df = store.get("test")
        assert len(df) == 11  # 5th to 15th inclusive

    def test_start_only(self, env):
        store, call = env
        result = call("datetime_filter", column="date", start="2024-01-20")
        df = store.get("test")
        assert len(df) == 11  # 20th to 30th

    def test_end_only(self, env):
        store, call = env
        result = call("datetime_filter", column="date", end="2024-01-10")
        df = store.get("test")
        assert len(df) == 10  # 1st to 10th

    def test_missing_column(self, env):
        _, call = env
        result = call("datetime_filter", column="ZZZ", start="2024-01-01")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestSetDatetimeIndex:
    def test_happy(self, env):
        store, call = env
        result = call("set_datetime_index", column="date")
        df = store.get("test")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_missing_column(self, env):
        _, call = env
        result = call("set_datetime_index", column="ZZZ")
        assert "Error" in str(result) or "error" in str(result).lower()
