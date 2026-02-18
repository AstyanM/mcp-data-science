"""Tests for loading tools (loading.py)."""
import pandas as pd
import pytest
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import loading
from conftest import ToolCaller, SAMPLE_CSV


@pytest.fixture
def env():
    mcp = FastMCP("test")
    store = DataStore()
    loading.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestLoadCsv:
    def test_happy(self, env):
        store, call = env
        result = call("load_csv", file_path=SAMPLE_CSV)
        assert "12" in result  # 12 rows
        assert store.current_name == "sample_data"

    def test_custom_name(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV, name="mydata")
        assert "mydata" in store.list_names()

    def test_file_not_found(self, env):
        _, call = env
        result = call("load_csv", file_path="/nonexistent/file.csv")
        assert "Error" in str(result) or "error" in str(result).lower()

    def test_separator(self, env, tmp_path):
        store, call = env
        csv = tmp_path / "semi.csv"
        csv.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
        result = call("load_csv", file_path=str(csv), separator=";")
        assert "2 rows" in result or "2" in result


class TestSaveCsv:
    def test_happy(self, env, tmp_path):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV)
        out = tmp_path / "out.csv"
        result = call("save_csv", file_path=str(out))
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 12

    def test_no_df(self, env, tmp_path):
        _, call = env
        result = call("save_csv", file_path=str(tmp_path / "x.csv"))
        assert "Error" in str(result) or "error" in str(result).lower()


class TestListDataframes:
    def test_empty(self, env):
        _, call = env
        result = call("list_dataframes")
        assert "No" in result or "0" in result or "empty" in result.lower()

    def test_multiple(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV, name="a")
        call("load_csv", file_path=SAMPLE_CSV, name="b")
        result = call("list_dataframes")
        assert "a" in result and "b" in result


class TestSetCurrentDataframe:
    def test_happy(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV, name="a")
        call("load_csv", file_path=SAMPLE_CSV, name="b")
        call("set_current_dataframe", name="a")
        assert store.current_name == "a"

    def test_nonexistent(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV)
        result = call("set_current_dataframe", name="zzz")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestCopyDataframe:
    def test_happy(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV, name="orig")
        result = call("copy_dataframe", source_name="orig", new_name="copy")
        assert "copy" in store.list_names()
        assert store.get("copy").shape == store.get("orig").shape

    def test_nonexistent(self, env):
        _, call = env
        result = call("copy_dataframe", source_name="zzz", new_name="copy")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestMergeDataframes:
    def test_inner(self, env):
        store, call = env
        store.add("left", pd.DataFrame({"key": [1, 2, 3], "v1": ["a", "b", "c"]}))
        store.add("right", pd.DataFrame({"key": [2, 3, 4], "v2": ["x", "y", "z"]}))
        result = call("merge_dataframes", left_name="left", right_name="right", on=["key"])
        assert "2 rows" in result or "2" in result

    def test_left(self, env):
        store, call = env
        store.add("left", pd.DataFrame({"key": [1, 2, 3], "v1": ["a", "b", "c"]}))
        store.add("right", pd.DataFrame({"key": [2, 3, 4], "v2": ["x", "y", "z"]}))
        result = call("merge_dataframes", left_name="left", right_name="right",
                       on=["key"], how="left")
        assert "3" in result


class TestPivotTable:
    def test_happy(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV)
        result = call("pivot_table", index=["City"], columns="Category",
                       values="Score", agg_func="mean")
        assert "Error" not in str(result)

    def test_missing_column(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV)
        result = call("pivot_table", index=["ZZZ"], columns="Category",
                       values="Score")
        assert "Error" in str(result) or "error" in str(result).lower()


class TestMeltDataframe:
    def test_happy(self, env):
        store, call = env
        call("load_csv", file_path=SAMPLE_CSV)
        result = call("melt_dataframe", id_vars=["Name", "City"])
        assert "Error" not in str(result)


class TestConcatDataframes:
    def test_rows(self, env):
        store, call = env
        store.add("a", pd.DataFrame({"x": [1, 2]}))
        store.add("b", pd.DataFrame({"x": [3, 4]}))
        result = call("concat_dataframes", names=["a", "b"], axis=0)
        assert "4" in result

    def test_columns(self, env):
        store, call = env
        store.add("a", pd.DataFrame({"x": [1, 2]}))
        store.add("b", pd.DataFrame({"y": [3, 4]}))
        result = call("concat_dataframes", names=["a", "b"], axis=1)
        assert "2" in result
