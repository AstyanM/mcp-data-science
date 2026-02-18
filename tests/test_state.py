"""Tests for DataStore (state.py)."""
import pandas as pd
import pytest

from mcp_data_science.state import DataStore


@pytest.fixture
def store():
    return DataStore()


@pytest.fixture
def df_a():
    return pd.DataFrame({"x": [1, 2, 3]})


@pytest.fixture
def df_b():
    return pd.DataFrame({"x": [4, 5, 6]})


class TestDataFrameManagement:
    def test_add_and_get(self, store, df_a):
        store.add("a", df_a)
        result = store.get("a")
        assert result.equals(df_a)

    def test_add_sets_current(self, store, df_a):
        store.add("a", df_a)
        assert store.current_name == "a"

    def test_get_empty_returns_current(self, store, df_a):
        store.add("a", df_a)
        result = store.get("")
        assert result.equals(df_a)

    def test_get_no_df_raises(self, store):
        with pytest.raises(ValueError, match="No dataframe loaded"):
            store.get("")

    def test_get_nonexistent_raises(self, store, df_a):
        store.add("a", df_a)
        with pytest.raises(KeyError, match="No dataframe named 'z'"):
            store.get("z")

    def test_current_name_setter(self, store, df_a, df_b):
        store.add("a", df_a)
        store.add("b", df_b)
        store.current_name = "a"
        assert store.current_name == "a"

    def test_current_name_setter_invalid(self, store, df_a):
        store.add("a", df_a)
        with pytest.raises(KeyError):
            store.current_name = "z"

    def test_set_updates_frame(self, store, df_a, df_b):
        store.add("a", df_a)
        store.set("a", df_b)
        assert store.get("a").equals(df_b)

    def test_resolve_name_default(self, store, df_a):
        store.add("a", df_a)
        assert store.resolve_name("") == "a"

    def test_resolve_name_explicit(self, store, df_a):
        store.add("a", df_a)
        assert store.resolve_name("a") == "a"

    def test_resolve_name_empty_raises(self, store):
        with pytest.raises(ValueError):
            store.resolve_name("")

    def test_list_names(self, store, df_a, df_b):
        store.add("a", df_a)
        store.add("b", df_b)
        assert set(store.list_names()) == {"a", "b"}

    def test_remove(self, store, df_a, df_b):
        store.add("a", df_a)
        store.add("b", df_b)
        store.remove("b")
        assert "b" not in store.list_names()

    def test_remove_current_updates(self, store, df_a, df_b):
        store.add("a", df_a)
        store.add("b", df_b)
        store.current_name = "b"
        store.remove("b")
        assert store.current_name == "a"

    def test_copy(self, store, df_a):
        store.add("a", df_a)
        store.copy("a", "a_copy")
        copy = store.get("a_copy")
        assert copy.equals(df_a)
        # Verify independence
        copy.iloc[0, 0] = 999
        assert store.get("a").iloc[0, 0] != 999

    def test_copy_nonexistent_raises(self, store):
        with pytest.raises(KeyError):
            store.copy("z", "z_copy")


class TestModelManagement:
    def test_add_and_get_model(self, store):
        store.add_model("m1", {"type": "linear", "model": None})
        result = store.get_model("m1")
        assert result["type"] == "linear"

    def test_get_model_nonexistent_raises(self, store):
        with pytest.raises(KeyError, match="No model named"):
            store.get_model("z")

    def test_list_model_names(self, store):
        store.add_model("m1", {"type": "a"})
        store.add_model("m2", {"type": "b"})
        assert set(store.list_model_names()) == {"m1", "m2"}

    def test_remove_model(self, store):
        store.add_model("m1", {"type": "a"})
        store.remove_model("m1")
        assert "m1" not in store.list_model_names()


class TestPlotStorage:
    def test_save_and_get_plot(self, store):
        png_bytes = b"\x89PNG\r\n\x1a\nfake_png_data"
        store.save_plot("histogram_Revenue", png_bytes)
        result = store.get_plot("histogram_Revenue")
        assert result == png_bytes

    def test_get_plot_nonexistent_raises(self, store):
        with pytest.raises(KeyError, match="No plot named"):
            store.get_plot("nonexistent")

    def test_list_plot_names(self, store):
        store.save_plot("plot_a", b"a")
        store.save_plot("plot_b", b"b")
        assert set(store.list_plot_names()) == {"plot_a", "plot_b"}

    def test_list_plot_names_empty(self, store):
        assert store.list_plot_names() == []

    def test_overwrite_plot(self, store):
        store.save_plot("p1", b"old")
        store.save_plot("p1", b"new")
        assert store.get_plot("p1") == b"new"

    def test_csv_dir_default_empty(self, store):
        assert store._csv_dir == ""

    def test_plots_default_empty(self, store):
        assert store._plots == {}
