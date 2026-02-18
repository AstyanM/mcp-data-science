"""Tests for visualization tools (visualization.py)."""
import pytest
import matplotlib
matplotlib.use("Agg")

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image

from mcp_data_science.state import DataStore
from mcp_data_science.tools import visualization
from conftest import ToolCaller


@pytest.fixture
def env(sample_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", sample_df.copy())
    visualization.register_tools(mcp, store)
    return store, ToolCaller(mcp)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _assert_image(result):
    assert isinstance(result, Image)
    assert len(result.data) > 100  # non-trivial PNG


class TestPlotHistogram:
    def test_happy(self, env):
        _, call = env
        _assert_image(call("plot_histogram", column="Revenue"))

    def test_log_scale(self, env):
        _, call = env
        _assert_image(call("plot_histogram", column="Revenue", log_scale=True))

    def test_no_kde(self, env):
        _, call = env
        _assert_image(call("plot_histogram", column="Revenue", kde=False))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_histogram", column="Revenue")
        assert "histogram_Revenue" in store.list_plot_names()
        assert len(store.get_plot("histogram_Revenue")) > 100

    def test_save_path(self, env, tmp_path):
        store, call = env
        out = tmp_path / "plots" / "hist.png"
        call("plot_histogram", column="Revenue", save_path=str(out))
        assert out.exists()
        assert len(out.read_bytes()) > 100


class TestPlotBar:
    def test_vertical(self, env):
        _, call = env
        _assert_image(call("plot_bar", column="Category"))

    def test_horizontal(self, env):
        _, call = env
        _assert_image(call("plot_bar", column="Category", orientation="horizontal"))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_bar", column="Category")
        assert "bar_Category" in store.list_plot_names()


class TestPlotScatter:
    def test_happy(self, env):
        _, call = env
        _assert_image(call("plot_scatter", x="Age", y="Revenue"))

    def test_with_hue(self, env):
        _, call = env
        _assert_image(call("plot_scatter", x="Age", y="Revenue", hue="Category"))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_scatter", x="Age", y="Revenue")
        assert "scatter_Age_vs_Revenue" in store.list_plot_names()


class TestPlotBox:
    def test_simple(self, env):
        _, call = env
        _assert_image(call("plot_box", column="Revenue"))

    def test_with_by(self, env):
        _, call = env
        _assert_image(call("plot_box", column="Revenue", by="Category"))

    def test_stores_plot_simple(self, env):
        store, call = env
        call("plot_box", column="Revenue")
        assert "box_Revenue" in store.list_plot_names()

    def test_stores_plot_with_by(self, env):
        store, call = env
        call("plot_box", column="Revenue", by="Category")
        assert "box_Revenue_by_Category" in store.list_plot_names()


class TestPlotCorrelationMatrix:
    def test_default(self, env):
        _, call = env
        _assert_image(call("plot_correlation_matrix"))

    def test_specific_columns(self, env):
        _, call = env
        _assert_image(call("plot_correlation_matrix",
                            columns=["Age", "Revenue", "Score"]))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_correlation_matrix")
        assert "correlation_matrix" in store.list_plot_names()


class TestPlotPairplot:
    def test_happy(self, env):
        _, call = env
        _assert_image(call("plot_pairplot", columns=["Age", "Revenue", "Score"]))

    def test_with_hue(self, env):
        _, call = env
        _assert_image(call("plot_pairplot",
                            columns=["Age", "Revenue", "Score"], hue="Category"))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_pairplot", columns=["Age", "Revenue", "Score"])
        assert "pairplot" in store.list_plot_names()


class TestPlotMissingValues:
    def test_happy(self, env):
        _, call = env
        _assert_image(call("plot_missing_values"))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_missing_values")
        assert "missing_values" in store.list_plot_names()


class TestPlotLine:
    def test_single_y(self, env):
        _, call = env
        _assert_image(call("plot_line", x="Age", y="Revenue"))

    def test_multi_y(self, env):
        _, call = env
        _assert_image(call("plot_line", x="Age", y=["Revenue", "Score"]))

    def test_stores_plot(self, env):
        store, call = env
        call("plot_line", x="Age", y="Revenue")
        assert "line_Age_vs_Revenue" in store.list_plot_names()
