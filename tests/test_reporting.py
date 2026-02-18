"""Tests for reporting tools (reporting.py)."""
import pytest
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import reporting
from conftest import ToolCaller


@pytest.fixture
def env(sample_df):
    mcp = FastMCP("test")
    store = DataStore()
    store.add("test", sample_df.copy())
    reporting.register_tools(mcp, store)
    return store, ToolCaller(mcp)


class TestSaveReport:
    def test_with_explicit_output_dir(self, env, tmp_path):
        store, call = env
        out = tmp_path / "reports"
        result = call("save_report", content="# Test Report\nHello", output_dir=str(out))
        assert "Report saved" in result
        assert (out / "report.md").exists()
        assert (out / "report.md").read_text(encoding="utf-8") == "# Test Report\nHello"

    def test_default_output_dir_from_csv(self, env, tmp_path):
        store, call = env
        store._csv_dir = str(tmp_path)
        result = call("save_report", content="# Report")
        expected_dir = tmp_path / "reports"
        assert expected_dir.exists()
        assert (expected_dir / "report.md").exists()

    def test_no_csv_dir_no_output_dir_errors(self, env):
        store, call = env
        store._csv_dir = ""
        result = call("save_report", content="# Report")
        assert "Error" in result or "error" in result.lower()

    def test_with_plots(self, env, tmp_path):
        store, call = env
        # Store some fake plot data
        store.save_plot("histogram_Revenue", b"\x89PNGfake_histogram")
        store.save_plot("correlation_matrix", b"\x89PNGfake_corr")

        out = tmp_path / "reports"
        result = call(
            "save_report",
            content="# Report\n![](histogram_Revenue.png)",
            output_dir=str(out),
            include_plots=["histogram_Revenue", "correlation_matrix"],
        )
        assert "2 plots" in result
        assert (out / "histogram_Revenue.png").exists()
        assert (out / "correlation_matrix.png").exists()
        assert (out / "histogram_Revenue.png").read_bytes() == b"\x89PNGfake_histogram"

    def test_missing_plot_warns(self, env, tmp_path):
        store, call = env
        out = tmp_path / "reports"
        result = call(
            "save_report",
            content="# Report",
            output_dir=str(out),
            include_plots=["nonexistent_plot"],
        )
        assert "Warning" in result or "not found" in result
        assert (out / "report.md").exists()  # report is still saved

    def test_partial_plots_saves_what_exists(self, env, tmp_path):
        store, call = env
        store.save_plot("good_plot", b"\x89PNGdata")
        out = tmp_path / "reports"
        result = call(
            "save_report",
            content="# Report",
            output_dir=str(out),
            include_plots=["good_plot", "bad_plot"],
        )
        assert (out / "good_plot.png").exists()
        assert not (out / "bad_plot.png").exists()
        assert "1 plots" in result or "1 plot" in result
        assert "not found" in result

    def test_empty_include_plots(self, env, tmp_path):
        store, call = env
        out = tmp_path / "reports"
        result = call("save_report", content="# Report", output_dir=str(out), include_plots=[])
        assert "Report saved" in result
        assert (out / "report.md").exists()

    def test_creates_nested_directories(self, env, tmp_path):
        store, call = env
        out = tmp_path / "deep" / "nested" / "reports"
        result = call("save_report", content="# Report", output_dir=str(out))
        assert "Report saved" in result
        assert (out / "report.md").exists()
