"""Tests for server bootstrap (server.py)."""
import asyncio


def test_server_import():
    from mcp_data_science.server import mcp  # noqa: F401


def test_instructions_loaded():
    from mcp_data_science.server import mcp
    assert isinstance(mcp.instructions, str)
    assert len(mcp.instructions) > 1000
    assert "Pipeline" in mcp.instructions or "Phase" in mcp.instructions


def test_tool_count():
    from mcp_data_science.server import mcp
    tools = asyncio.run(mcp.list_tools())
    assert len(tools) == 102


def test_expected_tools_registered():
    from mcp_data_science.server import mcp
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "load_csv", "save_csv", "get_head", "quality_report",
        "drop_duplicates", "create_column", "one_hot_encode",
        "plot_histogram", "get_correlation", "train_model",
        "correlation_filter", "extract_datetime_parts",
        "save_report",
    }
    assert expected.issubset(names)


def test_main_entry_point():
    from mcp_data_science import main  # noqa: F401
