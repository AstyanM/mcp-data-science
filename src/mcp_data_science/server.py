import logging
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import (
    loading, inspection, cleaning, transformation, encoding,
    visualization, analysis, modeling, feature_selection, datetime_tools,
    reporting, statistical_tests, interpretation, clustering, dimensionality,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

_instructions = (Path(__file__).parent / "instructions.md").read_text(encoding="utf-8")

mcp = FastMCP(
    "mcp-data-science",
    instructions=_instructions,
)

store = DataStore()

loading.register_tools(mcp, store)
inspection.register_tools(mcp, store)
cleaning.register_tools(mcp, store)
transformation.register_tools(mcp, store)
encoding.register_tools(mcp, store)
visualization.register_tools(mcp, store)
analysis.register_tools(mcp, store)
modeling.register_tools(mcp, store)
feature_selection.register_tools(mcp, store)
datetime_tools.register_tools(mcp, store)
statistical_tests.register_tools(mcp, store)
interpretation.register_tools(mcp, store)
clustering.register_tools(mcp, store)
dimensionality.register_tools(mcp, store)
reporting.register_tools(mcp, store)
