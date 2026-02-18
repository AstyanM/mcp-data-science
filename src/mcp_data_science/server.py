import logging
import sys

from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import (
    loading, inspection, cleaning, transformation, encoding,
    visualization, analysis, modeling, feature_selection, datetime_tools,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

mcp = FastMCP(
    "mcp-data-science",
    instructions=(
        "Data science server: load, clean, transform, encode, analyze, visualize CSV datasets, "
        "train ML models, select features, and handle datetime operations. 65 tools for complete DS pipelines."
    ),
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
