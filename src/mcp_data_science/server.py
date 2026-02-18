import logging
import sys

from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore
from mcp_data_science.tools import loading, inspection, cleaning, transformation, encoding, visualization, analysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

mcp = FastMCP(
    "mcp-data-science",
    instructions=(
        "Data science preprocessing server: load, clean, transform, encode, "
        "analyze, and visualize CSV datasets. 48 tools for complete EDA pipelines."
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
