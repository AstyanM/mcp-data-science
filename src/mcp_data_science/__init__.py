from mcp_data_science.server import mcp


def main():
    mcp.run(transport="stdio")
