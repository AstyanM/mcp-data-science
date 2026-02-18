"""Test script for Phase 2 — Inspection tools.
Run: python tests/test_inspection.py
"""
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcp_data_science.state import DataStore
from mcp_data_science.tools import inspection

# ── Setup ───────────────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")
store = DataStore()
inspection.register_tools(mcp, store)

# Grab inner tool functions from the MCP registry
tools = mcp._tool_manager._tools


def call(tool_name: str, **kwargs) -> str:
    """Call a registered MCP tool by name (sync wrapper)."""
    import asyncio
    ctx = None  # inspection tools don't use ctx
    result = asyncio.run(tools[tool_name].run(kwargs))
    return result


# ── Load test CSV ───────────────────────────────────────────────────────
from mcp_data_science.tools import loading
loading.register_tools(mcp, store)

csv_path = str(Path(__file__).resolve().parent / "sample_data.csv")
print("=" * 70)
print("LOADING CSV")
print("=" * 70)
print(call("load_csv", file_path=csv_path))

# ── Test each inspection tool ───────────────────────────────────────────
tests = [
    ("get_head", {}),
    ("get_tail", {"n": 3}),
    ("get_info", {}),
    ("get_statistics", {}),
    ("get_shape", {}),
    ("quality_report", {}),
    ("get_unique_values", {"column": "Category"}),
    ("get_column_profile", {"column": "Revenue"}),
    ("get_column_profile", {"column": "City"}),
    ("sample_data", {"n": 4}),
]

passed = 0
failed = 0

for tool_name, kwargs in tests:
    print()
    print("=" * 70)
    label = f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})"
    print(f"TEST: {label}")
    print("=" * 70)
    try:
        result = call(tool_name, **kwargs)
        if isinstance(result, str) and result.startswith("Error"):
            print(f"  FAIL: {result}")
            failed += 1
        else:
            # For string results, print them; for other types just confirm
            print(str(result)[:500])
            passed += 1
    except Exception as e:
        print(f"  EXCEPTION: {type(e).__name__} - {e}")
        failed += 1

print()
print("=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed} tests")
print("=" * 70)
