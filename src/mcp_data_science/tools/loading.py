import pandas as pd
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def load_csv(file_path: str, name: str = "", separator: str = ",") -> str:
        """Load a CSV file into memory. The name defaults to the filename.
        This becomes the current active dataframe."""
        try:
            resolved_name = name if name else Path(file_path).stem
            df = pd.read_csv(file_path, sep=separator)
            store.add(resolved_name, df)
            dtype_summary = df.dtypes.value_counts().to_dict()
            dtype_str = ", ".join(f"{v} {k}" for k, v in dtype_summary.items())
            return (
                f"Loaded '{resolved_name}': {df.shape[0]} rows, {df.shape[1]} columns.\n"
                f"Columns: {', '.join(df.columns.tolist())}\n"
                f"Dtypes: {dtype_str}"
            )
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"
        except Exception as e:
            return f"Error loading CSV: {type(e).__name__} - {e}"

    @mcp.tool()
    def save_csv(file_path: str, df_name: str = "") -> str:
        """Save a dataframe to a CSV file (without row index)."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            df.to_csv(file_path, index=False)
            return f"Saved '{name}' to {file_path} ({df.shape[0]} rows, {df.shape[1]} columns)."
        except Exception as e:
            return f"Error saving CSV: {type(e).__name__} - {e}"

    @mcp.tool()
    def list_dataframes() -> str:
        """List all loaded dataframes with their shapes."""
        names = store.list_names()
        if not names:
            return "No dataframes loaded."
        lines = []
        for n in names:
            df = store.get(n)
            marker = " (current)" if n == store.current_name else ""
            lines.append(f"  - {n}: {df.shape[0]} rows, {df.shape[1]} columns{marker}")
        return "Loaded dataframes:\n" + "\n".join(lines)

    @mcp.tool()
    def set_current_dataframe(name: str) -> str:
        """Switch the current active dataframe."""
        try:
            store.current_name = name
            return f"Current dataframe set to '{name}'."
        except KeyError as e:
            return f"Error: {e}"

    @mcp.tool()
    def copy_dataframe(source_name: str, new_name: str) -> str:
        """Create a deep copy of a dataframe under a new name.
        Useful to save a snapshot before destructive operations."""
        try:
            store.copy(source_name, new_name)
            df = store.get(new_name)
            return f"Copied '{source_name}' -> '{new_name}' ({df.shape[0]} rows, {df.shape[1]} columns)."
        except KeyError as e:
            return f"Error: {e}"

    @mcp.tool()
    def merge_dataframes(
        left_name: str,
        right_name: str,
        on: list[str],
        how: str = "inner",
        result_name: str = "",
    ) -> str:
        """Merge two dataframes (like SQL JOIN). How: 'inner', 'left', 'right', 'outer'.
        Result is stored as a new dataframe."""
        try:
            left = store.get(left_name)
            right = store.get(right_name)
            res_name = result_name if result_name else f"{left_name}_{right_name}_merged"
            merged = pd.merge(left, right, on=on, how=how)
            store.add(res_name, merged)
            return (
                f"Merged '{left_name}' ({left.shape[0]} rows) + '{right_name}' ({right.shape[0]} rows) "
                f"on {on} ({how}).\n"
                f"Result '{res_name}': {merged.shape[0]} rows, {merged.shape[1]} columns."
            )
        except Exception as e:
            return f"Error merging: {type(e).__name__} - {e}"
