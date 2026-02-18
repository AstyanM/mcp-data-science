import pandas as pd
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def load_csv(file_path: str, name: str = "", separator: str = ",") -> str:
        """Load a CSV file into memory. The name defaults to the filename.
        This becomes the current active dataframe.
        Always the first step in any pipeline. After loading, immediately run quality_report to understand the data."""
        try:
            resolved_name = name if name else Path(file_path).stem
            df = pd.read_csv(file_path, sep=separator)
            store.add(resolved_name, df)
            if not store._csv_dir:
                store._csv_dir = str(Path(file_path).resolve().parent)
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
        """Save a dataframe to a CSV file (without row index).
        Use at the end of a pipeline to export cleaned/processed data."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            df.to_csv(file_path, index=False)
            return f"Saved '{name}' to {file_path} ({df.shape[0]} rows, {df.shape[1]} columns)."
        except Exception as e:
            return f"Error saving CSV: {type(e).__name__} - {e}"

    @mcp.tool()
    def list_dataframes() -> str:
        """List all loaded dataframes with their shapes.
        Check what datasets are available before switching or merging."""
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
        """Switch the current active dataframe.
        Switch active dataset when working with multiple dataframes."""
        try:
            store.current_name = name
            return f"Current dataframe set to '{name}'."
        except KeyError as e:
            return f"Error: {e}"

    @mcp.tool()
    def copy_dataframe(source_name: str, new_name: str) -> str:
        """Create a deep copy of a dataframe under a new name.
        Use BEFORE any destructive operation (dropping, filtering, encoding) if you may need the original data later."""
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
        Result is stored as a new dataframe.
        Use when data is split across files. Always verify join keys exist in both dataframes with get_info first."""
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

    @mcp.tool()
    def pivot_table(
        index: list[str],
        columns: str,
        values: str,
        agg_func: str = "mean",
        result_name: str = "",
        df_name: str = "",
    ) -> str:
        """Create a pivot table. Stores result as a new dataframe.
        Creates summary tables. Use for reporting or reshaping data from long to wide format.
        Example: pivot_table(index=["City"], columns="Category", values="Revenue", agg_func="mean")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in index + [columns, values]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            pivot = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=agg_func)
            pivot = pivot.reset_index()
            # Flatten MultiIndex columns
            pivot.columns = [str(c) if not isinstance(c, tuple) else "_".join(str(x) for x in c) for c in pivot.columns]

            res_name = result_name if result_name else f"{name}_pivot"
            store.add(res_name, pivot, set_current=False)
            return (
                f"Pivot table created as '{res_name}': {pivot.shape[0]} rows, {pivot.shape[1]} columns.\n"
                f"Preview:\n{pivot.head(10).to_string()}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def melt_dataframe(
        id_vars: list[str],
        value_vars: list[str] | None = None,
        var_name: str = "variable",
        value_name: str = "value",
        df_name: str = "",
    ) -> str:
        """Unpivot (wide to long format). id_vars are columns to keep, value_vars are columns to melt.
        Reverse of pivot. Use when data is in wide format but tools expect long format.
        Example: melt_dataframe(id_vars=["Name","City"], value_vars=["Score","Revenue"])"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            val_vars = value_vars if value_vars else None
            before_shape = df.shape
            df = pd.melt(df, id_vars=id_vars, value_vars=val_vars, var_name=var_name, value_name=value_name)
            store.set(name, df)
            return (
                f"Melted '{name}': {before_shape[0]}x{before_shape[1]} -> {df.shape[0]}x{df.shape[1]}.\n"
                f"Columns: {df.columns.tolist()}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def concat_dataframes(
        names: list[str],
        axis: int = 0,
        result_name: str = "",
    ) -> str:
        """Concatenate multiple dataframes along rows (axis=0) or columns (axis=1).
        Combine train and test sets back together, or append new data.
        Example: concat_dataframes(names=["data_train", "data_test"], axis=0)"""
        try:
            dfs = []
            for n in names:
                dfs.append(store.get(n))

            res_name = result_name if result_name else "_".join(names) + "_concat"
            concatenated = pd.concat(dfs, axis=axis, ignore_index=True)
            store.add(res_name, concatenated, set_current=False)
            return (
                f"Concatenated {len(names)} dataframes (axis={axis}) as '{res_name}'.\n"
                f"Result: {concatenated.shape[0]} rows, {concatenated.shape[1]} columns."
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
