import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def extract_datetime_parts(
        column: str,
        parts: list[str] | None = None,
        df_name: str = "",
    ) -> str:
        """Extract datetime components as new columns ({column}_year, {column}_month, etc.).
        Parts: year, month, day, dayofweek, hour, minute, weekofyear, quarter, is_weekend.
        Example: extract_datetime_parts(column="date", parts=["year", "month", "dayofweek", "is_weekend"])"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = pd.to_datetime(df[column])

            default_parts = ["year", "month", "day", "dayofweek", "hour"]
            extract_parts = parts if parts else default_parts
            dt = df[column].dt

            created = []
            for part in extract_parts:
                new_col = f"{column}_{part}"
                if part == "year":
                    df[new_col] = dt.year
                elif part == "month":
                    df[new_col] = dt.month
                elif part == "day":
                    df[new_col] = dt.day
                elif part == "dayofweek":
                    df[new_col] = dt.dayofweek
                elif part == "hour":
                    df[new_col] = dt.hour
                elif part == "minute":
                    df[new_col] = dt.minute
                elif part == "weekofyear":
                    df[new_col] = dt.isocalendar().week.astype(int)
                elif part == "quarter":
                    df[new_col] = dt.quarter
                elif part == "is_weekend":
                    df[new_col] = (dt.dayofweek >= 5).astype(int)
                else:
                    return f"Error: Unknown part '{part}'. Use: year, month, day, dayofweek, hour, minute, weekofyear, quarter, is_weekend."
                created.append(new_col)

            store.set(name, df)
            return f"Extracted {len(created)} datetime parts from '{column}' in '{name}': {created}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def datetime_diff(
        column_a: str,
        column_b: str,
        new_column: str,
        unit: str = "days",
        df_name: str = "",
    ) -> str:
        """Compute time difference between two datetime columns (column_a - column_b).
        Units: days, hours, minutes, seconds. Creates a new numeric column.
        Example: datetime_diff(column_a="end_date", column_b="start_date", new_column="duration_days", unit="days")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column_a, column_b]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            # Ensure datetime
            for col in [column_a, column_b]:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])

            delta = df[column_a] - df[column_b]

            unit_map = {
                "days": delta.dt.total_seconds() / 86400,
                "hours": delta.dt.total_seconds() / 3600,
                "minutes": delta.dt.total_seconds() / 60,
                "seconds": delta.dt.total_seconds(),
            }

            if unit not in unit_map:
                return f"Error: Unknown unit '{unit}'. Use: days, hours, minutes, seconds."

            df[new_column] = unit_map[unit]
            store.set(name, df)

            col = df[new_column]
            return (
                f"Created '{new_column}' = '{column_a}' - '{column_b}' in {unit}.\n"
                f"  min={col.min():.2f}, max={col.max():.2f}, mean={col.mean():.2f}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def datetime_filter(
        column: str,
        start: str = "",
        end: str = "",
        df_name: str = "",
    ) -> str:
        """Filter rows by datetime range. Start/end as ISO strings (e.g., '2024-01-01').
        Leave start or end empty for open-ended range.
        Example: datetime_filter(column="date", start="2024-01-01", end="2024-06-30")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = pd.to_datetime(df[column])

            before = len(df)
            mask = pd.Series(True, index=df.index)

            if start:
                mask &= df[column] >= pd.to_datetime(start)
            if end:
                mask &= df[column] <= pd.to_datetime(end)

            df = df[mask]
            store.set(name, df)

            range_str = f"[{start or '...'} to {end or '...'}]"
            return f"Filtered '{name}' on '{column}' {range_str}. Rows: {before} -> {len(df)}."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def set_datetime_index(column: str, df_name: str = "") -> str:
        """Set a datetime column as the DataFrame index. Useful before time-series operations.
        Example: set_datetime_index(column="date")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            if not pd.api.types.is_datetime64_any_dtype(df[column]):
                df[column] = pd.to_datetime(df[column])

            df = df.set_index(column).sort_index()
            store.set(name, df)

            return (
                f"Set '{column}' as datetime index for '{name}'.\n"
                f"  Range: {df.index.min()} to {df.index.max()}\n"
                f"  Remaining columns: {df.columns.tolist()}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
