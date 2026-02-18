import io

import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def get_head(n: int = 5, df_name: str = "") -> str:
        """Return the first N rows of the dataframe as a formatted table.
        Quick first look after loading. Verify data loaded correctly and understand structure.
        Example: get_head(n=10)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            return f"First {n} rows of '{name}':\n{df.head(n).to_string()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_tail(n: int = 5, df_name: str = "") -> str:
        """Return the last N rows of the dataframe as a formatted table.
        Check the last rows. Useful for time-series to see the most recent data.
        Example: get_tail(n=10)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            return f"Last {n} rows of '{name}':\n{df.tail(n).to_string()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_info(df_name: str = "") -> str:
        """Get column dtypes, non-null counts, and memory usage for the dataframe.
        Similar to pandas df.info() output.
        Check column types and non-null counts. Run when a tool reports 'column not found' to verify names."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            buf = io.StringIO()
            df.info(buf=buf, memory_usage="deep")
            return f"Info for '{name}':\n{buf.getvalue()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_statistics(df_name: str = "") -> str:
        """Get descriptive statistics for all columns (numeric and categorical).
        Returns count, mean, std, min, quartiles, max for numeric; count, unique, top, freq for categorical.
        Understand distributions. Run after quality_report to decide cleaning strategies."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            return f"Statistics for '{name}':\n{df.describe(include='all').to_string()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_shape(df_name: str = "") -> str:
        """Get the number of rows and columns in the dataframe.
        Quick dimension check. Always run after filtering or dropping to verify not too many rows lost."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            return f"'{name}': {df.shape[0]} rows, {df.shape[1]} columns."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def quality_report(df_name: str = "") -> str:
        """Comprehensive data quality report: dtype per column, missing value count and %,
        unique value count, and total duplicate rows. ESSENTIAL FIRST STEP after loading data.
        Run before any cleaning or transformation. Reveals the full data quality landscape in one call."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)

            lines = [f"Data Quality Report for '{name}' ({df.shape[0]} rows, {df.shape[1]} columns)", ""]

            # Per-column report
            lines.append(f"{'Column':<30} {'Dtype':<12} {'Missing':>8} {'Miss%':>7} {'Unique':>8}")
            lines.append("-" * 70)
            for col in df.columns:
                dtype = str(df[col].dtype)
                missing = int(df[col].isna().sum())
                miss_pct = f"{missing / len(df) * 100:.1f}%"
                unique = int(df[col].nunique())
                lines.append(f"{col:<30} {dtype:<12} {missing:>8} {miss_pct:>7} {unique:>8}")

            # Duplicates
            dup_count = int(df.duplicated().sum())
            lines.append("")
            lines.append(f"Duplicate rows: {dup_count} ({dup_count / len(df) * 100:.1f}%)")
            lines.append(f"Total memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_unique_values(column: str, top_n: int = 50, df_name: str = "") -> str:
        """List unique values with their frequencies for a column, sorted by count descending.
        Understand categorical columns before encoding. Reveals rare categories that may need grouping.
        Example: get_unique_values(column="CargoType", top_n=20)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"
            vc = df[column].value_counts(dropna=False).head(top_n)
            total = len(df)
            lines = [f"Unique values for '{column}' in '{name}' (top {top_n}):", ""]
            lines.append(f"{'Value':<40} {'Count':>8} {'%':>7}")
            lines.append("-" * 58)
            for val, count in vc.items():
                pct = f"{count / total * 100:.1f}%"
                lines.append(f"{str(val):<40} {count:>8} {pct:>7}")
            lines.append(f"\nTotal unique: {df[column].nunique(dropna=False)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_column_profile(column: str, df_name: str = "") -> str:
        """Detailed single-column analysis: dtype, nulls, unique count,
        statistics (if numeric: min/max/mean/median/std/skew/kurtosis),
        and top 10 value frequencies.
        Deep-dive into one column. Use when quality_report flags something. Provides skewness for deciding if log_transform is needed.
        Example: get_column_profile(column="Revenue")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            col = df[column]
            lines = [f"Column Profile: '{column}' in '{name}'", ""]
            lines.append(f"Dtype:    {col.dtype}")
            lines.append(f"Non-null: {col.notna().sum()} / {len(col)}")
            lines.append(f"Missing:  {col.isna().sum()} ({col.isna().sum() / len(col) * 100:.1f}%)")
            lines.append(f"Unique:   {col.nunique(dropna=False)}")

            if pd.api.types.is_numeric_dtype(col):
                lines.append("")
                lines.append("Numeric Statistics:")
                lines.append(f"  Min:      {col.min()}")
                lines.append(f"  Max:      {col.max()}")
                lines.append(f"  Mean:     {col.mean():.4f}")
                lines.append(f"  Median:   {col.median():.4f}")
                lines.append(f"  Std:      {col.std():.4f}")
                lines.append(f"  Skewness: {col.skew():.4f}")
                lines.append(f"  Kurtosis: {col.kurtosis():.4f}")

            lines.append("")
            lines.append("Top 10 values:")
            vc = col.value_counts(dropna=False).head(10)
            for val, count in vc.items():
                pct = f"{count / len(col) * 100:.1f}%"
                lines.append(f"  {str(val):<35} {count:>8} ({pct})")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def sample_data(n: int = 10, random_state: int = 42, df_name: str = "") -> str:
        """Return a random sample of N rows. Use when head/tail are not representative.
        Random sampling reveals data variety better.
        Example: sample_data(n=5)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            sample_n = min(n, len(df))
            return f"Random sample of {sample_n} rows from '{name}':\n{df.sample(n=sample_n, random_state=random_state).to_string()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
