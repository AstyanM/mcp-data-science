import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def get_correlation(
        col_a: str,
        col_b: str,
        method: str = "pearson",
        df_name: str = "",
    ) -> str:
        """Compute pairwise correlation between two numeric columns.
        Methods: pearson, spearman, kendall. Returns value and interpretation.
        Example: get_correlation(col_a="Revenue", col_b="Weight", method="pearson")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [col_a, col_b]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            corr = df[col_a].corr(df[col_b], method=method)
            abs_corr = abs(corr)
            if abs_corr >= 0.8:
                strength = "very strong"
            elif abs_corr >= 0.6:
                strength = "strong"
            elif abs_corr >= 0.4:
                strength = "moderate"
            elif abs_corr >= 0.2:
                strength = "weak"
            else:
                strength = "very weak / none"

            direction = "positive" if corr > 0 else "negative"
            return (
                f"Correlation ({method}) between '{col_a}' and '{col_b}': {corr:.4f}\n"
                f"Interpretation: {strength} {direction} correlation."
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def get_value_counts(column: str, top_n: int = 20, df_name: str = "") -> str:
        """Value counts with percentages, sorted descending. Shows top_n values.
        Example: get_value_counts(column="City", top_n=10)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            vc = df[column].value_counts(dropna=False).head(top_n)
            total = len(df)
            lines = [f"Value counts for '{column}' in '{name}' (top {top_n}):", ""]
            lines.append(f"{'Value':<40} {'Count':>8} {'%':>7} {'Cumul%':>8}")
            lines.append("-" * 66)
            cumul = 0.0
            for val, count in vc.items():
                pct = count / total * 100
                cumul += pct
                lines.append(f"{str(val):<40} {count:>8} {pct:>6.1f}% {cumul:>7.1f}%")

            lines.append(f"\nTotal rows: {total}, Unique values: {df[column].nunique(dropna=False)}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def detect_outliers(
        column: str,
        method: str = "iqr",
        threshold: float = 1.5,
        df_name: str = "",
    ) -> str:
        """Detect outliers in a numeric column. Methods: 'iqr' (threshold=IQR multiplier)
        or 'zscore' (threshold=z-score cutoff). Returns count, %, and boundary values.
        Example: detect_outliers(column="Revenue", method="iqr", threshold=1.5)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"Error: Column '{column}' is not numeric (dtype: {df[column].dtype})."

            col = df[column].dropna()

            if method == "iqr":
                q1 = col.quantile(0.25)
                q3 = col.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                outliers = col[(col < lower) | (col > upper)]
                method_desc = f"IQR method (multiplier={threshold})"
            elif method == "zscore":
                mean = col.mean()
                std = col.std()
                z_scores = (col - mean) / std
                outliers = col[z_scores.abs() > threshold]
                lower = mean - threshold * std
                upper = mean + threshold * std
                method_desc = f"Z-score method (threshold={threshold})"
            else:
                return f"Error: Unknown method '{method}'. Use: iqr, zscore."

            pct = len(outliers) / len(col) * 100 if len(col) > 0 else 0
            return (
                f"Outlier Detection for '{column}' in '{name}' ({method_desc}):\n"
                f"  Outliers found: {len(outliers)} ({pct:.1f}%)\n"
                f"  Lower bound: {lower:.4f}\n"
                f"  Upper bound: {upper:.4f}\n"
                f"  Column range: [{col.min():.4f}, {col.max():.4f}]\n"
                f"  Outlier values (min/max): [{outliers.min():.4f}, {outliers.max():.4f}]"
                if len(outliers) > 0 else
                f"Outlier Detection for '{column}' in '{name}' ({method_desc}):\n"
                f"  Outliers found: 0\n"
                f"  Lower bound: {lower:.4f}\n"
                f"  Upper bound: {upper:.4f}\n"
                f"  Column range: [{col.min():.4f}, {col.max():.4f}]"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def group_aggregate(
        group_by: list[str],
        agg_column: str,
        agg_func: str = "mean",
        df_name: str = "",
    ) -> str:
        """GroupBy aggregation. Functions: mean, sum, count, min, max, median, std.
        Example: group_aggregate(group_by=["City"], agg_column="Revenue", agg_func="mean")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in group_by + [agg_column]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            valid_funcs = ["mean", "sum", "count", "min", "max", "median", "std"]
            if agg_func not in valid_funcs:
                return f"Error: Unknown function '{agg_func}'. Use: {valid_funcs}"

            result = df.groupby(group_by)[agg_column].agg(agg_func).reset_index()
            result.columns = group_by + [f"{agg_column}_{agg_func}"]
            result = result.sort_values(f"{agg_column}_{agg_func}", ascending=False)

            return (
                f"GroupBy {group_by} -> {agg_func}({agg_column}) in '{name}':\n\n"
                f"{result.to_string(index=False)}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def crosstab(
        index_col: str,
        columns_col: str,
        normalize: bool = False,
        df_name: str = "",
    ) -> str:
        """Cross-tabulation of two categorical columns. Optional normalization (percentages).
        Example: crosstab(index_col="City", columns_col="Category", normalize=True)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [index_col, columns_col]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            norm = "all" if normalize else False
            ct = pd.crosstab(df[index_col], df[columns_col], normalize=norm)

            if normalize:
                ct = ct.round(4) * 100
                label = "Cross-tabulation (%) "
            else:
                label = "Cross-tabulation "

            return f"{label}of '{index_col}' x '{columns_col}' in '{name}':\n\n{ct.to_string()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def add_row_index(
        column_name: str = "index",
        start: int = 0,
        df_name: str = "",
    ) -> str:
        """Add or reset a numeric row index as a column. Useful after filtering operations.
        Example: add_row_index(column_name="row_id", start=1)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            df[column_name] = range(start, start + len(df))
            store.set(name, df)
            return f"Added index column '{column_name}' to '{name}' (range {start} to {start + len(df) - 1})."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
