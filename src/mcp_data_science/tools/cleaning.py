import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def drop_duplicates(
        subset: list[str] | None = None,
        keep: str = "first",
        df_name: str = "",
    ) -> str:
        """Remove duplicate rows. subset: columns to check (None = all).
        keep: 'first', 'last', or 'none' (drop all duplicates).
        Example: drop_duplicates(subset=["col1","col2"], keep="first")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            before = len(df)
            keep_val = False if keep == "none" else keep
            sub = subset if subset else None
            df = df.drop_duplicates(subset=sub, keep=keep_val)
            store.set(name, df)
            dropped = before - len(df)
            return f"Dropped {dropped} duplicate rows from '{name}'. Rows: {before} -> {len(df)}."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def drop_columns(columns: list[str], df_name: str = "") -> str:
        """Drop specified columns from the dataframe.
        Example: drop_columns(columns=["col1","col2"])"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"
            df = df.drop(columns=columns)
            store.set(name, df)
            return f"Dropped {len(columns)} columns from '{name}'. Remaining: {df.columns.tolist()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def drop_missing(
        subset: list[str] | None = None,
        how: str = "any",
        df_name: str = "",
    ) -> str:
        """Drop rows with missing values. how: 'any' (any null) or 'all' (all null).
        subset: limit check to specific columns (None = all).
        Example: drop_missing(subset=["col1"], how="any")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            before = len(df)
            sub = subset if subset else None
            df = df.dropna(subset=sub, how=how)
            store.set(name, df)
            dropped = before - len(df)
            return f"Dropped {dropped} rows with missing values from '{name}'. Rows: {before} -> {len(df)}."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def fill_missing(
        columns: list[str] | None = None,
        strategy: str = "value",
        value: str = "",
        df_name: str = "",
    ) -> str:
        """Fill missing values. Strategies: 'value' (literal), 'mean', 'median', 'mode', 'ffill', 'bfill'.
        columns: which columns to fill (None = all columns with nulls).
        Example: fill_missing(columns=["Revenue"], strategy="median")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)

            if columns:
                missing_cols = [c for c in columns if c not in df.columns]
                if missing_cols:
                    return f"Error: Columns not found: {missing_cols}. Available: {df.columns.tolist()}"
                target_cols = columns
            else:
                target_cols = [c for c in df.columns if df[c].isna().any()]

            filled_counts = {}
            for col in target_cols:
                null_count = int(df[col].isna().sum())
                if null_count == 0:
                    continue
                is_numeric = pd.api.types.is_numeric_dtype(df[col])
                if strategy == "value":
                    try:
                        fill_val = float(value) if is_numeric else value
                    except ValueError:
                        fill_val = value
                    df[col] = df[col].fillna(fill_val)
                elif strategy == "mean":
                    if not is_numeric:
                        continue
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    if not is_numeric:
                        continue
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                elif strategy == "ffill":
                    df[col] = df[col].ffill()
                elif strategy == "bfill":
                    df[col] = df[col].bfill()
                else:
                    return f"Error: Unknown strategy '{strategy}'. Use: value, mean, median, mode, ffill, bfill."
                filled_counts[col] = null_count

            store.set(name, df)
            if not filled_counts:
                return f"No missing values found in the specified columns of '{name}'."
            summary = ", ".join(f"{col}: {n} filled" for col, n in filled_counts.items())
            return f"Filled missing values in '{name}': {summary}."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def filter_rows(column: str, operator: str, value: str, df_name: str = "") -> str:
        """Filter rows by condition. Operators: ==, !=, >, <, >=, <=, contains, isin.
        For isin: provide comma-separated values. Value is auto-cast for numeric columns.
        Example: filter_rows(column="Revenue", operator=">", value="1000")
        Example: filter_rows(column="CargoType", operator="isin", value="GCR,SCR")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            before = len(df)
            col = df[column]

            def cast_val(v):
                if pd.api.types.is_numeric_dtype(col):
                    try:
                        return float(v)
                    except ValueError:
                        pass
                return v

            if operator == "==":
                mask = col == cast_val(value)
            elif operator == "!=":
                mask = col != cast_val(value)
            elif operator == ">":
                mask = col > cast_val(value)
            elif operator == "<":
                mask = col < cast_val(value)
            elif operator == ">=":
                mask = col >= cast_val(value)
            elif operator == "<=":
                mask = col <= cast_val(value)
            elif operator == "contains":
                mask = col.astype(str).str.contains(value, na=False)
            elif operator == "isin":
                values = [v.strip() for v in value.split(",")]
                if pd.api.types.is_numeric_dtype(col):
                    values = [float(v) for v in values]
                mask = col.isin(values)
            else:
                return f"Error: Unknown operator '{operator}'. Use: ==, !=, >, <, >=, <=, contains, isin."

            df = df[mask]
            store.set(name, df)
            return f"Filtered '{name}' on {column} {operator} {value}. Rows: {before} -> {len(df)}."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def rename_columns(mapping: dict[str, str], df_name: str = "") -> str:
        """Rename columns using an old->new mapping.
        Example: rename_columns(mapping={"old_name": "new_name"})"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in mapping if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"
            df = df.rename(columns=mapping)
            store.set(name, df)
            return f"Renamed {len(mapping)} columns in '{name}'. Columns: {df.columns.tolist()}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def clip_outliers(
        column: str,
        method: str = "iqr",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        iqr_multiplier: float = 1.5,
        df_name: str = "",
    ) -> str:
        """Clip outlier values in a numeric column.
        Methods: 'iqr' (Q1 - multiplier*IQR, Q3 + multiplier*IQR) or 'quantile' (lower/upper quantile).
        Example: clip_outliers(column="Revenue", method="iqr", iqr_multiplier=1.5)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"Error: Column '{column}' is not numeric (dtype: {df[column].dtype})."

            original_min, original_max = df[column].min(), df[column].max()

            if method == "iqr":
                q1 = df[column].quantile(0.25)
                q3 = df[column].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - iqr_multiplier * iqr
                upper = q3 + iqr_multiplier * iqr
            elif method == "quantile":
                lower = df[column].quantile(lower_quantile)
                upper = df[column].quantile(upper_quantile)
            else:
                return f"Error: Unknown method '{method}'. Use: iqr, quantile."

            clipped_count = int(((df[column] < lower) | (df[column] > upper)).sum())
            df[column] = df[column].clip(lower=lower, upper=upper)
            store.set(name, df)

            return (
                f"Clipped '{column}' in '{name}': {clipped_count} values adjusted.\n"
                f"Bounds: [{lower:.4f}, {upper:.4f}]\n"
                f"Range: [{original_min} .. {original_max}] -> [{df[column].min()} .. {df[column].max()}]"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def sort_values(columns: list[str], ascending: bool = True, df_name: str = "") -> str:
        """Sort the dataframe by one or more columns.
        Example: sort_values(columns=["Revenue"], ascending=False)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"
            df = df.sort_values(by=columns, ascending=ascending).reset_index(drop=True)
            store.set(name, df)
            preview = df[columns].head(5).to_string()
            return f"Sorted '{name}' by {columns} (ascending={ascending}). Preview:\n{preview}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
