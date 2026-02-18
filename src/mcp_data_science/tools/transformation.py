import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def create_column(new_column: str, expression: str, df_name: str = "") -> str:
        """Create a new column using a pandas eval expression. Columns are referenced by name.
        For domain-relevant feature engineering: ratios, differences, interactions. Always add epsilon (1e-5) to denominators to avoid division by zero.
        Example: create_column(new_column="Yield", expression="Revenue / (ChargeableWeight + 1e-5)")
        Example: create_column(new_column="WeightPerPiece", expression="ChargeableWeight / (Pieces + 1e-5)")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            df[new_column] = df.eval(expression)
            store.set(name, df)
            col = df[new_column]
            if pd.api.types.is_numeric_dtype(col):
                stats = f"min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}"
            else:
                stats = f"unique={col.nunique()}, dtype={col.dtype}"
            return f"Created column '{new_column}' in '{name}'. {stats}"
        except Exception as e:
            return f"Error creating column: {type(e).__name__} - {e}"

    @mcp.tool()
    def log_transform(
        columns: list[str],
        method: str = "log1p",
        df_name: str = "",
    ) -> str:
        """Apply log transform to columns. Creates new columns named Log_{column}.
        Methods: 'log1p' (recommended, handles zeros), 'log' (natural log), 'log10'.
        Use on right-skewed distributions (|skewness| > 1). 'log1p' is safest (handles zeros). Check result with plot_histogram.
        Example: log_transform(columns=["Revenue","ChargeableWeight"], method="log1p")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            created = []
            warnings = []
            for col in columns:
                new_col = f"Log_{col}"
                if method == "log1p":
                    neg_count = int((df[col] < 0).sum())
                    if neg_count > 0:
                        warnings.append(f"  {col}: {neg_count} negative values (log1p may produce NaN)")
                    df[new_col] = np.log1p(df[col])
                elif method == "log":
                    non_pos = int((df[col] <= 0).sum())
                    if non_pos > 0:
                        warnings.append(f"  {col}: {non_pos} non-positive values (will become NaN)")
                    df[new_col] = np.log(df[col])
                elif method == "log10":
                    non_pos = int((df[col] <= 0).sum())
                    if non_pos > 0:
                        warnings.append(f"  {col}: {non_pos} non-positive values (will become NaN)")
                    df[new_col] = np.log10(df[col])
                else:
                    return f"Error: Unknown method '{method}'. Use: log1p, log, log10."
                created.append(new_col)

            store.set(name, df)
            result = f"Created {len(created)} log-transformed columns in '{name}': {created}"
            if warnings:
                result += "\nWarnings:\n" + "\n".join(warnings)
            return result
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def normalize(
        columns: list[str],
        method: str = "minmax",
        df_name: str = "",
    ) -> str:
        """Scale numeric columns in-place. Methods: 'minmax' (0-1), 'standard' (z-score), 'robust' (IQR-based).
        Only needed for distance-based models (linear, logistic). Tree-based models do NOT need normalization. Use 'standard' by default, 'robust' if outliers remain.
        Example: normalize(columns=["Revenue","Weight"], method="standard")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            if method == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif method == "standard":
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
            elif method == "robust":
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                return f"Error: Unknown method '{method}'. Use: minmax, standard, robust."

            df[columns] = scaler.fit_transform(df[columns])
            store.set(name, df)

            stats_lines = []
            for col in columns:
                stats_lines.append(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")
            return f"Normalized {len(columns)} columns in '{name}' ({method}):\n" + "\n".join(stats_lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def apply_mapping(column: str, mapping: dict[str, str], df_name: str = "") -> str:
        """Map values in a column using a dictionary. Unmapped values stay as-is.
        Values are auto-converted to match the column's type.
        Map categorical values to new values. Useful for translating codes, merging similar categories, or creating ordinal mappings.
        Example: apply_mapping(column="FlownMonth", mapping={"SEPTEMBER":"9","OCTOBER":"10"})"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            converted_mapping = {}
            for k, v in mapping.items():
                try:
                    converted_mapping[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    converted_mapping[k] = v

            before_unique = df[column].nunique()
            df[column] = df[column].map(lambda x: converted_mapping.get(x, x))
            store.set(name, df)
            after_unique = df[column].nunique()
            return (
                f"Applied mapping to '{column}' in '{name}' ({len(converted_mapping)} rules).\n"
                f"Unique values: {before_unique} -> {after_unique}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def convert_dtype(columns: list[str], dtype: str, df_name: str = "") -> str:
        """Convert column types. Dtypes: 'int', 'float', 'str', 'category', 'datetime', 'bool'.
        Fix incorrect types (numeric stored as string). Do this BEFORE statistical analysis or encoding.
        Example: convert_dtype(columns=["age","score"], dtype="float")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            dtype_map = {
                "int": "int64",
                "float": "float64",
                "str": "str",
                "category": "category",
                "datetime": "datetime64[ns]",
                "bool": "bool",
            }
            if dtype not in dtype_map:
                return f"Error: Unknown dtype '{dtype}'. Use: {list(dtype_map.keys())}"

            results = []
            for col in columns:
                old_dtype = str(df[col].dtype)
                if dtype == "datetime":
                    df[col] = pd.to_datetime(df[col])
                else:
                    df[col] = df[col].astype(dtype_map[dtype])
                results.append(f"  {col}: {old_dtype} -> {df[col].dtype}")

            store.set(name, df)
            return f"Converted {len(columns)} columns in '{name}':\n" + "\n".join(results)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def replace_values(column: str, old_value: str, new_value: str, df_name: str = "") -> str:
        """Replace a specific value with another in a column. Types are auto-cast.
        Fix known data entry errors or standardize values. Use after identifying issues with get_unique_values.
        Example: replace_values(column="Status", old_value="N/A", new_value="Unknown")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            def cast(val, series):
                if pd.api.types.is_numeric_dtype(series):
                    try:
                        return float(val)
                    except ValueError:
                        pass
                return val

            old_cast = cast(old_value, df[column])
            new_cast = cast(new_value, df[column])
            count = int((df[column] == old_cast).sum())
            df[column] = df[column].replace(old_cast, new_cast)
            store.set(name, df)
            return f"Replaced {count} occurrences of '{old_value}' with '{new_value}' in '{column}' of '{name}'."
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def select_dtypes(
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        df_name: str = "",
    ) -> str:
        """Keep only columns matching specified dtypes. Drops non-matching columns.
        Filter columns by type. Use include=['number'] before modeling to keep only numeric features.
        Example: select_dtypes(include=["number"]) — keeps only numeric columns.
        Example: select_dtypes(exclude=["object"]) — drops string columns."""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            before_cols = len(df.columns)
            inc = include if include else None
            exc = exclude if exclude else None
            df = df.select_dtypes(include=inc, exclude=exc)
            store.set(name, df)
            return (
                f"Selected dtypes in '{name}': {before_cols} -> {len(df.columns)} columns.\n"
                f"Kept: {df.columns.tolist()}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def string_clean(
        column: str,
        operations: list[str] | None = None,
        replace_old: str = "",
        replace_new: str = "",
        df_name: str = "",
    ) -> str:
        """Clean a string column. Operations applied in order: 'strip', 'lower', 'upper', 'title', 'replace'.
        For 'replace': uses replace_old and replace_new parameters.
        Run on text columns BEFORE any encoding. Inconsistent casing or whitespace creates spurious categories.
        Example: string_clean(column="Name", operations=["strip","lower"])"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"

            ops = operations if operations else ["strip"]
            col = df[column].astype(str)

            for op in ops:
                if op == "strip":
                    col = col.str.strip()
                elif op == "lower":
                    col = col.str.lower()
                elif op == "upper":
                    col = col.str.upper()
                elif op == "title":
                    col = col.str.title()
                elif op == "replace":
                    col = col.str.replace(replace_old, replace_new, regex=False)
                else:
                    return f"Error: Unknown operation '{op}'. Use: strip, lower, upper, title, replace."

            df[column] = col
            store.set(name, df)
            sample = df[column].head(3).tolist()
            return f"Cleaned '{column}' in '{name}' (operations: {ops}). Sample: {sample}"
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
