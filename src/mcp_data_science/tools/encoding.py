import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def one_hot_encode(
        columns: list[str],
        drop_first: bool = True,
        df_name: str = "",
    ) -> str:
        """One-hot encode categorical columns (creates binary columns).
        drop_first=True avoids multicollinearity (recommended for modeling).
        Example: one_hot_encode(columns=["CargoType"], drop_first=True)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            before_cols = df.columns.tolist()
            df = pd.get_dummies(df, columns=columns, drop_first=drop_first)
            store.set(name, df)
            new_cols = [c for c in df.columns if c not in before_cols]
            return (
                f"One-hot encoded {len(columns)} columns in '{name}'.\n"
                f"New columns ({len(new_cols)}): {new_cols}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def target_encode(
        columns: list[str],
        target_column: str,
        smoothing: float = 10.0,
        df_name: str = "",
    ) -> str:
        """Target-encode high-cardinality categorical columns. Each category is replaced
        by the smoothed mean of the target variable. Good for columns with many unique values.
        Example: target_encode(columns=["AgentCode","OriginCode"], target_column="Log_Revenue", smoothing=10.0)"""
        try:
            import category_encoders as ce

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"
            if target_column not in df.columns:
                return f"Error: Target column '{target_column}' not found. Available: {df.columns.tolist()}"

            encoder = ce.TargetEncoder(cols=columns, smoothing=smoothing)
            df[columns] = encoder.fit_transform(df[columns], df[target_column])
            store.set(name, df)

            summaries = []
            for col in columns:
                summaries.append(
                    f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}"
                )
            return (
                f"Target-encoded {len(columns)} columns in '{name}' (target='{target_column}', smoothing={smoothing}).\n"
                + "\n".join(summaries)
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def label_encode(columns: list[str], df_name: str = "") -> str:
        """Label-encode categorical columns: map each unique value to an integer (0, 1, 2, ...).
        Example: label_encode(columns=["City","Category"])"""
        try:
            from sklearn.preprocessing import LabelEncoder

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            mappings = {}
            for col in columns:
                le = LabelEncoder()
                non_null = df[col].dropna()
                le.fit(non_null.astype(str))
                encoded = le.transform(non_null.astype(str))
                df[col] = pd.array([pd.NA] * len(df), dtype=pd.Int64Dtype())
                df.loc[non_null.index, col] = encoded
                mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))

            store.set(name, df)

            lines = [f"Label-encoded {len(columns)} columns in '{name}':"]
            for col, mapping in mappings.items():
                preview = dict(list(mapping.items())[:10])
                lines.append(f"  {col}: {preview}" + (" ..." if len(mapping) > 10 else ""))
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def frequency_encode(columns: list[str], df_name: str = "") -> str:
        """Replace each category with its frequency (count / total rows).
        Example: frequency_encode(columns=["City","Category"])"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            total = len(df)
            lines = [f"Frequency-encoded {len(columns)} columns in '{name}':"]
            for col in columns:
                freq_map = df[col].value_counts(normalize=True).to_dict()
                df[col] = df[col].map(freq_map)
                preview = dict(list(freq_map.items())[:5])
                preview_str = {str(k): f"{v:.4f}" for k, v in preview.items()}
                lines.append(f"  {col}: {preview_str}" + (" ..." if len(freq_map) > 5 else ""))

            store.set(name, df)
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
