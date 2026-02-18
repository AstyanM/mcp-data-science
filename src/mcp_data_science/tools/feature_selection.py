import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def correlation_filter(
        target_column: str,
        threshold: float = 0.05,
        method: str = "pearson",
        df_name: str = "",
    ) -> str:
        """Identify features with correlation to target below threshold.
        Returns list of low-importance columns with their correlation values. Does NOT auto-drop.
        Non-destructive: identifies low-correlation features but does NOT drop them. Review results before using drop_low_importance.
        Example: correlation_filter(target_column="Revenue", threshold=0.05, method="pearson")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}"

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            if target_column not in numeric_cols:
                return f"Error: Target '{target_column}' is not numeric."

            correlations = {}
            for col in numeric_cols:
                if col == target_column:
                    continue
                corr = abs(df[col].corr(df[target_column], method=method))
                correlations[col] = corr

            # Sort by correlation ascending
            sorted_corr = sorted(correlations.items(), key=lambda x: x[1])
            low = [(col, corr) for col, corr in sorted_corr if corr < threshold]
            high = [(col, corr) for col, corr in sorted_corr if corr >= threshold]

            lines = [
                f"Correlation filter (target='{target_column}', threshold={threshold}, method={method}):",
                f"  Total numeric features: {len(correlations)}",
                f"  Below threshold: {len(low)}",
                f"  Above threshold: {len(high)}",
                "",
                "Low-correlation features (candidates for removal):",
            ]
            for col, corr in low:
                lines.append(f"  {col:<35} {corr:.4f}")

            if not low:
                lines.append("  (none)")

            lines.append("")
            lines.append("Top correlated features:")
            for col, corr in sorted(high, key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  {col:<35} {corr:.4f}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def variance_filter(threshold: float = 0.0, df_name: str = "") -> str:
        """Identify columns with variance at or below threshold (constant/near-constant).
        Returns list of low-variance columns. Does NOT auto-drop.
        Non-destructive: identifies constant/near-constant columns. These provide no useful information for modeling.
        Example: variance_filter(threshold=0.01)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)

            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            variances = {}
            for col in numeric_cols:
                variances[col] = df[col].var()

            sorted_var = sorted(variances.items(), key=lambda x: x[1])
            low = [(col, var) for col, var in sorted_var if var <= threshold]

            lines = [
                f"Variance filter (threshold={threshold}) for '{name}':",
                f"  Total numeric columns: {len(numeric_cols)}",
                f"  Low-variance columns: {len(low)}",
                "",
            ]

            if low:
                lines.append("Low-variance columns (candidates for removal):")
                for col, var in low:
                    lines.append(f"  {col:<35} variance={var:.6f}")
            else:
                lines.append("No low-variance columns found.")

            # Show all variances sorted
            lines.append("")
            lines.append("All variances (ascending):")
            for col, var in sorted_var[:20]:
                lines.append(f"  {col:<35} {var:.6f}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def feature_importance(
        target_column: str,
        method: str = "random_forest",
        top_n: int = 20,
        df_name: str = "",
    ) -> str:
        """Compute feature importance using tree-based model or mutual information.
        Methods: 'random_forest', 'mutual_info_classif', 'mutual_info_regression'.
        Returns ranked list of features with importance scores.
        Run after all encoding and feature engineering. Ranks features by predictive power. Helps focus modeling on most important features.
        Example: feature_importance(target_column="Revenue", method="random_forest", top_n=20)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if target_column not in df.columns:
                return f"Error: Column '{target_column}' not found. Available: {df.columns.tolist()}"

            feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
            if not feature_cols:
                return "Error: No numeric feature columns found (excluding target)."

            X = df[feature_cols].fillna(0)
            y = df[target_column]

            if method == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X, y)
                importances = dict(zip(feature_cols, model.feature_importances_))
            elif method == "mutual_info_classif":
                from sklearn.feature_selection import mutual_info_classif
                mi = mutual_info_classif(X, y, random_state=42)
                importances = dict(zip(feature_cols, mi))
            elif method == "mutual_info_regression":
                from sklearn.feature_selection import mutual_info_regression
                mi = mutual_info_regression(X, y, random_state=42)
                importances = dict(zip(feature_cols, mi))
            else:
                return f"Error: Unknown method '{method}'. Use: random_forest, mutual_info_classif, mutual_info_regression."

            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

            lines = [
                f"Feature Importance ({method}) for target='{target_column}' in '{name}':",
                f"  Total features: {len(feature_cols)}",
                "",
                f"Top {min(top_n, len(sorted_imp))} features:",
                f"{'Rank':<6} {'Feature':<35} {'Importance':>12}",
                "-" * 56,
            ]
            for i, (col, imp) in enumerate(sorted_imp[:top_n], 1):
                lines.append(f"{i:<6} {col:<35} {imp:>12.6f}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def drop_low_importance(
        target_column: str,
        method: str = "variance",
        threshold: float = 0.01,
        df_name: str = "",
    ) -> str:
        """Drop columns below a computed importance threshold.
        Methods: 'variance' (drop cols with variance <= threshold),
                 'correlation' (drop cols with abs correlation to target <= threshold).
        Destructive: actually drops columns. Use AFTER reviewing results from correlation_filter or variance_filter.
        Example: drop_low_importance(target_column="Revenue", method="correlation", threshold=0.05)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            before_cols = len(df.columns)

            numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_column]
            to_drop = []

            if method == "variance":
                for col in numeric_cols:
                    if df[col].var() <= threshold:
                        to_drop.append(col)
            elif method == "correlation":
                if target_column not in df.columns:
                    return f"Error: Target column '{target_column}' not found."
                for col in numeric_cols:
                    corr = abs(df[col].corr(df[target_column]))
                    if corr <= threshold:
                        to_drop.append(col)
            else:
                return f"Error: Unknown method '{method}'. Use: variance, correlation."

            if not to_drop:
                return f"No columns to drop (all above threshold={threshold})."

            df = df.drop(columns=to_drop)
            store.set(name, df)
            return (
                f"Dropped {len(to_drop)} low-importance columns from '{name}' ({method}, threshold={threshold}).\n"
                f"  Dropped: {to_drop}\n"
                f"  Columns: {before_cols} -> {len(df.columns)}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
