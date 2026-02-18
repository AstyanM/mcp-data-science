import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mcp.server.fastmcp import FastMCP, Image

from mcp_data_science.state import DataStore
from mcp_data_science.tools._plot_helpers import fig_to_image


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def plot_histogram(
        column: str,
        bins: int = 30,
        kde: bool = True,
        log_scale: bool = False,
        df_name: str = "",
    ) -> Image:
        """Plot a histogram for a numeric column with optional KDE overlay and log-scaled x-axis.
        Example: plot_histogram(column="Revenue", bins=50, kde=True, log_scale=True)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")

            fig, ax = plt.subplots(figsize=(10, 6))
            data = df[column].dropna()
            sns.histplot(data, bins=bins, kde=kde, ax=ax)
            if log_scale:
                ax.set_xscale("log")
            ax.set_title(f"Histogram of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            return fig_to_image(fig)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_bar(
        column: str,
        top_n: int = 10,
        orientation: str = "vertical",
        df_name: str = "",
    ) -> Image:
        """Bar plot of value counts (top N categories). Vertical or horizontal.
        Example: plot_bar(column="CargoType", top_n=10, orientation="horizontal")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")

            vc = df[column].value_counts().head(top_n)
            fig, ax = plt.subplots(figsize=(10, 6))

            if orientation == "horizontal":
                sns.barplot(x=vc.values, y=vc.index, ax=ax, orient="h")
                ax.set_xlabel("Count")
                ax.set_ylabel(column)
            else:
                sns.barplot(x=vc.index, y=vc.values, ax=ax)
                ax.set_xlabel(column)
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)

            ax.set_title(f"Top {top_n} values of {column}")
            fig.tight_layout()
            return fig_to_image(fig)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_scatter(
        x: str,
        y: str,
        hue: str = "",
        df_name: str = "",
    ) -> Image:
        """Scatter plot between two numeric columns. Optional color grouping by hue column.
        Example: plot_scatter(x="Revenue", y="Weight", hue="CargoType")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [x, y]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

            fig, ax = plt.subplots(figsize=(10, 6))
            hue_col = hue if hue and hue in df.columns else None
            sns.scatterplot(data=df, x=x, y=y, hue=hue_col, alpha=0.6, ax=ax)
            ax.set_title(f"{y} vs {x}")
            fig.tight_layout()
            return fig_to_image(fig)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_box(
        column: str,
        by: str = "",
        df_name: str = "",
    ) -> Image:
        """Box plot for outlier visualization. Optional grouping by a categorical column.
        Example: plot_box(column="Revenue", by="CargoType")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")

            fig, ax = plt.subplots(figsize=(10, 6))
            if by and by in df.columns:
                sns.boxplot(data=df, x=by, y=column, ax=ax)
                ax.set_title(f"{column} by {by}")
                ax.tick_params(axis="x", rotation=45)
            else:
                sns.boxplot(data=df, y=column, ax=ax)
                ax.set_title(f"Box plot of {column}")

            fig.tight_layout()
            return fig_to_image(fig)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_correlation_matrix(
        columns: list[str] | None = None,
        method: str = "pearson",
        df_name: str = "",
    ) -> Image:
        """Correlation heatmap with lower triangle mask and annotations.
        Methods: pearson, spearman, kendall. Empty columns = all numeric.
        Example: plot_correlation_matrix(method="pearson")"""
        try:
            import numpy as np

            name = store.resolve_name(df_name)
            df = store.get(name)

            if columns:
                missing = [c for c in columns if c not in df.columns]
                if missing:
                    raise ValueError(f"Columns not found: {missing}")
                data = df[columns]
            else:
                data = df.select_dtypes(include="number")

            corr = data.corr(method=method)
            mask = np.triu(np.ones_like(corr, dtype=bool))

            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                ax=ax,
            )
            ax.set_title(f"Correlation Matrix ({method})")
            fig.tight_layout()
            return fig_to_image(fig)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_pairplot(
        columns: list[str],
        hue: str = "",
        df_name: str = "",
    ) -> Image:
        """Seaborn pairplot for up to 6 columns. Shows distributions on diagonal and
        scatter plots on off-diagonal. Limit to 6 columns for readability.
        Example: plot_pairplot(columns=["Revenue","Weight","Score"], hue="Category")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)

            # Limit to 6 columns
            cols = columns[:6]
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")

            hue_col = hue if hue and hue in df.columns else None
            plot_cols = cols + ([hue_col] if hue_col and hue_col not in cols else [])

            g = sns.pairplot(df[plot_cols].dropna(), hue=hue_col, diag_kind="kde")
            g.figure.suptitle(f"Pair Plot ({len(cols)} columns)", y=1.02)
            return fig_to_image(g.figure)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_missing_values(df_name: str = "") -> Image:
        """Heatmap showing missing value patterns across all columns.
        White = present, colored = missing. Helps identify systematic missing data.
        Example: plot_missing_values()"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(
                df.isna().astype(int),
                cbar_kws={"label": "Missing"},
                yticklabels=False,
                cmap="YlOrRd",
                ax=ax,
            )
            ax.set_title(f"Missing Values Pattern — '{name}'")
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            fig.tight_layout()
            return fig_to_image(fig)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)
