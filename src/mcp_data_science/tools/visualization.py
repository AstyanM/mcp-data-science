import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mcp.server.fastmcp import FastMCP, Image

from mcp_data_science.state import DataStore
from mcp_data_science.tools._plot_helpers import fig_to_image, fig_to_image_and_store


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def plot_histogram(
        column: str,
        bins: int = 30,
        kde: bool = True,
        log_scale: bool = False,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Plot a histogram for a numeric column with optional KDE overlay and log-scaled x-axis.
        Use during EDA to understand numeric distributions. Set log_scale=True for skewed data (revenue, prices) — otherwise histogram is unreadable. Check for bimodality indicating subgroups.
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
            return fig_to_image_and_store(fig, store, f"histogram_{column}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_bar(
        column: str,
        top_n: int = 10,
        orientation: str = "vertical",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Bar plot of value counts (top N categories). Vertical or horizontal.
        Understand categorical distributions during EDA. Reveals rare categories for potential grouping and dominant categories for stratified sampling.
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
            return fig_to_image_and_store(fig, store, f"bar_{column}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_scatter(
        x: str,
        y: str,
        hue: str = "",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Scatter plot between two numeric columns. Optional color grouping by hue column.
        Explore relationships between two numeric variables. Add hue for categorical grouping to reveal subgroup patterns.
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
            return fig_to_image_and_store(fig, store, f"scatter_{x}_vs_{y}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_box(
        column: str,
        by: str = "",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Box plot for outlier visualization. Optional grouping by a categorical column.
        Visualize outliers and compare distributions across groups. Complements detect_outliers with a visual perspective.
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
            plot_name = f"box_{column}_by_{by}" if by else f"box_{column}"
            return fig_to_image_and_store(fig, store, plot_name, save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_correlation_matrix(
        columns: list[str] | None = None,
        method: str = "pearson",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Correlation heatmap with lower triangle mask and annotations.
        Methods: pearson, spearman, kendall. Empty columns = all numeric.
        Run after all numeric features are prepared. Identifies multicollinearity (features correlated with each other) and features correlated with the target.
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
            return fig_to_image_and_store(fig, store, "correlation_matrix", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_pairplot(
        columns: list[str],
        hue: str = "",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Seaborn pairplot for up to 6 columns. Shows distributions on diagonal and
        scatter plots on off-diagonal. Limit to 6 columns for readability.
        Expensive computation. Use after feature_importance to focus on top features.
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
            return fig_to_image_and_store(g.figure, store, "pairplot", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_missing_values(save_path: str = "", df_name: str = "") -> Image:
        """Heatmap showing missing value patterns across all columns.
        White = present, colored = missing. Use when quality_report shows significant missingness.
        Patterns in the heatmap reveal if data is missing randomly or systematically.
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
            return fig_to_image_and_store(fig, store, "missing_values", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_line(
        x: str,
        y: str | list[str],
        hue: str = "",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Line plot. Supports multiple y columns overlaid. Essential for time-series and trends.
        For time-series data and trends. Sort data by x-axis column first for meaningful results. Supports multiple y columns overlaid.
        Example: plot_line(x="date", y="Revenue")
        Example: plot_line(x="month", y=["Revenue", "Cost"], hue="Category")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if x not in df.columns:
                raise ValueError(f"Column '{x}' not found. Available: {df.columns.tolist()}")

            y_cols = [y] if isinstance(y, str) else y
            for col in y_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found. Available: {df.columns.tolist()}")

            fig, ax = plt.subplots(figsize=(12, 6))
            hue_col = hue if hue and hue in df.columns else None

            if len(y_cols) == 1 and hue_col:
                sns.lineplot(data=df, x=x, y=y_cols[0], hue=hue_col, ax=ax)
            elif len(y_cols) == 1:
                sns.lineplot(data=df, x=x, y=y_cols[0], ax=ax)
            else:
                for col in y_cols:
                    sns.lineplot(data=df, x=x, y=col, label=col, ax=ax)
                ax.legend()

            ax.set_title(f"Line Plot: {', '.join(y_cols)} vs {x}")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            y_label = "_".join(y_cols) if len(y_cols) > 1 else y_cols[0]
            return fig_to_image_and_store(fig, store, f"line_{x}_vs_{y_label}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)
