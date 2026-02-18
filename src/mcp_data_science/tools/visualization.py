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

    @mcp.tool()
    def plot_violin(
        column: str,
        by: str = "",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Violin plot: combines box plot with KDE to show full distribution shape.
        Better than boxplot for skewed or multimodal distributions. Optional grouping.
        Example: plot_violin(column="Revenue", by="CargoType")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found. Available: {df.columns.tolist()}")

            fig, ax = plt.subplots(figsize=(10, 6))
            if by and by in df.columns:
                sns.violinplot(data=df, x=by, y=column, ax=ax, inner="box")
                ax.set_title(f"{column} by {by}")
                ax.tick_params(axis="x", rotation=45)
            else:
                sns.violinplot(data=df, y=column, ax=ax, inner="box")
                ax.set_title(f"Violin plot of {column}")

            fig.tight_layout()
            plot_name = f"violin_{column}_by_{by}" if by else f"violin_{column}"
            return fig_to_image_and_store(fig, store, plot_name, save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_qq(
        column: str,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Q-Q plot to visually assess if a column follows a normal distribution.
        Points on the diagonal = normal. Deviations show skewness or heavy tails.
        Use alongside normality_test for visual confirmation.
        Example: plot_qq(column="Revenue")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found.")

            fig, ax = plt.subplots(figsize=(8, 8))
            data = df[column].dropna()
            stats.probplot(data, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot — {column}")
            ax.get_lines()[0].set_markersize(3)
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"qq_{column}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_stacked_bar(
        column: str,
        by: str,
        normalize: bool = True,
        top_n: int = 10,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Stacked bar chart showing composition of one categorical within another.
        normalize=True shows percentages (100% stacked), False shows raw counts.
        Example: plot_stacked_bar(column="ProductCode", by="CargoType", normalize=True)"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column, by]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found.")

            ct = pd.crosstab(df[column], df[by])
            ct = ct.loc[ct.sum(axis=1).nlargest(top_n).index]

            if normalize:
                ct = ct.div(ct.sum(axis=1), axis=0) * 100

            fig, ax = plt.subplots(figsize=(12, 6))
            ct.plot(kind="bar", stacked=True, ax=ax)
            ax.set_title(f"{'Proportions' if normalize else 'Counts'}: {column} by {by}")
            ax.set_ylabel("%" if normalize else "Count")
            ax.legend(title=by, bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"stacked_bar_{column}_by_{by}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_heatmap(
        index_col: str,
        columns_col: str,
        values_col: str,
        agg_func: str = "mean",
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Heatmap of aggregated values in a pivot table format.
        Shows a color-coded matrix of one metric across two categorical dimensions.
        Example: plot_heatmap(index_col="OriginCode", columns_col="FlownMonth", values_col="Revenue", agg_func="sum")"""
        try:
            import numpy as np

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [index_col, columns_col, values_col]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found.")

            pivot = df.pivot_table(index=index_col, columns=columns_col, values=values_col, aggfunc=agg_func)
            # Limit to top 20 rows by total
            if len(pivot) > 20:
                pivot = pivot.loc[pivot.sum(axis=1).nlargest(20).index]

            fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
            sns.heatmap(pivot, annot=len(pivot) <= 15 and len(pivot.columns) <= 10,
                        fmt=".1f", cmap="YlOrRd", ax=ax)
            ax.set_title(f"Heatmap: {agg_func}({values_col}) by {index_col} x {columns_col}")
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"heatmap_{index_col}_{columns_col}_{values_col}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_distribution_comparison(
        column: str,
        by: str,
        top_n_groups: int = 5,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Overlay KDE distributions of a numeric column for different groups.
        Better than separate histograms for comparing distribution shapes across categories.
        Example: plot_distribution_comparison(column="Revenue", by="CargoType")"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column, by]:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found.")

            top_groups = df[by].value_counts().head(top_n_groups).index
            fig, ax = plt.subplots(figsize=(10, 6))

            for group in top_groups:
                subset = df[df[by] == group][column].dropna()
                if len(subset) > 1:
                    sns.kdeplot(subset, label=str(group), ax=ax, fill=True, alpha=0.3)

            ax.set_xlabel(column)
            ax.set_ylabel("Density")
            ax.set_title(f"Distribution Comparison: {column} by {by}")
            ax.legend(title=by)
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"dist_compare_{column}_by_{by}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def plot_cumulative(
        column: str,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Cumulative distribution function (CDF) plot.
        Shows what percentage of data falls below each value. Useful for understanding thresholds.
        Example: plot_cumulative(column="Revenue")"""
        try:
            import numpy as np

            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found.")

            data = df[column].dropna().sort_values()
            cdf = np.arange(1, len(data) + 1) / len(data) * 100

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.values, cdf, linewidth=2)
            ax.set_xlabel(column)
            ax.set_ylabel("Cumulative %")
            ax.set_title(f"Cumulative Distribution — {column}")
            ax.grid(True, alpha=0.3)

            # Add reference lines at key percentiles
            for pct in [25, 50, 75, 90]:
                val = data.quantile(pct / 100)
                ax.axhline(y=pct, color="gray", linestyle="--", alpha=0.5)
                ax.axvline(x=val, color="gray", linestyle="--", alpha=0.5)
                ax.annotate(f"P{pct}={val:.1f}", xy=(val, pct), fontsize=8,
                            xytext=(5, 5), textcoords="offset points")

            fig.tight_layout()
            return fig_to_image_and_store(fig, store, f"cdf_{column}", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)
