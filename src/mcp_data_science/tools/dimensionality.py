import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mcp.server.fastmcp import FastMCP, Image

from mcp_data_science.state import DataStore
from mcp_data_science.tools._plot_helpers import fig_to_image, fig_to_image_and_store


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def pca_transform(
        columns: list[str],
        n_components: int = 2,
        plot: bool = True,
        hue: str = "",
        save_path: str = "",
        df_name: str = "",
    ) -> str:
        """PCA dimensionality reduction. Projects numeric columns onto principal components.
        New columns PC1, PC2, ... are added to the dataframe. Features are auto-standardized.
        If plot=True and n_components>=2, also stores a 2D scatter plot (viewable via save_report).
        Use to visualize high-dimensional data, reduce multicollinearity, or compress features.
        Example: pca_transform(columns=["Revenue","Weight","Pieces"], n_components=2, hue="CargoType")"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}"

            X = df[columns].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            pca = PCA(n_components=min(n_components, len(columns)))
            components = pca.fit_transform(X_scaled)

            # Add PC columns to dataframe
            pc_cols = []
            for i in range(components.shape[1]):
                col_name = f"PC{i+1}"
                df[col_name] = components[:, i]
                pc_cols.append(col_name)
            store.set(name, df)

            # Build summary
            lines = [f"PCA on '{name}': {len(columns)} features -> {components.shape[1]} components\n"]
            lines.append(f"{'Component':<12} {'Variance %':>12} {'Cumulative %':>14}")
            lines.append("-" * 40)
            cum_var = 0
            for i, var in enumerate(pca.explained_variance_ratio_):
                cum_var += var * 100
                lines.append(f"PC{i+1:<10} {var*100:>11.2f}% {cum_var:>13.2f}%")

            lines.append(f"\nTotal variance explained: {cum_var:.2f}%")
            lines.append(f"Columns added: {pc_cols}")

            if plot and n_components >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                hue_col = hue if hue and hue in df.columns else None

                scatter_data = pd.DataFrame({"PC1": components[:, 0], "PC2": components[:, 1]})
                if hue_col:
                    scatter_data[hue_col] = df[hue_col].values

                sns.scatterplot(data=scatter_data, x="PC1", y="PC2", hue=hue_col, alpha=0.6, ax=ax)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                ax.set_title("PCA — First 2 Principal Components")
                fig.tight_layout()
                fig_to_image_and_store(fig, store, "pca_scatter", save_path)
                lines.append("\nPCA scatter plot stored as 'pca_scatter'.")

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def tsne_plot(
        columns: list[str],
        n_components: int = 2,
        perplexity: float = 30.0,
        hue: str = "",
        random_state: int = 42,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """t-SNE visualization. Non-linear dimensionality reduction for 2D visualization.
        Better than PCA for revealing clusters and local structure. Features are auto-standardized.
        Slower than PCA — best on datasets < 10,000 rows or use sample_data first.
        Example: tsne_plot(columns=["Revenue","Weight","Pieces"], perplexity=30, hue="CargoType")"""
        try:
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")

            X = df[columns].fillna(0)

            # Sample if too large
            max_rows = 10000
            if len(X) > max_rows:
                sample_idx = X.sample(max_rows, random_state=random_state).index
                X_sample = X.loc[sample_idx]
                hue_data = df.loc[sample_idx, hue] if hue and hue in df.columns else None
            else:
                X_sample = X
                sample_idx = X.index
                hue_data = df.loc[sample_idx, hue] if hue and hue in df.columns else None

            X_scaled = StandardScaler().fit_transform(X_sample)

            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            embedding = tsne.fit_transform(X_scaled)

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter_data = pd.DataFrame({"t-SNE 1": embedding[:, 0], "t-SNE 2": embedding[:, 1]})
            if hue_data is not None:
                scatter_data["hue"] = hue_data.values
                sns.scatterplot(data=scatter_data, x="t-SNE 1", y="t-SNE 2", hue="hue", alpha=0.6, ax=ax)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=10)

            n_used = len(X_sample)
            ax.set_title(f"t-SNE Visualization (n={n_used}, perplexity={perplexity})")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, "tsne_plot", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)
