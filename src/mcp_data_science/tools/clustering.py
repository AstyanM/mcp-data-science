import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mcp.server.fastmcp import FastMCP, Image

from mcp_data_science.state import DataStore
from mcp_data_science.tools._plot_helpers import fig_to_image, fig_to_image_and_store


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def kmeans_cluster(
        columns: list[str],
        n_clusters: int = 3,
        cluster_column: str = "cluster",
        random_state: int = 42,
        df_name: str = "",
    ) -> str:
        """K-Means clustering. Assigns each row to one of n_clusters groups based on numeric columns.
        Features are auto-standardized before clustering. Result is stored as a new column.
        Use for customer segmentation, anomaly grouping, or discovering natural data groups.
        Example: kmeans_cluster(columns=["Revenue","ChargeableWeight","Pieces"], n_clusters=4)"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            X = df[columns].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            df[cluster_column] = labels
            store.set(name, df)

            # Cluster summary
            lines = [f"K-Means clustering on '{name}': {n_clusters} clusters, features={columns}\n"]
            lines.append(f"{'Cluster':<10} {'Size':>8} {'%':>7}")
            lines.append("-" * 27)
            for i in range(n_clusters):
                count = int((labels == i).sum())
                pct = count / len(labels) * 100
                lines.append(f"{i:<10} {count:>8} {pct:>6.1f}%")

            lines.append(f"\nInertia: {km.inertia_:.2f}")
            lines.append(f"Labels stored as column '{cluster_column}'.")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def dbscan_cluster(
        columns: list[str],
        eps: float = 0.5,
        min_samples: int = 5,
        cluster_column: str = "cluster",
        df_name: str = "",
    ) -> str:
        """DBSCAN clustering. Density-based clustering that finds arbitrarily shaped clusters.
        Does not require specifying n_clusters. Labels outliers as -1.
        Features are auto-standardized. Use when clusters have irregular shapes or when you need outlier detection.
        Example: dbscan_cluster(columns=["Revenue","Weight"], eps=0.5, min_samples=5)"""
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                return f"Error: Columns not found: {missing}. Available: {df.columns.tolist()}"

            X = df[columns].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_scaled)
            df[cluster_column] = labels
            store.set(name, df)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())

            lines = [f"DBSCAN clustering on '{name}': eps={eps}, min_samples={min_samples}\n"]
            lines.append(f"  Clusters found: {n_clusters}")
            lines.append(f"  Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
            for label in sorted(set(labels)):
                count = int((labels == label).sum())
                lbl = "Noise" if label == -1 else f"Cluster {label}"
                lines.append(f"  {lbl}: {count} points ({count/len(labels)*100:.1f}%)")

            lines.append(f"\nLabels stored as column '{cluster_column}'.")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def elbow_plot(
        columns: list[str],
        max_k: int = 10,
        random_state: int = 42,
        save_path: str = "",
        df_name: str = "",
    ) -> Image:
        """Elbow plot to find optimal number of clusters for K-Means.
        Plots inertia (within-cluster sum of squares) vs number of clusters.
        The 'elbow' point where the curve bends is the optimal K.
        Example: elbow_plot(columns=["Revenue","Weight"], max_k=10)"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler

            name = store.resolve_name(df_name)
            df = store.get(name)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found: {missing}")

            X = df[columns].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            inertias = []
            k_range = range(2, max_k + 1)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                km.fit(X_scaled)
                inertias.append(km.inertia_)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(list(k_range), inertias, "bo-", linewidth=2, markersize=8)
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Plot — Optimal K for K-Means")
            ax.grid(True)
            fig.tight_layout()
            return fig_to_image_and_store(fig, store, "elbow_plot", save_path)
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes)
            return fig_to_image(fig)

    @mcp.tool()
    def silhouette_score(
        cluster_column: str,
        feature_columns: list[str],
        df_name: str = "",
    ) -> str:
        """Compute silhouette score to evaluate clustering quality.
        Score ranges from -1 to 1: higher is better. >0.5 = good, >0.7 = excellent, <0.25 = poor.
        Run after kmeans_cluster or dbscan_cluster.
        Example: silhouette_score(cluster_column="cluster", feature_columns=["Revenue","Weight"])"""
        try:
            from sklearn.metrics import silhouette_score as sk_sil
            from sklearn.preprocessing import StandardScaler

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [cluster_column] + feature_columns:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            labels = df[cluster_column].values
            n_labels = len(set(labels) - {-1})
            if n_labels < 2:
                return "Error: Need at least 2 clusters to compute silhouette score."

            # Exclude noise points (-1)
            mask = labels != -1
            X = df.loc[mask, feature_columns].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)
            valid_labels = labels[mask]

            score = sk_sil(X_scaled, valid_labels)

            if score >= 0.7:
                quality = "Excellent"
            elif score >= 0.5:
                quality = "Good"
            elif score >= 0.25:
                quality = "Fair"
            else:
                quality = "Poor"

            return (
                f"Silhouette Score for '{cluster_column}' in '{name}': {score:.4f}\n"
                f"  Quality: {quality}\n"
                f"  Clusters: {n_labels}, Points used: {mask.sum()} (excluded {(~mask).sum()} noise points)"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def cluster_profile(
        cluster_column: str,
        feature_columns: list[str] | None = None,
        df_name: str = "",
    ) -> str:
        """Descriptive statistics per cluster: mean of each feature for each cluster.
        Use to understand what characterizes each cluster. If feature_columns is empty, uses all numeric columns.
        Example: cluster_profile(cluster_column="cluster", feature_columns=["Revenue","Weight","Pieces"])"""
        try:
            name = store.resolve_name(df_name)
            df = store.get(name)
            if cluster_column not in df.columns:
                return f"Error: Column '{cluster_column}' not found. Available: {df.columns.tolist()}"

            if feature_columns:
                missing = [c for c in feature_columns if c not in df.columns]
                if missing:
                    return f"Error: Columns not found: {missing}"
                cols = feature_columns
            else:
                cols = [c for c in df.select_dtypes(include="number").columns if c != cluster_column]

            profile = df.groupby(cluster_column)[cols].agg(["mean", "median", "count"]).round(2)

            # Flatten multi-level columns for readable output
            lines = [f"Cluster Profile for '{cluster_column}' in '{name}':\n"]

            cluster_counts = df[cluster_column].value_counts().sort_index()
            lines.append("Cluster sizes:")
            for cl, count in cluster_counts.items():
                lbl = "Noise" if cl == -1 else f"Cluster {cl}"
                lines.append(f"  {lbl}: {count} ({count/len(df)*100:.1f}%)")

            lines.append("\nMean values per cluster:")
            mean_table = df.groupby(cluster_column)[cols].mean().round(4)
            lines.append(mean_table.to_string())

            lines.append("\nMedian values per cluster:")
            median_table = df.groupby(cluster_column)[cols].median().round(4)
            lines.append(median_table.to_string())

            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
