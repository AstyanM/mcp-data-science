import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def ttest_independent(
        column: str,
        group_column: str,
        df_name: str = "",
    ) -> str:
        """Independent two-sample t-test. Compares means of a numeric column across two groups.
        Use to determine if the difference between two group means is statistically significant.
        Requires exactly 2 groups in group_column. For 3+ groups, use anova_test instead.
        Example: ttest_independent(column="Revenue", group_column="CargoType")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column, group_column]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            groups = df[group_column].dropna().unique()
            if len(groups) != 2:
                return (
                    f"Error: t-test requires exactly 2 groups, found {len(groups)}: {groups.tolist()}. "
                    f"Use anova_test for 3+ groups, or filter_rows first."
                )

            g1 = df[df[group_column] == groups[0]][column].dropna()
            g2 = df[df[group_column] == groups[1]][column].dropna()

            t_stat, p_value = stats.ttest_ind(g1, g2, equal_var=False)

            sig = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
            effect_size = abs(g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)

            return (
                f"Independent t-test: '{column}' by '{group_column}' in '{name}'\n"
                f"  Group '{groups[0]}': n={len(g1)}, mean={g1.mean():.4f}, std={g1.std():.4f}\n"
                f"  Group '{groups[1]}': n={len(g2)}, mean={g2.mean():.4f}, std={g2.std():.4f}\n"
                f"  t-statistic: {t_stat:.4f}\n"
                f"  p-value: {p_value:.6f}\n"
                f"  Significant difference: {sig}\n"
                f"  Cohen's d (effect size): {effect_size:.4f}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def anova_test(
        column: str,
        group_column: str,
        df_name: str = "",
    ) -> str:
        """One-way ANOVA test. Compares means of a numeric column across 3+ groups.
        Use to determine if at least one group mean differs significantly from the others.
        For exactly 2 groups, ttest_independent is more appropriate.
        Example: anova_test(column="Revenue", group_column="FlownMonth")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column, group_column]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            groups = df[group_column].dropna().unique()
            if len(groups) < 2:
                return f"Error: ANOVA requires at least 2 groups, found {len(groups)}."

            group_data = [df[df[group_column] == g][column].dropna() for g in groups]
            f_stat, p_value = stats.f_oneway(*group_data)

            sig = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"

            # Group summaries
            lines = [
                f"One-way ANOVA: '{column}' by '{group_column}' in '{name}'\n",
                f"{'Group':<30} {'N':>6} {'Mean':>12} {'Std':>12}",
                "-" * 62,
            ]
            for g, data in zip(groups, group_data):
                lines.append(f"{str(g):<30} {len(data):>6} {data.mean():>12.4f} {data.std():>12.4f}")

            # Eta-squared effect size
            grand_mean = df[column].dropna().mean()
            ss_between = sum(len(d) * (d.mean() - grand_mean) ** 2 for d in group_data)
            ss_total = sum((d - grand_mean).pow(2).sum() for d in group_data)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            lines.extend([
                "",
                f"  F-statistic: {f_stat:.4f}",
                f"  p-value: {p_value:.6f}",
                f"  Significant difference: {sig}",
                f"  Eta-squared (effect size): {eta_sq:.4f}",
            ])
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def chi_square_test(
        col_a: str,
        col_b: str,
        df_name: str = "",
    ) -> str:
        """Chi-square test of independence between two categorical columns.
        Tests whether two categorical variables are statistically associated or independent.
        Example: chi_square_test(col_a="CargoType", col_b="ProductCode")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [col_a, col_b]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            ct = pd.crosstab(df[col_a], df[col_b])
            chi2, p_value, dof, expected = stats.chi2_contingency(ct)

            sig = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"
            n = ct.values.sum()
            cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0

            return (
                f"Chi-square test: '{col_a}' vs '{col_b}' in '{name}'\n"
                f"  Contingency table: {ct.shape[0]} x {ct.shape[1]}\n"
                f"  Chi-square statistic: {chi2:.4f}\n"
                f"  Degrees of freedom: {dof}\n"
                f"  p-value: {p_value:.6f}\n"
                f"  Significant association: {sig}\n"
                f"  Cramér's V (effect size): {cramers_v:.4f}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def normality_test(
        column: str,
        method: str = "shapiro",
        df_name: str = "",
    ) -> str:
        """Test if a numeric column follows a normal distribution.
        Methods: 'shapiro' (best for n < 5000), 'ks' (Kolmogorov-Smirnov, any sample size),
        'dagostino' (D'Agostino-Pearson, n >= 20).
        Use before deciding on parametric vs non-parametric tests. If p < 0.05, data is NOT normal.
        Example: normality_test(column="Revenue", method="shapiro")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            if column not in df.columns:
                return f"Error: Column '{column}' not found. Available: {df.columns.tolist()}"
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"Error: Column '{column}' is not numeric."

            data = df[column].dropna()

            if method == "shapiro":
                sample = data.sample(min(5000, len(data)), random_state=42) if len(data) > 5000 else data
                stat, p_value = stats.shapiro(sample)
                test_name = "Shapiro-Wilk"
            elif method == "ks":
                standardized = (data - data.mean()) / data.std()
                stat, p_value = stats.kstest(standardized, "norm")
                test_name = "Kolmogorov-Smirnov"
            elif method == "dagostino":
                if len(data) < 20:
                    return "Error: D'Agostino test requires at least 20 samples."
                stat, p_value = stats.normaltest(data)
                test_name = "D'Agostino-Pearson"
            else:
                return f"Error: Unknown method '{method}'. Use: shapiro, ks, dagostino."

            is_normal = "YES (p >= 0.05)" if p_value >= 0.05 else "NO (p < 0.05)"
            skew = data.skew()
            kurt = data.kurtosis()

            return (
                f"Normality test ({test_name}): '{column}' in '{name}'\n"
                f"  n={len(data)}, skewness={skew:.4f}, kurtosis={kurt:.4f}\n"
                f"  Test statistic: {stat:.6f}\n"
                f"  p-value: {p_value:.6f}\n"
                f"  Normally distributed: {is_normal}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def mann_whitney_test(
        column: str,
        group_column: str,
        df_name: str = "",
    ) -> str:
        """Mann-Whitney U test (non-parametric alternative to t-test).
        Compares distributions of a numeric column across two groups without assuming normality.
        Use when normality_test indicates non-normal data. Requires exactly 2 groups.
        Example: mann_whitney_test(column="Revenue", group_column="CargoType")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column, group_column]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            groups = df[group_column].dropna().unique()
            if len(groups) != 2:
                return (
                    f"Error: Mann-Whitney requires exactly 2 groups, found {len(groups)}. "
                    f"Use kruskal_wallis_test for 3+ groups."
                )

            g1 = df[df[group_column] == groups[0]][column].dropna()
            g2 = df[df[group_column] == groups[1]][column].dropna()

            u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            sig = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"

            # Rank-biserial correlation as effect size
            n1, n2 = len(g1), len(g2)
            r = 1 - (2 * u_stat) / (n1 * n2)

            return (
                f"Mann-Whitney U test: '{column}' by '{group_column}' in '{name}'\n"
                f"  Group '{groups[0]}': n={n1}, median={g1.median():.4f}\n"
                f"  Group '{groups[1]}': n={n2}, median={g2.median():.4f}\n"
                f"  U-statistic: {u_stat:.4f}\n"
                f"  p-value: {p_value:.6f}\n"
                f"  Significant difference: {sig}\n"
                f"  Rank-biserial correlation: {r:.4f}"
            )
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"

    @mcp.tool()
    def kruskal_wallis_test(
        column: str,
        group_column: str,
        df_name: str = "",
    ) -> str:
        """Kruskal-Wallis H test (non-parametric alternative to ANOVA).
        Compares distributions of a numeric column across 3+ groups without assuming normality.
        Use when normality_test indicates non-normal data. For 2 groups, use mann_whitney_test.
        Example: kruskal_wallis_test(column="Revenue", group_column="FlownMonth")"""
        try:
            from scipy import stats

            name = store.resolve_name(df_name)
            df = store.get(name)
            for col in [column, group_column]:
                if col not in df.columns:
                    return f"Error: Column '{col}' not found. Available: {df.columns.tolist()}"

            groups = df[group_column].dropna().unique()
            if len(groups) < 2:
                return f"Error: Kruskal-Wallis requires at least 2 groups, found {len(groups)}."

            group_data = [df[df[group_column] == g][column].dropna() for g in groups]
            h_stat, p_value = stats.kruskal(*group_data)

            sig = "YES (p < 0.05)" if p_value < 0.05 else "NO (p >= 0.05)"

            # Epsilon-squared effect size
            n = sum(len(d) for d in group_data)
            eps_sq = (h_stat - len(groups) + 1) / (n - len(groups)) if n > len(groups) else 0

            lines = [
                f"Kruskal-Wallis H test: '{column}' by '{group_column}' in '{name}'\n",
                f"{'Group':<30} {'N':>6} {'Median':>12}",
                "-" * 50,
            ]
            for g, data in zip(groups, group_data):
                lines.append(f"{str(g):<30} {len(data):>6} {data.median():>12.4f}")

            lines.extend([
                "",
                f"  H-statistic: {h_stat:.4f}",
                f"  p-value: {p_value:.6f}",
                f"  Significant difference: {sig}",
                f"  Epsilon-squared (effect size): {eps_sq:.4f}",
            ])
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {type(e).__name__} - {e}"
