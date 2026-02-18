# MCP Data Science Server — Agent Workflow Guide

## Core Principle

**Understand before acting.** Never transform, clean, or model data without inspecting it first. Every decision (what to drop, how to encode, which model to train) must be informed by data exploration.

---

## Standard Pipeline (9 Phases)

### Phase 1 — Loading and First Look

Start every analysis here. Load the data, then immediately verify it loaded correctly.

1. `load_csv` — load the dataset
2. `get_shape` — confirm row/column counts
3. `get_head` — verify structure and column names
4. `get_info` — check dtypes and non-null counts

### Phase 2 — Exploratory Data Analysis (EDA)

Build a complete picture of the data before any modifications.

1. `quality_report` — **ALWAYS RUN FIRST**. Reveals missing values, duplicates, dtypes, and unique counts in one call.
2. `get_statistics` — numeric and categorical distributions (mean, std, quartiles, top values).
3. `plot_histogram` — on key numeric columns. Use `log_scale=True` for skewed data (revenue, prices, weights) — otherwise the histogram is unreadable.
4. `plot_bar` — on key categorical columns to see category frequencies.
5. `get_column_profile` — deep-dive on the target variable and any column flagged by quality_report.
6. `plot_missing_values` — if quality_report shows significant missingness (>5% in any column).
7. `plot_correlation_matrix` — identify multicollinearity and features correlated with the target.
8. `detect_outliers` — on numeric columns of interest, before deciding whether to clip.

### Phase 3 — Data Cleaning

**Order matters.** Follow this sequence to avoid cascading errors.

1. **`drop_duplicates`** — always first. Duplicates inflate statistics and bias models.
2. **`rename_columns`** — standardize to snake_case, remove spaces and special characters, use explicit names. Do this early to avoid reference errors later.
3. **Missing values** — decision tree:
   - <5% missing → `fill_missing` with `median` (numeric) or `mode` (categorical)
   - 5–30% missing → evaluate column importance before deciding
   - >50% missing → consider `drop_columns` unless the column is critical
4. **`string_clean`** — on text columns (`strip` + `lower`), BEFORE any encoding.
5. **`replace_values`** — fix known data entry errors.
6. **`clip_outliers`** — only if outliers are errors, not legitimate extreme values. Always run `detect_outliers` first.
7. **`convert_dtype`** — fix incorrect types (e.g., numeric stored as string). Do this before statistical analysis or encoding.

### Phase 4 — Feature Engineering

Create new features that capture domain knowledge.

- `extract_datetime_parts` — if datetime columns exist, extract `year`, `month`, `dayofweek`, `is_weekend`. These are often highly predictive.
- `create_column` — for domain-relevant ratios and combinations. **Always add epsilon (1e-5) to denominators** to avoid division by zero. Example: `"Revenue / (Weight + 1e-5)"`.
- `log_transform` — use `log1p` on right-skewed features (|skewness| > 1). Verify with `plot_histogram` after transforming.
- `drop_columns` — remove useless columns: constants, IDs, redundant features. Always understand a column before dropping it.

### Phase 5 — Categorical Encoding

Choose the right encoding based on cardinality:

| Scenario | Tool | Notes |
|----------|------|-------|
| 2–10 unique values | `one_hot_encode` | Always `drop_first=True` for modeling |
| >10 unique values | `target_encode` | **After train/test split** to avoid data leakage |
| Natural ordinal order | `label_encode` | e.g., low/medium/high, small/large |
| Quick exploration | `frequency_encode` | When no clear target variable exists |

**WARNING**: Never use `target_encode` before `train_test_split` — this causes data leakage and inflated model performance.

### Phase 6 — Feature Selection

Remove noise features to improve model performance and reduce overfitting.

1. `variance_filter` — identify constant or near-constant columns (zero variance = zero information).
2. `correlation_filter` — identify features with very low correlation to the target.
3. `feature_importance` — rank features by predictive power using a tree-based model.
4. `drop_low_importance` — remove features below the threshold. **Review results before dropping.**

### Phase 7 — Normalization

**Only for distance-based models.** Tree-based models do NOT need normalization.

| Model type | Needs normalization? |
|------------|---------------------|
| `linear_regression`, `logistic_regression` | Yes |
| `random_forest`, `gradient_boosting`, `decision_tree` | No |

- Use `"standard"` (z-score) by default.
- Use `"robust"` if outliers remain after cleaning.

### Phase 8 — Modeling

1. `train_test_split` — always first. Use `stratify=True` for classification, fixed `random_state=42` for reproducibility.
2. Start with a **simple baseline** (`linear_regression` or `logistic_regression`).
3. Then try complex models (`random_forest`, `gradient_boosting`).
4. `evaluate_model` — always on the **test set**, never on training data.
5. Compare metrics against the baseline to justify added complexity.

**Model selection guide:**
- Predicting a continuous number → regression (`linear_regression`, `random_forest`, `gradient_boosting`)
- Predicting a category → classification (`logistic_regression`, `random_forest`, `gradient_boosting`)

### Phase 9 — Report

Use `save_report` as the final step to export a structured, presentable analysis.

Required sections:
1. **Executive Summary** — key findings in 3–5 bullet points
2. **Data Overview** — source, shape, columns, data types
3. **Exploratory Analysis** — key distributions, correlations, outliers (with plots)
4. **Cleaning Log** — what was removed/fixed and why
5. **Modeling Results** — baseline vs. final model, metrics comparison

Include key plots by referencing them via `include_plots`.

---

## Decision Frameworks

### When to Visualize vs. Compute

- **Statistics** (`get_statistics`, `detect_outliers`, `get_correlation`) → when you need precise numbers to make decisions.
- **Visualizations** (`plot_histogram`, `plot_box`, `plot_scatter`) → when understanding distributions, spotting patterns, or communicating findings to users.

Use both together: compute first, then visualize to confirm and communicate.

### Handling Skew

1. Check via `get_column_profile` → look at the skewness value.
2. If |skewness| > 1 → apply `log_transform(method="log1p")`.
3. Verify the result with `plot_histogram` — the distribution should look more symmetric.
4. Use the log-transformed version for modeling, keep the original for interpretability.

### Missing Data Strategy

| Missing % | Strategy |
|-----------|----------|
| 0% | Nothing to do |
| <5% | `fill_missing` — median (numeric), mode (categorical) |
| 5–30% | Investigate first: is the missingness random or systematic? Check with `plot_missing_values` |
| 30–50% | Consider if the column is important enough to keep. If kept, use median/mode fill |
| >50% | `drop_columns` unless the column is critical and missingness is informative |

---

## 10 Common Pitfalls

1. **NEVER** encode before cleaning — dirty strings create spurious categories.
2. **NEVER** target-encode before train/test split — this is data leakage.
3. **NEVER** normalize for tree-based models — it adds complexity with no benefit.
4. **NEVER** skip EDA — blind transformations lead to bad models and wrong conclusions.
5. **NEVER** drop columns without understanding them first — use `get_column_profile` before `drop_columns`.
6. **ALWAYS** use `copy_dataframe` before destructive operations if the original may be needed later.
7. **ALWAYS** check `get_shape` after filtering or dropping to ensure not too many rows were lost.
8. **ALWAYS** use `log_scale=True` in `plot_histogram` for heavily skewed distributions.
9. **ALWAYS** add epsilon (1e-5) to denominators in `create_column` expressions.
10. **ALWAYS** use median over mean for `fill_missing` on skewed data — the mean is pulled by outliers.

---

## Error Recovery

| Error | What to do |
|-------|------------|
| "Column not found" | Run `get_info` or `get_head` to check exact column names (case-sensitive). |
| "No dataframe loaded" | Call `load_csv` first, or `list_dataframes` to check what's available. |
| log_transform warnings about negatives | Check with `get_column_profile`, the column may need clipping first. |
| Training R² < 0.1 | Features are not predictive. Run `feature_importance` to investigate. |
| Test score much worse than training | Overfitting. Reduce features with `drop_low_importance` or use a simpler model. |
| merge_dataframes unexpected shape | Verify join keys exist and are consistent with `get_info` on both dataframes. |
| Encoding creates too many columns | The column has too many categories for one-hot. Switch to `target_encode` or `frequency_encode`. |
