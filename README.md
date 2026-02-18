# MCP Data Science

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![MCP](https://img.shields.io/badge/MCP-1.6+-8A2BE2)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3+-f89939)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458)
![License](https://img.shields.io/badge/License-MIT-yellow)

MCP server (Model Context Protocol) that gives an LLM **102 tools** for complete data science pipelines.
Load, inspect, clean, transform, encode, visualize, analyze datasets, train ML models, select features, handle datetime operations, run statistical tests, interpret models, perform clustering, and reduce dimensionality — all through natural language.
Built-in **agent workflow guide** with a 13-phase pipeline, decision frameworks, and common pitfall warnings.
Zero configuration — point any MCP-compatible client at the server and start analyzing data.

---

## Motivation

Data science workflows require dozens of repetitive steps: loading CSVs, checking for missing values, encoding categoricals, training models, evaluating results.
MCP Data Science wraps the entire pipeline into a single server that any LLM agent can drive end-to-end — no notebooks, no boilerplate, no context switching.

| Phase                 | Tools | What it covers                                                          |
| --------------------- | ----- | ----------------------------------------------------------------------- |
| **Loading**           | 11    | CSV, Excel, Parquet, merge, pivot, melt, concat                         |
| **Inspection**        | 9     | Head, tail, info, stats, quality report, profiling                      |
| **Cleaning**          | 9     | Duplicates, missing values, outliers, filtering, binning                |
| **Transformation**    | 9     | Computed columns, log transform, normalization, string cleaning         |
| **Encoding**          | 4     | One-hot, target, label, frequency encoding                              |
| **Visualization**     | 14    | Histogram, scatter, box, violin, QQ, correlation matrix, heatmap…       |
| **Analysis**          | 8     | Correlation, outliers, group aggregation, crosstab                      |
| **Modeling**          | 8     | Train/test split, train, predict, evaluate, cross-validate, grid search |
| **Feature Selection** | 4     | Variance filter, correlation filter, importance ranking                 |
| **Datetime**          | 4     | Extract parts, diff, filter, set index                                  |
| **Statistical Tests** | 6     | t-test, ANOVA, chi-square, normality, Mann-Whitney, Kruskal-Wallis      |
| **Interpretation**    | 7     | Feature importance, residuals, confusion matrix, ROC, learning curve    |
| **Clustering**        | 5     | K-Means, DBSCAN, elbow plot, silhouette score, cluster profiling        |
| **Dimensionality**    | 2     | PCA, t-SNE                                                              |
| **Reporting**         | 2     | Markdown report, self-contained HTML report                             |

All 102 tools follow the same pattern: accept a DataFrame name (or use the current one), perform the operation, return a concise text summary (or a PNG image for visualizations).

---

## Features

- **102 data science tools** — complete pipeline from raw CSV to trained model with interpretability
- **Stateful DataStore** — named DataFrames, trained models, and plot images persist across tool calls
- **13-phase agent workflow** — built-in instructions guide the LLM through EDA → cleaning → feature engineering → modeling → reporting
- **Decision frameworks** — when to use parametric vs. non-parametric tests, how to handle missing data, which encoding to pick
- **Visualization as images** — matplotlib figures converted to PNG and returned as MCP Image objects
- **ML model management** — train, evaluate, compare, and tune scikit-learn models (linear/logistic regression, random forest, gradient boosting, decision tree)
- **Statistical testing** — t-test, ANOVA, chi-square, normality, Mann-Whitney, Kruskal-Wallis with effect sizes
- **Clustering** — K-Means, DBSCAN, elbow method, silhouette analysis, cluster profiling
- **Dimensionality reduction** — PCA (with variance explained) and t-SNE (with auto-sampling for large datasets)
- **Report generation** — Markdown (`.md` + PNG plots) or self-contained HTML with embedded base64 images
- **Pitfall prevention** — 12 documented common mistakes the agent is warned about (e.g., target encoding before split = data leakage)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              MCP Client (Claude, etc.)              │
│                                                     │
│   "Load sales.csv, show a quality report,           │
│    clean missing values, train a random forest"     │
└──────────────────────┬──────────────────────────────┘
                       │  stdio (JSON-RPC)
          ┌────────────┴────────────┐
          │   FastMCP Server        │
          │   (server.py)           │
          │                         │
          │   ┌───────────────┐     │
          │   │  DataStore    │     │
          │   │  · DataFrames │     │
          │   │  · ML Models  │     │
          │   │  · Plot cache │     │
          │   └───────────────┘     │
          │                         │
          │   15 tool modules       │
          │   (102 tools total)     │
          │                         │
          │   · pandas              │
          │   · scikit-learn        │
          │   · matplotlib/seaborn  │
          │   · category-encoders   │
          │   · scipy               │
          └─────────────────────────┘
```

**No external services required** — all computation happens in-process. The server communicates via stdio (standard MCP transport).

---

## Tech Stack

| Layer         | Technology                          | Details                                                    |
| ------------- | ----------------------------------- | ---------------------------------------------------------- |
| Protocol      | MCP (FastMCP)                       | stdio transport, JSON-RPC                                  |
| Data          | pandas 2.0+, numpy 1.24+            | DataFrame manipulation, numerical computing                |
| ML            | scikit-learn 1.3+                   | Regression, classification, clustering, preprocessing      |
| Visualization | matplotlib 3.7+, seaborn 0.13+      | 14 plot types, Agg backend → PNG export                    |
| Encoding      | category-encoders 2.6+              | Target encoding, frequency encoding                        |
| Statistics    | scipy                               | t-test, ANOVA, chi-square, normality, non-parametric tests |
| Images        | Pillow 10+                          | PNG encoding for MCP Image responses                       |
| Build         | hatchling                           | PEP 517 build system                                       |
| Testing       | pytest 7+                           | 200+ test functions across 13 test files                   |
| Formats       | openpyxl (Excel), pyarrow (Parquet) | Optional dependencies for additional file formats          |

---

## Project Structure

```
mcp-data-science/
├── pyproject.toml                       # Build config + dependencies
├── CLAUDE.md                            # Architecture reference
│
├── src/mcp_data_science/
│   ├── __init__.py                      # main() entry point → mcp.run(transport="stdio")
│   ├── __main__.py                      # python -m support
│   ├── server.py                        # FastMCP instance + registers all 15 tool modules
│   ├── state.py                         # DataStore: named DataFrames, ML models, plot cache
│   ├── instructions.md                  # 13-phase agent workflow guide (loaded as MCP instructions)
│   └── tools/
│       ├── _plot_helpers.py             # fig_to_image() — matplotlib Agg → PNG → MCP Image
│       ├── loading.py                   # 11 tools: load/save CSV/Excel/Parquet, merge, pivot, melt, concat
│       ├── inspection.py                # 9 tools: head, tail, info, stats, shape, quality_report, profile
│       ├── cleaning.py                  # 9 tools: duplicates, missing, filter, rename, clip, sort, bin
│       ├── transformation.py            # 9 tools: create_column, log, normalize, mapping, dtype, string_clean
│       ├── encoding.py                  # 4 tools: one-hot, target, label, frequency encoding
│       ├── visualization.py             # 14 tools: histogram, bar, scatter, box, violin, QQ, heatmap…
│       ├── analysis.py                  # 8 tools: correlation, outliers, group_aggregate, crosstab
│       ├── modeling.py                  # 8 tools: train/test split, train, predict, evaluate, grid_search
│       ├── feature_selection.py         # 4 tools: correlation_filter, variance_filter, importance
│       ├── datetime_tools.py            # 4 tools: extract_parts, diff, filter, set_index
│       ├── statistical_tests.py         # 6 tools: ttest, anova, chi_square, normality, mann_whitney, kruskal
│       ├── interpretation.py            # 7 tools: feature_importance, residuals, confusion_matrix, ROC, PR
│       ├── clustering.py                # 5 tools: kmeans, dbscan, elbow, silhouette, cluster_profile
│       ├── dimensionality.py            # 2 tools: pca_transform, tsne_plot
│       └── reporting.py                 # 2 tools: save_report (md), save_report_html
│
└── tests/
    ├── conftest.py                      # Fixtures: sample_df, numeric_df, datetime_df, classification_df
    ├── sample_data.csv                  # Test fixture data
    ├── test_server.py                   # Server bootstrap, tool count verification
    ├── test_state.py                    # DataStore management (25 tests)
    ├── test_loading.py                  # CSV/Excel/Parquet loading (18 tests)
    ├── test_inspection.py               # Inspection tools (15 tests)
    ├── test_cleaning.py                 # Cleaning tools (23 tests)
    ├── test_transformation.py           # Transformation tools (22 tests)
    ├── test_encoding.py                 # Encoding tools (9 tests)
    ├── test_visualization.py            # Visualization tools (26 tests)
    ├── test_analysis.py                 # Analysis tools (18 tests)
    ├── test_modeling.py                 # Modeling tools (14 tests)
    ├── test_feature_selection.py        # Feature selection tools (9 tests)
    ├── test_datetime_tools.py           # Datetime tools (11 tests)
    └── test_reporting.py                # Reporting tools (8 tests)
```

---

## Prerequisites

| Requirement            | Version | Install                                                          |
| ---------------------- | ------- | ---------------------------------------------------------------- |
| **Python**             | 3.10+   | [python.org](https://www.python.org/downloads/)                  |
| **uv** _(recommended)_ | latest  | `pip install uv` or [docs.astral.sh](https://docs.astral.sh/uv/) |

> **Note**: No GPU needed. All computation runs on CPU. Optional dependencies (`openpyxl` for Excel, `pyarrow` for Parquet) are installed automatically.

---

## Installation

### Quick setup (recommended)

```bash
git clone https://github.com/AstyanM/mcp-data-science.git
cd mcp-data-science
pip install -e .
```

### With test dependencies

```bash
pip install -e ".[test]"
```

### Using uv

```bash
uv venv --python 3.13
uv pip install -e ".[test]"
```

---

## Usage

### As an MCP server

Add to your MCP client configuration (e.g., `.mcp.json`, Claude Desktop config):

```json
{
  "mcpServers": {
    "mcp-data-science": {
      "command": "python",
      "args": ["-m", "mcp_data_science"]
    }
  }
}
```

The server starts on stdio and exposes all 102 tools to your MCP client.

### With MCP Inspector (interactive testing)

```bash
mcp dev src/mcp_data_science/server.py
```

Opens a web UI at `http://localhost:6274`:

1. Set Command = `python`, Arguments = `-m mcp_data_science`, then click **Connect**
2. Go to the **Tools** tab — all 102 tools are listed
3. Call `load_csv` first with a CSV path, then call any other tool

### Direct execution

```bash
python -m mcp_data_science
```

### Run tests

```bash
pytest tests/
```

---

## Agent Workflow

The server includes a built-in 13-phase workflow guide that instructs the LLM how to approach data science tasks systematically:

```
Phase 1:  Loading & First Look      →  load_csv, get_shape, get_head, get_info
Phase 2:  Exploratory Data Analysis  →  quality_report, get_statistics, plot_histogram, plot_bar
Phase 3:  Statistical Testing        →  normality_test, ttest, anova, chi_square
Phase 4:  Data Cleaning              →  drop_duplicates, fill_missing, string_clean, clip_outliers
Phase 5:  Feature Engineering        →  create_column, extract_datetime_parts, log_transform
Phase 6:  Categorical Encoding       →  one_hot_encode, target_encode, label_encode
Phase 7:  Feature Selection          →  variance_filter, correlation_filter, feature_importance
Phase 8:  Dimensionality Reduction   →  pca_transform, tsne_plot
Phase 9:  Normalization              →  normalize (only for distance-based models)
Phase 10: Modeling                   →  train_test_split, train_model, evaluate_model
Phase 11: Model Interpretation       →  plot_residuals, plot_confusion_matrix, plot_roc_curve
Phase 12: Clustering                 →  elbow_plot, kmeans_cluster, silhouette_score
Phase 13: Reporting                  →  save_report, save_report_html
```

### Decision frameworks

The workflow includes decision trees for common choices:

| Decision                  | Framework                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Missing data**          | <5% → fill with median/mode · 5–30% → investigate · >50% → consider dropping                            |
| **Group comparison test** | 2 groups + normal → t-test · 2 groups + non-normal → Mann-Whitney · 3+ groups → ANOVA or Kruskal-Wallis |
| **Categorical encoding**  | 2–10 unique → one-hot · >10 unique → target encode · ordinal → label encode                             |
| **Normalization**         | Linear/logistic models → yes · Tree-based models → no                                                   |

---

## Key Patterns

- **Tool registration**: each module exports `register_tools(mcp, store)`, called in `server.py`
- **State**: `DataStore` holds `dict[str, pd.DataFrame]` + `dict[str, dict]` for ML models. All tools accept `df_name: str = ""` (empty = current)
- **Mutations**: tools modify the DataFrame in the store via `store.set(name, df)` and return a concise text summary
- **Visualization**: tools create matplotlib figures, convert via `fig_to_image()` → `Image(data=png_bytes, format="png")`, always `plt.close(fig)` after
- **Modeling**: models stored as `{"model": sklearn_obj, "type": str, "features": list, "target": str}` in `DataStore._models`
- **Errors**: tools return readable error strings, never raw exceptions

---

## Supported Models

| Task                         | Models                                                               |
| ---------------------------- | -------------------------------------------------------------------- |
| **Regression**               | Linear Regression, Random Forest, Gradient Boosting, Decision Tree   |
| **Classification**           | Logistic Regression, Random Forest, Gradient Boosting, Decision Tree |
| **Clustering**               | K-Means, DBSCAN                                                      |
| **Dimensionality Reduction** | PCA, t-SNE                                                           |

---

## Contributing

Contributions are welcome! Please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Rules for new tools

- Follow the `register_tools(mcp, store)` pattern
- Never print to stdout — it's reserved for MCP stdio protocol (use `logging` → stderr)
- Visualization tools must always close figures (`plt.close(fig)`) to prevent memory leaks
- Tool docstrings are LLM-facing — keep them clear and include parameter examples
- `create_column` uses `df.eval()` (safe pandas expressions), never Python `eval()`

---

## Author

- **Martin Astyan** — [GitHub](https://github.com/AstyanM)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
