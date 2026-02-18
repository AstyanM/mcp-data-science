# MCP Data Science Server

## Project Purpose
MCP server (Model Context Protocol) that gives an LLM 65 tools for complete data science pipelines: load, inspect, clean, transform, encode, visualize, analyze datasets, train ML models, select features, and handle datetime operations.

## Architecture

```
src/mcp_data_science/
├── __init__.py              # main() entry point → mcp.run(transport="stdio")
├── __main__.py              # python -m support
├── server.py                # FastMCP instance + registers all 10 tool modules
├── state.py                 # DataStore: holds named DataFrames + trained ML models
└── tools/
    ├── _plot_helpers.py     # fig_to_image() — matplotlib Agg → PNG → MCP Image
    ├── loading.py           # 9 tools: load/save CSV, list/set/copy/merge dataframes, pivot, melt, concat
    ├── inspection.py        # 9 tools: head, tail, info, stats, shape, quality_report, unique, profile, sample
    ├── cleaning.py          # 8 tools: drop_duplicates, drop_columns, drop/fill_missing, filter, rename, clip, sort
    ├── transformation.py    # 8 tools: create_column (eval), log_transform, normalize, mapping, dtype, replace, select_dtypes, string_clean
    ├── encoding.py          # 4 tools: one_hot, target, label, frequency encoding
    ├── visualization.py     # 8 tools: histogram, bar, scatter, box, correlation_matrix, pairplot, missing_values, line
    ├── analysis.py          # 6 tools: correlation, value_counts, outliers, group_aggregate, crosstab, add_row_index
    ├── modeling.py          # 5 tools: train_test_split, train_model, predict, evaluate_model, list_models
    ├── feature_selection.py # 4 tools: correlation_filter, variance_filter, feature_importance, drop_low_importance
    └── datetime_tools.py    # 4 tools: extract_datetime_parts, datetime_diff, datetime_filter, set_datetime_index
```

## Key Patterns

- **Tool registration**: each module exports `register_tools(mcp, store)`, called in `server.py`
- **State**: `DataStore` holds `dict[str, pd.DataFrame]` + `dict[str, dict]` for ML models. All tools accept `df_name: str = ""` (empty = current)
- **Mutations**: tools modify the DataFrame in the store via `store.set(name, df)` and return a concise text summary
- **Visualization**: tools create matplotlib figures, convert via `fig_to_image()` → `Image(data=png_bytes, format="png")`, always `plt.close(fig)` after
- **Modeling**: models are stored as `{"model": sklearn_obj, "type": str, "features": list, "target": str}` in `DataStore._models`
- **Errors**: tools return readable error strings, never raw exceptions

## Development

```bash
pip install -e .              # Install in editable mode
python -m mcp_data_science    # Run server (stdio transport)
mcp dev src/mcp_data_science/server.py  # Launch MCP Inspector for interactive testing
```

## Testing with MCP Inspector

1. Run `mcp dev src/mcp_data_science/server.py` — opens web UI at http://localhost:6274
2. In the Inspector, set Command=`python`, Arguments=`-m mcp_data_science`, then click Connect
3. Go to "Tools" tab — all 65 tools are listed
4. Call `load_csv` first with a CSV path, then call any other tool

## Rules

- **Never print to stdout** — it's reserved for MCP stdio protocol. Use `logging` (goes to stderr)
- All new tools must follow the `register_tools(mcp, store)` pattern
- Visualization tools must always close figures (`plt.close(fig)`) to prevent memory leaks
- Tool docstrings are LLM-facing descriptions — keep them clear and include parameter examples
- `create_column` uses `df.eval()` (safe pandas expressions), never Python `eval()`

## Dependencies
mcp, pandas, numpy, matplotlib, seaborn, category-encoders, scikit-learn, Pillow

## Reference
- PRD: `.taskmaster/docs/prd.txt` — original spec with 48 tool signatures and 5 implementation phases
- Example notebook: `examples/explore_data.ipynb` — manual EDA workflow this server replicates
