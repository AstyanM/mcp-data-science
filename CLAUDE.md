# MCP Data Science Server

## Project Purpose
MCP server (Model Context Protocol) that gives an LLM 48 tools to perform complete CSV data cleaning and preprocessing pipelines: load, inspect, clean, transform, encode, visualize, and analyze datasets.

## Architecture

```
src/mcp_data_science/
├── __init__.py          # main() entry point → mcp.run(transport="stdio")
├── __main__.py          # python -m support
├── server.py            # FastMCP instance + registers all tool modules
├── state.py             # DataStore: holds named DataFrames in memory
└── tools/
    ├── _plot_helpers.py # fig_to_image() — matplotlib Agg → PNG → MCP Image
    ├── loading.py       # 6 tools: load_csv, save_csv, list/set/copy/merge dataframes
    ├── inspection.py    # 9 tools: head, tail, info, stats, shape, quality_report, unique, profile, sample
    ├── cleaning.py      # 8 tools: drop_duplicates, drop_columns, drop/fill_missing, filter, rename, clip, sort
    ├── transformation.py# 8 tools: create_column (eval), log_transform, normalize, mapping, dtype, replace, select_dtypes, string_clean
    ├── encoding.py      # 4 tools: one_hot, target, label, frequency encoding
    ├── visualization.py # 7 tools: histogram, bar, scatter, box, correlation_matrix, pairplot, missing_values
    └── analysis.py      # 6 tools: correlation, value_counts, outliers, group_aggregate, crosstab, add_row_index
```

## Key Patterns

- **Tool registration**: each module exports `register_tools(mcp, store)`, called in `server.py`
- **State**: `DataStore` holds a `dict[str, pd.DataFrame]` with a `current_name` property. All tools accept `df_name: str = ""` (empty = current)
- **Mutations**: tools modify the DataFrame in the store via `store.set(name, df)` and return a concise text summary
- **Visualization**: tools create matplotlib figures, convert via `fig_to_image()` → `Image(data=png_bytes, format="png")`, always `plt.close(fig)` after
- **Errors**: tools return readable error strings, never raw exceptions

## Development

```bash
pip install -e .              # Install in editable mode
python -m mcp_data_science    # Run server (stdio transport)
```

## Rules

- **Never print to stdout** — it's reserved for MCP stdio protocol. Use `logging` (goes to stderr)
- All new tools must follow the `register_tools(mcp, store)` pattern
- Visualization tools must always close figures (`plt.close(fig)`) to prevent memory leaks
- Tool docstrings are LLM-facing descriptions — keep them clear and include parameter examples
- `create_column` uses `df.eval()` (safe pandas expressions), never Python `eval()`

## Dependencies
mcp, pandas, numpy, matplotlib, seaborn, category-encoders, scikit-learn, Pillow

## Reference
- PRD: `.taskmaster/docs/prd.txt` — full spec with 48 tool signatures and 5 implementation phases
- Example notebook: `examples/explore_data.ipynb` — manual EDA workflow this server replicates
