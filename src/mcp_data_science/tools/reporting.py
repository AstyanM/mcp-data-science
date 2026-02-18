import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from mcp_data_science.state import DataStore

logger = logging.getLogger(__name__)


def register_tools(mcp: FastMCP, store: DataStore) -> None:

    @mcp.tool()
    def save_report(
        content: str,
        output_dir: str = "",
        include_plots: list[str] | None = None,
    ) -> str:
        """Save a data science report as a markdown file with associated plot images.
        Use this as the FINAL STEP of an analysis to export a structured, presentable report.
        The report and plots are saved to a 'reports/' folder next to the original CSV file.
        If output_dir is empty, defaults to 'reports/' in the directory of the first loaded CSV.
        If include_plots is provided, saves those named plots as PNG files in the same folder
        and you can reference them in the markdown content as ![](plot_name.png).
        Example: save_report(content="# Analysis Report\\n...", include_plots=["histogram_Revenue", "correlation_matrix"])"""
        try:
            # Determine output directory
            if output_dir:
                out = Path(output_dir)
            elif store._csv_dir:
                out = Path(store._csv_dir) / "reports"
            else:
                return "Error: No output_dir specified and no CSV has been loaded yet. Load a CSV first or provide output_dir."

            out.mkdir(parents=True, exist_ok=True)

            # Write markdown report
            report_path = out / "report.md"
            report_path.write_text(content, encoding="utf-8")

            # Save requested plots
            saved_plots = []
            warnings = []
            if include_plots:
                for plot_name in include_plots:
                    try:
                        png_bytes = store.get_plot(plot_name)
                        plot_path = out / f"{plot_name}.png"
                        plot_path.write_bytes(png_bytes)
                        saved_plots.append(plot_name)
                    except KeyError:
                        warnings.append(f"Plot '{plot_name}' not found (available: {store.list_plot_names()})")

            result = f"Report saved to {report_path}"
            if saved_plots:
                result += f" with {len(saved_plots)} plots: {', '.join(saved_plots)}."
            else:
                result += "."

            if warnings:
                result += "\nWarnings:\n" + "\n".join(f"  - {w}" for w in warnings)

            logger.info("Report saved to %s (%d plots)", report_path, len(saved_plots))
            return result
        except Exception as e:
            return f"Error saving report: {type(e).__name__} - {e}"
