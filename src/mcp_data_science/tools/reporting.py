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

    @mcp.tool()
    def save_report_html(
        content: str,
        output_dir: str = "",
        include_plots: list[str] | None = None,
    ) -> str:
        """Save a data science report as an HTML file with embedded plot images (base64).
        Unlike save_report (markdown + separate PNGs), this produces a single self-contained HTML file.
        Plots are embedded inline using base64 encoding — no separate image files needed.
        Reference plots in content as ![](plot_name.png) — they'll be auto-replaced with embedded images.
        Example: save_report_html(content="# Analysis\\n![](histogram_Revenue.png)", include_plots=["histogram_Revenue"])"""
        try:
            import base64
            import re

            if output_dir:
                out = Path(output_dir)
            elif store._csv_dir:
                out = Path(store._csv_dir) / "reports"
            else:
                return "Error: No output_dir specified and no CSV loaded."

            out.mkdir(parents=True, exist_ok=True)

            # Replace markdown image references with base64
            html_content = content
            embedded_plots = []
            if include_plots:
                for plot_name in include_plots:
                    try:
                        png_bytes = store.get_plot(plot_name)
                        b64 = base64.b64encode(png_bytes).decode("utf-8")
                        img_tag = f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;" alt="{plot_name}">'
                        # Replace markdown image syntax
                        html_content = html_content.replace(f"![]({plot_name}.png)", img_tag)
                        embedded_plots.append(plot_name)
                    except KeyError:
                        pass

            # Convert remaining markdown to basic HTML
            lines = html_content.split("\n")
            html_lines = []
            in_table = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# "):
                    html_lines.append(f"<h1>{stripped[2:]}</h1>")
                elif stripped.startswith("## "):
                    html_lines.append(f"<h2>{stripped[3:]}</h2>")
                elif stripped.startswith("### "):
                    html_lines.append(f"<h3>{stripped[4:]}</h3>")
                elif stripped.startswith("---"):
                    html_lines.append("<hr>")
                elif stripped.startswith("|"):
                    if not in_table:
                        html_lines.append("<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;'>")
                        in_table = True
                    cells = [c.strip() for c in stripped.strip("|").split("|")]
                    if all(set(c) <= {"-", " ", ":"} for c in cells):
                        continue
                    tag = "th" if not any("td" in l for l in html_lines[-3:]) else "td"
                    row = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
                    html_lines.append(f"<tr>{row}</tr>")
                elif stripped.startswith("- **"):
                    html_lines.append(f"<li>{stripped[2:]}</li>")
                elif stripped.startswith("- "):
                    html_lines.append(f"<li>{stripped[2:]}</li>")
                elif stripped.startswith("<img"):
                    html_lines.append(stripped)
                elif stripped == "":
                    if in_table:
                        html_lines.append("</table>")
                        in_table = False
                    html_lines.append("<br>")
                else:
                    # Bold
                    processed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
                    html_lines.append(f"<p>{processed}</p>")

            if in_table:
                html_lines.append("</table>")

            html_body = "\n".join(html_lines)

            full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Data Science Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #2980b9; margin-top: 30px; }}
  h3 {{ color: #7f8c8d; }}
  table {{ width: 100%; margin: 15px 0; }}
  th {{ background: #3498db; color: white; text-align: left; }}
  td {{ border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  img {{ margin: 15px 0; border: 1px solid #ddd; border-radius: 4px; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 30px 0; }}
  li {{ margin: 5px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

            report_path = out / "report.html"
            report_path.write_text(full_html, encoding="utf-8")

            result = f"HTML report saved to {report_path}"
            if embedded_plots:
                result += f" with {len(embedded_plots)} embedded plots."
            return result
        except Exception as e:
            return f"Error saving HTML report: {type(e).__name__} - {e}"
