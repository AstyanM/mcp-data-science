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
            table_row_count = 0
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
                        html_lines.append("<table>")
                        in_table = True
                        table_row_count = 0
                    cells = [c.strip() for c in stripped.strip("|").split("|")]
                    if all(set(c) <= {"-", " ", ":"} for c in cells):
                        continue
                    tag = "th" if table_row_count == 0 else "td"
                    table_row_count += 1
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
                        table_row_count = 0
                    html_lines.append("<br>")
                else:
                    # Markdown image ![alt](src) → <img>
                    img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', stripped)
                    if img_match:
                        alt, src = img_match.group(1), img_match.group(2)
                        html_lines.append(f'<img src="{src}" alt="{alt}" style="max-width:100%; height:auto;">')
                    else:
                        # Bold
                        processed = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
                        html_lines.append(f"<p>{processed}</p>")

            if in_table:
                html_lines.append("</table>")

            html_body = "\n".join(html_lines)

            full_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Science Report</title>
<style>
  /* ── Warm stone + amber design system ── */
  :root {{
    --bg:          rgb(252, 250, 247);
    --bg-secondary:rgb(245, 241, 235);
    --fg:          rgb(28, 25, 23);
    --fg-secondary:rgb(87, 83, 78);
    --border:      rgb(214, 211, 209);
    --accent:      rgb(217, 119, 6);
    --accent-hover:rgb(180, 83, 9);
    --accent-10:   rgba(217, 119, 6, 0.10);
    --accent-20:   rgba(217, 119, 6, 0.20);
    --radius-sm:   6px;
    --radius-md:   10px;
    --radius-lg:   14px;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.7;
    max-width: 900px;
    margin: 0 auto;
    padding: 48px 28px 96px;
    -webkit-font-smoothing: antialiased;
  }}

  /* Subtle grain overlay */
  body::after {{
    content: '';
    position: fixed;
    inset: 0;
    z-index: 9999;
    pointer-events: none;
    opacity: 0.025;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.7' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 256px 256px;
  }}

  /* ── Typography ── */
  h1 {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--fg);
    margin: 0 0 8px;
    letter-spacing: -0.02em;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 12px;
  }}

  h2 {{
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--fg);
    margin: 48px 0 12px;
    padding-left: 12px;
    border-left: 3px solid var(--accent);
  }}

  h3 {{
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--fg-secondary);
    margin: 28px 0 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.8rem;
  }}

  p {{
    color: var(--fg-secondary);
    margin: 10px 0;
  }}

  strong {{ color: var(--fg); font-weight: 600; }}

  /* ── Lists ── */
  li {{
    color: var(--fg-secondary);
    margin: 6px 0;
    padding-left: 20px;
    list-style: none;
    position: relative;
  }}

  li::before {{
    content: '▸';
    color: var(--accent);
    font-size: 0.75em;
    position: absolute;
    left: 0;
    top: 0.15em;
  }}

  /* ── Tables ── */
  table {{
    width: 100%;
    margin: 18px 0;
    border-collapse: collapse;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border);
    font-size: 0.9rem;
  }}

  th {{
    background: var(--bg-secondary);
    color: var(--fg);
    font-weight: 600;
    text-align: left;
    padding: 10px 14px;
    border-bottom: 2px solid var(--border);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--border);
    color: var(--fg-secondary);
  }}

  tr:last-child td {{ border-bottom: none; }}

  tr:hover td {{
    background: var(--accent-10);
    color: var(--fg);
  }}

  /* ── Images ── */
  img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 24px auto;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    box-shadow: 0 4px 24px rgba(28, 25, 23, 0.08);
  }}

  /* ── Dividers ── */
  hr {{
    border: none;
    border-top: 1px solid var(--border);
    margin: 40px 0;
  }}

  /* ── Accent chip on h2 ── */
  h2 span.badge {{
    display: inline-block;
    background: var(--accent-10);
    color: var(--accent);
    border-radius: 99px;
    font-size: 0.7rem;
    padding: 2px 8px;
    font-weight: 600;
    vertical-align: middle;
    margin-left: 8px;
  }}

  /* ── br spacing ── */
  br {{ display: block; margin: 4px 0; content: ''; }}
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
