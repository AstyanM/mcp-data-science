import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from mcp.server.fastmcp import Image  # noqa: E402

sns.set_theme(style="whitegrid")


def fig_to_image(fig: plt.Figure) -> Image:
    """Convert a matplotlib Figure to an MCP Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(data=buf.getvalue(), format="png")


def fig_to_image_and_store(fig: plt.Figure, store, plot_name: str, save_path: str = "") -> Image:
    """Convert a matplotlib Figure to an MCP Image, store bytes in DataStore, optionally save to disk."""
    from pathlib import Path

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.getvalue()

    # Store for report generation
    store.save_plot(plot_name, png_bytes)

    # Optionally save to disk
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_bytes(png_bytes)

    return Image(data=png_bytes, format="png")
