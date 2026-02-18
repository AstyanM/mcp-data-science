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
