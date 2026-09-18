"""Shared utilities for publication figures."""

import json
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from .config import OUTPUT_DIR, SUPP_OUTPUT_DIR, PANEL_LABEL_SIZE, DPI_PNG
except ImportError:
    from config import OUTPUT_DIR, SUPP_OUTPUT_DIR, PANEL_LABEL_SIZE, DPI_PNG


def load_json(path):
    """Load a JSON result file."""
    with open(path) as f:
        return json.load(f)


def save_fig(fig, name, supplementary=False):
    """Save figure as both PDF (vector) and PNG (600 dpi)."""
    out_dir = SUPP_OUTPUT_DIR if supplementary else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"

    fig.savefig(png_path, dpi=DPI_PNG, facecolor='white', edgecolor='none')
    fig.savefig(pdf_path, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {png_path.name}, {pdf_path.name}")


def panel_label(ax, label, x=-0.12, y=1.12):
    """Add bold lowercase panel label (Nature style)."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE, fontweight='bold',
            va='top', ha='left')


def add_ci_errorbar(ax, x, y, ci_lo, ci_hi, color, **kwargs):
    """Add a point with CI error bar."""
    yerr_lo = y - ci_lo
    yerr_hi = ci_hi - y
    ax.errorbar(x, y, yerr=[[yerr_lo], [yerr_hi]], fmt='o',
                color=color, capsize=3, capthick=0.8, linewidth=0.8,
                markersize=4, **kwargs)
