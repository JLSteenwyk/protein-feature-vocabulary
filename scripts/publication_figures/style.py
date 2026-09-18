"""Nature/Science figure styling using pypubfigs."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from pypubfigs.themes import theme_simple
try:
    from .config import AXIS_LABEL_SIZE, TICK_SIZE, PANEL_LABEL_SIZE, DPI_PNG
except ImportError:
    from config import AXIS_LABEL_SIZE, TICK_SIZE, PANEL_LABEL_SIZE, DPI_PNG


def setup_style():
    """Apply Nature/Science publication style."""
    theme_simple()
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.labelsize': AXIS_LABEL_SIZE,
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        'legend.fontsize': TICK_SIZE,
        'legend.title_fontsize': AXIS_LABEL_SIZE,
        'figure.dpi': 150,
        'savefig.dpi': DPI_PNG,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
        'axes.linewidth': 0.5,
        # Tick marks: visible, outward-facing, on bottom/left only
        'xtick.bottom': True,
        'ytick.left': True,
        'xtick.top': False,
        'ytick.right': False,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.minor.size': 0,
        'ytick.minor.size': 0,
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
    })
