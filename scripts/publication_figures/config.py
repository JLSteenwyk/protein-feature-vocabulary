"""Shared configuration for publication figures.

All color schemes from pypubfigs colorblind-friendly palettes.
"""

from pathlib import Path
from pypubfigs.palettes import friendly_pal

# === Paths ===
ROOT = Path(__file__).parent.parent.parent
RESULTS_SCALED = ROOT / "results" / "scaled_1.5M"
RESULTS_UNIFIED_ESM3 = ROOT / "results" / "unified" / "esm3"
RESULTS_UNIFIED_ESM2 = ROOT / "results" / "unified" / "esm2"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
OUTPUT_DIR = ROOT / "results" / "figures" / "publication"
SUPP_OUTPUT_DIR = OUTPUT_DIR / "supplementary"

# === Colorblind-friendly palettes from pypubfigs ===
# contrast_three: best for 2-3 category comparisons (model A vs B)
CONTRAST = friendly_pal("contrast_three")  # ['#004488', '#BB5566', '#DDAA33']
C_ESM3 = CONTRAST[0]   # '#004488' dark blue
C_ESM2 = CONTRAST[1]   # '#BB5566' rose
C_ACCENT = CONTRAST[2]  # '#DDAA33' gold

# bright_seven: for multi-category comparisons
BRIGHT = friendly_pal("bright_seven")  # ['#4477AA', '#228833', '#AA3377', '#BBBBBB', '#66CCEE', '#CCBB44', '#EE6677']

# Modality conditions (from bright_seven — visually distinct)
C_S = BRIGHT[0]         # '#4477AA' blue — sequence-only
C_SST = BRIGHT[6]       # '#EE6677' red-pink — sequence + structure
C_SF = BRIGHT[1]        # '#228833' green — sequence + function
C_SALL = BRIGHT[2]      # '#AA3377' purple — all modalities

# Cross-modal feature categories (from bright_seven)
C_ENHANCED = BRIGHT[6]   # '#EE6677' red-pink
C_SUPPRESSED = BRIGHT[4] # '#66CCEE' light blue
C_INVARIANT = BRIGHT[3]  # '#BBBBBB' grey

# Baseline comparisons (from vibrant_seven for maximum distinctness)
VIBRANT = friendly_pal("vibrant_seven")  # ['#0077BB', '#EE7733', '#33BBEE', '#CC3311', '#009988', '#EE3377', '#BBBBBB']
C_SAE = VIBRANT[0]      # '#0077BB' blue
C_RAW = VIBRANT[4]      # '#009988' teal
C_RANDOM = VIBRANT[6]   # '#BBBBBB' grey
C_SHUFFLED = VIBRANT[1]  # '#EE7733' orange

# Heatmap colormaps (from pypubfigs continuous palettes)
VIRIDIS = friendly_pal("viridis_eight")

# === Figure dimensions (inches) ===
SINGLE_COL = 3.5    # 89mm
ONE_HALF_COL = 4.7  # 120mm
DOUBLE_COL = 7.2    # 183mm

# === Font sizes (Nature style) ===
PANEL_LABEL_SIZE = 10
AXIS_LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7
TITLE_SIZE = 9

# === Output ===
DPI_PNG = 600
