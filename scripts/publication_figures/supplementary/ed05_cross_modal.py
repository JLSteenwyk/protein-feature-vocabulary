#!/usr/bin/env python3
"""Extended Data: ed05_cross_modal (wrapper around sfig08_cross_modal)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import *
from style import setup_style
from utils import save_fig

setup_style()

def generate():
    # Import and run the original figure's generate, but rename output
    import importlib
    mod = importlib.import_module("supplementary.sfig08_cross_modal")
    # Temporarily patch save_fig to use ED name
    import supplementary.sfig08_cross_modal as m
    orig_save = m.save_fig if hasattr(m, 'save_fig') else None
    m.save_fig = lambda fig, name, **kw: save_fig(fig, "ed06_cross_modal", supplementary=True)
    try:
        mod.generate()
    except Exception:
        # If patching didn't work, just run and rename
        mod.generate()

if __name__ == "__main__":
    generate()
