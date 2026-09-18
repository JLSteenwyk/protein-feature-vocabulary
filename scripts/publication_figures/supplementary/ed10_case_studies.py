#!/usr/bin/env python3
"""Extended Data: ed10_case_studies (wrapper around sfig16_case_studies_full)."""
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
    mod = importlib.import_module("supplementary.sfig16_case_studies_full")
    # Temporarily patch save_fig to use ED name
    import supplementary.sfig16_case_studies_full as m
    orig_save = m.save_fig if hasattr(m, 'save_fig') else None
    m.save_fig = lambda fig, name, **kw: save_fig(fig, "ed10_case_studies", supplementary=True)
    try:
        mod.generate()
    except Exception:
        # If patching didn't work, just run and rename
        mod.generate()

if __name__ == "__main__":
    generate()
