#!/usr/bin/env python3
"""Generate all publication figures.

Usage:
    ./env/bin/python scripts/publication_figures/generate_all.py
    ./env/bin/python scripts/publication_figures/generate_all.py --main-only
    ./env/bin/python scripts/publication_figures/generate_all.py --supp-only
    ./env/bin/python scripts/publication_figures/generate_all.py --fig 1
    ./env/bin/python scripts/publication_figures/generate_all.py --sfig 5
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent to path so imports work
sys.path.insert(0, str(Path(__file__).parent))

from style import setup_style

# Main figure modules
MAIN_FIGURES = {
    1: ("fig1_interpretability", "SAE features are interpretable and biologically grounded"),
    2: ("fig2_convergence", "ESM-2 and ESM-3 converge on shared biological representations"),
    3: ("fig3_multimodal", "How ESM-3 integrates structural information"),
    4: ("fig4_causal", "Causal validation confirms feature function"),
}

# Extended Data figure modules (10 figures for NMI format)
SUPP_FIGURES = {
    1: ("supplementary.ed01_training_reconstruction", "SAE Training & Reconstruction Quality"),
    2: ("supplementary.ed02_negative_controls", "Negative Controls Detail"),
    3: ("supplementary.ed03_autointerpretability_go", "Autointerpretability & GO Enrichment"),
    4: ("supplementary.ed04_feature_biology", "Feature Biology: Residue Examples & Probing"),
    5: ("supplementary.ed05_cross_modal", "Cross-Modal Feature Detail"),
    6: ("supplementary.ed06_lens_decomposition", "Logit Lens, Tuned Lens & Residual Decomposition"),
    7: ("supplementary.ed07_patching_ablation_attention", "Attribution Patching, Ablation & Attention Atlas"),
    8: ("supplementary.ed08_contact_circuits", "Contact Prediction, OV/QK & Sparse Circuits"),
    9: ("supplementary.ed09_decoder_dms_phylo", "Decoder Geometry, Co-activation, DMS & Phylogenetic"),
    10: ("supplementary.ed10_case_studies", "All 5 Case Studies"),
}


def run_figure(module_name, title, fig_num, is_supp=False):
    """Import and run a figure module's generate() function."""
    prefix = f"ED Fig {fig_num}" if is_supp else f"Fig {fig_num}"
    print(f"\n{'='*60}")
    print(f"  {prefix}: {title}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        mod = __import__(module_name, fromlist=["generate"])
        mod.generate()
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
    except ImportError as e:
        print(f"  SKIPPED (not implemented): {e}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--main-only", action="store_true", help="Only main figures")
    parser.add_argument("--supp-only", action="store_true", help="Only supplementary figures")
    parser.add_argument("--fig", type=int, help="Generate specific main figure (1-4)")
    parser.add_argument("--sfig", type=int, help="Generate specific supplementary figure (1-16)")
    args = parser.parse_args()

    setup_style()

    t_total = time.time()

    if args.fig:
        if args.fig in MAIN_FIGURES:
            mod_name, title = MAIN_FIGURES[args.fig]
            run_figure(mod_name, title, args.fig)
        else:
            print(f"Unknown figure {args.fig}. Available: {list(MAIN_FIGURES.keys())}")
        return

    if args.sfig:
        if args.sfig in SUPP_FIGURES:
            mod_name, title = SUPP_FIGURES[args.sfig]
            run_figure(mod_name, title, args.sfig, is_supp=True)
        else:
            print(f"Unknown sfig {args.sfig}. Available: {list(SUPP_FIGURES.keys())}")
        return

    if not args.supp_only:
        print("\n" + "#" * 60)
        print("  MAIN FIGURES")
        print("#" * 60)
        for num, (mod_name, title) in MAIN_FIGURES.items():
            run_figure(mod_name, title, num)

    if not args.main_only:
        print("\n" + "#" * 60)
        print("  SUPPLEMENTARY FIGURES")
        print("#" * 60)
        for num, (mod_name, title) in SUPP_FIGURES.items():
            run_figure(mod_name, title, num, is_supp=True)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  All figures generated in {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
