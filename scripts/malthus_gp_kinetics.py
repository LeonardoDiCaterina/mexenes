"""
malthus_gp_kinetics.py

Autonomous Symbolic Kinetics Discovery for MXene Thermal Transformation.
Pairs Functional Data Analysis (FDA) with Malthus-GP (Differentiable Cartesian Genetic
Programming & JAX AutoDiff) to extract closed-form solid-state kinetic rate laws.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import config
    from scripts.fda_analysis import (
        load_dataset,
        fit_functional_splines,
        compute_functional_pca,
        discover_kinetics_with_malthus_gp
    )
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config
    from scripts.fda_analysis import (
        load_dataset,
        fit_functional_splines,
        compute_functional_pca,
        discover_kinetics_with_malthus_gp
    )


def run_kinetics_discovery():
    print("=" * 70)
    print("  MALTHUS-GP & FDA: SOLID-STATE REACTION KINETICS DISCOVERY")
    print("=" * 70)

    # 1. FDA Extraction
    print("\n[Step 1/3] Extracting Continuous Reaction Trajectories via FDA...")
    t_eval, temps, raw_y, _ = load_dataset(crop_min=7.5, crop_max=75.0)
    splines, y_func, d1_func, d2_func = fit_functional_splines(t_eval, raw_y)
    fpca_res = compute_functional_pca(t_eval, y_func, n_components=2)
    print(f"           Primary phase conversion coordinate captured ({fpca_res['var_explained'][0]:.1f}% variance)")

    # 2. Malthus-GP Discovery
    print("\n[Step 2/3] Running Malthus-GP Differentiable Symbolic Regression...")
    t0 = time.time()
    gp_res = discover_kinetics_with_malthus_gp(temps, fpca_res, config.OUTPUT_DIR)
    elapsed = time.time() - t0

    if gp_res is None:
        print("[-] Discovery aborted: Malthus-GP not available.")
        return

    print(f"\n[Step 3/3] Discovery Finished in {elapsed:.2f} s")
    print(f"           R² Score:            {gp_res['r2']:.4f}")
    print(f"           Peak Rate Temp:      {gp_res['t_peak_rate']:.1f} °C")
    print(f"           Diagnostic Plot:     {gp_res['plot_path']}")
    print(f"           Formula Summary:     {gp_res['summary_path']}")
    print("=" * 70)


if __name__ == "__main__":
    run_kinetics_discovery()
