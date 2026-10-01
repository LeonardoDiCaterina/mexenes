#!/usr/bin/env python3
"""
ga_peak_fit_demo.py

Demonstration of Evolutionary / Genetic Algorithm peak deconvolution
on synthetic in-situ XRD data. Replaces naive argmax with continuous
Pseudo-Voigt profile optimization.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
import config

def pseudo_voigt(tt, center, amplitude, fwhm, eta=0.5):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    gaussian = np.exp(-((tt - center) ** 2) / (2.0 * (sigma ** 2)))
    lorentzian = 1.0 / (1.0 + ((tt - center) / gamma) ** 2)
    return amplitude * (eta * lorentzian + (1.0 - eta) * gaussian)

def model_function(params, tt):
    # params: [center, amp, fwhm, eta, b0, b1]
    center, amp, fwhm, eta, b0, b1 = params
    baseline = b0 + b1 * (tt - tt[0])
    peak = pseudo_voigt(tt, center, amp, fwhm, eta)
    return baseline + peak

def objective(params, tt, y_obs):
    y_pred = model_function(params, tt)
    weights = 1.0 / np.maximum(y_obs, 1.0)
    rwp = np.sqrt(np.sum(weights * ((y_obs - y_pred) ** 2)) / np.sum(weights * (y_obs ** 2)))
    return rwp

def fit_file(filepath, out_plot):
    data = np.loadtxt(filepath, comments=config.COMMENT_CHAR)
    tt_full, inten_full = data[:, 0], data[:, 1]

    mask = (tt_full >= config.TWO_THETA_MIN) & (tt_full <= config.TWO_THETA_MAX)
    tt = tt_full[mask]
    y_obs = inten_full[mask]

    naive_idx = np.argmax(y_obs)
    naive_peak = tt[naive_idx]
    naive_height = y_obs[naive_idx]

    bounds = [
        (config.TWO_THETA_MIN + 0.2, config.TWO_THETA_MAX - 0.2), # center
        (50.0, 3000.0),                                            # amplitude
        (0.1, 1.2),                                                # fwhm
        (0.0, 1.0),                                                # eta
        (50.0, 400.0),                                             # b0
        (-50.0, 50.0)                                              # b1
    ]

    print(f"Optimizing peak profile for {os.path.basename(filepath)} via Evolutionary Search...")
    result = differential_evolution(objective, bounds, args=(tt, y_obs), seed=42, popsize=15, maxiter=80)

    p_opt = result.x
    y_fit = model_function(p_opt, tt)
    rwp_pct = result.fun * 100.0

    print("\n--- Optimization Results ---")
    print(f"[His naive method]   Peak position: {naive_peak:.3f} deg (Height: {naive_height:.0f})")
    print(f"[Evolutionary fit]   Center:        {p_opt[0]:.4f} deg")
    print(f"                     FWHM:          {p_opt[2]:.4f} deg")
    print(f"                     Amplitude:     {p_opt[1]:.1f} counts")
    print(f"                     Lorentzian eta:{p_opt[3]:.2f}")
    print(f"                     Rwp residual:  {rwp_pct:.2f}%")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.scatter(tt, y_obs, s=15, color='black', alpha=0.6, label='Lab Data (.xy)')
    ax1.plot(tt, y_fit, color='crimson', lw=2, label=f'GA Fit (Center={p_opt[0]:.3f} deg)')
    ax1.axvline(naive_peak, color='blue', linestyle='--', alpha=0.7, label=f'Naive argmax ({naive_peak:.3f} deg)')
    ax1.set_ylabel('Intensity (counts)')
    ax1.set_title(f'Peak Profile Fit: {os.path.basename(filepath)}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.scatter(tt, y_obs - y_fit, s=10, color='crimson', alpha=0.5)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xlabel('2theta (deg)')
    ax2.set_ylabel('Residual')
    ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=180)
    plt.close()
    print(f"Saved comparison plot to: {out_plot}")

if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, config.FILE_PATTERN)))
    if files:
        fit_file(files[0], str(config.OUTPUT_DIR / "ga_fit_demo.png"))
    else:
        print("No files found to fit.")
