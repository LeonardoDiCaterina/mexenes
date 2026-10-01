#!/usr/bin/env python3
"""
generate_synthetic_xrd.py

Generates synthetic in-situ temperature-resolved X-ray Powder Diffraction (XRD) .xy files
simulating MXene Ti2AlC0.5N0.5 oxidation to TiO2 Rutile over a temperature ramp.
Compatible with findpeaks_V4.ipynb.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
import config

def pseudo_voigt(tt, center, intensity, fwhm, eta=0.5):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0
    gaussian = np.exp(-((tt - center) ** 2) / (2.0 * (sigma ** 2)))
    lorentzian = 1.0 / (1.0 + ((tt - center) / gamma) ** 2)
    return intensity * (eta * lorentzian + (1.0 - eta) * gaussian)

def thermal_shift(tt0, temp, ref_t=25.0, alpha=1.0e-5):
    rad0 = np.radians(tt0 / 2.0)
    denom = 1.0 + alpha * (temp - ref_t)
    sin_new = np.sin(rad0) / denom
    sin_new = np.clip(sin_new, -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(sin_new))

def generate_scan(temp, tt_grid, seed=None):
    rng = np.random.default_rng(seed)

    # 1. Background
    background = (
        120.0
        + 80.0 * np.exp(-tt_grid / 15.0)
        + 150.0 * np.exp(-((tt_grid - 22.0) ** 2) / (2.0 * (8.0 ** 2)))
    )

    y_clean = background.copy()

    # 2. MXene (002) reflection around 9.8 deg
    mxene_weight = 1.0 / (1.0 + np.exp((temp - 520.0) / 60.0))
    mxene_shift = 9.80 + 0.35 * (1.0 - np.exp(-temp / 350.0))
    mxene_peak = pseudo_voigt(tt_grid, center=mxene_shift, intensity=1200.0 * mxene_weight, fwhm=0.32, eta=0.6)
    y_clean += mxene_peak

    # Minor MXene peaks
    y_clean += pseudo_voigt(tt_grid, center=32.0, intensity=180.0 * mxene_weight, fwhm=0.4, eta=0.5)
    y_clean += pseudo_voigt(tt_grid, center=38.0, intensity=120.0 * mxene_weight, fwhm=0.45, eta=0.5)

    # 3. Ti2AlC0.5N0.5 MAX phase peaks
    max_phase_weight = 1.0 / (1.0 + np.exp((temp - 680.0) / 70.0))
    max_peaks = [
        (13.008, 200.0, 0.22),
        (34.021, 150.0, 0.25),
        (34.673, 60.0, 0.25),
        (39.545, 800.0, 0.28),
        (39.727, 160.0, 0.28),
        (43.428, 40.0, 0.30),
        (53.287, 120.0, 0.32),
        (60.897, 130.0, 0.35),
        (71.626, 30.0, 0.38),
        (71.992, 80.0, 0.38),
        (75.204, 100.0, 0.40)
    ]
    for tt0, raw_i, fwhm in max_peaks:
        shifted_tt = thermal_shift(tt0, temp, ref_t=25.0, alpha=8.5e-6)
        y_clean += pseudo_voigt(tt_grid, center=shifted_tt, intensity=raw_i * max_phase_weight, fwhm=fwhm, eta=0.5)

    # 4. TiO2 Rutile phase peaks
    rutile_weight = 1.0 / (1.0 + np.exp(-(temp - 550.0) / 50.0))
    rutile_peaks = [
        (27.446, 1100.0, 0.26),
        (36.085, 550.0, 0.28),
        (39.187, 90.0, 0.28),
        (41.225, 280.0, 0.29),
        (44.050, 110.0, 0.30),
        (54.322, 650.0, 0.32),
        (56.640, 220.0, 0.34),
        (62.740, 110.0, 0.36),
        (64.038, 110.0, 0.36),
        (69.008, 220.0, 0.38),
    ]
    for tt0, raw_i, fwhm in rutile_peaks:
        shifted_tt = thermal_shift(tt0, temp, ref_t=500.0, alpha=9.0e-6)
        y_clean += pseudo_voigt(tt_grid, center=shifted_tt, intensity=raw_i * rutile_weight, fwhm=fwhm, eta=0.6)

    # 5. Poisson noise
    y_noisy = rng.poisson(np.maximum(y_clean, 1.0)).astype(float)
    return y_noisy

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic in-situ XRD .xy files.")
    parser.add_argument("--output-dir", type=str, default=str(config.DATA_DIR))
    parser.add_argument("--num-scans", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tt_grid = np.arange(7.0, 80.01, 0.02)
    temps = np.linspace(25, 900, args.num_scans, dtype=int)

    print(f"Generating {args.num_scans} synthetic XRD scans in {args.output_dir}...")

    for i, t in enumerate(temps):
        y_noisy = generate_scan(t, tt_grid, seed=args.seed + i)
        filename = f"sample_temp{t:04d}.xy"
        filepath = os.path.join(args.output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# In-situ XRD scan at T = {t} C\n")
            f.write("# Column 1: 2theta (degrees)\n")
            f.write("# Column 2: Intensity (photon counts)\n")
            for tt, inten in zip(tt_grid, y_noisy):
                f.write(f"{tt:.3f}   {inten:.1f}\n")

        print(f"  [{i+1:02d}/{args.num_scans}] Saved {filename} (T={t:3d} C, {len(tt_grid)} points)")

    print("Done!")

if __name__ == "__main__":
    main()
