#!/usr/bin/env python3
"""
run_analysis.py

Automated in-situ XRD temperature ramp pipeline.
Imports settings from config.py, applies robust preprocessing (despiking,
baseline subtraction, Savitzky-Golay smoothing, sub-pixel apex interpolation),
tracks peaks, and produces publication-ready overlays, heatmaps, and GIFs.
"""

import os
import sys
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
import matplotlib.patheffects as pe

# Add project root and scripts directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import config
from preprocessing import preprocess_scan, peak_apex_subpixel

try:
    from PIL import Image
except ImportError:
    Image = None

def extract_temp(filename):
    match = config.TEMP_PATTERN.search(filename)
    if match is None:
        print(f"[{filename}] warning: could not extract temperature from filename")
        return np.nan
    return float(match.group(1))

def read_data(path):
    try:
        data = np.loadtxt(path, comments=config.COMMENT_CHAR, ndmin=2)
    except ValueError:
        data = np.loadtxt(path, comments=config.COMMENT_CHAR, delimiter=",", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("File must contain at least 2 columns (2theta, intensity)")
    return data

def peak_in_range(two_theta, intensity, lo, hi, name=""):
    if config.SUBPIXEL_APEX:
        center, _ = peak_apex_subpixel(two_theta, intensity, lo, hi)
        return center
    mask = (two_theta >= lo) & (two_theta <= hi)
    if not mask.any():
        print(f"[{name}] warning: no points found in range [{lo}, {hi}] deg")
        return np.nan
    return two_theta[mask][np.argmax(intensity[mask])]

def shifted_two_theta(tt0, temp, alpha):
    if not np.isfinite(temp) or alpha == 0:
        return tt0
    rad0 = np.radians(tt0 / 2.0)
    sin_new = np.sin(rad0) / (1.0 + alpha * (temp - config.REF_T_C))
    sin_new = np.clip(sin_new, -1.0, 1.0)
    return np.degrees(2.0 * np.arcsin(sin_new))

def hkl_label(h, k, l):
    parts = []
    for v in (h, k, l):
        parts.append(r"\bar{{{0}}}".format(abs(v)) if v < 0 else str(v))
    return r"$({0})$".format("".join(parts))

def draw_refs(ax, lo, hi, temp):
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.5, len(config.PHASES) * 1.0)
    ax.set_yticks([])
    ax.set_xlabel("2theta (deg)")
    placed = []
    for idx_p, ph in enumerate(config.PHASES):
        base = idx_p * 1.0
        color = ph.get("color", "k")
        alpha = ph.get("alpha", 0.0)
        pks = [(tt, inten, h, k, l) for (tt, inten, h, k, l) in ph["peaks"]
               if lo <= shifted_two_theta(tt, temp, alpha) <= hi]
        if not pks:
            continue
        max_i = max(p[1] for p in pks) or 1.0
        for tt0, raw_i, h, k, l in pks:
            x = shifted_two_theta(tt0, temp, alpha)
            h_rel = (raw_i / float(max_i)) * 0.75
            ax.vlines(x, base, base + h_rel, color=color, linewidth=1.2)
            if raw_i >= config.LABEL_MIN_I and not any(abs(x - p) < config.MIN_LABEL_SEP * (hi - lo) for p in placed):
                lbl = hkl_label(h, k, l)
                ax.text(x, base + h_rel + 0.03, lbl, ha="center", va="bottom",
                        fontsize=6, color=color, rotation=90)
                placed.append(x)
        ax.text(hi, base + 0.98, ph["name"], ha="right", va="top", fontsize=7,
                color=ph["color"], fontweight="bold")

def run():
    print(f"=== MXene XRD In-Situ Analysis ===")
    print(f"Input Directory:       {config.DATA_DIR}")
    print(f"Output Directory:      {config.OUTPUT_DIR}")
    print(f"Preprocessing Enabled: {config.ENABLE_PREPROCESSING} (Baseline={config.BASELINE_METHOD}, SavGol={config.ENABLE_SAVGOL}, SubPixel={config.SUBPIXEL_APEX})")

    files = sorted(glob.glob(os.path.join(config.DATA_DIR, config.FILE_PATTERN)))
    if not files:
        print(f"ERROR: No files matching '{config.FILE_PATTERN}' found in {config.DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} scan files.")

    results = []
    patterns = []
    peak_by_name = {}

    for f in files:
        name = os.path.basename(f)
        temp = extract_temp(name)
        try:
            data = read_data(f)
        except Exception as e:
            print(f"[{name}] error reading file: {e}")
            continue

        if config.TWO_THETA_CROP_MIN is not None:
            data = data[data[:, 0] >= config.TWO_THETA_CROP_MIN]
        if data.shape[0] == 0:
            print(f"[{name}] warning: no points left above {config.TWO_THETA_CROP_MIN} deg")
            results.append((temp, np.nan))
            continue

        tt, raw_inten = data[:, 0], data[:, 1]
        
        # Apply robust preprocessing
        if config.ENABLE_PREPROCESSING:
            clean_inten, baseline, _ = preprocess_scan(
                tt, raw_inten,
                remove_spike=config.REMOVE_SPIKES,
                smooth=config.ENABLE_SAVGOL,
                subtract_bg=config.SUBTRACT_BASELINE,
                bg_method=config.BASELINE_METHOD,
                savgol_win=config.SAVGOL_WINDOW,
                savgol_poly=config.SAVGOL_POLYORDER,
                snip_iters=config.BASELINE_SNIP_ITERATIONS
            )
            inten_for_tracking = clean_inten
        else:
            inten_for_tracking = raw_inten

        peak = peak_in_range(tt, inten_for_tracking, config.TWO_THETA_MIN, config.TWO_THETA_MAX, name)
        results.append((temp, peak))
        patterns.append((temp, tt, inten_for_tracking, name))
        peak_by_name[name] = peak
        print(f"  [{name}] T = {temp:4.0f} °C -> Peak at 2theta = {peak:.4f}°")

    # Save summary table
    table_file = config.OUTPUT_FILE_PREFIX + "_peak_positions.txt"
    table = np.array(results, dtype=float)
    table = table[np.argsort(table[:, 0], kind="mergesort")]
    header = (
        f"# Peak position in range [{config.TWO_THETA_MIN}, {config.TWO_THETA_MAX}] deg vs temperature\n"
        f"# Preprocessing: Baseline={config.BASELINE_METHOD}, SavGol={config.ENABLE_SAVGOL}, SubPixel={config.SUBPIXEL_APEX}\n"
        "# columns: temperature(C)  peak_position_2theta(deg)"
    )
    np.savetxt(table_file, table, fmt="%8.2f  %10.4f", header=header)
    print(f"\nSaved peak summary table: {table_file}")

    # Plot 1: Overlay
    plot_pats = sorted([p for p in patterns if np.isfinite(p[0])], key=lambda p: p[0])
    temps = np.array([p[0] for p in plot_pats])
    lo = config.PLOT_2THETA_MIN if config.PLOT_2THETA_MIN is not None else min(p[1].min() for p in plot_pats)
    hi = config.PLOT_2THETA_MAX if config.PLOT_2THETA_MAX is not None else max(p[1].max() for p in plot_pats)

    cmap = plt.get_cmap(config.CMAP_NAME)
    norm = colors.Normalize(vmin=temps.min(), vmax=temps.max())
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for temp, tt, inten, name in plot_pats:
        ax1.plot(tt, inten, color=cmap(norm(temp)), alpha=0.85, lw=1.2)
    ax1.set_xlim(lo, hi)
    ax1.set_xlabel("2theta (deg)")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.set_title(f"Diffractograms Overlay - Preprocessed ({len(plot_pats)} scans)")
    fig1.colorbar(sm, ax=ax1, label=f"Temperature ({config.TEMP_UNIT})")
    fig1.tight_layout()
    overlay_path = config.OUTPUT_FILE_PREFIX + "_overlay.png"
    fig1.savefig(overlay_path, dpi=200)
    plt.close(fig1)
    print(f"Saved overlay plot: {overlay_path}")

    # Plot 2: 2D Heatmap
    if len(plot_pats) >= 2:
        grid = np.linspace(lo, hi, 1000)
        rows = []
        for temp, tt, inten, name in plot_pats:
            order = np.argsort(tt)
            rows.append(np.interp(grid, tt[order], inten[order], left=np.nan, right=np.nan))
        Z = np.ma.masked_invalid(np.vstack(rows))

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        mesh = ax2.pcolormesh(grid, temps, Z, cmap=cmap, shading="gouraud",
                              vmin=0, vmax=np.percentile(Z.compressed(), 99.5))
        fig2.colorbar(mesh, ax=ax2, label="Intensity (a.u.)")

        if config.SHOW_PEAK_TRACK:
            t = table[np.isfinite(table[:, 0]) & np.isfinite(table[:, 1])]
            ax2.plot(t[:, 1], t[:, 0], color="red", lw=2, linestyle="--", label="Peak Track (Sub-pixel)")
            ax2.legend(loc="upper right")

        ax2.set_xlim(lo, hi)
        ax2.set_xlabel("2theta (deg)")
        ax2.set_ylabel(f"Temperature ({config.TEMP_UNIT})")
        ax2.set_title("Diffractogram Map (Heatmap with Sub-pixel Peak Track)")
        fig2.tight_layout()
        heatmap_path = config.OUTPUT_FILE_PREFIX + "_heatmap.png"
        fig2.savefig(heatmap_path, dpi=200)
        plt.close(fig2)
        print(f"Saved heatmap plot: {heatmap_path}")

    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    run()
