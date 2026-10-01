"""
fda_analysis.py

Functional Data Analysis (FDA) for in-situ temperature-resolved X-ray diffraction.
Treats each diffractogram not as discrete channels, but as an observation of a
continuous smooth function x_i(2theta) in L^2 space.

Key FDA Capabilities:
1. Continuous B-spline Basis Smoothing & Representation.
2. Analytical Functional Derivatives (1st derivative = slope/roots, 2nd derivative = curvature).
3. Continuous Functional Principal Component Analysis (fPCA) with quadrature L2 weighting.
4. Curve Registration: Decoupling Phase Variation (thermal lattice expansion shift)
   from Amplitude Variation (chemical phase transition / peak growth & decay).
5. Autonomous Symbolic Kinetic Law Discovery using Malthus-GP (dCGP + JAX).
6. Comprehensive multi-panel diagnostic visualizations.
"""

import os
import sys
import glob
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize

try:
    import config
    from scripts.preprocessing import preprocess_scan
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config
    from scripts.preprocessing import preprocess_scan

# Optional Malthus-GP Integration
try:
    from malthus_gp.learn import DCGPRegressor, CGPRegressor
    HAS_MALTHUS_GP = True
except ImportError:
    HAS_MALTHUS_GP = False


def load_dataset(data_dir=None, crop_min=7.5, crop_max=75.0, apply_preprocessing=True):
    """
    Loads all .xy files, extracts temperature metadata, crops to 2theta ROI,
    and optionally applies baseline subtraction and spike removal.
    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    files = sorted(glob.glob(os.path.join(data_dir, config.FILE_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No files matching '{config.FILE_PATTERN}' in {data_dir}")

    temps = []
    data_list = []
    
    for f in files:
        m = re.search(r"temp(\d+)", os.path.basename(f), re.IGNORECASE)
        t_val = float(m.group(1)) if m else np.nan
        temps.append(t_val)
        
        arr = np.loadtxt(f, comments=config.COMMENT_CHAR)
        data_list.append(arr)

    # Sort strictly by temperature
    temps = np.array(temps)
    order = np.argsort(temps)
    temps = temps[order]
    data_list = [data_list[o] for o in order]
    filenames = [os.path.basename(files[o]) for o in order]

    tt = data_list[0][:, 0]
    mask = (tt >= crop_min) & (tt <= crop_max)
    t_eval = tt[mask]
    
    raw_intensities = np.array([d[mask, 1] for d in data_list])
    
    if apply_preprocessing:
        processed_intensities = []
        for i in range(len(temps)):
            clean, _, _ = preprocess_scan(
                t_eval, raw_intensities[i],
                remove_spike=config.REMOVE_SPIKES,
                smooth=False, # Continuous B-splines handle functional smoothing
                subtract_bg=config.SUBTRACT_BASELINE,
                bg_method=config.BASELINE_METHOD,
                snip_iters=config.BASELINE_SNIP_ITERATIONS
            )
            processed_intensities.append(clean)
        intensities = np.array(processed_intensities)
    else:
        intensities = raw_intensities

    return t_eval, temps, intensities, filenames


def fit_functional_splines(t_eval, intensities, smoothing_factor=None):
    """
    Fits continuous cubic B-spline functions to discrete observations.
    Computes continuous functional values, 1st analytical derivatives,
    and 2nd analytical derivatives.
    """
    N = intensities.shape[0]
    P = len(t_eval)
    if smoothing_factor is None:
        smoothing_factor = P * 25.0

    splines = []
    y_func = np.zeros_like(intensities)
    d1_func = np.zeros_like(intensities)
    d2_func = np.zeros_like(intensities)

    for i in range(N):
        tck = splrep(t_eval, intensities[i], s=smoothing_factor, k=3)
        splines.append(tck)
        y_func[i] = splev(t_eval, tck, der=0)
        d1_func[i] = splev(t_eval, tck, der=1)
        d2_func[i] = splev(t_eval, tck, der=2)

    return splines, y_func, d1_func, d2_func


def compute_functional_pca(t_eval, functional_curves, n_components=4):
    r"""
    Mathematically rigorous continuous Functional Principal Component Analysis (fPCA)
    using numerical quadrature weights for the L^2 functional inner product:
        <f, g> = \int f(t) g(t) dt \approx f^T W^2 g
    
    Returns:
    - mean_curve: continuous mean function \mu(t)
    - harmonics: continuous eigenfunctions \phi_j(t) with \int \phi_j(t)^2 dt = 1
    - eigenvalues: variance associated with each harmonic
    - var_explained: percentage of total functional variance
    - scores: functional projection scores \xi_{ij}
    """
    N, P = functional_curves.shape
    dt = np.gradient(t_eval)
    W = np.sqrt(dt)

    # 1. Functional Mean
    mean_curve = np.mean(functional_curves, axis=0)
    centered = functional_curves - mean_curve

    # 2. Quadrature-weighted SVD
    weighted_centered = centered * W[np.newaxis, :]
    U, S, Vt = np.linalg.svd(weighted_centered, full_matrices=False)

    # 3. Covariance operator eigenvalues
    eigenvalues = (S ** 2) / (N - 1)
    total_var = np.sum(eigenvalues)
    var_explained = (eigenvalues / total_var) * 100.0

    # 4. Continuous functional harmonics (eigenfunctions)
    harmonics = Vt[:n_components] / W[np.newaxis, :]
    
    # 5. Functional PC scores: \xi_ij = \int (x_i(t) - \mu(t)) \phi_j(t) dt
    scores = (U * S)[:, :n_components]

    return {
        "mean_curve": mean_curve,
        "harmonics": harmonics,
        "eigenvalues": eigenvalues[:n_components],
        "var_explained": var_explained[:n_components],
        "scores": scores,
        "cumulative_var": np.cumsum(var_explained)[:n_components]
    }


def compute_curve_registration(t_eval, functional_curves, ref_idx=0):
    r"""
    Performs continuous functional curve registration to decouple:
    1. Phase Variation (horizontal shift / thermal lattice strain): h_i(t)
    2. Amplitude Variation (vertical intensity changes / phase conversion): \tilde{x}_i(t)
    
    Uses smooth affine/deformation warping parameterized by continuous monotonic mapping.
    """
    N, P = functional_curves.shape
    ref_curve = functional_curves[ref_idx]
    
    registered_curves = np.zeros_like(functional_curves)
    warping_functions = np.zeros((N, P))
    strain_fields = np.zeros((N, P))

    domain_width = t_eval[-1] - t_eval[0]
    t_norm = (t_eval - t_eval[0]) / domain_width

    for i in range(N):
        if i == ref_idx:
            registered_curves[i] = functional_curves[i].copy()
            warping_functions[i] = t_eval.copy()
            strain_fields[i] = np.zeros(P)
            continue

        cur = functional_curves[i]
        
        # Warp parameterized by [shift_0, shift_slope]: w(t) = a + b * t_norm
        def obj(params):
            w = params[0] + params[1] * t_norm
            warped_t = np.clip(t_eval + w, t_eval[0], t_eval[-1])
            warped_cur = np.interp(warped_t, t_eval, cur)
            corr = np.corrcoef(warped_cur, ref_curve)[0, 1]
            if np.isnan(corr):
                return 0.0
            reg = 1e-4 * np.sum(w ** 2)
            return -corr + reg

        res = minimize(obj, [0.0, 0.0], method="Nelder-Mead")
        w_opt = res.x[0] + res.x[1] * t_norm
        warped_t = np.clip(t_eval + w_opt, t_eval[0], t_eval[-1])
        
        registered_curves[i] = np.interp(warped_t, t_eval, cur)
        warping_functions[i] = warped_t
        theta_rad = np.radians(t_eval / 2.0)
        strain_fields[i] = - w_opt / (2.0 * np.tan(np.maximum(theta_rad, 1e-3)))

    return {
        "registered_curves": registered_curves,
        "warping_functions": warping_functions,
        "strain_fields": strain_fields
    }


def plot_fda_diagnostics(t_eval, temps, raw_y, y_func, d1_func, d2_func,
                         fpca_res, reg_res, output_dir=None):
    """
    Generates three high-resolution diagnostic figures:
    1. Spline smoothing and continuous 1st/2nd functional derivatives
    2. Continuous Functional PCA harmonics and temperature score trajectories
    3. Curve registration: Phase vs Amplitude separation
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    norm = colors.Normalize(vmin=temps.min(), vmax=temps.max())
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Figure 1: Functional Representation & Analytical Derivatives
    fig1, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    
    for i in range(len(temps)):
        col = cmap(norm(temps[i]))
        axs[0].plot(t_eval, raw_y[i], color=col, alpha=0.35, lw=0.8)
        axs[0].plot(t_eval, y_func[i], color=col, alpha=0.9, lw=1.5)
        axs[1].plot(t_eval, d1_func[i], color=col, alpha=0.85, lw=1.2)
        axs[2].plot(t_eval, d2_func[i], color=col, alpha=0.85, lw=1.2)

    axs[0].set_ylabel(r"Intensity $x_i(2\theta)$ (a.u.)", fontsize=11, fontweight="bold")
    axs[0].set_title(r"A. Continuous B-Spline Basis Representation $x_i(2\theta)$", fontsize=12, fontweight="bold", loc="left")
    axs[0].grid(True, linestyle=":", alpha=0.6)

    axs[1].axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.7)
    axs[1].set_ylabel(r"1st Deriv $\frac{dx_i}{d(2\theta)}$", fontsize=11, fontweight="bold")
    axs[1].set_title("B. Continuous Analytical First Derivative (Zero-crossings = Peak Apices)", fontsize=12, fontweight="bold", loc="left")
    axs[1].grid(True, linestyle=":", alpha=0.6)

    axs[2].axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.7)
    axs[2].set_ylabel(r"2nd Deriv $\frac{d^2x_i}{d(2\theta)^2}$", fontsize=11, fontweight="bold")
    axs[2].set_title("C. Continuous Analytical Second Derivative (Curvature / Peak Broadening)", fontsize=12, fontweight="bold", loc="left")
    axs[2].set_xlabel(r"Diffraction Angle $2\theta$ (degrees)", fontsize=11, fontweight="bold")
    axs[2].grid(True, linestyle=":", alpha=0.6)

    cbar1 = fig1.colorbar(sm, ax=axs, orientation="vertical", fraction=0.02, pad=0.02)
    cbar1.set_label("Temperature (°C)", fontsize=10, fontweight="bold")
    
    fig1.savefig(output_dir / "fda_splines_derivatives.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Functional Principal Component Analysis (fPCA)
    fig2, axs2 = plt.subplots(2, 2, figsize=(13, 9))

    # Panel A: Mean function
    axs2[0, 0].plot(t_eval, fpca_res["mean_curve"], color="#2b5c8f", lw=2.2)
    axs2[0, 0].fill_between(t_eval, 0, fpca_res["mean_curve"], color="#2b5c8f", alpha=0.15)
    axs2[0, 0].set_title(r"A. Functional Mean Curve $\mu(2\theta)$", fontsize=11, fontweight="bold", loc="left")
    axs2[0, 0].set_xlabel(r"$2\theta$ (deg)")
    axs2[0, 0].set_ylabel("Intensity (a.u.)")
    axs2[0, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel B: Harmonics / Eigenfunctions
    colors_harm = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    for j in range(min(3, len(fpca_res["harmonics"]))):
        axs2[0, 1].plot(
            t_eval, fpca_res["harmonics"][j],
            label=f"Harmonic {j+1} ({fpca_res['var_explained'][j]:.1f}%)",
            color=colors_harm[j], lw=1.8
        )
    axs2[0, 1].axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.7)
    axs2[0, 1].set_title(r"B. Continuous Functional Harmonics $\phi_j(2\theta)$", fontsize=11, fontweight="bold", loc="left")
    axs2[0, 1].set_xlabel(r"$2\theta$ (deg)")
    axs2[0, 1].set_ylabel("Harmonic Amplitude")
    axs2[0, 1].legend(loc="upper right", frameon=True)
    axs2[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Panel C: Scree Plot
    comps = np.arange(1, len(fpca_res["var_explained"]) + 1)
    axs2[1, 0].bar(comps, fpca_res["var_explained"], color="#41b6c4", edgecolor="black", alpha=0.85, label="Individual")
    axs2[1, 0].step(comps, fpca_res["cumulative_var"], where="mid", color="#d7301f", lw=2, marker="o", label="Cumulative")
    for idx_c, v in enumerate(fpca_res["var_explained"]):
        axs2[1, 0].text(idx_c + 1, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axs2[1, 0].set_title("C. Functional Variance Explained (fPCA Scree)", fontsize=11, fontweight="bold", loc="left")
    axs2[1, 0].set_xlabel("Functional Principal Component")
    axs2[1, 0].set_ylabel("Variance Explained (%)")
    axs2[1, 0].set_ylim(0, 110)
    axs2[1, 0].legend(loc="center right")
    axs2[1, 0].grid(True, linestyle=":", alpha=0.6)

    # Panel D: Score Trajectories vs Temperature
    axs2[1, 1].plot(temps, fpca_res["scores"][:, 0], "o-", color=colors_harm[0], lw=2, label=r"Score $\xi_1(T)$ (Phase Degradation)")
    axs2[1, 1].plot(temps, fpca_res["scores"][:, 1], "s--", color=colors_harm[1], lw=2, label=r"Score $\xi_2(T)$ (Lattice Shift / Phase Emergence)")
    axs2[1, 1].set_title(r"D. Functional Score Trajectories $\xi_j(T)$ vs Temperature", fontsize=11, fontweight="bold", loc="left")
    axs2[1, 1].set_xlabel("Temperature (°C)")
    axs2[1, 1].set_ylabel(r"Functional Score $\xi_j$")
    axs2[1, 1].legend(loc="best", frameon=True)
    axs2[1, 1].grid(True, linestyle=":", alpha=0.6)

    fig2.tight_layout()
    fig2.savefig(output_dir / "fda_fpca_modes.png", dpi=200, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Phase vs Amplitude Separation (Curve Registration)
    fig3, axs3 = plt.subplots(1, 3, figsize=(16, 5))

    for i in range(len(temps)):
        col = cmap(norm(temps[i]))
        axs3[0].plot(t_eval, y_func[i], color=col, alpha=0.85, lw=1.3)
        axs3[1].plot(t_eval, reg_res["warping_functions"][i] - t_eval, color=col, alpha=0.85, lw=1.4)
        axs3[2].plot(t_eval, reg_res["registered_curves"][i], color=col, alpha=0.85, lw=1.3)

    axs3[0].set_title("A. Unregistered Scans (Phase + Amplitude Mixed)", fontsize=10, fontweight="bold")
    axs3[0].set_xlabel(r"$2\theta$ (deg)")
    axs3[0].set_ylabel("Intensity (a.u.)")
    axs3[0].grid(True, linestyle=":", alpha=0.6)

    axs3[1].axhline(0, color="k", linestyle="--", lw=0.8)
    axs3[1].set_title(r"B. Warping Function $w_i(2\theta) = h_i(2\theta) - 2\theta$" + "\n(Pure Thermal Lattice Expansion / Strain Field)", fontsize=10, fontweight="bold")
    axs3[1].set_xlabel(r"$2\theta$ (deg)")
    axs3[1].set_ylabel(r"Displacement $\Delta(2\theta)$ (deg)")
    axs3[1].grid(True, linestyle=":", alpha=0.6)

    axs3[2].set_title(r"C. Registered Curves $\tilde{x}_i(2\theta)$" + "\n(Pure Amplitude / Chemical Phase Conversion)", fontsize=10, fontweight="bold")
    axs3[2].set_xlabel(r"$2\theta$ (deg)")
    axs3[2].set_ylabel("Aligned Intensity (a.u.)")
    axs3[2].grid(True, linestyle=":", alpha=0.6)

    fig3.tight_layout()
    fig3.savefig(output_dir / "fda_registration.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)


def export_fda_summary(temps, fpca_res, reg_res, output_path=None):
    """
    Saves a tabular summary of functional scores and thermal lattice strain parameters.
    """
    if output_path is None:
        output_path = Path(config.OUTPUT_DIR) / "fda_scores_summary.txt"
    
    scores = fpca_res["scores"]
    header = (
        "========================================================================\n"
        "FUNCTIONAL DATA ANALYSIS (FDA) - IN-SITU TEMPERATURE-RESOLVED XRD\n"
        "========================================================================\n"
        f"Variance Explained by Harmonics:\n"
        f"  fPC1 (Harmonic 1): {fpca_res['var_explained'][0]:.2f}%\n"
        f"  fPC2 (Harmonic 2): {fpca_res['var_explained'][1]:.2f}%\n"
        f"  fPC3 (Harmonic 3): {fpca_res['var_explained'][2]:.2f}%\n"
        f"  Cumulative (1-3):  {fpca_res['cumulative_var'][min(2, len(fpca_res['cumulative_var'])-1)]:.2f}%\n"
        "------------------------------------------------------------------------\n"
        "Columns:\n"
        "  1. Temperature (°C)\n"
        "  2. fPC1 Score (Primary Reaction Extent)\n"
        "  3. fPC2 Score (Secondary Phase / Peak Shifting)\n"
        "  4. Mean Thermal Strain Delta(2theta)\n"
        "========================================================================"
    )
    
    mean_displacements = np.mean(reg_res["warping_functions"] - reg_res["warping_functions"][0], axis=1)

    table = np.column_stack([
        temps,
        scores[:, 0],
        scores[:, 1],
        mean_displacements
    ])
    
    np.savetxt(output_path, table, fmt="%8.1f  %12.4f  %12.4f  %12.4f", header=header)
    return output_path


def discover_kinetics_with_malthus_gp(temps, fpca_res, output_dir=None, random_state=42):
    r"""
    Symbolic Kinetic Law Discovery using Malthus-GP (Differentiable Cartesian Genetic Programming).
    Normalizes the primary functional reaction coordinate (fPC1 score) into the solid-state
    conversion fraction \alpha(T) in [0, 1], and solves for the closed-form analytical rate law.
    """
    if not HAS_MALTHUS_GP:
        print("[-] Malthus-GP is not available in the current environment. Skipping GP discovery.")
        return None

    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  MALTHUS-GP: AUTONOMOUS KINETIC LAW DISCOVERY (dCGP + JAX)")
    print("=" * 65)

    scores1 = fpca_res["scores"][:, 0]
    alpha = (scores1 - scores1[0]) / (scores1[-1] - scores1[0])

    # Feature: Normalized temperature T / 1000
    X = (temps / 1000.0).reshape(-1, 1)
    y = alpha

    print(f"[*] Training dCGP on {len(temps)} in-situ temperature points...")
    dcgp = DCGPRegressor(
        n_rows=1,
        n_cols=25,
        pop_size=5,
        max_generations=150,
        lsmf_epochs=12,
        learning_rate=0.03,
        random_state=random_state
    )
    dcgp.fit(X, y)

    r2 = dcgp.score(X, y)
    formula = dcgp.to_formula()

    # Dense prediction and analytical temperature derivative
    dense_T = np.linspace(temps.min(), temps.max(), 200)
    dense_X = (dense_T / 1000.0).reshape(-1, 1)
    dense_pred = dcgp.predict(dense_X)
    d_pred_dT = np.gradient(dense_pred, dense_T)
    t_peak_rate = dense_T[np.argmax(d_pred_dT)]

    print(f"[+] Discovered Closed-Form Rate Model (R² = {r2:.4f}):")
    print(f"    alpha(T) = {formula}")
    print(f"[+] Peak Reaction Rate Temperature: T_max = {t_peak_rate:.1f} °C")

    # Generate 2-panel Diagnostic Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Conversion Extent alpha(T)
    axes[0].scatter(temps, alpha, color="#d95f02", s=70, edgecolor="black", zorder=4, label=r"FDA Empirical Coordinate $\xi_1(T)$")
    axes[0].plot(dense_T, dense_pred, color="#1b9e77", lw=2.5, label=f"Malthus-GP dCGP Model ($R^2={r2:.4f}$)")
    axes[0].set_title(r"A. Solid-State Conversion Progress $\alpha(T)$" + f"\nFormula: {formula[:55]}...", fontsize=11, fontweight="bold", loc="left")
    axes[0].set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"Reaction Extent $\alpha(T)$", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", frameon=True)

    # Panel B: Reaction Rate Derivative d(alpha)/dT
    axes[1].plot(dense_T, d_pred_dT * 1000.0, color="#7570b3", lw=2.5, label=r"dCGP Conversion Rate $\frac{d\alpha}{dT}$")
    axes[1].axvline(t_peak_rate, color="red", linestyle="--", lw=1.5, label=f"Peak Rate: $T_{{max}} = {t_peak_rate:.1f}^\\circ\\text{{C}}$")
    axes[1].scatter([t_peak_rate], [np.max(d_pred_dT) * 1000.0], color="red", s=80, zorder=5)
    axes[1].set_title(r"B. Reaction Rate Derivative $\frac{d\alpha}{dT}$ (Transformation Kinetics)", fontsize=11, fontweight="bold", loc="left")
    axes[1].set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel(r"Rate $\frac{d\alpha}{dT} \times 10^3$ ($^\circ\text{C}^{-1}$)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", frameon=True)

    fig.tight_layout()
    plot_path = output_dir / "malthus_gp_kinetic_discovery.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Export Text Summary
    summary_path = output_dir / "malthus_gp_discovered_formula.txt"
    with open(summary_path, "w") as f:
        f.write("========================================================================\n")
        f.write("MALTHUS-GP: SYMBOLIC KINETIC RATE LAW DISCOVERY VIA dCGP & JAX AUTODIFF\n")
        f.write("========================================================================\n")
        f.write(f"R² Fit Score:                    {r2:.5f}\n")
        f.write(f"Peak Conversion Temperature:     {t_peak_rate:.1f} °C\n")
        f.write(f"Normalized Feature:              x0 = T_celsius / 1000.0\n")
        f.write(f"Discovered Closed-Form Formula:\n  alpha(T) = {formula}\n\n")
        f.write("Temperature (°C)   Empirical alpha   Predicted alpha   Residual\n")
        f.write("------------------------------------------------------------------------\n")
        pred_pts = dcgp.predict(X)
        for t_val, a_emp, a_pred in zip(temps, alpha, pred_pts):
            f.write(f"{t_val:12.1f}   {a_emp:15.4f}   {a_pred:15.4f}   {abs(a_emp - a_pred):10.4f}\n")
        f.write("========================================================================\n")

    print(f"[+] Saved Malthus-GP plot to:    {plot_path}")
    print(f"[+] Saved Malthus-GP summary to: {summary_path}")

    return {
        "model": dcgp,
        "r2": r2,
        "formula": formula,
        "t_peak_rate": t_peak_rate,
        "plot_path": plot_path,
        "summary_path": summary_path
    }


def run_fda_pipeline(run_gp=True):
    """
    Executes the full FDA pipeline end-to-end.
    """
    print("=" * 65)
    print("  FUNCTIONAL DATA ANALYSIS (FDA) PIPELINE FOR IN-SITU XRD")
    print("=" * 65)

    print(f"[1/6] Loading data from {config.DATA_DIR}...")
    t_eval, temps, raw_y, filenames = load_dataset(crop_min=7.5, crop_max=75.0)
    print(f"      Loaded {len(temps)} diffractograms across {len(t_eval)} angular points.")
    print(f"      Temperature range: {temps.min():.0f} °C -> {temps.max():.0f} °C")

    print(f"[2/6] Fitting Continuous B-Splines & Analytical Derivatives...")
    splines, y_func, d1_func, d2_func = fit_functional_splines(t_eval, raw_y)
    print("      Continuous L^2 functional representations and derivatives computed.")

    print(f"[3/6] Computing Continuous Functional PCA (fPCA)...")
    fpca_res = compute_functional_pca(t_eval, y_func, n_components=4)
    print(f"      fPC1: {fpca_res['var_explained'][0]:.2f}% variance explained")
    print(f"      fPC2: {fpca_res['var_explained'][1]:.2f}% variance explained")
    print(f"      fPC3: {fpca_res['var_explained'][2]:.2f}% variance explained")
    print(f"      Total (1-2): {fpca_res['cumulative_var'][1]:.2f}% of structural evolution captured!")

    print(f"[4/6] Computing Continuous Curve Registration (Phase-Amplitude Separation)...")
    reg_res = compute_curve_registration(t_eval, y_func, ref_idx=0)
    print("      Lattice expansion strain field decoupled from chemical reaction conversion.")

    print(f"[5/6] Exporting Diagnostic Visualizations and Summary Table...")
    plot_fda_diagnostics(t_eval, temps, raw_y, y_func, d1_func, d2_func,
                         fpca_res, reg_res, config.OUTPUT_DIR)
    
    out_table = export_fda_summary(temps, fpca_res, reg_res)
    print(f"      Saved: {out_table}")
    print(f"      Saved: {config.OUTPUT_DIR}/fda_splines_derivatives.png")
    print(f"      Saved: {config.OUTPUT_DIR}/fda_fpca_modes.png")
    print(f"      Saved: {config.OUTPUT_DIR}/fda_registration.png")

    if run_gp and HAS_MALTHUS_GP:
        print(f"[6/6] Executing Symbolic Kinetics Discovery via Malthus-GP...")
        discover_kinetics_with_malthus_gp(temps, fpca_res, config.OUTPUT_DIR)
    else:
        print("[6/6] Malthus-GP kinetics step skipped.")

    print("=" * 65)
    print("FDA Pipeline completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_fda_pipeline(run_gp=True)
