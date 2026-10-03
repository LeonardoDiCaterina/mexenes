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



def fit_rutile_phase_evolution(t_eval, temps, raw_y, output_dir=None):
    """
    Performs explicit crystallographic peak deconvolution of the primary TiO2 Rutile (110)
    reflection (2theta ~ 27.45 deg) across in-situ temperature scans.
    
    Uses a pseudo-Voigt profile with linear local background:
        I(2theta) = b0 + b1*(2theta - 27.5) + A * [eta*L(2theta) + (1-eta)*G(2theta)]
        
    Extracts:
        - Integrated Peak Area A_rutile(T)
        - Peak centroid position 2theta(T) (tracking thermal lattice expansion)
        - Peak FWHM w(T) (crystallite size broadening)
        - Direct physical conversion fraction alpha_rutile(T) in [0, 1] (strictly non-negative)
    """
    from scipy.optimize import curve_fit

    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def pseudo_voigt(x, b0, b1, A, mu, w, eta):
        sigma = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        G = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        gamma = w / 2.0
        L = (1.0 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
        return b0 + b1 * (x - 27.5) + A * (eta * L + (1.0 - eta) * G)

    mask_fit = (t_eval >= 26.2) & (t_eval <= 28.8)
    x_sub = t_eval[mask_fit]

    rutile_areas = []
    rutile_centers = []
    rutile_fwhms = []
    fits = []

    for idx, temp in enumerate(temps):
        y_sub = raw_y[idx, mask_fit]
        b0_init = float(np.min(y_sub))
        b1_init = float((y_sub[-1] - y_sub[0]) / (x_sub[-1] - x_sub[0]))
        A_init = float(max(np.max(y_sub) - b0_init, 0.0) * 0.4)
        bounds = ([-100, -100, 0.0, 27.0, 0.1, 0.0], [2000, 100, 5000, 27.8, 1.5, 1.0])
        try:
            popt, _ = curve_fit(
                pseudo_voigt, x_sub, y_sub,
                p0=[b0_init, b1_init, A_init, 27.42, 0.35, 0.5],
                bounds=bounds,
                maxfev=5000
            )
            b0, b1, A, mu, w, eta = popt
            if A < 5.0 or np.max(y_sub) - np.min(y_sub) < 15.0:
                A = 0.0
                mu = np.nan
                w = np.nan
                fit_curve = b0_init + b1_init * (x_sub - 27.5)
            else:
                fit_curve = pseudo_voigt(x_sub, *popt)
        except Exception:
            A = 0.0
            mu = np.nan
            w = np.nan
            fit_curve = b0_init + b1_init * (x_sub - 27.5)

        rutile_areas.append(A)
        rutile_centers.append(mu)
        rutile_fwhms.append(w)
        fits.append(fit_curve)

    rutile_areas = np.array(rutile_areas)
    rutile_centers = np.array(rutile_centers)
    rutile_fwhms = np.array(rutile_fwhms)

    denom = np.max(rutile_areas) - np.min(rutile_areas)
    if denom > 0:
        alpha_rutile = (rutile_areas - np.min(rutile_areas)) / denom
    else:
        alpha_rutile = np.zeros_like(rutile_areas)
    alpha_rutile = np.clip(alpha_rutile, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(temps)))
    for idx, temp in enumerate(temps):
        y_sub = raw_y[idx, mask_fit]
        offset = idx * 120
        axes[0].plot(x_sub, y_sub + offset, color=colors[idx], lw=1.2, alpha=0.7)
        axes[0].plot(x_sub, fits[idx] + offset, color="black", lw=1.5, linestyle="--")
        axes[0].text(28.85, y_sub[-1] + offset, f"{temp:.0f} °C", color=colors[idx], fontsize=8, va="center")

    axes[0].axvline(27.446, color="#0072B2", linestyle=":", lw=1.5, label="Rutile (110) ICDD (27.446°)")
    axes[0].set_title("A. In-Situ Rutile (110) Peak Deconvolution (Pseudo-Voigt)", fontsize=11, fontweight="bold", loc="left")
    axes[0].set_xlabel("2θ (degrees)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Intensity + Offset (a.u.)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    p1 = ax2.plot(temps, alpha_rutile, "o-", color="#d95f02", lw=2.2, ms=7, label=r"Measured Rutile Fraction $lpha_{\mathrm{rutile}}(T)$")
    valid_mu = [(t, m) for t, m in zip(temps, rutile_centers) if np.isfinite(m)]
    if valid_mu:
        t_v, m_v = zip(*valid_mu)
        p2 = ax2_twin.plot(t_v, m_v, "s--", color="#2b83ba", lw=1.8, ms=6, label=r"Peak Centroid $2	heta$ (Lattice Expansion)")
    else:
        p2 = []

    ax2.set_title("B. Quantitative Phase Evolution & Lattice Shift", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Measured Conversion $lpha_{\mathrm{rutile}} \in [0, 1]$", color="#d95f02", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel(r"Peak Position $2	heta$ (°)", color="#2b83ba", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)

    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center left", frameon=True)

    fig.tight_layout()
    plot_path = output_dir / "rutile_peak_deconvolution.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "rutile_areas": rutile_areas,
        "rutile_centers": rutile_centers,
        "rutile_fwhms": rutile_fwhms,
        "alpha_rutile": alpha_rutile,
        "plot_path": plot_path
    }



def fit_rutile_phase_evolution(t_eval, temps, raw_y, output_dir=None):
    """
    Performs explicit crystallographic peak deconvolution of the primary TiO2 Rutile (110)
    reflection (2theta ~ 27.45 deg) across in-situ temperature scans.
    
    Uses a pseudo-Voigt profile with linear local background:
        I(2theta) = b0 + b1*(2theta - 27.5) + A * [eta*L(2theta) + (1-eta)*G(2theta)]
        
    Extracts:
        - Integrated Peak Area A_rutile(T)
        - Peak centroid position 2theta(T) (tracking thermal lattice expansion)
        - Peak FWHM w(T) (crystallite size broadening)
        - Direct physical conversion fraction alpha_rutile(T) in [0, 1] (strictly non-negative)
    """
    from scipy.optimize import curve_fit

    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def pseudo_voigt(x, b0, b1, A, mu, w, eta):
        sigma = w / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        G = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        gamma = w / 2.0
        L = (1.0 / np.pi) * (gamma / ((x - mu) ** 2 + gamma ** 2))
        return b0 + b1 * (x - 27.5) + A * (eta * L + (1.0 - eta) * G)

    mask_fit = (t_eval >= 26.2) & (t_eval <= 28.8)
    x_sub = t_eval[mask_fit]

    rutile_areas = []
    rutile_centers = []
    rutile_fwhms = []
    fits = []

    for idx, temp in enumerate(temps):
        y_sub = raw_y[idx, mask_fit]
        b0_init = float(np.min(y_sub))
        b1_init = float((y_sub[-1] - y_sub[0]) / (x_sub[-1] - x_sub[0]))
        A_init = float(max(np.max(y_sub) - b0_init, 0.0) * 0.4)
        bounds = ([-100, -100, 0.0, 27.0, 0.1, 0.0], [2000, 100, 5000, 27.8, 1.5, 1.0])
        try:
            popt, _ = curve_fit(
                pseudo_voigt, x_sub, y_sub,
                p0=[b0_init, b1_init, A_init, 27.42, 0.35, 0.5],
                bounds=bounds,
                maxfev=5000
            )
            b0, b1, A, mu, w, eta = popt
            if A < 5.0 or np.max(y_sub) - np.min(y_sub) < 15.0:
                A = 0.0
                mu = np.nan
                w = np.nan
                fit_curve = b0_init + b1_init * (x_sub - 27.5)
            else:
                fit_curve = pseudo_voigt(x_sub, *popt)
        except Exception:
            A = 0.0
            mu = np.nan
            w = np.nan
            fit_curve = b0_init + b1_init * (x_sub - 27.5)

        rutile_areas.append(A)
        rutile_centers.append(mu)
        rutile_fwhms.append(w)
        fits.append(fit_curve)

    rutile_areas = np.array(rutile_areas)
    rutile_centers = np.array(rutile_centers)
    rutile_fwhms = np.array(rutile_fwhms)

    denom = np.max(rutile_areas) - np.min(rutile_areas)
    if denom > 0:
        alpha_rutile = (rutile_areas - np.min(rutile_areas)) / denom
    else:
        alpha_rutile = np.zeros_like(rutile_areas)
    alpha_rutile = np.clip(alpha_rutile, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    colors = plt.cm.inferno(np.linspace(0.15, 0.85, len(temps)))
    for idx, temp in enumerate(temps):
        y_sub = raw_y[idx, mask_fit]
        offset = idx * 120
        axes[0].plot(x_sub, y_sub + offset, color=colors[idx], lw=1.2, alpha=0.7)
        axes[0].plot(x_sub, fits[idx] + offset, color="black", lw=1.5, linestyle="--")
        axes[0].text(28.85, y_sub[-1] + offset, f"{temp:.0f} °C", color=colors[idx], fontsize=8, va="center")

    axes[0].axvline(27.446, color="#0072B2", linestyle=":", lw=1.5, label="Rutile (110) ICDD (27.446°)")
    axes[0].set_title("A. In-Situ Rutile (110) Peak Deconvolution (Pseudo-Voigt)", fontsize=11, fontweight="bold", loc="left")
    axes[0].set_xlabel("2θ (degrees)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Intensity + Offset (a.u.)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.5)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    p1 = ax2.plot(temps, alpha_rutile, "o-", color="#d95f02", lw=2.2, ms=7, label=r"Measured Rutile Fraction $\alpha_{\mathrm{rutile}}(T)$")
    valid_mu = [(t, m) for t, m in zip(temps, rutile_centers) if np.isfinite(m)]
    if valid_mu:
        t_v, m_v = zip(*valid_mu)
        p2 = ax2_twin.plot(t_v, m_v, "s--", color="#2b83ba", lw=1.8, ms=6, label=r"Peak Centroid $2\theta$ (Lattice Expansion)")
    else:
        p2 = []

    ax2.set_title("B. Quantitative Phase Evolution & Lattice Shift", fontsize=11, fontweight="bold", loc="left")
    ax2.set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(r"Measured Conversion $\alpha_{\mathrm{rutile}} \in [0, 1]$", color="#d95f02", fontsize=11, fontweight="bold")
    ax2_twin.set_ylabel(r"Peak Position $2\theta$ (°)", color="#2b83ba", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.5)

    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc="center left", frameon=True)

    fig.tight_layout()
    plot_path = output_dir / "rutile_peak_deconvolution.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "rutile_areas": rutile_areas,
        "rutile_centers": rutile_centers,
        "rutile_fwhms": rutile_fwhms,
        "alpha_rutile": alpha_rutile,
        "plot_path": plot_path
    }


def distill_clean_rate_law(temps, dense_T, dense_pred, alpha_emp=None):
    """
    Fits the standard empirical 2-parameter logistic/tanh phase transition model:
        alpha(T) = 0.5 * [1 + tanh((T - T_1/2) / Delta_T)]
    parameterizing the non-isothermal transformation midpoint (T_1/2) and width (Delta_T).

    NOTE (ICTAC Kinetics Standard): In non-isothermal thermal analysis at a single heating rate,
    alpha(T) parameterizes the empirical phase transition curve. Decoupling the full kinetic triplet
    (Ea, A, f(alpha)) requires multi-heating-rate isoconversional analysis.
    """
    from scipy.optimize import curve_fit

    def clean_tanh(T, T_half, Delta_T):
        return 0.5 * (1.0 + np.tanh((T - T_half) / Delta_T))

    # Dynamically estimate initial parameter guesses from data without hardcoding
    if alpha_emp is not None and temps is not None and len(temps) > 1:
        idx_half = int(np.argmin(np.abs(alpha_emp - 0.5)))
        p0_thalf = float(temps[idx_half])
        p0_delta = max(float(temps[-1] - temps[0]) / 8.0, 20.0)
    else:
        p0_thalf = float(np.median(dense_T))
        p0_delta = max(float(dense_T[-1] - dense_T[0]) / 8.0, 20.0)

    popt, _ = curve_fit(clean_tanh, dense_T, dense_pred, p0=[p0_thalf, p0_delta])
    t_half = float(np.round(popt[0]))
    delta_t = float(np.round(popt[1]))

    pred_clean = clean_tanh(dense_T, t_half, delta_t)
    r2_clean = 1.0 - np.sum((dense_pred - pred_clean) ** 2) / np.sum((dense_pred - np.mean(dense_pred)) ** 2)

    if alpha_emp is not None and temps is not None:
        emp_clean = clean_tanh(temps, t_half, delta_t)
        denom = np.sum((alpha_emp - np.mean(alpha_emp)) ** 2)
        r2_emp = 1.0 - np.sum((alpha_emp - emp_clean) ** 2) / denom if denom > 0 else r2_clean
    else:
        r2_emp = r2_clean

    formula_clean = f"0.5 * (1 + tanh((T - {t_half:.0f}) / {delta_t:.0f}))"
    latex_formula = rf"\alpha(T) = \frac{{1}}{{2}} \left[ 1 + \tanh\left( \frac{{T - {t_half:.0f}}}{{{delta_t:.0f}}} \right) \right]"
    latex_deriv = rf"\frac{{d\alpha}}{{dT}} = \frac{{1}}{{{2*delta_t:.0f}}} \operatorname{{sech}}^2\left( \frac{{T - {t_half:.0f}}}{{{delta_t:.0f}}} \right)"

    return {
        "formula_clean": formula_clean,
        "latex_formula": latex_formula,
        "latex_deriv": latex_deriv,
        "t_half": t_half,
        "delta_t": delta_t,
        "r2_clean": r2_clean,
        "r2_emp": r2_emp,
        "clean_pred": pred_clean
    }

def discover_kinetics_with_malthus_gp(temps, fpca_res=None, alpha_custom=None, output_dir=None, random_state=42):
    r"""
    Phase Transition Modeling using Malthus-GP (dCGP + JAX).
    Takes either:
      1. alpha_custom: direct crystallographic phase conversion (e.g. from Rutile (110) deconvolution)
      2. fpca_res: continuous 1st functional principal component score, baseline-corrected to [0, 1].

    Solves for the continuous transformation progress alpha(T) and analytical derivative d(alpha)/dT.
    """
    if not HAS_MALTHUS_GP:
        print("[-] Malthus-GP is not available in the current environment. Skipping GP discovery.")
        return None

    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("  MALTHUS-GP: PHASE TRANSITION MODELING (dCGP + JAX)")
    print("=" * 65)

    if alpha_custom is not None:
        alpha = np.asarray(alpha_custom, dtype=float)
        alpha_source = "Rutile (110) Bragg Peak Deconvolution"
    elif fpca_res is not None:
        scores1 = fpca_res["scores"][:, 0]
        pre_reaction_base = np.min(scores1[:3])
        denom = scores1[-1] - pre_reaction_base
        alpha = (scores1 - pre_reaction_base) / denom if denom > 0 else np.zeros_like(scores1)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha_source = "Baseline-Corrected fPC1 Coordinate"
    else:
        raise ValueError("Must provide either alpha_custom or fpca_res.")

    # Feature: Normalized temperature T / 1000
    X = (temps / 1000.0).reshape(-1, 1)
    y = alpha

    print(f"[*] Modeling phase transition on {len(temps)} in-situ scans (Source: {alpha_source})...")
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

    # Parameterize canonical transition model
    clean_info = distill_clean_rate_law(temps, dense_T, dense_pred, alpha)

    print(f"[+] Evolved Transition Model (R² = {r2:.4f}):")
    print(f"    Raw dCGP Genome: alpha(T) = {formula}")
    print(f"    Canonical Model: alpha(T) = {clean_info['formula_clean']} (R² = {clean_info['r2_emp']:.4f})")
    print(f"[+] Transition Midpoint: T_1/2 = {clean_info['t_half']:.0f} °C (Width ΔT = {clean_info['delta_t']:.0f} °C)")

    # Generate 2-panel Diagnostic Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Conversion Extent alpha(T)
    axes[0].scatter(temps, alpha, color="#d95f02", s=70, edgecolor="black", zorder=4, label=f"Measured {alpha_source}")
    axes[0].plot(dense_T, dense_pred, color="#1b9e77", lw=2.5, label=f"Malthus-GP dCGP ($R^2={r2:.4f}$)")
    axes[0].plot(dense_T, clean_info["clean_pred"], color="#2b83ba", lw=2.0, linestyle="--", label=f"Canonical Model ($R^2={clean_info['r2_emp']:.4f}$)")
    axes[0].set_title(r"A. Solid-State Conversion Progress $\alpha(T)$" + f"\nModel: $\\alpha(T) = \\frac{{1}}{{2}}[1 + \\tanh((T - {clean_info['t_half']:.0f}) / {clean_info['delta_t']:.0f})]$", fontsize=11, fontweight="bold", loc="left")
    axes[0].set_xlabel("Temperature (°C)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(r"Phase Conversion $\alpha(T) \in [0, 1]$", fontsize=11, fontweight="bold")
    axes[0].set_ylim(-0.02, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", frameon=True)

    # Panel B: Reaction Rate Derivative d(alpha)/dT
    axes[1].plot(dense_T, d_pred_dT * 1000.0, color="#7570b3", lw=2.5, label=r"Conversion Velocity $\frac{d\alpha}{dT}$")
    axes[1].axvline(t_peak_rate, color="red", linestyle="--", lw=1.5, label=f"Peak Velocity: $T_{{max}} = {t_peak_rate:.1f}^\\circ\\text{{C}}$")
    axes[1].scatter([t_peak_rate], [np.max(d_pred_dT) * 1000.0], color="red", s=80, zorder=5)
    axes[1].set_title(r"B. Analytical Transformation Velocity $\frac{d\alpha}{dT}$", fontsize=11, fontweight="bold", loc="left")
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
        f.write("================================================================================\n")
        f.write("MALTHUS-GP: SOLID-STATE PHASE TRANSITION MODELING (dCGP + JAX)\n")
        f.write("================================================================================\n")
        f.write(f"Input Phase Coordinate:          {alpha_source}\n")
        f.write(f"Canonical Model R² Fit:          {clean_info['r2_emp']:.5f}\n")
        f.write(f"Raw dCGP Genome R² Fit:          {r2:.5f}\n")
        f.write(f"Peak Rate Temperature (T_max):   {t_peak_rate:.1f} °C\n")
        f.write(f"Transition Midpoint (T_1/2):     {clean_info['t_half']:.1f} °C\n")
        f.write(f"Thermal Transition Width (ΔT):   {clean_info['delta_t']:.1f} °C\n\n")
        f.write("CANONICAL TRANSITION MODEL (PUBLICATION FORM):\n")
        f.write(f"  alpha(T) = {clean_info['formula_clean']}\n")
        f.write(f"  LaTeX:     ${clean_info['latex_formula']}$\n\n")
        f.write("RAW dCGP GENOME DAG (UNREDUCED FORM):\n")
        f.write(f"  alpha(T) = {formula}\n\n")
        f.write("Temperature (°C)   Measured alpha   Fitted alpha   dCGP alpha   Residual\n")
        f.write("--------------------------------------------------------------------------------\n")
        pred_pts = dcgp.predict(X)
        clean_pts = 0.5 * (1.0 + np.tanh((temps - clean_info['t_half']) / clean_info['delta_t']))
        for t_val, a_emp, a_clean, a_pred in zip(temps, alpha, clean_pts, pred_pts):
            f.write(f"{t_val:12.1f}   {a_emp:14.4f}   {a_clean:14.4f}   {a_pred:10.4f}   {abs(a_emp - a_clean):10.4f}\n")
        f.write("================================================================================\n")
        f.write("NOTE ON CHEMICAL KINETICS (ICTAC COMMITTEE STANDARDS):\n")
        f.write("This model parameterizes the non-isothermal cumulative conversion alpha(T) at\n")
        f.write("a single heating rate. True decoupling of the kinetic triplet (Ea, A, f(alpha))\n")
        f.write("requires multi-heating-rate isoconversional series (e.g. 5, 10, 15, 20 °C/min).\n")
        f.write("================================================================================\n")

    print(f"[+] Saved model plot to:    {plot_path}")
    print(f"[+] Saved model summary to: {summary_path}")

    return {
        "model": dcgp,
        "r2": r2,
        "formula": clean_info["formula_clean"],
        "raw_formula": formula,
        "clean_formula": clean_info["formula_clean"],
        "latex_formula": clean_info["latex_formula"],
        "latex_deriv": clean_info["latex_deriv"],
        "t_half": clean_info["t_half"],
        "delta_t": clean_info["delta_t"],
        "r2_clean": clean_info["r2_emp"],
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

    print(f"[5b/6] Deconvolving Rutile (110) Phase Evolution across Temperature...")
    rutile_res = fit_rutile_phase_evolution(t_eval, temps, raw_y, config.OUTPUT_DIR)
    print(f"      Saved: {rutile_res['plot_path']}")

    if run_gp and HAS_MALTHUS_GP:
        print(f"[6/6] Executing Phase Transition Modeling via Malthus-GP...")
        discover_kinetics_with_malthus_gp(temps, fpca_res, alpha_custom=rutile_res["alpha_rutile"], output_dir=config.OUTPUT_DIR)
    else:
        print("[6/6] Malthus-GP step skipped.")

    print("=" * 65)
    print("FDA Pipeline completed successfully!")
    print("=" * 65)

if __name__ == '__main__':
    run_fda_pipeline(run_gp=True)
