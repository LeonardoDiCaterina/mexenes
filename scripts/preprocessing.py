"""
preprocessing.py

Advanced, mathematically rigorous preprocessing pipeline for in-situ XRD scans.
Provides:
1. Anscombe Variance-Stabilizing Transformation (Poisson -> Gaussian).
2. Spatial Outlier / Cosmic Ray Spike Removal.
3. Savitzky-Golay Peak-Preserving Smoothing.
4. Baseline / Background Subtraction (SNIP and AsLS).
5. Continuous Sub-Pixel Peak Apex Interpolation.
"""

import numpy as np
import scipy.signal

def anscombe_transform(y):
    """
    Anscombe Variance-Stabilizing Transformation.
    Converts heteroscedastic Poisson counting noise (sigma ~ sqrt(I))
    into homoscedastic unit Gaussian noise (sigma ~ 1).
    """
    return 2.0 * np.sqrt(np.maximum(y, 0.0) + 3.0 / 8.0)

def inverse_anscombe(y_prime):
    """
    Algebraic exact inverse of the Anscombe transformation back to counts.
    """
    return np.maximum((y_prime / 2.0) ** 2 - 3.0 / 8.0, 0.0)

def remove_spikes(intensity, threshold_sigma=5.0):
    """
    Identifies and removes isolated single-point spikes (cosmic rays or hot pixels)
    using a median difference test with difference-based MAD noise scaling.
    """
    med = scipy.signal.medfilt(intensity, kernel_size=5)
    diff = intensity - med
    d_diff = np.diff(diff)
    mad = 1.4826 * np.median(np.abs(d_diff - np.median(d_diff))) / np.sqrt(2.0)
    is_spike = diff > (threshold_sigma * mad)
    
    cleaned = intensity.copy()
    cleaned[is_spike] = med[is_spike]
    return cleaned

def smooth_savgol(intensity, window_length=9, polyorder=2):
    """
    Savitzky-Golay filtering. Fits local low-degree polynomials across a sliding window.
    Attenuates high-frequency noise while strictly preserving peak apex height,
    area, and FWHM (unlike moving average filters which flatten and broaden peaks).
    """
    if len(intensity) <= window_length:
        return intensity
    if window_length % 2 == 0:
        window_length += 1
    return scipy.signal.savgol_filter(intensity, window_length=window_length, polyorder=polyorder)

def baseline_snip(intensity, num_iterations=40):
    """
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm.
    Iteratively clips positive peak reflections to extract the underlying
    amorphous / background envelope without human intervention.
    """
    z = np.copy(intensity)
    for p in range(1, num_iterations + 1):
        left = np.pad(z[:-p], (p, 0), mode='edge')
        right = np.pad(z[p:], (0, p), mode='edge')
        z = np.minimum(z, 0.5 * (left + right))
    return z

def baseline_asls(intensity, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares (AsLS) baseline estimator (Eilers & Boelens, 2005).
    Penalizes positive residuals (peaks) with weight p and negative residuals with 1-p.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    L = len(intensity)
    D = sp.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)
    for _ in range(niter):
        W = sp.diags(w, 0, shape=(L, L))
        Z = W + lam * D.dot(D.transpose())
        z = spla.spsolve(Z, w * intensity)
        w = p * (intensity > z) + (1 - p) * (intensity <= z)
    return z

def peak_apex_subpixel(two_theta, intensity, lo, hi):
    """
    Locates the true continuous peak apex with sub-step precision using a
    3-point parabolic vertex interpolation around the local discrete maximum.
    Corrects the quantization error of raw np.argmax.
    """
    mask = (two_theta >= lo) & (two_theta <= hi)
    if not mask.any():
        return np.nan, 0.0
    
    tt_sub = two_theta[mask]
    inten_sub = intensity[mask]
    idx = np.argmax(inten_sub)
    
    # Boundary guard: fallback to discrete coordinate
    if idx == 0 or idx == len(inten_sub) - 1:
        return tt_sub[idx], inten_sub[idx]
    
    y1, y2, y3 = inten_sub[idx - 1], inten_sub[idx], inten_sub[idx + 1]
    denom = y1 - 2.0 * y2 + y3
    if denom == 0:
        return tt_sub[idx], y2
    
    d_tt = tt_sub[1] - tt_sub[0]
    delta = 0.5 * (y1 - y3) / denom
    delta = np.clip(delta, -1.0, 1.0)
    
    sub_center = tt_sub[idx] + delta * d_tt
    sub_height = y2 - 0.25 * (y1 - y3) * delta
    return sub_center, sub_height

def preprocess_scan(two_theta, intensity,
                    remove_spike=True,
                    smooth=True,
                    subtract_bg=True,
                    bg_method="snip",
                    savgol_win=9,
                    savgol_poly=2,
                    snip_iters=40):
    """
    Full integrated preprocessing pipeline returning:
    (cleaned_intensity, baseline, diagnostic_dict)
    """
    current = intensity.copy()
    
    # 1. Spike removal
    if remove_spike:
        current = remove_spikes(current)
    
    # 2. Baseline extraction
    if subtract_bg:
        if bg_method.lower() == "snip":
            baseline = baseline_snip(current, num_iterations=snip_iters)
        else:
            baseline = baseline_asls(current)
        net_intensity = np.maximum(current - baseline, 0.0)
    else:
        baseline = np.zeros_like(current)
        net_intensity = current
        
    # 3. Savitzky-Golay smoothing on net intensity
    if smooth:
        smoothed_net = smooth_savgol(net_intensity, window_length=savgol_win, polyorder=savgol_poly)
    else:
        smoothed_net = net_intensity
        
    diag = {
        "raw": intensity,
        "despiked": current,
        "baseline": baseline,
        "net": net_intensity,
        "final": smoothed_net
    }
    return smoothed_net, baseline, diag
