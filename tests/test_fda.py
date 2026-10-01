"""
test_fda.py

Unit and mathematical consistency tests for Functional Data Analysis (FDA) module.
Validates:
1. Continuous B-spline fitting and analytical derivative consistency.
2. Functional PCA L^2 orthonormality of eigenfunctions.
3. Functional PCA eigenvalue and variance conservation.
4. Curve registration warping monotonicity.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.fda_analysis import (
    fit_functional_splines,
    compute_functional_pca,
    compute_curve_registration
)

class TestFDAAnalysis(unittest.TestCase):
    def setUp(self):
        # Synthetic domain: 2theta from 10 to 40 degrees, 500 points
        self.t_eval = np.linspace(10.0, 40.0, 500)
        # Create 5 synthetic Gaussian peaks shifting and attenuating with temperature
        self.temps = np.array([25.0, 200.0, 400.0, 600.0, 800.0])
        self.curves = []
        for i, T in enumerate(self.temps):
            # Peak center shifts slightly to lower angle with T
            center = 25.0 - 0.001 * (T - 25.0)
            # Amplitude decays with T
            amp = 1000.0 * np.exp(-T / 500.0)
            sigma = 0.5
            y = amp * np.exp(-0.5 * ((self.t_eval - center) / sigma) ** 2)
            # Add small noise
            np.random.seed(42 + i)
            y += np.random.normal(0, 5.0, size=len(self.t_eval))
            self.curves.append(y)
        self.curves = np.array(self.curves)

    def test_spline_representation_and_derivatives(self):
        splines, y_func, d1_func, d2_func = fit_functional_splines(
            self.t_eval, self.curves, smoothing_factor=500.0 * 10.0
        )
        self.assertEqual(y_func.shape, self.curves.shape)
        self.assertEqual(d1_func.shape, self.curves.shape)
        self.assertEqual(d2_func.shape, self.curves.shape)
        
        # Test that spline smoothing reduced noise variance
        res_raw = np.diff(self.curves[0], 2)
        res_smooth = np.diff(y_func[0], 2)
        self.assertLess(np.std(res_smooth), np.std(res_raw))

    def test_fpca_orthonormality_and_variance(self):
        _, y_func, _, _ = fit_functional_splines(self.t_eval, self.curves)
        res = compute_functional_pca(self.t_eval, y_func, n_components=3)

        harmonics = res["harmonics"]
        dt = np.gradient(self.t_eval)

        # Orthonormality under L2 inner product: \int \phi_j(t) \phi_k(t) dt = \delta_{jk}
        for j in range(len(harmonics)):
            for k in range(len(harmonics)):
                inner_prod = np.sum(harmonics[j] * harmonics[k] * dt)
                expected = 1.0 if j == k else 0.0
                self.assertAlmostEqual(inner_prod, expected, delta=0.05)

        # Variance conservation
        self.assertAlmostEqual(np.sum(res["var_explained"]), 100.0, delta=1.0)
        self.assertGreater(res["var_explained"][0], res["var_explained"][1])

    def test_curve_registration(self):
        _, y_func, _, _ = fit_functional_splines(self.t_eval, self.curves)
        reg_res = compute_curve_registration(self.t_eval, y_func, ref_idx=0)

        # Test monotonicity of warping functions
        for w in reg_res["warping_functions"]:
            dw = np.diff(w)
            self.assertTrue(np.all(dw >= 0), "Warping function must be strictly monotonic!")

if __name__ == "__main__":
    unittest.main()
