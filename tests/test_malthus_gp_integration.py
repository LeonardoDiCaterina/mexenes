"""
test_malthus_gp_integration.py

Unit tests for pairing Functional Data Analysis (FDA) with Malthus-GP
dCGP regression for solid-state reaction kinetics discovery.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from malthus_gp.learn import DCGPRegressor
    HAS_MALTHUS_GP = True
except ImportError:
    HAS_MALTHUS_GP = False


class TestMalthusGPIntegration(unittest.TestCase):
    def setUp(self):
        # Synthetic kinetics data: Sigmoidal conversion alpha(T) from 25 C to 900 C
        self.temps = np.linspace(25.0, 900.0, 10)
        # Synthetic Avrami curve: alpha(T) = 1 - exp(- ((T - 200)/300)^2) for T > 200
        t_shift = np.maximum(self.temps - 200.0, 0.0) / 350.0
        self.alpha = 1.0 - np.exp(-(t_shift ** 2.2))
        self.alpha = np.clip(self.alpha, 0.0, 1.0)

    @unittest.skipUnless(HAS_MALTHUS_GP, "Malthus-GP package not installed")
    def test_dcgp_kinetics_fit(self):
        X = (self.temps / 1000.0).reshape(-1, 1)
        y = self.alpha

        reg = DCGPRegressor(
            n_rows=1,
            n_cols=15,
            pop_size=4,
            max_generations=40,
            lsmf_epochs=5,
            learning_rate=0.03,
            random_state=42
        )
        reg.fit(X, y)
        r2 = reg.score(X, y)
        self.assertTrue(np.isfinite(r2))
        self.assertGreater(r2, 0.70, "dCGP should capture sigmoidal conversion trend!")

        # Test formula export
        formula = reg.to_formula()
        self.assertIsInstance(formula, str)
        self.assertTrue(len(formula) > 0)

        # Test continuous extrapolation
        dense_x = np.linspace(0.025, 0.900, 50).reshape(-1, 1)
        pred = reg.predict(dense_x)
        self.assertEqual(pred.shape, (50,))
        self.assertTrue(np.all(np.isfinite(pred)))


if __name__ == "__main__":
    unittest.main()
