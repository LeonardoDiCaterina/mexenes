import unittest
import glob
import os
import sys
from pathlib import Path
import numpy as np

# Load central config
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
import config

class TestXRDDataIntegrity(unittest.TestCase):
    def setUp(self):
        self.data_dir = config.DATA_DIR
        self.files = sorted(glob.glob(os.path.join(self.data_dir, config.FILE_PATTERN)))
        self.temp_pattern = config.TEMP_PATTERN

    def test_file_count(self):
        self.assertGreaterEqual(len(self.files), 1, f"No files matching '{config.FILE_PATTERN}' in {self.data_dir}")

    def test_temperature_extraction(self):
        temps = []
        for f in self.files:
            match = self.temp_pattern.search(os.path.basename(f))
            self.assertIsNotNone(match, f"Filename {f} does not match temp regex {self.temp_pattern.pattern}")
            temps.append(int(match.group(1)))
        self.assertEqual(temps, sorted(temps), "Temperatures are not monotonically increasing")

    def test_data_shape_and_range(self):
        for f in self.files:
            data = np.loadtxt(f, comments=config.COMMENT_CHAR, ndmin=2)
            self.assertGreaterEqual(data.shape[1], 2, f"{f} has fewer than 2 columns")
            self.assertGreaterEqual(data.shape[0], 100, f"{f} has fewer than 100 data points")
            
            tt = data[:, 0]
            self.assertLessEqual(tt.min(), config.TWO_THETA_CROP_MIN, f"2theta min should be <= {config.TWO_THETA_CROP_MIN}")
            self.assertGreaterEqual(tt.max(), 60.0, "2theta max should be >= 60.0 deg")

    def test_mxene_peak_presence_at_low_temp(self):
        first_file = self.files[0]
        data = np.loadtxt(first_file, comments=config.COMMENT_CHAR, ndmin=2)
        tt, inten = data[:, 0], data[:, 1]
        mask = (tt >= config.TWO_THETA_MIN) & (tt <= config.TWO_THETA_MAX)
        self.assertTrue(mask.any(), f"No points found in [{config.TWO_THETA_MIN}, {config.TWO_THETA_MAX}] deg range")
        peak_tt = tt[mask][np.argmax(inten[mask])]
        self.assertAlmostEqual(peak_tt, 9.8, delta=0.5, msg="MXene (002) peak at room temp should be near 9.8 deg")

if __name__ == "__main__":
    unittest.main()
