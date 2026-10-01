"""
config.py

Central configuration for MXene in-situ temperature-resolved XRD analysis.
All file paths are dynamically resolved relative to the repository root.
No personal or absolute paths are hardcoded.
"""

import os
import re
from pathlib import Path

# --- 1. Directory & File Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("XRD_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("XRD_OUTPUT_DIR", PROJECT_ROOT / "output"))
OUTPUT_FILE_PREFIX = str(OUTPUT_DIR / "analysis")

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 2. Input Data Configuration ---
FILE_PATTERN = "*.xy"
COMMENT_CHAR = "#"
TEMP_PATTERN = re.compile(r"temp(\d+)", re.IGNORECASE)
TEMP_UNIT = "C"

# --- 3. Preprocessing Configuration ---
ENABLE_PREPROCESSING = True
REMOVE_SPIKES = True
SPIKE_THRESHOLD_SIGMA = 5.0

SUBTRACT_BASELINE = True
BASELINE_METHOD = "snip"             # Options: "snip" or "asls"
BASELINE_SNIP_ITERATIONS = 40        # Iteration clipping depth for SNIP

ENABLE_SAVGOL = True
SAVGOL_WINDOW = 9                   # Must be an odd integer
SAVGOL_POLYORDER = 2                # Polynomial order

SUBPIXEL_APEX = True                 # True: 3-point parabolic interpolation, False: raw discrete argmax

# --- 4. Angle Windows & Crop Limits (2theta in degrees) ---
TWO_THETA_CROP_MIN = 7.5
TWO_THETA_MIN = 9.0
TWO_THETA_MAX = 11.0

# Plot limits (None = auto-scale from data)
PLOT_2THETA_MIN = None
PLOT_2THETA_MAX = None
WATERFALL_OFFSET = 0.0
SHOW_PEAK_TRACK = True
CMAP_NAME = "viridis"

# --- 5. Animation & Presentation Settings ---
MAKE_GIF = True
SPLIT_TEMP = 500
GIF_CUMULATIVE = True
GIF_STEP = 3
GIF_FPS = 10
GIF_DPI = 120
GIF_HOLD_LAST_S = 1.5
GIF_SHARED_COLORS = False
GIF_LOOP = False
MAKE_MP4 = False
MAKE_PPTX = False

# --- 6. ICDD / Reference Phases ---
SHOW_REFS = True
LAMBDA_CU_KA1 = 1.540598
LABEL_MIN_I = 8
MIN_LABEL_SEP = 0.02
REF_T_C = 25.0

# --- 7. Peak Label Specifications ---
SHOW_PEAK_LABELS = True
LABEL_FONTSIZE = 12
LABELS_ON_ALL_SHOWN = False

# (index, 2theta min, 2theta max)
MXENE_LABELS = [
    ("002", 9.0, 11.0),
    ("100", 31.0, 33.0),
    ("103", 37.0, 39.0)
]
MXENE_LABEL_COLOR = "#8c1515"
MXENE_PRESENCE_SIGMA = 0.0

# (index, center 2theta, +/- window)
RUTILE_LABELS = [
    ("101", 36.085, 0.6),
    ("002", 62.740, 2.0),
    ("111", 41.225, 0.3)
]
RUTILE_LABEL_COLOR = "#0072B2"
RUTILE_PRESENCE_SIGMA = 3.0

# Reference Crystallographic Phases (PDF database entries)
PHASES = [
    {
        "name": "Ti2AlC0.5N0.5 (PDF 00-029-0095)",
        "color": "#d62728",
        "alpha": 0.0,
        "peaks": [
            (13.008, 39, 0, 0, 2),
            (34.021, 19, 1, 0, 0),
            (34.673, 6, 1, 0, 1),
            (39.545, 100, 1, 0, 3),
            (39.727, 18, 0, 0, 6),
            (43.428, 3, 1, 0, 4),
            (53.287, 14, 1, 0, 6),
            (60.897, 14, 1, 1, 0),
            (71.626, 2, 2, 0, 0),
            (71.992, 9, 1, 0, 9),
            (75.204, 11, 1, 1, 6)
        ]
    },
    {
        "name": "TiO2 Rutile (PDF 00-021-1276)",
        "color": "#0072B2",
        "alpha": 0.0,
        "peaks": [
            (27.446, 100, 1, 1, 0),
            (36.085, 50, 1, 0, 1),
            (39.187, 8, 2, 0, 0),
            (41.225, 25, 1, 1, 1),
            (44.050, 10, 2, 1, 0),
            (54.322, 60, 2, 1, 1),
            (56.640, 20, 2, 2, 0),
            (62.740, 10, 0, 0, 2),
            (64.038, 10, 3, 1, 0),
            (65.478, 2, 2, 2, 1),
            (69.008, 20, 3, 0, 1),
            (69.788, 12, 1, 1, 2),
            (72.408, 2, 3, 1, 1),
            (74.409, 1, 3, 2, 0),
            (76.508, 4, 2, 0, 2),
            (79.819, 2, 2, 1, 2)
        ]
    }
]
