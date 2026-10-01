# Mexenes: In-Situ Temperature-Resolved XRD Analysis Pipeline

An automated, cross-platform analysis pipeline and benchmark suite for in-situ temperature-resolved X-ray Powder Diffraction (XRD). 

Designed for tracking structural phase transitions, lattice thermal expansion, and peak deconvolution in MXene and MAX-phase materials (e.g. $\text{Ti}_2\text{AlC}_{0.5}\text{N}_{0.5}$ decomposing and oxidizing into rutile $\text{TiO}_2$).

---

## Table of Contents
1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Preparing Your Data](#2-preparing-your-data)
3. [How to Run the Analysis](#3-how-to-run-the-analysis)
   - [Option A: Terminal / Command Line (Fastest)](#option-a-terminal--command-line-fastest)
   - [Option B: Interactive Jupyter Notebook](#option-b-interactive-jupyter-notebook)
   - [Option C: Evolutionary (GA) Peak Fitting Demo](#option-c-evolutionary-ga-peak-fitting-demo)
   - [Option D: Functional Data Analysis (FDA) Pipeline](#option-d-functional-data-analysis-fda-pipeline)
   - [Option E: Generating Synthetic Demo Data](#option-e-generating-synthetic-demo-data)
4. [Output Files Explained](#4-output-files-explained)
5. [Customizing Configuration (`config.py`)](#5-customizing-configuration-configpy)
6. [Troubleshooting & FAQ](#6-troubleshooting--faq)

---

## 1. Prerequisites & Installation

### Step 1.1: Clone the Repository
Open your terminal (macOS/Linux) or Command Prompt / PowerShell (Windows) and run:
```bash
git clone https://github.com/LeonardoDiCaterina/mexenes.git
cd mexenes
```

### Step 1.2: Set Up a Python Environment (Recommended)
Python 3.8 or newer is required. Setting up a virtual environment ensures clean dependencies:

* **On macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **On Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **On Windows (Command Prompt):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### Step 1.3: Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: If you want to export `.mp4` video animations or PowerPoint `.pptx` presentations, run: `pip install "imageio[ffmpeg]" python-pptx`)*

---

## 2. Preparing Your Data

1. Place your experimental diffraction scan files directly into the **`data/`** directory.
   *(The `data/` folder is kept clean in Git. If you want to test the pipeline with synthetic demo data first, run: `python scripts/generate_synthetic_xrd.py`).*

2. **File Requirements**:
   * **Extension**: Must be `.xy` (e.g., `scan01.xy`).
   * **Columns**: At least 2 columns separated by spaces, tabs, or commas:
     * **Column 1**: $2\theta$ diffraction angle (in degrees).
     * **Column 2**: Measured intensity (counts).
   * **Comments**: Any header lines starting with `#` are automatically ignored.

3. **Filename Convention (Important)**:
   The temperature must be present in the filename preceded by `temp` (case-insensitive). 
   
   Examples of valid filenames:
   * `sample_temp0025.xy` $\rightarrow$ parsed as $25\,^\circ\text{C}$
   * `ramp_temp100.xy` $\rightarrow$ parsed as $100\,^\circ\text{C}$
   * `MXene_S1_Temp0500_air.xy` $\rightarrow$ parsed as $500\,^\circ\text{C}$

---

## 3. How to Run the Analysis

### Option A: Terminal / Command Line (Fastest)

Run the automated standalone runner:
```bash
python scripts/run_analysis.py
```

**What it does:**
1. Automatically scans the `data/` folder and orders files by temperature.
2. Crops low-angle air scatter ($< 7.5^\circ$).
3. Locates the primary peak position in the target window ($9.0^\circ$ to $11.0^\circ$).
4. Generates:
   * A summary text table (`output/analysis_peak_positions.txt`).
   * A 1D overlay plot colored by temperature (`output/analysis_overlay.png`).
   * A 2D intensity heatmap with the peak track trajectory (`output/analysis_heatmap.png`).

---

### Option B: Interactive Jupyter Notebook

If you prefer an interactive notebook to inspect curves, plot reference ICDD lines, and create animated GIFs:

1. Launch Jupyter or open the project folder in VS Code / Antigravity IDE:
   ```bash
   jupyter lab
   # or: jupyter notebook
   ```
2. Open [`insitu_xrd_analysis.ipynb`](insitu_xrd_analysis.ipynb).
3. Select your Python kernel.
4. Run all cells (`Cell` $\rightarrow$ `Run All`, or `Shift + Enter` cell by cell).

*The notebook automatically imports all path and angle settings from `config.py` with zero manual path edits needed.*

---

### Option C: Evolutionary (GA) Peak Fitting Demo

To see how an Evolutionary / Genetic Algorithm fits the true continuous physical profile (Pseudo-Voigt) instead of a naive discrete maximum (`np.argmax`):
```bash
python scripts/ga_peak_fit_demo.py
```
This script will:
* Optimize the peak center ($\theta_0$), Full Width at Half Maximum (FWHM), amplitude, and Lorentzian/Gaussian ratio ($\eta$) using differential evolution.
* Print the comparison between naive `argmax` and the evolutionary fit.
* Save a residual and fit visualization to `output/ga_fit_demo.png`.

---

### Option D: Generating Synthetic Demo Data

If you want to simulate 10 in-situ temperature scans ($25\,^\circ\text{C}$ to $900\,^\circ\text{C}$) to test the pipeline before using real experimental data:
```bash
python scripts/generate_synthetic_xrd.py --num-scans 10
```

---

## 4. Output Files Explained

All generated outputs are saved to the **`output/`** directory:

| Output File | Description |
| :--- | :--- |
| `analysis_peak_positions.txt` | Tabulated 2-column data: `temperature (°C)` vs `peak_position_2theta (deg)`. |
| `analysis_overlay.png` | Stacked 1D diffraction patterns colored continuously by temperature using the `viridis` colormap. |
| `analysis_heatmap.png` | 2D color contour map showing $2\theta$ on the x-axis, temperature on the y-axis, and the red dashed peak tracking line. |
| `analysis_evolution_*.gif` | *(If enabled in notebook)* Animated GIFs showing the real-time evolution of the diffractograms during heating. |
| `ga_fit_demo.png` | Detailed Pseudo-Voigt peak fit with sub-step center estimation and residual difference plot. |
| `fda_splines_derivatives.png` | 3-panel continuous B-spline curves, analytical 1st derivative ($x'$, peak apices), and 2nd derivative ($x''$, curvature). |
| `fda_fpca_modes.png` | 4-panel fPCA results: mean function $\mu(2\theta)$, continuous functional harmonics $\phi_j(2\theta)$, scree variance plot, and score trajectories $\xi_j(T)$ vs temperature. |
| `fda_registration.png` | 3-panel curve registration: unregistered scans, pure thermal lattice strain field $w_i(2\theta)$, and registered pure-amplitude phase conversion curves $\tilde{x}_i(2\theta)$. |
| `fda_scores_summary.txt` | Tabulated functional PCA scores $\xi_1(T), \xi_2(T)$ and thermal strain metrics vs temperature. |

---

## 5. Customizing Configuration (`config.py`)

All parameters are centralized in [`config.py`](config.py). You never have to modify paths in multiple scripts.

### Common Parameters to Adjust:
* **Primary Peak Window**:
  ```python
  TWO_THETA_MIN = 9.0    # Lower boundary for peak search (deg)
  TWO_THETA_MAX = 11.0   # Upper boundary for peak search (deg)
  ```
* **Low-Angle Crop**:
  ```python
  TWO_THETA_CROP_MIN = 7.5  # Discards beamstop / air-scatter below this angle
  ```
* **Animation Toggles**:
  ```python
  MAKE_GIF = True        # Build animated GIFs
  SPLIT_TEMP = 500       # Temperature dividing Part 1 (RT to 500 °C) and Part 2 (500 °C to end)
  MAKE_MP4 = False       # Set True to export MP4 (requires ffmpeg)
  MAKE_PPTX = False      # Set True to build presentation slides (requires python-pptx)
  ```
* **Reference Phases**:
  Edit the `PHASES` list in `config.py` to add or update $(hkl)$ Bragg reflections from ICDD/PDF cards.

---

## 6. Troubleshooting & FAQ

#### Q: `ERROR: No files matching '*.xy' found in .../data`
* **Fix**: Place your `.xy` files into `data/`, or run `python scripts/generate_synthetic_xrd.py` to create demo files.

#### Q: `[filename] warning: could not extract temperature from filename`
* **Fix**: Ensure the filename contains `temp` followed by numbers. For example, rename `scan_room.xy` to `scan_temp0025.xy`.

#### Q: How do I run the unit tests?
* Run `python scripts/generate_synthetic_xrd.py` to generate the demo files, then run:
  ```bash
  python -m unittest discover tests
  ```

#### Q: Can I specify custom input/output folders outside this repository?
* **Yes!** Set environment variables before running:
  ```bash
  export XRD_DATA_DIR="/path/to/my/external/xy_files"
  export XRD_OUTPUT_DIR="/path/to/my/custom_output"
  python scripts/run_analysis.py
  ```
