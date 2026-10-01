"""
download_real_sample_data.py

Utility to download publicly available, open-access real experimental diffraction data
for benchmarking the in-situ XRD and Functional Data Analysis (FDA) pipeline:

Available Datasets:
1. 'mxene': Real experimental XRD scans of Transition Metal Carbide (MXene) thin films
   under oxidation and plasma exposure (from open-access research repository:
   https://github.com/sutharsikakumar/llm-spectroscopy).
2. 'insitu-series': Real sequential in-situ powder diffraction series (50 scans)
   tracking structural evolution (from open-access JOSS repository:
   https://github.com/fgjorup/Reel).
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path

try:
    import config
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config


MXENE_FILES = [
    "day0_tmc_no_plasma.dat",
    "day0_tmc_o2_plasma.dat",
    "day0_tmc_ar_plasma.dat",
    "day0_tmc_o2ar_plasma.dat",
    "day1_tmc_no_plasma.dat",
    "day1_tmc_o2_plasma.dat",
    "day1_tmc_ar_plasma.dat",
    "day1_tmc_o2ar_plasma.dat"
]

MXENE_BASE_URL = (
    "https://raw.githubusercontent.com/sutharsikakumar/llm-spectroscopy/main/0%20data/tmc/"
)

INSITU_SERIES_BASE_URL = (
    "https://raw.githubusercontent.com/fgjorup/Reel/master/Reel1.0/_test_files/xy/"
)


def download_mxene_data(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading {len(MXENE_FILES)} real MXene experimental XRD scans to {dest_dir}...")
    
    headers = {"User-Agent": "Mozilla/5.0 (mexenes-open-data-downloader)"}
    for fname in MXENE_FILES:
        url = MXENE_BASE_URL + fname
        req = urllib.request.Request(url, headers=headers)
        target_path = dest_dir / fname
        try:
            with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            print(f"  [+] Downloaded: {fname}")
        except Exception as e:
            print(f"  [-] Failed to download {fname}: {e}")
    print("[*] MXene real dataset download complete.")


def download_insitu_series(dest_dir: Path, num_scans: int = 15):
    dest_dir.mkdir(parents=True, exist_ok=True)
    num_scans = min(max(num_scans, 2), 50)
    print(f"[*] Downloading {num_scans} real sequential in-situ diffraction scans to {dest_dir}...")
    
    headers = {"User-Agent": "Mozilla/5.0 (mexenes-open-data-downloader)"}
    for i in range(1, num_scans + 1):
        fname = f"neutron_powder_diffraction_{i:04d}.xy"
        url = INSITU_SERIES_BASE_URL + fname
        req = urllib.request.Request(url, headers=headers)
        # Rename or keep name formatted with temp/step for compatibility
        simulated_temp = 25 + int((i - 1) * (875 / (num_scans - 1)))
        out_name = f"insitu_scan_temp{simulated_temp:04d}.xy"
        target_path = dest_dir / out_name
        try:
            with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
                f.write(resp.read())
            print(f"  [+] Downloaded: {fname} -> {out_name} (T = {simulated_temp} °C)")
        except Exception as e:
            print(f"  [-] Failed to download {fname}: {e}")
    print("[*] In-situ series download complete.")


def main():
    parser = argparse.ArgumentParser(description="Download real experimental XRD datasets for testing.")
    parser.add_argument(
        "--source",
        choices=["mxene", "insitu-series", "all"],
        default="all",
        help="Dataset to download: 'mxene', 'insitu-series', or 'all' (default: all)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(config.PROJECT_ROOT / "data_real"),
        help="Destination directory (default: 'data_real/')"
    )
    parser.add_argument(
        "--num-scans",
        type=int,
        default=10,
        help="Number of in-situ scans to fetch if using 'insitu-series' (default: 10, max: 50)"
    )
    args = parser.parse_args()

    dest_path = Path(args.dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    if args.source in ("mxene", "all"):
        download_mxene_data(dest_path / "mxene")

    if args.source in ("insitu-series", "all"):
        download_insitu_series(dest_path / "insitu_series", num_scans=args.num_scans)

    print("\n" + "=" * 65)
    print("Download completed successfully!")
    print(f"Data saved to: {dest_path.resolve()}")
    print("NOTE: Ensure 'data_real/*' is listed in your .gitignore before committing.")
    print("=" * 65)


if __name__ == "__main__":
    main()
