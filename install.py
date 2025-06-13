#!/usr/bin/env python3
""" Installation helper script to a) ensure non-python dependency OpenMP is available and b) install the Python package.
Optionally install in development mode using python install.py --dev"""

import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _parse_args():
    parser = argparse.ArgumentParser(description="Install the package and its dependencies.")
    parser.add_argument("--dev", action="store_true", help="Install in development mode.")
    return parser.parse_args()


def _ensure_openmp():
    """ Ensure that an OpenMP runtime is available. If not, attempt to install it. """

    def __openmp_is_available():
        """Check for common OpenMP runtime libraries."""
        if sys.platform.startswith("darwin"):
            # macOS: typically uses libomp
            lib_names = [
                "libomp.dylib",
                "/opt/homebrew/opt/libomp/lib/libomp.dylib",
                "/usr/local/lib/libomp.dylib"
            ]
        elif sys.platform.startswith("linux"):
            # Linux: usually uses GNU OpenMP (libgomp)
            lib_names = ["libgomp.so.1", "libgomp.so"]
        elif sys.platform.startswith("win"):
            # Windows: OpenMP is usually provided by the MSVC runtime.
            lib_names = ["vcomp140.dll", "vcomp140_1.dll"]
        else:
            lib_names = []

        for name in lib_names:
            try:
                ctypes.CDLL(name)
                logging.info(f"Found OpenMP runtime: {name}")
                return True
            except Exception:
                continue
        return False

    def __install_openmp():
        """Install the OpenMP runtime using the appropriate system package manager."""
        if sys.platform.startswith("darwin"):
            logging.info("Detected macOS. Installing libomp via Homebrew...")
            subprocess.run(["brew", "install", "libomp"], check=True)
        elif sys.platform.startswith("linux"):
            logging.info("Detected Linux. Installing libgomp via apt-get...")
            subprocess.run(["sudo", "apt-get", "install", "-y", "libgomp1"], check=True)
        elif sys.platform.startswith("win"):
            logging.info("Detected Windows. Please ensure you have the latest Visual C++ Build Tools installed.")
        else:
            logging.error("Unsupported OS. Please install an OpenMP runtime manually.")

    logging.info("Checking for OpenMP runtime...")
    if __openmp_is_available():
        logging.info("OpenMP runtime is available.")
    else:
        logging.info("OpenMP runtime is not available. Attempting installation...")
        try:
            __install_openmp()
        except Exception as e:
            logging.error("Error during OpenMP installation: %s", e)
            logging.error("Please install OpenMP manually and re-run this script.")
            sys.exit(1)

        if not __openmp_is_available():  # re-ensure installation succeeded
            logging.error("OpenMP runtime installation failed or is still unavailable.")
            sys.exit(1)
        else:
            logging.info("OpenMP runtime is now available after installation.")


def _set_experiment_root():
    _exp_root = Path(__file__).parents[1].resolve()
    with open(Path(__file__).parent.joinpath('src/aktiRBD/config/experiment_root.json'), 'w') as f:
        json.dump({"EXPERIMENT_ROOT": str(_exp_root)}, f, indent=4)


def main():
    args = _parse_args()

    # Ensure we're in the right directory by checking for pyproject.toml.
    if not os.path.isfile("pyproject.toml"):
        logging.error("pyproject.toml not found. Please run this script from the project root directory.")
        sys.exit(1)

    # 1: XGBoost depends on OpenMP runtime for multiprocessing, ensure it's available
    _ensure_openmp()

    # 2: Dynamically set the experiment root path
    _set_experiment_root()

    #  3: Install the Python package using pip.
    pip_command = ["pip", "install", "."]
    if args.dev:
        pip_command = ["pip", "install", "-e", ".[dev]"]

    try:
        logging.info("Installing Python package using pip command: %s", " ".join(pip_command))
        subprocess.run(pip_command, check=True)

        logging.info("Verifying package installation...")
        try:
            subprocess.run(["python", "-c", "import aktiRBD", "from xgboost import XGBClassifier"], check=True)
            logging.info("Installation successful!")
        except subprocess.CalledProcessError:
            logging.error("Quality-control check failed: aktiRBD could not be imported.")
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        logging.error("Error during pip installation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
