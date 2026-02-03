#!/usr/bin/env python3
""" Installation helper script to a) ensure non-python dependency OpenMP is available and b) install the Python package.
If only the core actigraphy analysis is needed, add --core-only to skip the RBD prediction part.
Optionally install in development mode using: python install.py --dev
"""

import argparse
import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _parse_args():
    parser = argparse.ArgumentParser(description="Install the package and its dependencies.")
    parser.add_argument("-d", "--dev", action="store_true", help="Install in development mode.")
    parser.add_argument("-c", "--core-only", action="store_true",
                        help="Install only the core (skip RBDisco/OpenMP checks).")
    return parser.parse_args()


def _ensure_openmp():
    """ Ensure that an OpenMP runtime is available. If not, attempt to install it. """

    def __openmp_is_available():
        if sys.platform.startswith("darwin"):
            lib_names = [
                "libomp.dylib",
                "/opt/homebrew/opt/libomp/lib/libomp.dylib",
                "/usr/local/opt/libomp/lib/libomp.dylib",
                "/usr/local/lib/libomp.dylib",
            ]
        elif sys.platform.startswith("linux"):
            lib_names = ["libgomp.so.1", "libgomp.so"]
        elif sys.platform.startswith("win"):
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
        if sys.platform.startswith("darwin"):
            logging.info("Detected macOS. Will install libomp via Homebrew...")
            if shutil.which("brew") is None:
                logging.error("Homebrew not found. Please install Homebrew first and then run:\n"
                              "  brew install libomp\n"
                              "Alternative (conda): conda install -c conda-forge libomp")
                sys.exit(1)
            subprocess.run(["brew", "install", "libomp"], check=True)
        elif sys.platform.startswith("linux"):
            logging.info("Detected Linux. Installing libgomp via apt-get (if available)...")
            if shutil.which("apt-get"):
                subprocess.run(["sudo", "apt-get", "update", "-y"], check=True)
                subprocess.run(["sudo", "apt-get", "install", "-y", "libgomp1"], check=True)
            else:
                logging.error("apt-get not found. Please install your distro's GNU OpenMP runtime (e.g., libgomp) "
                              "using the appropriate package manager, then re-run this script.")
                sys.exit(1)
        elif sys.platform.startswith("win"):
            logging.info("Detected Windows. Please ensure you have the latest Visual C++ Build Tools installed.")
        else:
            logging.error("Unsupported OS. Please install an OpenMP runtime manually.")
            sys.exit(1)

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

        if not __openmp_is_available():
            logging.error("OpenMP runtime installation failed or is still unavailable.")
            sys.exit(1)
        else:
            logging.info("OpenMP runtime is now available after installation.")


def _resolve_repo_root() -> Path:
    """ Resolve the actitect repository root robustly, even if this code lives inside a git subtree.
    Strategy:
      1) Walk upwards from this install.py location and find a directory that contains:
         - libs/actitect/pyproject.toml  (monorepo layout)
         OR
         - pyproject.toml                (single-package layout)
      2) If not found, error out with a helpful message."""
    here = Path(__file__).resolve().parent

    # Sentinel patterns that define a "repo root" for this installer
    sentinels = ('pyproject.toml', os.path.join('libs', 'actitect', 'pyproject.toml'),)

    for p in [here, *here.parents]:
        if any((p / s).exists() for s in sentinels):
            return p

    logging.error("Could not locate actitect repository root from %s. "
        "Expected to find pyproject.toml or libs/actitect/pyproject.toml in a parent directory.",
        here)
    sys.exit(1)


def _set_experiment_root(repo_root: Path):
    """Experiment root which will be used for default data paths."""
    _exp_root = repo_root.parent.resolve()
    cfg_dir = repo_root / 'libs' / 'actitect' / 'src' / 'actitect' / 'config'
    local_root_json = cfg_dir / 'experiment_root.local.json'  # untracked local sidecar
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        with open(local_root_json, 'w') as f:
            json.dump({'EXPERIMENT_ROOT': str(_exp_root)}, f, indent=4)
        logging.info('Wrote experiment root to %s', local_root_json)
        return
    except Exception:
        logging.warning('Could not write experiment_root.local.json. Skipping.')


def _pip_install(path: Path, dev: bool):
    cmd = [sys.executable, '-m', 'pip', 'install']
    if dev:
        cmd += ['-e', str(path)]
    else:
        cmd += [str(path)]
    logging.info("Installing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _prepare_pkg_docs(repo_root: Path):
    """Copy main README and license to each libs root dir. PEP 517 build regulations."""
    src_readme = repo_root / "README.md"
    src_license = repo_root / "LICENSE"
    src_citation = repo_root / "CITATION.cff"
    pkg_dirs = [repo_root / "libs" / "actitect", repo_root / "libs" / "actitect-rbdisco"]
    for pkg in pkg_dirs:
        if pkg.exists():
            if src_readme.exists():
                shutil.copy2(src_readme, pkg / "README.md")
            if src_license.exists():
                shutil.copy2(src_license, pkg / "LICENSE")
            if src_citation.exists():
                shutil.copy2(src_citation, pkg / "CITATION.cff")


def main():
    args = _parse_args()

    repo_root = _resolve_repo_root()
    logging.info("Repository root: %s", repo_root)

    # Accept root or monorepo pyproject presence
    if not ((repo_root / 'pyproject.toml').exists() or (repo_root / 'libs' / 'actitect' / 'pyproject.toml').exists()):
        logging.error('pyproject.toml not found under resolved repository root.')
        sys.exit(1)

    # 1: If installing with RBDisco (default), ensure OpenMP is available
    if not args.core_only:
        _ensure_openmp()
    else:
        logging.info('--core-only specified: skipping OpenMP checks (RBDisco not installed).')

    # 2: Dynamically set the experiment root path
    _set_experiment_root(repo_root)

    # 3: ensure README/LICENCE present for isolated builds
    _prepare_pkg_docs(repo_root)

    # 4: Install the Python package(s) using pip.
    try:
        core_pkg = repo_root / 'libs' / 'actitect'
        rbdisco_pkg = repo_root / 'libs' / 'actitect-rbdisco'
        if args.core_only:  # core only: install the core actigraphy dist
            _pip_install(core_pkg, args.dev)
        else:  # with RBDisco (default): install core, then plugin
            _pip_install(core_pkg, args.dev)
            _pip_install(rbdisco_pkg, args.dev)

    except subprocess.CalledProcessError as e:
        logging.error("Error during pip installation: %s", e)
        sys.exit(1)

    # 5: Post-install verification (separate try/except)
    try:
        if args.core_only:
            subprocess.run([sys.executable, "-c", "import actitect"], check=True)
        else:
            subprocess.run(
                [sys.executable, "-c", "import actitect, actitect.rbdisco; from xgboost import XGBClassifier"],
                check=True
            )
        logging.info("Installation successful!")
    except subprocess.CalledProcessError:
        logging.error("Quality-control check failed: unable to import expected modules.")
        sys.exit(1)


if __name__ == "__main__":
    main()
