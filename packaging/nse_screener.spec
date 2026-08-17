# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec: builds BOTH executables into one dist folder.

    pyinstaller packaging/nse_screener.spec --clean --noconfirm

See BUILD_EXE.md for why the dashboard needs one-dir rather than one-file, and
why the copy_metadata calls below are not optional.

VERIFIED: built and run with PyInstaller 6.22 on Windows / Python 3.11. Both
executables start and work. The one real trap it encodes is the bundle path of
the dashboard script - see the DATAS comment below.
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent          # noqa: F821 - SPECPATH is injected
SRC = ROOT / "src"

# collect_submodules() IMPORTS the package to walk it, and it runs here at
# spec-parse time - before Analysis(pathex=...) has any effect. Without this
# line it cannot find nse_screener, returns almost nothing, and does so SILENTLY:
# no warning, no error, just a dashboard exe missing the whole package.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# --------------------------------------------------------------------------
# Streamlit needs its static frontend AND its distribution metadata. It calls
# importlib.metadata.version() at import time, so a build without metadata dies
# with PackageNotFoundError before printing anything useful.
# --------------------------------------------------------------------------
st_datas, st_binaries, st_hidden = collect_all("streamlit")

metadata = []
for pkg in ("streamlit", "pandas", "numpy", "scikit-learn", "altair",
            "pyarrow", "joblib", "packaging"):
    try:
        metadata += copy_metadata(pkg)
    except Exception:                  # noqa: BLE001 - optional package
        pass

# scikit-learn imports these dynamically when unpickling an estimator, so static
# analysis cannot see them. Without them the exe runs until it loads the model.
SKLEARN_HIDDEN = [
    "sklearn.ensemble._forest",
    "sklearn.tree._tree",
    "sklearn.tree._utils",
    "sklearn.tree._splitter",
    "sklearn.tree._criterion",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "sklearn.metrics._pairwise_distances_reduction._datasets_pair",
    "sklearn.metrics._pairwise_distances_reduction._middle_term_computer",
]

# EVERY nse_screener submodule, collected explicitly.
#
# This is not belt-and-braces, it is load-bearing. The dashboard entry point
# imports only streamlit - Streamlit locates and execs app.py by PATH at runtime,
# so PyInstaller's static analysis never sees `nse_screener.backtest` and friends
# and leaves them out of that executable's archive. The CLI exe works because its
# entry script imports the package directly, which is exactly what makes the bug
# so confusing: same spec, same Analysis settings, one exe works and one does not.
#
# Symptom if this line is removed:
#     ModuleNotFoundError: No module named 'nse_screener.backtest'
# raised only by the dashboard, only at runtime, only when a page is rendered.
NSE_MODULES = collect_submodules("nse_screener")

HIDDEN = st_hidden + SKLEARN_HIDDEN + NSE_MODULES + [
    "pandas._libs.tslibs.base",
    "sqlite3",
]

# Optional broker SDKs pull in a large tree. Remove from excludes and add the
# matching copy_metadata line if you ship a broker-enabled build.
EXCLUDES = ["matplotlib", "tkinter", "PyQt5", "PySide2", "notebook",
            "fyers_apiv3", "SmartApi", "xgboost", "shap"]

DATAS = st_datas + metadata + [
    # Bundled under a NEUTRAL directory. Placing it at "nse_screener/dashboard"
    # makes that folder shadow the real frozen package and breaks every
    # submodule import inside the dashboard script.
    (str(SRC / "nse_screener" / "dashboard" / "app.py"), "nse_dashboard"),
    (str(ROOT / ".env.example"), "."),
]


# ==========================================================================
# 1. CLI
# ==========================================================================
cli_a = Analysis(
    [str(ROOT / "scripts" / "run_live.py")],
    pathex=[str(SRC)],
    binaries=st_binaries,
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
cli_pyz = PYZ(cli_a.pure)              # noqa: F821
cli_exe = EXE(                          # noqa: F821
    cli_pyz,
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name="nse-screener",
    console=True,
    debug=False,
    strip=False,
    upx=False,                          # UPX raises AV false positives
)

# ==========================================================================
# 2. Dashboard launcher
# ==========================================================================
dash_a = Analysis(
    [str(ROOT / "packaging" / "dashboard_entry.py")],
    pathex=[str(SRC)],
    binaries=st_binaries,
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
dash_pyz = PYZ(dash_a.pure)             # noqa: F821
dash_exe = EXE(                          # noqa: F821
    dash_pyz,
    dash_a.scripts,
    [],
    exclude_binaries=True,
    name="nse-screener-dashboard",
    console=True,
    debug=False,
    strip=False,
    upx=False,
)

# ==========================================================================
# One COLLECT for both, so they share the runtime tree instead of duplicating
# ~250 MB of pandas/numpy/streamlit.
# ==========================================================================
coll = COLLECT(                          # noqa: F821
    cli_exe, cli_a.binaries, cli_a.datas,
    dash_exe, dash_a.binaries, dash_a.datas,
    strip=False,
    upx=False,
    name="nse-screener",
)
