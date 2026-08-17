"""
Frozen-app entry point for the Streamlit dashboard.

Streamlit runs your app BY PATH and spawns a subprocess for the script runner.
Under a frozen build:
  * the app file lives inside the bundle, located via sys._MEIPASS
  * sys.executable points at the bootloader, so letting Streamlit re-spawn it
    would relaunch the exe instead of the script - an infinite fork
Calling bootstrap.run() directly starts the server IN-PROCESS and sidesteps both.

    nse-screener-dashboard.exe                 # 8501, or the next free port
    nse-screener-dashboard.exe --port 8600
    DASHBOARD_PORT=8600 nse-screener-dashboard.exe
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Force UTF-8 on stdio BEFORE anything imports streamlit.
#
# The Windows console defaults to cp1252, which cannot encode the emoji several
# libraries print. Streamlit 1.61's optional "skills install" step emits a U+26A0
# through click and dies with UnicodeEncodeError - harmless to the app, but it
# prints a full traceback that looks like a crash in a screen recording.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

# Import the package here even though this file never calls it.
#
# Streamlit locates and execs app.py BY PATH at runtime, so PyInstaller's static
# analysis has no way to know app.py needs nse_screener - and a frozen build
# would omit the entire package. Referencing it here puts it in the dependency
# graph the only way that is guaranteed to work.
#
# Belt and braces with collect_submodules() in the spec: that one runs at
# spec-parse time and fails silently if the package is not importable yet.
try:  # pragma: no cover - import side effect only
    import nse_screener  # noqa: F401
    from nse_screener import (backtest, config, engine, features,  # noqa: F401
                              indicators, ingestion, models, screener,
                              signals, store, universe, utils)
    from nse_screener.ml import model  # noqa: F401
    from nse_screener.providers import simulated  # noqa: F401
except ImportError:
    pass


def requested_port(argv: list[str]) -> int:
    """`--port N` / `-p N` on the command line, else DASHBOARD_PORT, else 8501."""
    for flag in ("--port", "-p"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                try:
                    return int(argv[i + 1])
                except ValueError:
                    print(f"ignoring non-numeric {flag} {argv[i + 1]!r}", file=sys.stderr)
    return int(os.environ.get("DASHBOARD_PORT", 8501))


def free_port(preferred: int, attempts: int = 20) -> int:
    """
    First free port at or after `preferred`.

    Without this the exe prints "Port 8501 is not available" and exits - which
    is what happens to anyone who already has a Streamlit app, a Jupyter server,
    or a previous copy of this dashboard running. A binary handed to a reviewer
    should move to the next port and say so, not die on a detail like that.
    """
    import socket

    for offset in range(attempts):
        candidate = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", candidate))
            except OSError:
                continue
        if offset:
            print(f"port {preferred} is in use - starting on {candidate} instead")
        return candidate

    print(f"no free port in {preferred}-{preferred + attempts - 1}; trying {preferred} anyway",
          file=sys.stderr)
    return preferred


def app_path() -> Path:
    """Locate app.py inside the bundle, or in the source tree when not frozen."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        return base / "nse_dashboard" / "app.py"
    return Path(__file__).resolve().parents[1] / "src" / "nse_screener" / "dashboard" / "app.py"


def main() -> int:
    target = app_path()
    if not target.exists():
        print(f"dashboard app not found at {target}", file=sys.stderr)
        return 1

    # Runtime files sit NEXT TO the exe, not inside the temp bundle.
    if getattr(sys, "frozen", False):
        here = Path(sys.executable).parent
        os.environ.setdefault("DB_PATH", str(here / "data" / "screener.db"))
        os.environ.setdefault("MODEL_PATH", str(here / "data" / "trade_model.joblib"))
        (here / "data").mkdir(exist_ok=True)
        (here / "logs").mkdir(exist_ok=True)

    from streamlit.web import bootstrap

    flag_options = {
        "server.port": free_port(requested_port(sys.argv[1:])),
        "server.headless": False,
        "browser.gatherUsageStats": False,
        "global.developmentMode": False,
    }
    bootstrap.load_config_options(flag_options=flag_options)
    bootstrap.run(str(target), False, [], flag_options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
