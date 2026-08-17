#!/usr/bin/env python
"""
Launch the Streamlit dashboard.

    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8502

A thin wrapper so you do not have to remember the path to app.py, and so the
frozen executable has a single entry point to call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "nse_screener" / "dashboard" / "app.py"


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the NSE screener dashboard")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--headless", action="store_true", help="do not open a browser")
    args = parser.parse_args()

    if not APP.exists():
        print(f"dashboard not found at {APP}", file=sys.stderr)
        return 1

    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("streamlit is not installed. pip install streamlit", file=sys.stderr)
        return 1

    sys.argv = [
        "streamlit", "run", str(APP),
        "--server.port", str(args.port),
        "--server.headless", "true" if args.headless else "false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
