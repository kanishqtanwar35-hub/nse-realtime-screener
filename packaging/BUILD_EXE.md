# Packaging to an executable

Two deliverables, because they have genuinely different constraints:

| Target | Approach | Status |
|---|---|---|
| **CLI** (`run_live`, `train_model`) | PyInstaller one-file | Works cleanly |
| **Dashboard** (Streamlit) | PyInstaller one-**dir** + launcher | Works, with caveats |

---

## Why Streamlit cannot be a plain one-file exe

Streamlit is not a library you call — it is a **web server that re-executes your
script** on every interaction. Three consequences:

1. It runs your app file **by path**, so the `.py` must exist on disk at runtime.
   A one-file build unpacks to a temp directory that changes every launch.
2. It ships a large tree of **static assets** (the React frontend) plus
   `metadata.json` files that its `importlib.metadata` calls need. PyInstaller does
   not collect those by default, and the app fails at import with a
   `PackageNotFoundError` that gives no hint about the cause.
3. It spawns a **subprocess** for the script runner. Under one-file, `sys.executable`
   points at the bootloader, so the child relaunches the bootloader instead of the app
   and you get an infinite process fork.

The fix for all three is a **one-dir** build plus an explicit entry point that starts
the server in-process. The spec in this folder does that.

---

## Build

```bash
pip install pyinstaller
cd nse-screener

# CLI - one file, fully self-contained
pyinstaller packaging/nse_screener.spec --clean --noconfirm

# output:
#   dist/nse-screener/nse-screener.exe          <- CLI
#   dist/nse-screener/nse-screener-dashboard.exe <- dashboard launcher
```

Run:

```bash
dist\nse-screener\nse-screener.exe --relax-liquidity --cycles 10
dist\nse-screener\nse-screener-dashboard.exe        # opens http://localhost:8501
```

---

## What the spec handles

- `collect_all("streamlit")` — binaries, data files **and** metadata.
- `copy_metadata` for `streamlit`, `pandas`, `numpy`, `scikit-learn`, `altair` and
  `pyarrow`. Streamlit calls `importlib.metadata.version()` on several of these at
  import time; without the metadata the exe dies before printing anything useful.
- `hiddenimports` for scikit-learn's tree/ensemble internals, which are imported
  dynamically at unpickle time and are therefore invisible to static analysis. Missing
  these produces an exe that runs fine until it loads `trade_model.joblib`.
- The dashboard app file is bundled as **data** and located at runtime via
  `sys._MEIPASS`.
- Broker SDKs are in `excludes` — they are optional, and bundling them pulls in a
  large dependency tree for something most users will not have configured. If you ship
  a broker build, remove them from `excludes` and add the corresponding
  `copy_metadata` line.

---

## Size and startup

| | One-file | One-dir |
|---|---|---|
| Size | ~180-260 MB | ~200-300 MB |
| First start | 5-15 s (unpacks to temp) | <2 s |
| Antivirus false positives | Common | Rare |

pandas + numpy + scikit-learn + streamlit is simply a large payload. To trim it:

- `--exclude-module matplotlib` if you are not plotting outside Streamlit
- Use UPX (`--upx-dir`) — cuts ~25%, but raises AV false positives further
- Drop the dashboard from the build and ship CLI-only if the UI is not needed

---

## Runtime files

The exe writes next to itself, not into the bundle:

```
dist/nse-screener/
  nse-screener.exe
  data/screener.db          created on first run
  data/trade_model.joblib   ship a pre-trained model here, or train on first use
  logs/screener.log
  .env                      optional, read at startup if present
```

**Ship a trained model.** Training needs history; a fresh exe with no model shows
`UNKNOWN` for every signal, which looks broken. Run `train_model.py` before packaging
and copy `data/trade_model.joblib` into the dist folder.

---

## Verification checklist

Test on a machine **without** Python installed — that is the whole point, and it is
where the metadata and hidden-import failures actually surface.

- [ ] `nse-screener.exe --help` prints usage
- [ ] `nse-screener.exe --cycles 2 --relax-liquidity` completes and writes `data/screener.db`
- [ ] `nse-screener-dashboard.exe` serves on :8501 and renders all four tabs
- [ ] The Model tab shows a loaded model (not "none")
- [ ] Killing the CLI with Ctrl-C shuts down cleanly rather than stack-tracing

---

## The three bugs this build actually hit

All three produced errors that pointed away from the cause. Worth reading in full -
none is in the PyInstaller FAQ, and the first two are invisible until runtime.

**Shared symptom.** The dashboard exe starts, answers HTTP 200, `/_stcore/health`
returns `ok` - and then every page render fails with `ModuleNotFoundError`. The CLI
exe, built from the *same spec*, works perfectly. **That asymmetry is the whole clue.**

### Bug 1 - the bundled script shadowed the package

The dashboard script must exist on disk for Streamlit to exec it, so it ships as a
data file. Bundling it to `nse_screener/dashboard/app.py` created a real directory
`_internal/nse_screener/` containing **only** `dashboard/app.py`. Python found it
first, treated it as an implicit namespace package, and every
`nse_screener.<submodule>` import failed - while `import nse_screener` appeared to
succeed. Compounding it, `app.py` did `sys.path.insert(0, parents[2])` to find the
package when run from source, putting the shadowing directory at the FRONT of the path.

**Fix:** bundle to a neutral directory (`nse_dashboard/`), and make the `sys.path`
insert conditional on `not getattr(sys, "frozen", False)`.

**Necessary, but did not fix the error.**

### Bug 2 - `collect_submodules` failed silently

Adding `collect_submodules("nse_screener")` to `hiddenimports` looked like the answer.
It changed the error from `No module named 'nse_screener.backtest'` to
`No module named 'nse_screener'` - *worse*, and baffling.

Cause: `collect_submodules()` **imports** the package to walk it, and it runs at
spec-parse time - before `Analysis(pathex=...)` has any effect. It could not import
`nse_screener`, returned almost nothing, and **raised no warning**.

The proof is in the build TOCs:

```
Analysis-00.toc (CLI)        43 nse_screener entries
Analysis-01.toc (dashboard)   3 nse_screener entries   <-- there it is
```

**Fix:** put `src/` on `sys.path` at the top of the spec, before the call. After that
both analyses show 69 entries.

### Bug 3 - the entry point never imported the package

`dashboard_entry.py` imports only `streamlit`. Streamlit locates and execs `app.py`
**by path at runtime**, so static analysis has no way to know the package is needed.

**Fix:** import it explicitly in the entry point, even though nothing calls it there.

### Bug 4 (cosmetic) - UnicodeEncodeError from Streamlit itself

Streamlit 1.61's optional "skills install" prints a `U+26A0` through click into a
cp1252 Windows console and dies with a full traceback. Harmless to the app, but it
looks like a crash on a screen recording. Fixed by forcing `PYTHONIOENCODING=utf-8`
and reconfiguring stdio before streamlit is imported.

### The lessons

1. **Never bundle a data file to a path matching an importable package name.**
2. **If your entry point does not import your package, PyInstaller will not bundle
   it.** Anything reached by runtime path lookup - Streamlit apps, plugin loaders,
   `importlib` by string - is invisible to static analysis.
3. **`collect_submodules` needs the package importable at spec-parse time**, and
   fails silently when it is not. Check `build/<name>/Analysis-*.toc` - comparing the
   entry counts between a working and a broken exe found this in one command.
4. **A frozen app that starts is not a frozen app that works.** All three broken
   builds returned HTTP 200. Always exercise a real code path.

---

