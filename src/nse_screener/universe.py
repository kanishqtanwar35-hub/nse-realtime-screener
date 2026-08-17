"""
The tradeable universe.

Requirement 1 is "scan ALL NSE-listed stocks", which is ~1,800 equity symbols.
Three sources, tried in order, so the app always has a universe:

  1. A local CSV (`data/nse_universe.csv`) - authoritative, offline, fastest.
  2. The broker's own instrument master - Angel One publishes a public scrip
     master; if that provider is active it already has the full list.
  3. The bundled fallback list - small, but never fails.

Why not hit NSE's website directly: the equity list endpoint requires a browser
session and rate-limits aggressively, so a scraper is the least reliable link in
the chain. The broker master is the same data from a source you are already
authenticated against.

SCALE WARNING. 1,800 symbols is not free:
  * Quote polling is batched (`QUOTE_BATCH_SIZE`, default 50) => 36 API calls
    per cycle. At a 5-second poll and an 8 req/s limiter that is ~5s of pure
    request time - the loop becomes request-bound.
  * History seeding is ONE call per symbol. 1,800 calls at ~8/s is ~4 minutes of
    warm-up before the first signal can fire.
  * Memory is fine (~2,000 bars x 1,800 symbols ~ 300 MB).

`MAX_SYMBOLS` (default 200) therefore caps the working set. Raise it
deliberately once you have measured your broker's real throughput.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .config import DEFAULT_UNIVERSE, settings
from .utils import get_logger

log = get_logger(__name__)

UNIVERSE_CSV = settings.paths.data / "nse_universe.csv"


def _from_csv(path: Path) -> list[str]:
    """
    Read symbols from a CSV. Accepts NSE's own EQUITY_L.csv layout or a plain
    one-column list - detected from the header rather than assumed.
    """
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            return []

        header = [h.strip().upper() for h in rows[0]]
        col = 0
        for candidate in ("SYMBOL", "TRADINGSYMBOL", "NAME"):
            if candidate in header:
                col = header.index(candidate)
                break
        else:
            # No recognised header - treat row 0 as data.
            rows.insert(0, [])

        symbols = []
        for row in rows[1:]:
            if not row or col >= len(row):
                continue
            sym = row[col].strip().upper().replace("-EQ", "")
            if sym and sym.isascii() and " " not in sym:
                symbols.append(sym)

        deduped = sorted(set(symbols))
        log.info("universe: %d symbols from %s", len(deduped), path.name)
        return deduped
    except (OSError, csv.Error) as exc:
        log.error("could not read %s: %s", path, exc)
        return []


def _from_broker() -> list[str]:
    """Pull the equity list from the active broker's instrument master."""
    provider = (settings.provider or "").lower()
    if provider not in {"angel", "angelone", "angel_one", "smartapi"}:
        return []
    try:
        from .providers.angelone import AngelOneProvider

        p = AngelOneProvider()
        p.authenticate()
        symbols = sorted(p._token_map.keys())  # noqa: SLF001 - documented use
        log.info("universe: %d symbols from the Angel One scrip master", len(symbols))
        return symbols
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read the broker instrument master: %s", exc)
        return []


def load_universe(limit: int | None = None, source: str = "auto") -> list[str]:
    """
    Resolve the symbol universe.

    `source`: auto | csv | broker | default
    `limit`:  cap the working set; defaults to settings.screener.max_symbols.
              Pass 0 for no cap (and read the scale warning above first).
    """
    limit = settings.screener.max_symbols if limit is None else limit

    # An explicit SYMBOLS env var always wins - it is how you pin a test set.
    if settings.symbols:
        log.info("universe: %d symbols from the SYMBOLS env var", len(settings.symbols))
        return settings.symbols

    symbols: list[str] = []
    if source in ("auto", "csv"):
        symbols = _from_csv(UNIVERSE_CSV)
    if not symbols and source in ("auto", "broker"):
        symbols = _from_broker()
    if not symbols:
        symbols = list(DEFAULT_UNIVERSE)
        log.warning(
            "universe: falling back to the bundled list of %d symbols. For the full "
            "NSE equity list, download EQUITY_L.csv from nseindia.com and save it as "
            "%s, or run against Angel One which publishes an instrument master.",
            len(symbols), UNIVERSE_CSV,
        )

    if limit and len(symbols) > limit:
        log.warning(
            "universe capped at %d of %d symbols (MAX_SYMBOLS). Warm-up is one "
            "history call per symbol, so the full list costs minutes before the "
            "first signal - raise MAX_SYMBOLS once you have measured your broker.",
            limit, len(symbols),
        )
        symbols = symbols[:limit]

    return symbols


def write_template(path: Path | None = None) -> Path:
    """Write a starter CSV so the expected format is unambiguous."""
    path = path or UNIVERSE_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["SYMBOL"])
        for s in DEFAULT_UNIVERSE:
            w.writerow([s])
    log.info("wrote universe template to %s", path)
    return path
