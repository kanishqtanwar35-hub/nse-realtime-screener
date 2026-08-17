"""
Fyers API v3 adapter.

Credentials come from the environment only (FYERS_CLIENT_ID, FYERS_ACCESS_TOKEN,
...). Nothing is written to disk and nothing is logged in the clear.

  !! UNVERIFIED AGAINST A LIVE ACCOUNT !!
  This adapter is written against the documented v3 request/response shapes but
  has not been executed against a real Fyers session. Response parsing is
  deliberately defensive (every field access is a .get with a fallback) so a
  schema difference degrades to a dropped symbol plus a warning rather than a
  crash. Validate against your own account before trusting it with capital.

Auth model: Fyers issues a short-lived access token via an OAuth redirect. That
flow needs a browser and cannot run headless, so this adapter expects the token
to already exist in the environment. `python -m nse_screener.providers.fyers`
prints the authorisation URL to help you obtain one.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from ..config import settings
from ..models import Quote
from ..utils import PermanentError, RateLimiter, TransientError, get_logger, retry
from .base import BAR_COLUMNS, MarketDataProvider

log = get_logger(__name__)

# Fyers publishes ~10 req/s on the data endpoints; stay comfortably under.
_LIMITER = RateLimiter(max_calls=8, period_seconds=1.0)


class FyersProvider(MarketDataProvider):
    name = "fyers"

    def __init__(self, exchange: str = "NSE", segment: str = "EQ") -> None:
        self.exchange = exchange
        self.segment = segment
        self._client = None
        self._creds = settings.credentials

    # ------------------------------------------------------------------
    def _symbol_to_broker(self, symbol: str) -> str:
        """RELIANCE -> NSE:RELIANCE-EQ"""
        if ":" in symbol:
            return symbol
        return f"{self.exchange}:{symbol.upper()}-{self.segment}"

    @staticmethod
    def _symbol_from_broker(broker_symbol: str) -> str:
        """NSE:RELIANCE-EQ -> RELIANCE"""
        s = broker_symbol.split(":")[-1]
        return s.rsplit("-", 1)[0] if "-" in s else s

    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        if not self._creds.fyers_client_id:
            raise PermanentError("FYERS_CLIENT_ID is not set - see .env.example")
        if not self._creds.fyers_access_token:
            raise PermanentError(
                "FYERS_ACCESS_TOKEN is not set. Fyers uses a browser OAuth redirect that "
                "cannot run headless; generate a token and export it. Run "
                "`python -m nse_screener.providers.fyers` for the authorisation URL."
            )
        try:
            from fyers_apiv3 import fyersModel
        except ImportError as exc:
            raise PermanentError(
                "fyers-apiv3 is not installed. `pip install fyers-apiv3`, or set "
                "DATA_PROVIDER=simulated to run without a broker."
            ) from exc

        try:
            self._client = fyersModel.FyersModel(
                client_id=self._creds.fyers_client_id,
                token=self._creds.fyers_access_token,
                is_async=False,
                log_path="",
            )
            profile = self._client.get_profile()
        except Exception as exc:  # noqa: BLE001 - SDK raises bare Exception
            raise PermanentError(f"Fyers authentication failed: {exc}") from exc

        if isinstance(profile, dict) and profile.get("s") == "error":
            raise PermanentError(f"Fyers rejected the token: {profile.get('message')}")

        log.info("Fyers session established (credentials: %s)", self._creds.redacted())

    def _require_client(self):
        if self._client is None:
            raise PermanentError("authenticate() must be called before fetching data")
        return self._client

    # ------------------------------------------------------------------
    @retry(attempts=4, base_delay=0.6)
    def _quotes_call(self, broker_symbols: list[str]) -> dict:
        client = self._require_client()
        with _LIMITER:
            try:
                resp = client.quotes({"symbols": ",".join(broker_symbols)})
            except Exception as exc:  # noqa: BLE001
                raise TransientError(f"Fyers quotes call failed: {exc}") from exc

        if not isinstance(resp, dict):
            raise TransientError(f"unexpected Fyers response type: {type(resp)!r}")
        if resp.get("s") == "error":
            message = str(resp.get("message", "")).lower()
            if any(k in message for k in ("token", "auth", "invalid app")):
                raise PermanentError(f"Fyers auth error: {resp.get('message')}")
            raise TransientError(f"Fyers error: {resp.get('message')}")
        return resp

    def fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        if not symbols:
            return []

        broker_symbols = [self._symbol_to_broker(s) for s in symbols]
        resp = self._quotes_call(broker_symbols)
        now = datetime.now()
        quotes: list[Quote] = []

        for entry in resp.get("d", []) or []:
            try:
                if entry.get("s") == "error":
                    continue
                payload = entry.get("v", {}) or {}
                symbol = self._symbol_from_broker(entry.get("n", "") or payload.get("symbol", ""))
                if not symbol:
                    continue

                # Depth lives under v.depth on v3; fall back to flat keys.
                depth = payload.get("depth", {}) or {}
                bids = depth.get("buy", []) or []
                asks = depth.get("sell", []) or []
                top_bid = bids[0] if bids else {}
                top_ask = asks[0] if asks else {}

                quote = Quote(
                    symbol=symbol,
                    ltp=float(payload.get("lp", 0.0) or 0.0),
                    bid_price=float(top_bid.get("price", payload.get("bid", 0.0)) or 0.0),
                    ask_price=float(top_ask.get("price", payload.get("ask", 0.0)) or 0.0),
                    bid_qty=int(top_bid.get("volume", 0) or 0),
                    ask_qty=int(top_ask.get("volume", 0) or 0),
                    ltq=int(payload.get("ltq", 0) or 0),
                    volume=int(payload.get("volume", 0) or 0),
                    timestamp=now,
                )
                if quote.is_valid():
                    quotes.append(quote)
                else:
                    log.debug("discarding malformed Fyers quote for %s", symbol)
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                # One bad symbol must not sink the batch.
                log.warning("could not parse Fyers quote entry: %s", exc)
        return quotes

    # ------------------------------------------------------------------
    @retry(attempts=4, base_delay=0.6)
    def _history_call(self, params: dict) -> dict:
        client = self._require_client()
        with _LIMITER:
            try:
                resp = client.history(params)
            except Exception as exc:  # noqa: BLE001
                raise TransientError(f"Fyers history call failed: {exc}") from exc
        if isinstance(resp, dict) and resp.get("s") == "error":
            raise TransientError(f"Fyers history error: {resp.get('message')}")
        return resp if isinstance(resp, dict) else {}

    def fetch_history(self, symbol: str, minutes: int) -> pd.DataFrame:
        end = datetime.now()
        # Pad generously: `minutes` is trading minutes, the API wants a calendar
        # range, and weekends/holidays fall inside it.
        start = end - timedelta(days=max(2, int(minutes / 375) + 3))

        params = {
            "symbol": self._symbol_to_broker(symbol),
            "resolution": str(settings.data.interval_minutes),
            "date_format": "1",
            "range_from": start.strftime("%Y-%m-%d"),
            "range_to": end.strftime("%Y-%m-%d"),
            "cont_flag": "1",
        }
        try:
            resp = self._history_call(params)
        except TransientError as exc:
            log.error("history unavailable for %s: %s", symbol, exc)
            return self._empty_bars()

        candles = resp.get("candles") or []
        if not candles:
            return self._empty_bars()

        df = pd.DataFrame(candles, columns=["epoch", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["epoch"], unit="s", errors="coerce")
        bars = self.normalise_bars(df[BAR_COLUMNS])
        return bars.tail(minutes).reset_index(drop=True)

    def close(self) -> None:
        self._client = None

    def health_check(self) -> bool:
        try:
            return bool(self._client and self._client.get_profile().get("s") != "error")
        except Exception:  # noqa: BLE001
            return False


def print_auth_url() -> None:  # pragma: no cover - operator utility
    """Print the OAuth URL you must visit to mint an access token."""
    creds = settings.credentials
    if not (creds.fyers_client_id and creds.fyers_redirect_uri):
        print("Set FYERS_CLIENT_ID and FYERS_REDIRECT_URI first (see .env.example).")
        return
    url = (
        "https://api-t1.fyers.in/api/v3/generate-authcode"
        f"?client_id={creds.fyers_client_id}"
        f"&redirect_uri={creds.fyers_redirect_uri}"
        "&response_type=code&state=nse_screener"
    )
    print("\n1. Open this URL and log in:\n")
    print(f"   {url}\n")
    print("2. Copy the auth_code from the redirect, exchange it for an access token")
    print("   using fyersModel.SessionModel(...).generate_token().")
    print("3. Export it:  set FYERS_ACCESS_TOKEN=<token>\n")


if __name__ == "__main__":  # pragma: no cover
    print_auth_url()
