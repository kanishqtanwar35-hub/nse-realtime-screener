"""
Angel One SmartAPI adapter.

Credentials from the environment only (ANGEL_API_KEY, ANGEL_CLIENT_ID,
ANGEL_PASSWORD, ANGEL_TOTP_SECRET). The TOTP secret is the base32 seed from
your authenticator enrolment; the six-digit code is derived at login time, so
nothing time-sensitive is ever stored.

  !! UNVERIFIED AGAINST A LIVE ACCOUNT !!
  Written against documented SmartAPI shapes, not executed against a real
  session. Parsing is defensive throughout. Validate before trusting it.

Angel One addresses instruments by numeric token, not by trading symbol, so the
adapter downloads and caches the public scrip master on first use.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from ..config import settings
from ..models import Quote
from ..utils import PermanentError, RateLimiter, TransientError, get_logger, retry
from .base import BAR_COLUMNS, MarketDataProvider

log = get_logger(__name__)

_LIMITER = RateLimiter(max_calls=8, period_seconds=1.0)

SCRIP_MASTER_URL = (
    "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
)
_CACHE_TTL_SECONDS = 24 * 3600


class AngelOneProvider(MarketDataProvider):
    name = "angelone"

    def __init__(self, exchange: str = "NSE") -> None:
        self.exchange = exchange
        self._client = None
        self._creds = settings.credentials
        self._token_map: dict[str, str] = {}
        self._cache_path: Path = settings.paths.data / "angel_scrip_master.json"

    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        missing = [
            n for n, v in (
                ("ANGEL_API_KEY", self._creds.angel_api_key),
                ("ANGEL_CLIENT_ID", self._creds.angel_client_id),
                ("ANGEL_PASSWORD", self._creds.angel_password),
                ("ANGEL_TOTP_SECRET", self._creds.angel_totp_secret),
            ) if not v
        ]
        if missing:
            raise PermanentError(f"missing Angel One credentials: {', '.join(missing)}")

        try:
            from SmartApi import SmartConnect
        except ImportError as exc:
            raise PermanentError(
                "smartapi-python is not installed. `pip install smartapi-python pyotp`, "
                "or set DATA_PROVIDER=simulated."
            ) from exc
        try:
            import pyotp
        except ImportError as exc:
            raise PermanentError("pyotp is required for Angel One TOTP login") from exc

        try:
            self._client = SmartConnect(api_key=self._creds.angel_api_key)
            otp = pyotp.TOTP(self._creds.angel_totp_secret).now()
            session = self._client.generateSession(
                self._creds.angel_client_id, self._creds.angel_password, otp
            )
        except Exception as exc:  # noqa: BLE001
            raise PermanentError(f"Angel One authentication failed: {exc}") from exc

        if not isinstance(session, dict) or not session.get("status"):
            raise PermanentError(f"Angel One rejected the login: {session}")

        log.info("Angel One session established (credentials: %s)", self._creds.redacted())
        self._load_token_map()

    def _require_client(self):
        if self._client is None:
            raise PermanentError("authenticate() must be called before fetching data")
        return self._client

    # ------------------------------------------------------------------
    def _load_token_map(self) -> None:
        """
        Build {TRADINGSYMBOL: token} for NSE cash equities.

        The scrip master is ~10MB, so it is cached on disk for a day. A stale
        cache is still usable - tokens are stable - so a download failure with
        a cache present is a warning, not an error.
        """
        raw = None
        if self._cache_path.exists():
            age = time.time() - self._cache_path.stat().st_mtime
            if age < _CACHE_TTL_SECONDS:
                try:
                    raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
                    log.info("scrip master loaded from cache (%.1f h old)", age / 3600)
                except (json.JSONDecodeError, OSError) as exc:
                    log.warning("scrip master cache unreadable (%s), refetching", exc)

        if raw is None:
            try:
                import requests

                resp = requests.get(SCRIP_MASTER_URL, timeout=60)
                resp.raise_for_status()
                raw = resp.json()
                try:
                    self._cache_path.write_text(json.dumps(raw), encoding="utf-8")
                except OSError as exc:
                    log.warning("could not cache scrip master: %s", exc)
            except Exception as exc:  # noqa: BLE001
                if self._cache_path.exists():
                    log.warning("scrip master download failed (%s); using stale cache", exc)
                    raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
                else:
                    raise TransientError(f"could not load Angel One scrip master: {exc}") from exc

        mapping: dict[str, str] = {}
        for row in raw or []:
            if row.get("exch_seg") != self.exchange:
                continue
            tsym = str(row.get("symbol", ""))
            # Cash equities are suffixed "-EQ"; skip F&O, indices, ETFs.
            if not tsym.endswith("-EQ"):
                continue
            mapping[tsym[:-3].upper()] = str(row.get("token", ""))

        self._token_map = mapping
        log.info("scrip master ready: %d NSE cash symbols", len(mapping))

    def _token(self, symbol: str) -> str | None:
        token = self._token_map.get(symbol.upper())
        if token is None:
            log.warning("no Angel One token for %s - skipping", symbol)
        return token

    # ------------------------------------------------------------------
    @retry(attempts=4, base_delay=0.6)
    def _market_data_call(self, tokens: list[str]) -> dict:
        client = self._require_client()
        with _LIMITER:
            try:
                resp = client.getMarketData(
                    mode="FULL", exchangeTokens={self.exchange: tokens}
                )
            except Exception as exc:  # noqa: BLE001
                raise TransientError(f"Angel One market data call failed: {exc}") from exc

        if not isinstance(resp, dict):
            raise TransientError(f"unexpected Angel One response: {type(resp)!r}")
        if not resp.get("status", True):
            message = str(resp.get("message", "")).lower()
            if any(k in message for k in ("token", "session", "unauthor")):
                raise PermanentError(f"Angel One auth error: {resp.get('message')}")
            raise TransientError(f"Angel One error: {resp.get('message')}")
        return resp

    def fetch_quotes(self, symbols: list[str]) -> list[Quote]:
        if not symbols:
            return []
        if not self._token_map:
            self._load_token_map()

        pairs = [(s, self._token(s)) for s in symbols]
        pairs = [(s, t) for s, t in pairs if t]
        if not pairs:
            return []

        reverse = {t: s for s, t in pairs}
        resp = self._market_data_call([t for _, t in pairs])
        now = datetime.now()
        quotes: list[Quote] = []

        for row in (resp.get("data", {}) or {}).get("fetched", []) or []:
            try:
                symbol = reverse.get(str(row.get("symbolToken", "")))
                if not symbol:
                    continue
                depth = row.get("depth", {}) or {}
                bids = depth.get("buy", []) or []
                asks = depth.get("sell", []) or []
                top_bid = bids[0] if bids else {}
                top_ask = asks[0] if asks else {}

                quote = Quote(
                    symbol=symbol,
                    ltp=float(row.get("ltp", 0.0) or 0.0),
                    bid_price=float(top_bid.get("price", 0.0) or 0.0),
                    ask_price=float(top_ask.get("price", 0.0) or 0.0),
                    bid_qty=int(top_bid.get("quantity", 0) or 0),
                    ask_qty=int(top_ask.get("quantity", 0) or 0),
                    ltq=int(row.get("lastTradeQty", 0) or 0),
                    volume=int(row.get("tradeVolume", 0) or 0),
                    timestamp=now,
                )
                if quote.is_valid():
                    quotes.append(quote)
            except (TypeError, ValueError, KeyError, IndexError) as exc:
                log.warning("could not parse Angel One quote row: %s", exc)
        return quotes

    # ------------------------------------------------------------------
    @retry(attempts=4, base_delay=0.6)
    def _candle_call(self, params: dict) -> dict:
        client = self._require_client()
        with _LIMITER:
            try:
                resp = client.getCandleData(params)
            except Exception as exc:  # noqa: BLE001
                raise TransientError(f"Angel One candle call failed: {exc}") from exc
        return resp if isinstance(resp, dict) else {}

    def fetch_history(self, symbol: str, minutes: int) -> pd.DataFrame:
        if not self._token_map:
            self._load_token_map()
        token = self._token(symbol)
        if not token:
            return self._empty_bars()

        end = datetime.now()
        start = end - timedelta(days=max(2, int(minutes / 375) + 3))
        params = {
            "exchange": self.exchange,
            "symboltoken": token,
            "interval": "ONE_MINUTE",
            "fromdate": start.strftime("%Y-%m-%d %H:%M"),
            "todate": end.strftime("%Y-%m-%d %H:%M"),
        }
        try:
            resp = self._candle_call(params)
        except TransientError as exc:
            log.error("history unavailable for %s: %s", symbol, exc)
            return self._empty_bars()

        candles = resp.get("data") or []
        if not candles:
            return self._empty_bars()

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        bars = self.normalise_bars(df)
        return bars.tail(minutes).reset_index(drop=True)

    def close(self) -> None:
        try:
            if self._client:
                self._client.terminateSession(self._creds.angel_client_id)
        except Exception as exc:  # noqa: BLE001
            log.debug("session terminate failed (harmless): %s", exc)
        finally:
            self._client = None

    def health_check(self) -> bool:
        return self._client is not None
