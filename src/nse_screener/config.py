"""
Configuration.

Every tunable lives here and every value can be overridden by an environment
variable or a .env file. NOTHING is hardcoded - in particular no credentials.
Import `settings` and read attributes; never call os.getenv() elsewhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# .env loading. python-dotenv is optional; we degrade to real env vars.
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


def _env(key: str, default: Any, cast=str):
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if cast is list:
        return [p.strip().upper() for p in raw.split(",") if p.strip()]
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BrokerCredentials:
    """
    Credentials are read from the environment ONLY.

    Missing values are left as empty strings rather than raising, so that the
    simulated provider works with no configuration at all. Each broker adapter
    validates what it actually needs in authenticate() and fails loudly there.
    """

    # Fyers
    fyers_client_id: str = field(default_factory=lambda: _env("FYERS_CLIENT_ID", ""))
    fyers_secret_key: str = field(default_factory=lambda: _env("FYERS_SECRET_KEY", ""))
    fyers_redirect_uri: str = field(default_factory=lambda: _env("FYERS_REDIRECT_URI", ""))
    fyers_access_token: str = field(default_factory=lambda: _env("FYERS_ACCESS_TOKEN", ""))

    # Angel One
    angel_api_key: str = field(default_factory=lambda: _env("ANGEL_API_KEY", ""))
    angel_client_id: str = field(default_factory=lambda: _env("ANGEL_CLIENT_ID", ""))
    angel_password: str = field(default_factory=lambda: _env("ANGEL_PASSWORD", ""))
    angel_totp_secret: str = field(default_factory=lambda: _env("ANGEL_TOTP_SECRET", ""))

    def redacted(self) -> dict[str, str]:
        """Safe for logging: shows only whether each value is present."""
        return {f.name: ("<set>" if getattr(self, f.name) else "<missing>") for f in fields(self)}


@dataclass(frozen=True)
class ScreenerConfig:
    """The universe filters from the brief."""

    min_ltp: float = field(default_factory=lambda: _env("MIN_LTP", 30.0, float))
    max_ltp: float = field(default_factory=lambda: _env("MAX_LTP", 500.0, float))
    # NOTE: 1,000,000 on BOTH sides of the book is extremely restrictive for
    # cash-segment NSE names in this price band - real top-of-book depth is
    # usually hundreds to low thousands. Kept as the documented default because
    # it is what the specification asks for; lower it via env for live testing.
    min_bid_qty: int = field(default_factory=lambda: _env("MIN_BID_QTY", 1_000_000, int))
    min_ask_qty: int = field(default_factory=lambda: _env("MIN_ASK_QTY", 1_000_000, int))
    max_symbols: int = field(default_factory=lambda: _env("MAX_SYMBOLS", 200, int))


@dataclass(frozen=True)
class IndicatorConfig:
    smma_fast: int = field(default_factory=lambda: _env("SMMA_FAST", 20, int))
    smma_slow: int = field(default_factory=lambda: _env("SMMA_SLOW", 120, int))
    # LTQ rolling windows, in minutes
    ltq_windows: tuple[int, ...] = (2, 5, 20)
    # ETQ (cumulative traded quantity) windows, in minutes
    etq_windows: tuple[int, ...] = (5, 20, 60)
    momentum_window: int = field(default_factory=lambda: _env("MOMENTUM_WINDOW", 5, int))
    volatility_window: int = field(default_factory=lambda: _env("VOLATILITY_WINDOW", 20, int))
    roc_window: int = field(default_factory=lambda: _env("ROC_WINDOW", 5, int))


@dataclass(frozen=True)
class DataConfig:
    """
    History lookback.

    The brief asks for "at least the last 60 minutes". That is NOT enough to
    seed SMMA(120) on 1-minute bars - you need at least 120 bars before the
    slow line means anything, and realistically several times that before it
    has settled. So the default lookback is deliberately larger, and
    `min_bars_required` gates any symbol that cannot supply it.
    """

    interval_minutes: int = field(default_factory=lambda: _env("BAR_INTERVAL_MIN", 1, int))
    history_minutes: int = field(default_factory=lambda: _env("HISTORY_MINUTES", 1500, int))
    min_bars_required: int = field(default_factory=lambda: _env("MIN_BARS_REQUIRED", 150, int))
    poll_seconds: float = field(default_factory=lambda: _env("POLL_SECONDS", 5.0, float))
    quote_batch_size: int = field(default_factory=lambda: _env("QUOTE_BATCH_SIZE", 50, int))
    max_forward_fill: int = field(default_factory=lambda: _env("MAX_FORWARD_FILL", 3, int))


@dataclass(frozen=True)
class TradingConfig:
    """Costs used when labelling simulated trades."""

    cost_bps: float = field(default_factory=lambda: _env("COST_BPS", 10.0, float))
    slippage_bps: float = field(default_factory=lambda: _env("SLIPPAGE_BPS", 2.0, float))
    max_hold_minutes: int = field(default_factory=lambda: _env("MAX_HOLD_MINUTES", 240, int))
    allow_short: bool = field(default_factory=lambda: _env("ALLOW_SHORT", True, bool))


@dataclass(frozen=True)
class ModelConfig:
    algorithm: str = field(default_factory=lambda: _env("ML_ALGORITHM", "random_forest"))
    accept_threshold: float = field(default_factory=lambda: _env("ACCEPT_THRESHOLD", 0.55, float))
    n_estimators: int = field(default_factory=lambda: _env("ML_N_ESTIMATORS", 300, int))
    max_depth: int = field(default_factory=lambda: _env("ML_MAX_DEPTH", 6, int))
    min_training_rows: int = field(default_factory=lambda: _env("ML_MIN_ROWS", 60, int))
    test_size: float = field(default_factory=lambda: _env("ML_TEST_SIZE", 0.25, float))
    random_state: int = field(default_factory=lambda: _env("ML_RANDOM_STATE", 42, int))
    top_explanations: int = field(default_factory=lambda: _env("ML_TOP_EXPLANATIONS", 4, int))


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    data: Path = PROJECT_ROOT / "data"
    logs: Path = PROJECT_ROOT / "logs"

    @property
    def db(self) -> Path:
        return Path(_env("DB_PATH", str(self.data / "screener.db")))

    @property
    def model(self) -> Path:
        return Path(_env("MODEL_PATH", str(self.data / "trade_model.joblib")))

    def ensure(self) -> None:
        self.data.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("DATA_PROVIDER", "simulated").lower())
    symbols: list[str] = field(default_factory=lambda: _env("SYMBOLS", [], list))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO").upper())
    dashboard_refresh_seconds: int = field(
        default_factory=lambda: _env("DASHBOARD_REFRESH_SECONDS", 15, int)
    )

    credentials: BrokerCredentials = field(default_factory=BrokerCredentials)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: Paths = field(default_factory=Paths)


settings = Settings()
settings.paths.ensure()

# A small default universe so the app runs with zero configuration.
DEFAULT_UNIVERSE: list[str] = [
    "IDEA", "YESBANK", "SUZLON", "IDFCFIRSTB", "PNB", "BANKBARODA", "SAIL",
    "NHPC", "IOC", "GAIL", "ONGC", "NMDC", "TATASTEEL", "ASHOKLEY", "CANBK",
    "UNIONBANK", "IRFC", "RVNL", "BHEL", "ZOMATO", "INDHOTEL", "FEDERALBNK",
    "MOTHERSON", "TATAPOWER", "HINDCOPPER", "JPPOWER", "TRIDENT", "GMRINFRA",
]


def resolve_universe() -> list[str]:
    """
    The active symbol universe.

    Delegates to `universe.load_universe`, which tries a local NSE CSV, then the
    broker instrument master, then the bundled fallback. Imported lazily to keep
    this module free of dependencies on the rest of the package.
    """
    from .universe import load_universe

    return load_universe()
