"""
Provider registry.

`get_provider()` is the only entry point the rest of the app uses. It resolves a
name to an authenticated provider and - critically - falls back to the simulator
when a broker cannot be reached, so a missing token degrades the system to
paper mode instead of killing it.
"""
from __future__ import annotations

from ..config import settings
from ..utils import PermanentError, get_logger
from .base import BAR_COLUMNS, MarketDataProvider
from .simulated import SimulatedProvider

log = get_logger(__name__)

__all__ = [
    "MarketDataProvider",
    "SimulatedProvider",
    "BAR_COLUMNS",
    "get_provider",
    "available_providers",
]


def available_providers() -> list[str]:
    return ["simulated", "fyers", "angelone"]


def get_provider(
    name: str | None = None,
    *,
    symbols: list[str] | None = None,
    allow_fallback: bool = True,
) -> MarketDataProvider:
    """
    Build and authenticate a provider.

    allow_fallback=True (the default) means a broker failure logs loudly and
    returns the simulator instead of raising. That is the right default for a
    dashboard you want to stay up; pass False in tests or when a live feed is a
    hard requirement.
    """
    name = (name or settings.provider or "simulated").lower().strip()

    if name in {"sim", "simulated", "fake", "mock"}:
        provider: MarketDataProvider = SimulatedProvider(symbols=symbols)
        provider.authenticate()
        return provider

    try:
        if name == "fyers":
            from .fyers import FyersProvider

            provider = FyersProvider()
        elif name in {"angel", "angelone", "angel_one", "smartapi"}:
            from .angelone import AngelOneProvider

            provider = AngelOneProvider()
        else:
            raise PermanentError(
                f"unknown provider {name!r}; expected one of {available_providers()}"
            )

        provider.authenticate()
        log.info("using live provider: %s", provider.name)
        return provider

    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the boundary
        if not allow_fallback:
            raise
        log.error(
            "provider %r unavailable (%s: %s) - FALLING BACK TO SIMULATED DATA. "
            "Numbers shown are synthetic and must not be traded on.",
            name, type(exc).__name__, exc,
        )
        fallback = SimulatedProvider(symbols=symbols)
        fallback.authenticate()
        return fallback
