"""Startup validation and initialization for OmniNode Bridge services."""

from .config_validator import startup_validation
from .graceful_shutdown import GracefulShutdownHandler

__all__ = ["GracefulShutdownHandler", "startup_validation"]
