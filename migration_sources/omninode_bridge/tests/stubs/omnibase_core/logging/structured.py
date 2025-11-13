"""Stub for omnibase_core.logging.structured"""

from typing import Any


def emit_log_event_sync(
    level: Any, message: str, context: dict[str, Any] = None
) -> None:
    """Stub for structured logging - no-op for tests."""
    pass


__all__ = ["emit_log_event_sync"]
