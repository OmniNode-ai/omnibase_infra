# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Required context keys for structured logging.

This module defines EnumRequiredLogContextKey, a namespace class containing
the required and recommended context keys for structured logging entries.

Design Decisions:
    - Implemented as a class with Literal-typed class attributes rather than
      a proper Enum to avoid import complexity while maintaining type safety
    - Provides helper methods for querying auto-added vs recommended keys

See Also:
    - ModelBufferedLogEntry: The model that uses these context keys
    - SinkLoggingStructured: The sink that validates context keys
"""

from __future__ import annotations

from typing import Literal


class EnumRequiredLogContextKey:
    """Required context keys for structured logging.

    This class defines the keys that are automatically added to every log entry
    by the sink infrastructure. Callers should NOT include these keys in their
    context dictionaries as they will be overwritten.

    Automatically Added Keys (by SinkLoggingStructured):
        - ORIGINAL_TIMESTAMP: ISO-8601 timestamp from emit() time
        - LEVEL: Log level string (added by structlog processor)
        - TIMESTAMP: ISO-8601 timestamp from flush() time (added by structlog)

    Recommended Keys (callers SHOULD include):
        - CORRELATION_ID: UUID for distributed tracing
        - NODE_ID: ONEX node identifier
        - OPERATION: Current operation name

    Note:
        This is implemented as a class with class attributes rather than
        a proper Enum to avoid import complexity with EnumLogLevel while
        maintaining a clear namespace for key constants.
    """

    # Automatically added by sink/structlog
    ORIGINAL_TIMESTAMP: Literal["original_timestamp"] = "original_timestamp"
    LEVEL: Literal["level"] = "level"
    TIMESTAMP: Literal["timestamp"] = "timestamp"

    # Recommended keys for callers
    CORRELATION_ID: Literal["correlation_id"] = "correlation_id"
    NODE_ID: Literal["node_id"] = "node_id"
    OPERATION: Literal["operation"] = "operation"

    @classmethod
    def auto_added_keys(cls) -> frozenset[str]:
        """Return the set of keys automatically added by the sink.

        Returns:
            Frozenset of key names that are automatically added.
        """
        return frozenset({cls.ORIGINAL_TIMESTAMP, cls.LEVEL, cls.TIMESTAMP})

    @classmethod
    def recommended_keys(cls) -> frozenset[str]:
        """Return the set of recommended context keys.

        Returns:
            Frozenset of key names that callers SHOULD include.
        """
        return frozenset({cls.CORRELATION_ID, cls.NODE_ID, cls.OPERATION})


__all__: list[str] = [
    "EnumRequiredLogContextKey",
]
