# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for session aggregators.

Defines the interface that session aggregators must implement
to work with the SessionEventConsumer.

TODO(OMN-1526): This protocol was originally in omniclaude.aggregators.
Consider moving to omnibase_infra.protocols for reuse.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol
from uuid import UUID


class ProtocolSessionAggregator(Protocol):
    """Protocol for session aggregators.

    Session aggregators process events from the SessionEventConsumer
    and maintain session state. Implementations must be idempotent
    to support at-least-once delivery semantics.

    Example:
        >>> class MyAggregator:
        ...     @property
        ...     def aggregator_id(self) -> str:
        ...         return "my-aggregator"
        ...
        ...     async def process_event(self, event: object, correlation_id: UUID) -> bool:
        ...         # Process event and return True if accepted
        ...         return True
    """

    @property
    def aggregator_id(self) -> str:
        """Get the unique aggregator identifier.

        Returns:
            Unique string identifier for this aggregator instance.
        """
        ...

    async def process_event(self, event: object, correlation_id: UUID) -> bool:
        """Process a session event.

        Args:
            event: The event to process. May be a JSON string or
                a pre-parsed event envelope, depending on consumer configuration.
            correlation_id: Correlation ID for tracing.

        Returns:
            True if processed successfully, False if rejected (e.g., duplicate).

        Raises:
            Exception: On processing failure. Consumer will handle retry/circuit breaker.
        """
        ...

    async def get_snapshot(
        self, session_id: str, correlation_id: UUID
    ) -> object | None:
        """Get current snapshot for a session.

        Args:
            session_id: The session identifier.
            correlation_id: Correlation ID for tracing.

        Returns:
            Session snapshot object, or None if session not found.
        """
        ...

    async def finalize_session(
        self, session_id: str, correlation_id: UUID, reason: str | None = None
    ) -> object | None:
        """Finalize a session, marking it as ended.

        Args:
            session_id: The session identifier.
            correlation_id: Correlation ID for tracing.
            reason: Optional reason for session end.

        Returns:
            Final session snapshot, or None if session not found.
        """
        ...

    async def get_active_sessions(self, correlation_id: UUID) -> list[str]:
        """Get list of active session IDs.

        Args:
            correlation_id: Correlation ID for tracing.

        Returns:
            List of active session ID strings.
        """
        ...

    async def get_session_last_activity(
        self, session_id: str, correlation_id: UUID
    ) -> datetime | None:
        """Get last activity timestamp for a session.

        Args:
            session_id: The session identifier.
            correlation_id: Correlation ID for tracing.

        Returns:
            Last activity datetime, or None if session not found.
        """
        ...


__all__ = ["ProtocolSessionAggregator"]
