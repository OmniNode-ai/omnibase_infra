# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definitions for Registry Effect Node dependencies.

These protocols define the interfaces for handler and event bus dependencies
using duck typing with Python's Protocol class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolEnvelopeExecutor(Protocol):
    """Protocol for envelope executor objects (Consul, PostgreSQL).

    Executors must implement an async execute method that accepts an envelope
    dictionary and returns a result dictionary.
    """

    async def execute(self, envelope: dict[str, object]) -> dict[str, object]:
        """Execute an operation based on the envelope contents.

        Args:
            envelope: Dictionary containing operation details with keys:
                - operation: The operation to perform (e.g., "consul.register", "db.execute")
                - payload: Operation-specific data
                - correlation_id: UUID for distributed tracing

        Returns:
            Dictionary with operation results, typically containing:
                - status: "success" or "failed"
                - payload: Operation-specific result data
        """
        ...


@runtime_checkable
class ProtocolEventBus(Protocol):
    """Protocol for event bus objects.

    Event bus must implement an async publish method for sending messages
    to topics.
    """

    async def publish(self, topic: str, key: bytes, value: bytes) -> None:
        """Publish a message to a topic.

        Args:
            topic: The topic name to publish to
            key: Message key as bytes
            value: Message value as bytes (typically JSON-encoded)
        """
        ...


__all__ = ["ProtocolEnvelopeExecutor", "ProtocolEventBus"]
