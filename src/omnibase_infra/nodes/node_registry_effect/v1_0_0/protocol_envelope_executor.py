# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for envelope executor objects.

This protocol defines the interface for handler dependencies (Consul, PostgreSQL)
using duck typing with Python's Protocol class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.nodes.node_registry_effect.v1_0_0.protocol_types import (
    EnvelopeDict,
    ResultDict,
)


@runtime_checkable
class ProtocolEnvelopeExecutor(Protocol):
    """Protocol for envelope executor objects (Consul, PostgreSQL).

    Executors must implement an async execute method that accepts an envelope
    dictionary and returns a result dictionary.
    """

    async def execute(self, envelope: EnvelopeDict) -> ResultDict:
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


__all__ = [
    "ProtocolEnvelopeExecutor",
]
