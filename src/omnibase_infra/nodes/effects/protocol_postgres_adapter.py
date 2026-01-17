# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for PostgreSQL registration persistence.

This module defines the protocol that PostgreSQL adapters must implement
to be used with the NodeRegistryEffect node.

Concurrency Safety:
    Implementations MUST be safe for concurrent async calls.
    Multiple coroutines may invoke upsert() simultaneously for
    different or identical node registrations. Implementations
    should use asyncio.Lock for coroutine-safety when protecting shared state.

Related:
    - NodeRegistryEffect: Effect node that uses this protocol
    - ProtocolConsulClient: Protocol for Consul backend
    - ModelBackendResult: Structured result model for backend operations
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.nodes.effects.models import ModelBackendResult


@runtime_checkable
class ProtocolPostgresAdapter(Protocol):
    """Protocol for PostgreSQL registration persistence.

    Implementations must provide async upsert capability for
    registration records.

    Concurrency Safety:
        Implementations MUST be safe for concurrent async coroutine calls.

        **Guarantees implementers MUST provide:**
            - Concurrent upsert() calls are coroutine-safe
            - Connection pooling (if used) is async-safe
            - Database transactions are properly isolated

        **What callers can assume:**
            - Multiple coroutines can call upsert() concurrently
            - Each upsert operation is independent
            - Failures in one upsert do not affect others

        Note: asyncio.Lock provides coroutine-safety, not thread-safety.
    """

    async def upsert(
        self,
        node_id: UUID,
        node_type: EnumNodeKind,
        node_version: ModelSemVer,
        endpoints: dict[str, str],
        metadata: dict[str, str],
    ) -> ModelBackendResult:
        """Upsert a node registration record.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of ONEX node (EnumNodeKind).
            node_version: Semantic version of the node.
            endpoints: Dict of endpoint type to URL.
            metadata: Additional metadata.

        Returns:
            ModelBackendResult with success status, optional error message,
            timing information, and correlation context.
        """
        ...

    async def deactivate(
        self,
        node_id: UUID,
    ) -> ModelBackendResult:
        """Deactivate a node registration record.

        Marks the registration as inactive (soft delete) rather than
        removing it entirely. This preserves historical data while
        stopping the node from appearing in active registrations.

        Args:
            node_id: Unique identifier for the node to deactivate.

        Returns:
            ModelBackendResult with success status, optional error message,
            timing information, and correlation context.
        """
        ...


__all__ = ["ProtocolPostgresAdapter"]
