# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for PostgreSQL registration persistence.

This module defines the protocol that PostgreSQL adapters must implement
to be used with the NodeRegistryEffect node.

Thread Safety:
    Implementations MUST be thread-safe for concurrent async calls.
    Multiple async tasks may invoke upsert() simultaneously for
    different or identical node registrations.

Related:
    - NodeRegistryEffect: Effect node that uses this protocol
    - ProtocolConsulClient: Protocol for Consul backend
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID


class ProtocolPostgresAdapter(Protocol):
    """Protocol for PostgreSQL registration persistence.

    Implementations must provide async upsert capability for
    registration records.

    Thread Safety:
        Implementations MUST be thread-safe for concurrent async calls.

        **Guarantees implementers MUST provide:**
            - Concurrent upsert() calls are safe
            - Connection pooling (if used) is async-safe
            - Database transactions are properly isolated

        **What callers can assume:**
            - Multiple coroutines can call upsert() concurrently
            - Each upsert operation is independent
            - Failures in one upsert do not affect others
    """

    async def upsert(
        self,
        node_id: UUID,
        node_type: str,
        node_version: str,
        endpoints: dict[str, str],
        metadata: dict[str, str],
    ) -> dict[str, bool | str]:
        """Upsert a node registration record.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of ONEX node.
            node_version: Semantic version of the node.
            endpoints: Dict of endpoint type to URL.
            metadata: Additional metadata.

        Returns:
            Dict with "success" bool and optional "error" string.
        """
        ...


__all__ = ["ProtocolPostgresAdapter"]
