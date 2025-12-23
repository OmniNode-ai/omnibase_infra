# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for PostgreSQL registration persistence.

This module defines the protocol that PostgreSQL adapters must implement
to be used with the NodeRegistryEffect node.

Related:
    - NodeRegistryEffect: Effect node that uses this protocol
    - ProtocolConsulClient: Protocol for Consul backend
    - ModelBackendResult: Structured result model for backend operations
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from omnibase_infra.nodes.effects.models import ModelBackendResult


class ProtocolPostgresAdapter(Protocol):
    """Protocol for PostgreSQL registration persistence.

    Implementations must provide async upsert capability for
    registration records.
    """

    async def upsert(
        self,
        node_id: UUID,
        node_type: str,
        node_version: str,
        endpoints: dict[str, str],
        metadata: dict[str, str],
    ) -> ModelBackendResult:
        """Upsert a node registration record.

        Args:
            node_id: Unique identifier for the node.
            node_type: Type of ONEX node.
            node_version: Semantic version of the node.
            endpoints: Dict of endpoint type to URL.
            metadata: Additional metadata.

        Returns:
            ModelBackendResult with success status, optional error message,
            timing information, and correlation context.
        """
        ...


__all__ = ["ProtocolPostgresAdapter"]
