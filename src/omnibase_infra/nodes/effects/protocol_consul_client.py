# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for Consul service registration client.

This module defines the protocol that Consul clients must implement
to be used with the NodeRegistryEffect node.

Related:
    - NodeRegistryEffect: Effect node that uses this protocol
    - ProtocolPostgresAdapter: Protocol for PostgreSQL backend
"""

from __future__ import annotations

from typing import Protocol


class ProtocolConsulClient(Protocol):
    """Protocol for Consul service registration client.

    Implementations must provide async service registration capability.
    """

    async def register_service(
        self,
        service_id: str,
        service_name: str,
        tags: list[str],
        health_check: dict[str, str] | None = None,
    ) -> dict[str, bool | str]:
        """Register a service in Consul.

        Args:
            service_id: Unique identifier for the service instance.
            service_name: Name of the service for discovery.
            tags: List of tags for filtering.
            health_check: Optional health check configuration.

        Returns:
            Dict with "success" bool and optional "error" string.
        """
        ...


__all__ = ["ProtocolConsulClient"]
