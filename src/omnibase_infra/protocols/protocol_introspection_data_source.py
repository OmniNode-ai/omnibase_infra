# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for a service that can report its own introspection data."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)


@runtime_checkable
class ProtocolIntrospectionDataSource(Protocol):
    """A service that reports the introspection data describing its own node."""

    async def get_introspection_data(self) -> ModelNodeIntrospectionEvent:
        """Return the node's current introspection event."""
        ...


__all__: list[str] = ["ProtocolIntrospectionDataSource"]
