#!/usr/bin/env python3
"""
CanonicalStoreService Protocol - Interface Definition.

Protocol (interface) for the CanonicalStoreService to enable dependency
injection and testing without requiring the full implementation.

This protocol defines the minimal interface required by ProjectionStoreService
for fallback scenarios when projections are stale.

ONEX v2.0 Compliance:
- Protocol-based dependency injection
- Pure interface definition (no implementation)
- Used by ProjectionStoreService for canonical fallback

Pure Reducer Refactor - Wave 2, Workstream 2B
"""

from typing import Protocol

from omninode_bridge.infrastructure.entities.model_workflow_state import (
    ModelWorkflowState,
)


class CanonicalStoreProtocol(Protocol):
    """
    Protocol for CanonicalStoreService interface.

    This protocol defines the minimal interface required for canonical
    store operations. It allows ProjectionStoreService to depend on
    the interface without requiring the full implementation.

    Methods:
        get_state: Retrieve canonical workflow state by key

    Example:
        >>> class CanonicalStoreService:
        ...     async def get_state(self, workflow_key: str) -> ModelWorkflowState:
        ...         # Implementation
        ...         pass
        >>> # CanonicalStoreService automatically satisfies the protocol
    """

    async def get_state(self, workflow_key: str) -> ModelWorkflowState:
        """
        Retrieve canonical workflow state.

        Args:
            workflow_key: Unique workflow identifier

        Returns:
            ModelWorkflowState: Canonical state record

        Raises:
            KeyError: If workflow_key does not exist
            DatabaseError: On database access errors

        Example:
            >>> service = CanonicalStoreService(...)
            >>> state = await service.get_state("workflow-123")
            >>> assert state.version >= 1
        """
        ...
