# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for intent executors in the contract persistence pipeline.

This module defines the protocol interface for intent executors that process
persistence intents from the ContractRegistryReducer.

Related:
    - IntentExecutionRouter: Uses this protocol for handler routing
    - OMN-1869: Implementation ticket
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.nodes.contract_registry_reducer.models import (
    ModelPayloadCleanupTopicReferences,
    ModelPayloadDeactivateContract,
    ModelPayloadMarkStale,
    ModelPayloadUpdateHeartbeat,
    ModelPayloadUpdateTopic,
    ModelPayloadUpsertContract,
)
from omnibase_infra.nodes.effects.models.model_backend_result import ModelBackendResult

# Type alias for payload types
IntentPayloadType = (
    ModelPayloadUpsertContract
    | ModelPayloadUpdateTopic
    | ModelPayloadMarkStale
    | ModelPayloadUpdateHeartbeat
    | ModelPayloadDeactivateContract
    | ModelPayloadCleanupTopicReferences
)


@runtime_checkable
class ProtocolIntentExecutor(Protocol):
    """Protocol for intent executors.

    All persistence executors implement this interface, enabling type-safe
    routing without tight coupling to specific implementations.

    The protocol defines:
    - An async handle method that takes a payload and correlation ID
    - Returns ModelBackendResult with execution status
    """

    async def handle(
        self,
        payload: IntentPayloadType,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute the handler operation.

        Args:
            payload: The typed payload model for this handler.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult with execution status.
        """
        ...


__all__ = ["ProtocolIntentExecutor", "IntentPayloadType"]
