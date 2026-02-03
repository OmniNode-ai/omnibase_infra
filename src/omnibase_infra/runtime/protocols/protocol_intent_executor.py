# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for intent executors in the contract persistence pipeline.

This module defines the protocol interface for intent executors that process
persistence intents from the ContractRegistryReducer.

Design:
    Uses a Generic Protocol with contravariant TypeVar to properly express that
    each handler accepts its specific payload type while the router can store
    any handler conforming to the protocol. This avoids the need for `object`
    workarounds and `cast()` at call sites.

Related:
    - IntentExecutionRouter: Uses this protocol for handler routing
    - OMN-1869: Implementation ticket
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable
from uuid import UUID

if TYPE_CHECKING:
    # These imports are only needed for type annotations.
    # Using TYPE_CHECKING avoids circular import during package initialization
    # (runtime.protocols is loaded before nodes is loaded).
    from omnibase_infra.nodes.contract_registry_reducer.models import (
        ModelPayloadCleanupTopicReferences,
        ModelPayloadDeactivateContract,
        ModelPayloadMarkStale,
        ModelPayloadUpdateHeartbeat,
        ModelPayloadUpdateTopic,
        ModelPayloadUpsertContract,
    )
    from omnibase_infra.nodes.effects.models.model_backend_result import (
        ModelBackendResult,
    )

    # Type alias for payload types (union of all supported payloads)
    # Defined inside TYPE_CHECKING since it references models only available there
    IntentPayloadType = (
        ModelPayloadUpsertContract
        | ModelPayloadUpdateTopic
        | ModelPayloadMarkStale
        | ModelPayloadUpdateHeartbeat
        | ModelPayloadDeactivateContract
        | ModelPayloadCleanupTopicReferences
    )

# Contravariant TypeVar for payload types - allows handlers with specific
# payload types to satisfy the protocol when used with broader type hints
PayloadT_contra = TypeVar("PayloadT_contra", contravariant=True)


@runtime_checkable
class ProtocolIntentExecutor(Protocol[PayloadT_contra]):
    """Generic protocol for intent executors.

    All persistence executors implement this interface, enabling type-safe
    routing without tight coupling to specific implementations.

    The protocol uses a contravariant TypeVar for the payload parameter,
    which correctly expresses that:
    - A handler accepting `ModelPayloadUpsertContract` can be stored where
      `ProtocolIntentExecutor[Any]` is expected
    - The router can call `handle()` with any payload that matches the
      handler's declared payload type

    Type Parameters:
        PayloadT_contra: The payload type this executor accepts (contravariant).

    Example:
        >>> class HandlerPostgresContractUpsert:
        ...     async def handle(
        ...         self,
        ...         payload: ModelPayloadUpsertContract,
        ...         correlation_id: UUID,
        ...     ) -> ModelBackendResult: ...
        >>>
        >>> # Handler satisfies ProtocolIntentExecutor[ModelPayloadUpsertContract]
        >>> # and can be stored as ProtocolIntentExecutor[Any]
        >>> handlers: dict[str, ProtocolIntentExecutor[Any]] = {
        ...     "upsert": HandlerPostgresContractUpsert(pool),
        ... }
    """

    async def handle(
        self,
        payload: PayloadT_contra,
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


# NOTE: IntentPayloadType is only available under TYPE_CHECKING.
# Import it with: if TYPE_CHECKING: from ...protocol_intent_executor import IntentPayloadType
__all__ = ["PayloadT_contra", "ProtocolIntentExecutor"]
