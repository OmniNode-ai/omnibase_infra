# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol for the effect node that executes intents.

The effect node performs the actual I/O operations (Consul registration,
PostgreSQL upsert, etc.) based on typed intents from the reducer.

Protocol Responsibilities:
    - Execute infrastructure operations described by intents
    - Propagate correlation IDs for distributed tracing
    - Return results capturing success/failure and timing

Design Notes:
    - Uses @runtime_checkable for duck typing support
    - Type hints use forward references to avoid circular imports
    - The orchestrator owns the workflow; effect is pluggable

Related Modules:
    - omnibase_infra.nodes.node_registry_effect: Effect implementation
    - OMN-912: Intent models in omnibase_core (pending)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_registration_intent import (
    ModelRegistrationIntent,
)

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models import (
        ModelIntentExecutionResult,
    )


@runtime_checkable
class ProtocolEffect(Protocol):
    """Protocol for the effect node that executes intents.

    The effect node performs the actual I/O operations (Consul registration,
    PostgreSQL upsert, etc.) based on typed intents from the reducer.

    Contract:
        - Effect MUST execute exactly the operation specified by the intent
        - Effect MUST propagate correlation_id for distributed tracing
        - Effect MUST return a result even on failure (with success=False)
        - Effect MAY implement retry logic internally

    Example:
        ```python
        class MyEffect:
            def __init__(self, consul_client, db_client):
                self._consul = consul_client
                self._db = db_client

            async def execute_intent(
                self,
                intent: ModelRegistrationIntent,
                correlation_id: UUID,
            ) -> ModelIntentExecutionResult:
                start_time = time.perf_counter()
                try:
                    if intent.kind == "consul":
                        await self._consul.register(intent.payload)
                    elif intent.kind == "postgres":
                        await self._db.upsert(intent.payload)

                    return ModelIntentExecutionResult(
                        intent_kind=intent.kind,
                        success=True,
                        error=None,
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
                except Exception as e:
                    return ModelIntentExecutionResult(
                        intent_kind=intent.kind,
                        success=False,
                        error=str(e),
                        execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    )
        ```
    """

    async def execute_intent(
        self,
        intent: ModelRegistrationIntent,
        correlation_id: UUID,
    ) -> ModelIntentExecutionResult:
        """Execute a single registration intent.

        Performs the infrastructure operation described by the intent and
        returns a result capturing success/failure and timing.

        Args:
            intent: The typed intent to execute. Contains:
                - kind: Target system ('consul', 'postgres', etc.)
                - operation: Action to perform ('register', 'upsert', etc.)
                - node_id: Target node identifier
                - correlation_id: Intent-level correlation ID
                - payload: Operation-specific data
            correlation_id: Request-level correlation ID for tracing.
                This may differ from intent.correlation_id in batched
                workflows.

        Returns:
            ModelIntentExecutionResult containing:
                - intent_kind: Echoed from intent.kind
                - success: True if operation completed without error
                - error: Error message if failed, None otherwise
                - execution_time_ms: Duration of the operation

        Raises:
            This method SHOULD NOT raise exceptions. All errors should be
            captured in the returned ModelIntentExecutionResult with
            success=False.
        """
        ...


__all__ = ["ProtocolEffect"]
