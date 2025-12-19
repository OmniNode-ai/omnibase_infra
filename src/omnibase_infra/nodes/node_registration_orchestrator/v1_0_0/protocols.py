# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for the registration orchestrator node.

These protocols define the contracts for reducer and effect interactions.
The orchestrator depends on these protocols, not concrete implementations,
enabling loose coupling and easy testing with mocks.

Protocol Responsibilities:
    ProtocolReducer: Pure function that computes intents from events
    ProtocolEffect: Side-effectful executor that performs infrastructure operations

Design Notes:
    - Both protocols use @runtime_checkable for duck typing support
    - Type hints use forward references to avoid circular imports
    - The orchestrator owns the workflow; reducer and effect are pluggable

Related Modules:
    - omnibase_infra.models.registration: Event models
    - omnibase_infra.nodes.node_registry_effect: Effect implementation
    - OMN-889: Reducer implementation (pending)
    - OMN-912: Intent models in omnibase_core (pending)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
    from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models import (
        ModelIntentExecutionResult,
    )


class ModelReducerState(BaseModel):
    """State model for the registration reducer.

    This model captures the reducer's internal state between reductions.
    The orchestrator treats this as an opaque container that it passes
    to the reducer but does not inspect.

    Attributes:
        last_event_timestamp: ISO timestamp of the last processed event.
        processed_node_ids: Set of node IDs that have been processed.
        pending_registrations: Count of registrations awaiting confirmation.

    Note:
        This is a minimal placeholder. The actual reducer (OMN-889) may
        extend this with additional state fields for deduplication,
        rate limiting, or batching logic.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    last_event_timestamp: str | None = Field(
        default=None,
        description="ISO timestamp of the last processed event",
    )
    processed_node_ids: frozenset[UUID] = Field(
        default_factory=frozenset,
        description="Set of node IDs that have been processed",
    )
    pending_registrations: int = Field(
        default=0,
        ge=0,
        description="Count of registrations awaiting confirmation",
    )

    @classmethod
    def initial(cls) -> ModelReducerState:
        """Create an initial empty state.

        Returns:
            A fresh ModelReducerState with default values.
        """
        return cls()


class ModelRegistrationIntent(BaseModel):
    """Base model for registration intents.

    Intents are typed instructions that the reducer produces and the
    effect node executes. Each intent represents a single infrastructure
    operation to perform.

    Attributes:
        kind: Discriminator for the intent type (e.g., 'consul', 'postgres').
        operation: The specific operation within that kind.
        node_id: Target node ID for the operation.
        correlation_id: Correlation ID for distributed tracing.
        payload: Operation-specific data as a dictionary.

    Note:
        This is a placeholder for the discriminated union of intents
        that will be defined in omnibase_core (OMN-912). The actual
        implementation will use a tagged union pattern like:
        ConsulRegisterIntent | ConsulDeregisterIntent | PostgresUpsertIntent
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    kind: str = Field(
        ...,
        min_length=1,
        description="Intent type discriminator (e.g., 'consul', 'postgres')",
    )
    operation: str = Field(
        ...,
        min_length=1,
        description="Operation to perform (e.g., 'register', 'upsert')",
    )
    node_id: UUID = Field(
        ...,
        description="Target node ID for the operation",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )
    payload: dict[str, str | int | float | bool | None | list | dict] = Field(
        default_factory=dict,
        description="Operation-specific data",
    )


@runtime_checkable
class ProtocolReducer(Protocol):
    """Protocol for the reducer that computes registration intents.

    The reducer implements a pure function pattern: given the current state
    and an incoming event, it returns updated state plus a list of typed
    intents describing what infrastructure operations should occur.

    Contract:
        - Reducer MUST be deterministic (same inputs -> same outputs)
        - Reducer MUST NOT perform I/O operations
        - Reducer MUST return valid intents that the effect node can execute
        - Reducer MAY filter duplicate or invalid events

    Example:
        ```python
        class MyReducer:
            async def reduce(
                self,
                state: ModelReducerState,
                event: ModelNodeIntrospectionEvent,
            ) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
                # Validate event
                if event.node_id in state.processed_node_ids:
                    return state, []  # Already processed, no-op

                # Generate intents
                intents = [
                    ModelRegistrationIntent(
                        kind="consul",
                        operation="register",
                        node_id=event.node_id,
                        correlation_id=event.correlation_id or uuid4(),
                        payload={"service_name": f"node-{event.node_type}"},
                    ),
                    ModelRegistrationIntent(
                        kind="postgres",
                        operation="upsert",
                        node_id=event.node_id,
                        correlation_id=event.correlation_id or uuid4(),
                        payload=event.model_dump(),
                    ),
                ]

                # Update state
                new_state = ModelReducerState(
                    last_event_timestamp=event.timestamp.isoformat(),
                    processed_node_ids=state.processed_node_ids | {event.node_id},
                    pending_registrations=state.pending_registrations + len(intents),
                )

                return new_state, intents
        ```
    """

    async def reduce(
        self,
        state: ModelReducerState,
        event: ModelNodeIntrospectionEvent,
    ) -> tuple[ModelReducerState, list[ModelRegistrationIntent]]:
        """Reduce an introspection event to state and intents.

        This method processes an incoming introspection event and produces:
        1. Updated reducer state (for deduplication, rate limiting, etc.)
        2. A list of intents describing infrastructure operations to perform

        Args:
            state: Current reducer state. Pass ModelReducerState.initial()
                for the first reduction.
            event: The introspection event to process. Contains node metadata,
                capabilities, and endpoints.

        Returns:
            A tuple of (new_state, intents) where:
                - new_state: Updated reducer state to pass to the next reduction
                - intents: List of intents for the effect node to execute.
                  May be empty if the event should be filtered.

        Raises:
            ValueError: If the event is malformed or cannot be processed.

        Note:
            This method is async to allow for potential future enhancements
            (e.g., async validation), but implementations MUST NOT perform
            actual I/O operations.
        """
        ...


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


__all__ = [
    "ModelReducerState",
    "ModelRegistrationIntent",
    "ProtocolEffect",
    "ProtocolReducer",
]
