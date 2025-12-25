# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocols for the registration orchestrator workflow.

This module defines the protocols for the reducer and effect nodes that
participate in the registration workflow. These protocols enable duck typing
and dependency injection.

Protocol Responsibilities:
    ProtocolReducer: Pure function that computes intents from events
    ProtocolEffect: Side-effectful executor that performs infrastructure operations

Thread Safety:
    All protocol implementations MUST be thread-safe for concurrent async calls.

    ProtocolReducer:
    - Same reducer instance may process multiple events concurrently
    - Treat ModelReducerState as immutable (return new instances)
    - Avoid instance-level caches that could cause race conditions

    ProtocolEffect:
    - Multiple async tasks may invoke execute_intent() simultaneously
    - Use asyncio.Lock for any shared mutable state
    - Ensure underlying clients (Consul, PostgreSQL) are async-safe

Error Handling and Sanitization:
    All implementations MUST follow ONEX error sanitization guidelines.

    NEVER include in error messages:
    - Passwords, API keys, tokens, secrets
    - Full connection strings with credentials
    - PII (names, emails, SSNs, phone numbers)
    - Internal IP addresses (in production)
    - Private keys or certificates
    - Raw event payload content (may contain secrets)

    SAFE to include in error messages:
    - Service names (e.g., "consul", "postgres")
    - Operation names (e.g., "register", "upsert")
    - Correlation IDs (always include for tracing)
    - Error codes (e.g., EnumCoreErrorCode values)
    - Sanitized hostnames (e.g., "db.example.com")
    - Port numbers, retry counts, timeout values
    - Field names that are invalid or missing
    - node_id (UUID, not PII)

    ProtocolEffect errors should use InfraError subclasses:
    - InfraConnectionError: Connection failures
    - InfraTimeoutError: Operation timeouts
    - InfraAuthenticationError: Auth failures
    - InfraUnavailableError: Service unavailable

Design Notes:
    - Uses @runtime_checkable for duck typing support
    - Type hints use forward references to avoid circular imports
    - The orchestrator owns the workflow; reducer and effect are pluggable
    - Domain-grouped protocols per ONEX conventions (CLAUDE.md)

Related Modules:
    - omnibase_infra.models.registration: Event models
    - omnibase_infra.nodes.node_registry_effect: Effect implementation
    - OMN-912: Intent models in omnibase_core (pending)
    - OMN-889: Reducer implementation (pending)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import UUID

from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_execution_result import (
    ModelReducerExecutionResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registration_intent import (
    ModelRegistrationIntent,
)

if TYPE_CHECKING:
    from omnibase_infra.models.registration import ModelNodeIntrospectionEvent
    from omnibase_infra.nodes.node_registration_orchestrator.models import (
        ModelIntentExecutionResult,
    )


@runtime_checkable
class ProtocolReducer(Protocol):
    """Protocol for the reducer that computes registration intents.

    The reducer implements a pure function pattern: given the current state
    and an incoming event, it returns updated state plus a list of typed
    intents describing what infrastructure operations should occur.

    Thread Safety:
        Implementations MUST be thread-safe for concurrent async calls.
        The same reducer instance may process multiple events concurrently.
        Follow these guidelines:
        - Treat ModelReducerState as immutable (create new instances)
        - Do not store event-specific data as instance attributes
        - Use frozenset for processed_node_ids to ensure immutability

    Error Handling:
        When validation fails, raise ValueError with sanitized messages:
        - NEVER include raw event payloads in error messages
        - NEVER expose PII from node metadata
        - SAFE to include: node_id, event_type, field names, correlation_id

    Contract:
        - Reducer MUST be deterministic (same inputs -> same outputs)
        - Reducer MUST NOT perform I/O operations
        - Reducer MUST return valid intents that the effect node can execute
        - Reducer MUST sanitize any data included in ValueError messages
        - Reducer MAY filter duplicate or invalid events

    Example:
        ```python
        from uuid import uuid4

        class MyReducer:
            async def reduce(
                self,
                state: ModelReducerState,
                event: ModelNodeIntrospectionEvent,
            ) -> ModelReducerExecutionResult:
                # Validate event - use sanitized error messages
                if not event.node_id:
                    raise ValueError("Missing required field: node_id")

                if event.node_id in state.processed_node_ids:
                    # Already processed - return no_change result
                    return ModelReducerExecutionResult.no_change(state)

                # Propagate correlation_id with uuid4() fallback for tracing.
                # This ensures every intent has a valid correlation_id even if
                # the source event lacks one (e.g., legacy events, test fixtures).
                correlation_id = event.correlation_id or uuid4()

                # Generate intents with typed payloads
                intents = [
                    ModelConsulRegistrationIntent(
                        operation="register",
                        node_id=event.node_id,
                        correlation_id=correlation_id,
                        payload=ModelConsulIntentPayload(
                            service_name=f"node-{event.node_type}",
                        ),
                    ),
                    ModelPostgresUpsertIntent(
                        operation="upsert",
                        node_id=event.node_id,
                        correlation_id=correlation_id,
                        payload=ModelPostgresIntentPayload(
                            node_id=event.node_id,
                            node_type=event.node_type,
                            node_version=event.node_version,
                            capabilities=event.capabilities,  # Strongly-typed model
                            endpoints=event.endpoints,
                            node_role=event.node_role,
                            metadata=event.metadata,  # Strongly-typed model
                            correlation_id=correlation_id,
                            network_id=event.network_id,
                            deployment_id=event.deployment_id,
                            epoch=event.epoch,
                            timestamp=event.timestamp.isoformat(),
                        ),
                    ),
                ]

                # Update state - create new immutable state
                new_state = ModelReducerState(
                    last_event_timestamp=event.timestamp.isoformat(),
                    processed_node_ids=state.processed_node_ids | {event.node_id},
                    pending_registrations=state.pending_registrations + len(intents),
                )

                return ModelReducerExecutionResult.with_intents(new_state, intents)
        ```
    """

    async def reduce(
        self,
        state: ModelReducerState,
        event: ModelNodeIntrospectionEvent,
    ) -> ModelReducerExecutionResult:
        """Reduce an introspection event to state and intents.

        This method processes an incoming introspection event and produces
        a ModelReducerExecutionResult containing:
        1. Updated reducer state (for deduplication, rate limiting, etc.)
        2. A list of intents describing infrastructure operations to perform

        Thread Safety:
            This method MUST be safe to call concurrently from multiple
            async tasks. Implementations should:
            - Not modify the input state object
            - Return a new ModelReducerState instance
            - Avoid instance-level mutation

        Error Sanitization:
            When raising ValueError, NEVER include:
            - Raw event payload content (may contain secrets)
            - Full node metadata dumps
            - PII or credentials

            SAFE to include in ValueError messages:
            - Field names that are invalid or missing
            - node_id (UUID, not PII)
            - event_type or correlation_id

        Args:
            state: Current reducer state. Pass ModelReducerState.initial()
                for the first reduction. Treat as immutable.
            event: The introspection event to process. Contains node metadata,
                capabilities, and endpoints. May contain sensitive data
                that MUST NOT appear in error messages.

        Returns:
            ModelReducerExecutionResult containing:
                - state: NEW reducer state instance (do not mutate input)
                - intents: List of intents for the effect node to execute.
                  May be empty if the event should be filtered.

            Use factory methods for common patterns:
                - ModelReducerExecutionResult.no_change(state) for filtered events
                - ModelReducerExecutionResult.with_intents(state, intents) for normal flow
                - ModelReducerExecutionResult.empty() for initial/reset state

        Raises:
            ValueError: If the event is malformed or cannot be processed.
                Error message MUST be sanitized - no payload content or PII.

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

    Thread Safety:
        Implementations MUST be thread-safe for concurrent async calls.
        Multiple async tasks may invoke execute_intent() simultaneously.
        Use asyncio.Lock for any shared mutable state.

    Error Handling:
        Implementations MUST follow error sanitization guidelines:
        - NEVER include credentials or secrets in error messages
        - ALWAYS include correlation_id for distributed tracing
        - Use InfraError subclasses (InfraConnectionError, InfraTimeoutError, etc.)
        - Capture errors in ModelIntentExecutionResult.error, do not raise

    Contract:
        - Effect MUST execute exactly the operation specified by the intent
        - Effect MUST propagate correlation_id for distributed tracing
        - Effect MUST return a result even on failure (with success=False)
        - Effect MUST sanitize error messages before storing in result
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
                    # Sanitize error - never include credentials
                    sanitized_error = f"{intent.kind} {intent.operation} failed"
                    return ModelIntentExecutionResult(
                        intent_kind=intent.kind,
                        success=False,
                        error=sanitized_error,
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

        Thread Safety:
            This method MUST be safe to call concurrently from multiple
            async tasks. Implementations should not rely on instance state
            that could be modified by concurrent calls.

        Error Sanitization:
            The returned ModelIntentExecutionResult.error field MUST NOT
            contain sensitive information. Sanitize all error messages:
            - Remove credentials, tokens, connection strings
            - Include only: service name, operation, correlation_id
            - Use generic error descriptions, not raw exception messages

        Args:
            intent: The typed intent to execute. Contains:
                - kind: Target system ('consul', 'postgres', etc.)
                - operation: Action to perform ('register', 'upsert', etc.)
                - node_id: Target node identifier
                - correlation_id: Intent-level correlation ID
                - payload: Operation-specific data (may contain sensitive
                  data that MUST NOT appear in error messages)
            correlation_id: Request-level correlation ID for tracing.
                This may differ from intent.correlation_id in batched
                workflows. ALWAYS include in logs and error context.

        Returns:
            ModelIntentExecutionResult containing:
                - intent_kind: Echoed from intent.kind
                - success: True if operation completed without error
                - error: Sanitized error message if failed, None otherwise.
                  MUST NOT contain credentials or PII.
                - execution_time_ms: Duration of the operation

        Raises:
            This method SHOULD NOT raise exceptions. All errors should be
            captured in the returned ModelIntentExecutionResult with
            success=False and a sanitized error message.
        """
        ...


__all__ = [
    "ProtocolEffect",
    "ProtocolReducer",
]
