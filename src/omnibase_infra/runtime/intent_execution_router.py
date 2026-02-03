# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Intent Execution Router for Contract Persistence Operations.

This module routes intents from ContractRegistryReducer to the appropriate
handler implementations in NodeContractPersistenceEffect for PostgreSQL
persistence.

Architecture:
    IntentExecutionRouter is the bridge between the REDUCER and EFFECT layers:
    - REDUCER (ContractRegistryReducer) emits intents based on event processing
    - ROUTER (this module) maps intent types to handlers and executes them
    - EFFECT (NodeContractPersistenceEffect handlers) perform PostgreSQL I/O

    This follows the ONEX unidirectional flow: EFFECT -> COMPUTE -> REDUCER -> ORCHESTRATOR,
    where the orchestrator layer uses this router to execute intents emitted by reducers.

Intent Routing:
    Each intent payload has an `intent_type` field that serves as the routing key:
    - postgres.upsert_contract -> HandlerPostgresContractUpsert
    - postgres.update_topic -> HandlerPostgresTopicUpdate
    - postgres.mark_stale -> HandlerPostgresMarkStale
    - postgres.update_heartbeat -> HandlerPostgresHeartbeat
    - postgres.deactivate_contract -> HandlerPostgresDeactivate
    - postgres.cleanup_topic_references -> HandlerPostgresCleanupTopics

Error Handling:
    The router executes each intent independently. A failure in one intent does
    not prevent execution of subsequent intents. This enables partial success
    scenarios where some operations complete while others need retry.

Coroutine Safety:
    This router is coroutine-safe for concurrent calls. Each handler execution
    acquires its own connection from the pool. Thread-safety depends on the
    underlying asyncpg.Pool implementation.

Related:
    - ContractRegistryReducer: Emits intents consumed by this router
    - NodeContractPersistenceEffect: Contains handler implementations
    - ServiceKernel: Integrates this router into the event processing pipeline
    - OMN-1869: Implementation ticket
    - OMN-1653: ContractRegistryReducer ticket (source of intents)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

# Direct import to avoid circular import through omnibase_infra.models
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_contract_persistence_effect.handlers import (
    HandlerPostgresCleanupTopics,
    HandlerPostgresContractUpsert,
    HandlerPostgresDeactivate,
    HandlerPostgresHeartbeat,
    HandlerPostgresMarkStale,
    HandlerPostgresTopicUpdate,
)
from omnibase_infra.runtime.models.model_intent_execution_summary import (
    ModelIntentExecutionSummary,
)
from omnibase_infra.runtime.protocols.protocol_intent_executor import (
    ProtocolIntentExecutor,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    import asyncpg

    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_core.models.reducer.model_intent import ModelIntent

_logger = logging.getLogger(__name__)


# Intent type routing constants
INTENT_UPSERT_CONTRACT = "postgres.upsert_contract"
INTENT_UPDATE_TOPIC = "postgres.update_topic"
INTENT_MARK_STALE = "postgres.mark_stale"
INTENT_UPDATE_HEARTBEAT = "postgres.update_heartbeat"
INTENT_DEACTIVATE_CONTRACT = "postgres.deactivate_contract"
INTENT_CLEANUP_TOPIC_REFERENCES = "postgres.cleanup_topic_references"


class IntentExecutionRouter:
    """Routes and executes intents from ContractRegistryReducer to persistence handlers.

    This router maps intent types to their corresponding PostgreSQL handlers and
    orchestrates execution. It handles errors gracefully per-intent, enabling
    partial success scenarios where some operations complete while others need
    retry.

    Attributes:
        _container: ONEX container for dependency injection (optional).
        _pool: asyncpg connection pool for database operations.
        _handlers: Cached handler instances keyed by intent type.

    Example:
        >>> import asyncpg
        >>> pool = await asyncpg.create_pool(dsn="...")
        >>> router = IntentExecutionRouter(container=None, postgres_pool=pool)
        >>> summary = await router.execute_intents(intents, correlation_id)
        >>> if summary.all_successful:
        ...     print("All intents executed successfully")

    Thread Safety:
        This router is coroutine-safe. Each handler execution acquires its own
        connection from the pool. The handler instances are created once during
        initialization and are stateless.

    See Also:
        - ContractRegistryReducer: Source of intents
        - NodeContractPersistenceEffect: Contains handler implementations
        - ServiceKernel: Integrates router into event processing
    """

    def __init__(
        self,
        container: ModelONEXContainer | None,
        postgres_pool: asyncpg.Pool,
    ) -> None:
        """Initialize the intent execution router.

        Args:
            container: ONEX container for dependency injection. May be None
                if router is used standalone without container DI.
            postgres_pool: asyncpg connection pool for database operations.
                The pool should be pre-configured and ready for use.

        Raises:
            ValueError: If postgres_pool is None.
        """
        if postgres_pool is None:
            raise ValueError("postgres_pool is required for IntentExecutionRouter")

        self._container = container
        self._pool = postgres_pool

        # Initialize handlers with the pool
        # Handlers implement ProtocolIntentExecutor[SpecificPayloadType] structurally.
        # Using object here per ONEX rules (Any is forbidden); handler.handle() call
        # below uses type: ignore since we guarantee correct payload routing at runtime.
        self._handlers: dict[str, object] = {
            INTENT_UPSERT_CONTRACT: HandlerPostgresContractUpsert(postgres_pool),
            INTENT_UPDATE_TOPIC: HandlerPostgresTopicUpdate(postgres_pool),
            INTENT_MARK_STALE: HandlerPostgresMarkStale(postgres_pool),
            INTENT_UPDATE_HEARTBEAT: HandlerPostgresHeartbeat(postgres_pool),
            INTENT_DEACTIVATE_CONTRACT: HandlerPostgresDeactivate(postgres_pool),
            INTENT_CLEANUP_TOPIC_REFERENCES: HandlerPostgresCleanupTopics(
                postgres_pool
            ),
        }

        _logger.info(
            "IntentExecutionRouter initialized",
            extra={
                "handler_count": len(self._handlers),
                "intent_types": list(self._handlers.keys()),
            },
        )

    async def execute_intents(
        self,
        intents: tuple[ModelIntent, ...],
        correlation_id: UUID,
    ) -> ModelIntentExecutionSummary:
        """Execute a batch of intents, routing each to its handler.

        Processes each intent independently, allowing partial success. A failure
        in one intent does not prevent execution of subsequent intents. This
        enables scenarios where some operations complete while others need retry.

        Args:
            intents: Tuple of intents to execute. Each intent contains a payload
                with an intent_type field used for routing.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelIntentExecutionSummary with:
                - total_intents: Number of intents processed
                - successful_count: Number that succeeded
                - failed_count: Number that failed
                - total_duration_ms: Batch execution time
                - results: Individual ModelBackendResult for each intent
                - correlation_id: Passed through for tracing

        Note:
            This method never raises exceptions. All errors are captured in the
            results for each intent. This enables callers to inspect partial
            failures and decide on retry strategies.

            Intents with unknown intent_type values are logged and marked as
            failed with an appropriate error message.

        Example:
            >>> summary = await router.execute_intents(intents, correlation_id)
            >>> for result in summary.results:
            ...     if not result.success:
            ...         print(f"Failed: {result.error}")
        """
        start_time = time.perf_counter()
        results: list[ModelBackendResult] = []
        successful_count = 0
        failed_count = 0

        _logger.info(
            "Starting intent batch execution",
            extra={
                "correlation_id": str(correlation_id),
                "intent_count": len(intents),
            },
        )

        for intent in intents:
            try:
                result = await self._execute_single_intent(intent, correlation_id)
                results.append(result)

                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1

            except Exception as e:  # ONEX: catch-all for unexpected errors
                # Should not happen since _execute_single_intent handles errors,
                # but defense-in-depth to ensure we never crash the batch
                failed_count += 1
                sanitized_error = sanitize_error_message(e)
                _logger.exception(
                    "Unexpected error during intent execution",
                    extra={
                        "correlation_id": str(correlation_id),
                        "intent_id": str(getattr(intent, "intent_id", "unknown")),
                        "error": sanitized_error,
                    },
                )
                results.append(
                    ModelBackendResult(
                        success=False,
                        error=f"Unexpected error: {sanitized_error}",
                        error_code="INTENT_EXECUTION_UNEXPECTED_ERROR",
                        duration_ms=0.0,
                        backend_id="intent_router",
                        correlation_id=correlation_id,
                    )
                )

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        summary = ModelIntentExecutionSummary(
            total_intents=len(intents),
            successful_count=successful_count,
            failed_count=failed_count,
            total_duration_ms=total_duration_ms,
            results=tuple(results),
            correlation_id=correlation_id,
        )

        _logger.info(
            "Intent batch execution completed",
            extra={
                "correlation_id": str(correlation_id),
                "total_intents": summary.total_intents,
                "successful_count": summary.successful_count,
                "failed_count": summary.failed_count,
                "total_duration_ms": summary.total_duration_ms,
                "all_successful": summary.all_successful,
            },
        )

        return summary

    async def _execute_single_intent(
        self,
        intent: ModelIntent,
        correlation_id: UUID,
    ) -> ModelBackendResult:
        """Execute a single intent by routing to the appropriate handler.

        Args:
            intent: The intent to execute. Contains a payload with intent_type.
            correlation_id: Request correlation ID for distributed tracing.

        Returns:
            ModelBackendResult from the handler, or an error result if routing
            or execution fails.

        Note:
            This method never raises exceptions. All errors are captured in
            the returned ModelBackendResult.
        """
        start_time = time.perf_counter()

        try:
            # Extract payload from intent
            payload = intent.payload
            if payload is None:
                _logger.warning(
                    "Intent has no payload",
                    extra={
                        "correlation_id": str(correlation_id),
                        "intent_id": str(intent.intent_id),
                    },
                )
                return ModelBackendResult(
                    success=False,
                    error="Intent has no payload",
                    error_code="INTENT_NO_PAYLOAD",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    backend_id="intent_router",
                    correlation_id=correlation_id,
                )

            # Get intent_type from payload
            intent_type = getattr(payload, "intent_type", None)
            if intent_type is None:
                _logger.warning(
                    "Payload has no intent_type field",
                    extra={
                        "correlation_id": str(correlation_id),
                        "intent_id": str(intent.intent_id),
                        "payload_type": type(payload).__name__,
                    },
                )
                return ModelBackendResult(
                    success=False,
                    error="Payload has no intent_type field",
                    error_code="INTENT_TYPE_MISSING",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    backend_id="intent_router",
                    correlation_id=correlation_id,
                )

            # Look up handler for this intent type
            handler = self._handlers.get(intent_type)
            if handler is None:
                _logger.warning(
                    "No handler registered for intent type",
                    extra={
                        "correlation_id": str(correlation_id),
                        "intent_id": str(intent.intent_id),
                        "intent_type": intent_type,
                        "registered_types": list(self._handlers.keys()),
                    },
                )
                return ModelBackendResult(
                    success=False,
                    error=f"No handler for intent type: {intent_type}",
                    error_code="INTENT_TYPE_UNKNOWN",
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                    backend_id="intent_router",
                    correlation_id=correlation_id,
                )

            # Execute the handler
            _logger.debug(
                "Executing handler for intent",
                extra={
                    "correlation_id": str(correlation_id),
                    "intent_id": str(intent.intent_id),
                    "intent_type": intent_type,
                    "handler_class": type(handler).__name__,
                },
            )

            # All handlers implement ProtocolIntentExecutor structurally with signature:
            # handle(payload: SpecificPayloadType, correlation_id: UUID) -> ModelBackendResult
            # Using type: ignore since dict value is object per ONEX rules (Any forbidden)
            result: ModelBackendResult = await handler.handle(payload, correlation_id)  # type: ignore[attr-defined]

            _logger.debug(
                "Handler execution completed",
                extra={
                    "correlation_id": str(correlation_id),
                    "intent_id": str(intent.intent_id),
                    "intent_type": intent_type,
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                },
            )

            return result

        except (
            Exception
        ) as e:  # ONEX: catch-all for handler errors not caught internally
            duration_ms = (time.perf_counter() - start_time) * 1000
            sanitized_error = sanitize_error_message(e)
            _logger.exception(
                "Handler execution failed with unexpected error",
                extra={
                    "correlation_id": str(correlation_id),
                    "intent_id": str(getattr(intent, "intent_id", "unknown")),
                    "error": sanitized_error,
                    "duration_ms": duration_ms,
                },
            )
            return ModelBackendResult(
                success=False,
                error=sanitized_error,
                error_code="HANDLER_EXECUTION_ERROR",
                duration_ms=duration_ms,
                backend_id="intent_router",
                correlation_id=correlation_id,
            )

    @property
    def supported_intent_types(self) -> tuple[str, ...]:
        """Get the list of intent types supported by this router.

        Returns:
            Tuple of supported intent type strings.
        """
        return tuple(self._handlers.keys())


__all__: list[str] = [
    "IntentExecutionRouter",
    "ModelIntentExecutionSummary",
    "ProtocolIntentExecutor",
    "INTENT_UPSERT_CONTRACT",
    "INTENT_UPDATE_TOPIC",
    "INTENT_MARK_STALE",
    "INTENT_UPDATE_HEARTBEAT",
    "INTENT_DEACTIVATE_CONTRACT",
    "INTENT_CLEANUP_TOPIC_REFERENCES",
]
