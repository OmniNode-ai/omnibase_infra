# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Message Dispatch Engine.

Runtime dispatch engine for routing messages based on topic category and
message type. Routes incoming messages to registered dispatchers and collects
dispatcher outputs for publishing.

Design Principles:
    - **Pure Routing**: Routes messages to dispatchers, no workflow inference
    - **Deterministic**: Same input always produces same dispatcher selection
    - **Fan-out Support**: Multiple dispatchers can process the same message type
    - **Freeze-After-Init**: Thread-safe after registration phase completes
    - **Observable**: Structured logging and comprehensive metrics

Architecture:
    The dispatch engine provides:
    - Route registration for topic pattern matching
    - Dispatcher registration by category and message type
    - Message dispatch with category validation
    - Metrics collection for observability
    - Structured logging for debugging and monitoring

    It does NOT:
    - Infer workflow semantics from message content
    - Manage dispatcher lifecycle (dispatchers are external)
    - Perform message transformation or enrichment
    - Make decisions about message ordering or priority

Data Flow:
    ```
    +------------------------------------------------------------------+
    |                   Message Dispatch Engine                         |
    +------------------------------------------------------------------+
    |                                                                  |
    |   1. Parse Topic       2. Validate          3. Match Dispatchers |
    |        |                   |                       |             |
    |        |  topic string     |  category match       |             |
    |        |-------------------|----------------------|             |
    |        |                   |                       |             |
    |        | EnumMessageCategory                       | dispatchers[]|
    |        |<------------------|                       |------------>|
    |        |                   |                       |             |
    |   4. Execute Dispatchers 5. Collect Outputs  6. Return Result    |
    |        |                   |                       |             |
    |        | dispatcher outputs|  aggregate           |             |
    |        |-------------------|----------------------|             |
    |        |                   |                       |             |
    |        |                   |  ModelDispatchResult  |             |
    |        |<------------------|<---------------------|             |
    |                                                                  |
    +------------------------------------------------------------------+
    ```

Thread Safety:
    MessageDispatchEngine follows the "freeze after init" pattern:

    1. **Registration Phase** (single-threaded): Register routes and dispatchers
    2. **Freeze**: Call freeze() to prevent further modifications
    3. **Dispatch Phase** (multi-threaded safe): Route messages to dispatchers

    After freeze(), the engine becomes read-only and can be safely shared
    across threads for concurrent dispatch operations.

Related:
    - OMN-934: Message dispatch engine implementation
    - EnvelopeRouter: Transport-agnostic orchestrator (reference for freeze pattern)

.. versionadded:: 0.4.0
"""

from __future__ import annotations

__all__ = ["MessageDispatchEngine"]

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_metrics import ModelDispatchMetrics
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.models.dispatch.model_dispatcher_metrics import (
    ModelDispatcherMetrics,
)

# Module-level logger for fallback when no custom logger is provided
_module_logger = logging.getLogger(__name__)

# Type alias for dispatcher functions
# Dispatchers can be sync or async, take an envelope and return Any (dispatcher output)
DispatcherFunc = Callable[[ModelEventEnvelope[Any]], Any | Awaitable[Any]]


class DispatchEntryInternal:
    """
    Internal storage for dispatcher registration metadata.

    This class is an implementation detail and not part of the public API.
    It stores the dispatcher callable and associated metadata for the
    MessageDispatchEngine's internal routing.
    """

    __slots__ = ("category", "dispatcher", "dispatcher_id", "message_types")

    def __init__(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None,
    ) -> None:
        self.dispatcher_id = dispatcher_id
        self.dispatcher = dispatcher
        self.category = category
        self.message_types = message_types  # None means "all types"


class MessageDispatchEngine:
    """
    Runtime dispatch engine for message routing.

    Routes messages based on topic category and message type to registered
    dispatchers. Supports fan-out (multiple dispatchers per message type) and
    collects dispatcher outputs for publishing.

    Key Characteristics:
        - **Pure Routing**: No workflow inference or semantic understanding
        - **Deterministic**: Same input always produces same dispatcher selection
        - **Fan-out**: Multiple dispatchers can process the same message type
        - **Observable**: Structured logging and comprehensive metrics

    Registration Semantics:
        - **Routes**: Keyed by route_id, duplicates raise error
        - **Dispatchers**: Keyed by dispatcher_id, duplicates raise error
        - Both must complete before freeze() is called

    Thread Safety:
        Follows the freeze-after-init pattern. All registrations must complete
        before calling freeze(). After freeze(), dispatch operations are
        thread-safe for concurrent access.

    Logging Levels:
        - **INFO**: Dispatch start/complete with topic, category, dispatcher count
        - **DEBUG**: Dispatcher execution details, routing decisions
        - **WARNING**: No dispatchers found, category mismatches
        - **ERROR**: Dispatcher exceptions, validation failures

    Example:
        >>> from omnibase_core.runtime import MessageDispatchEngine
        >>> from omnibase_infra.models.dispatch import ModelDispatchRoute
        >>> from omnibase_infra.enums import EnumMessageCategory
        >>>
        >>> # Create engine with optional custom logger
        >>> engine = MessageDispatchEngine(logger=my_logger)
        >>> engine.register_dispatcher(
        ...     dispatcher_id="user-dispatcher",
        ...     dispatcher=process_user_event,
        ...     category=EnumMessageCategory.EVENT,
        ...     message_types={"UserCreated", "UserUpdated"},
        ... )
        >>> engine.register_route(ModelDispatchRoute(
        ...     route_id="user-route",
        ...     topic_pattern="*.user.events.*",
        ...     message_category=EnumMessageCategory.EVENT,
        ...     dispatcher_id="user-dispatcher",
        ... ))
        >>> engine.freeze()
        >>>
        >>> # Dispatch (thread-safe after freeze)
        >>> result = await engine.dispatch("dev.user.events.v1", envelope)

    Attributes:
        _routes: Registry of routes by route_id
        _dispatchers: Registry of dispatchers by dispatcher_id
        _dispatchers_by_category: Index of dispatchers by category for fast lookup
        _frozen: If True, registration methods raise ModelOnexError
        _registration_lock: Lock protecting registration methods
        _structured_metrics: Pydantic-based metrics model for observability
        _logger: Optional custom logger for structured logging

    See Also:
        - :class:`~omnibase_infra.models.dispatch.ModelDispatchRoute`: Route model
        - :class:`~omnibase_infra.models.dispatch.ModelDispatchResult`: Result model
        - :class:`~omnibase_infra.models.dispatch.ModelDispatchMetrics`: Metrics model
        - :class:`~omnibase_core.runtime.EnvelopeRouter`: Reference implementation

    .. versionadded:: 0.4.0
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize MessageDispatchEngine with empty registries.

        Creates empty route and dispatcher registries and initializes metrics.
        Call freeze() after registration to enable thread-safe dispatch.

        Args:
            logger: Optional custom logger for structured logging.
                If not provided, uses module-level logger.
        """
        # Optional custom logger
        self._logger: logging.Logger = logger if logger is not None else _module_logger

        # Route storage: route_id -> ModelDispatchRoute
        self._routes: dict[str, ModelDispatchRoute] = {}

        # Dispatcher storage: dispatcher_id -> DispatchEntryInternal
        self._dispatchers: dict[str, DispatchEntryInternal] = {}

        # Index for fast dispatcher lookup by category
        # category -> list of dispatcher_ids
        self._dispatchers_by_category: dict[EnumMessageCategory, list[str]] = {
            EnumMessageCategory.EVENT: [],
            EnumMessageCategory.COMMAND: [],
            EnumMessageCategory.INTENT: [],
        }

        # Freeze state
        self._frozen: bool = False
        self._registration_lock: threading.Lock = threading.Lock()

        # Structured metrics (Pydantic model)
        self._structured_metrics: ModelDispatchMetrics = ModelDispatchMetrics()

        # Legacy metrics dict (for backwards compatibility)
        self._metrics: dict[str, Any] = {
            "dispatch_count": 0,
            "dispatch_success_count": 0,
            "dispatch_error_count": 0,
            "total_latency_ms": 0.0,
            "dispatcher_execution_count": 0,
            "dispatcher_error_count": 0,
            "routes_matched_count": 0,
            "no_dispatcher_count": 0,
            "category_mismatch_count": 0,
        }

    def register_route(self, route: ModelDispatchRoute) -> None:
        """
        Register a routing rule.

        Routes define how messages are matched to dispatchers based on topic
        pattern, message category, and optionally message type.

        Args:
            route: The routing rule to register. Must have unique route_id.

        Raises:
            ModelOnexError: If engine is frozen (INVALID_STATE)
            ModelOnexError: If route is None (INVALID_PARAMETER)
            ModelOnexError: If route with same route_id exists (DUPLICATE_REGISTRATION)
            ModelOnexError: If route.dispatcher_id references non-existent dispatcher
                (ITEM_NOT_REGISTERED) - only checked after freeze

        Example:
            >>> engine.register_route(ModelDispatchRoute(
            ...     route_id="order-events",
            ...     topic_pattern="*.order.events.*",
            ...     message_category=EnumMessageCategory.EVENT,
            ...     dispatcher_id="order-dispatcher",
            ... ))

        Note:
            Route-to-dispatcher consistency is NOT validated during registration
            to allow flexible registration order. Validation occurs at freeze()
            time or during dispatch.
        """
        if route is None:
            raise ModelOnexError(
                message="Cannot register None route. ModelDispatchRoute is required.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        with self._registration_lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot register route: MessageDispatchEngine is frozen. "
                    "Registration is not allowed after freeze() has been called.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )

            if route.route_id in self._routes:
                raise ModelOnexError(
                    message=f"Route with ID '{route.route_id}' is already registered. "
                    "Cannot register duplicate route ID.",
                    error_code=EnumCoreErrorCode.DUPLICATE_REGISTRATION,
                )

            self._routes[route.route_id] = route
            self._logger.debug(
                "Registered route '%s' for pattern '%s' (category=%s, dispatcher=%s)",
                route.route_id,
                route.topic_pattern,
                route.message_category,
                route.dispatcher_id,
            )

    def register_dispatcher(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None = None,
    ) -> None:
        """
        Register a message dispatcher.

        Dispatchers process messages that match their category and (optionally)
        message type. Multiple dispatchers can register for the same category
        and message type (fan-out pattern).

        Args:
            dispatcher_id: Unique identifier for this dispatcher
            dispatcher: Callable that processes messages. Can be sync or async.
                Signature: (envelope: ModelEventEnvelope[Any]) -> Any
            category: Message category this dispatcher processes
            message_types: Optional set of specific message types to handle.
                When None, handles all message types in the category.

        Raises:
            ModelOnexError: If engine is frozen (INVALID_STATE)
            ModelOnexError: If dispatcher_id is empty (INVALID_PARAMETER)
            ModelOnexError: If dispatcher is not callable (INVALID_PARAMETER)
            ModelOnexError: If dispatcher with same ID exists (DUPLICATE_REGISTRATION)

        Example:
            >>> async def process_user_event(envelope):
            ...     user_data = envelope.payload
            ...     # Process the event
            ...     return {"processed": True}
            >>>
            >>> engine.register_dispatcher(
            ...     dispatcher_id="user-event-dispatcher",
            ...     dispatcher=process_user_event,
            ...     category=EnumMessageCategory.EVENT,
            ...     message_types={"UserCreated", "UserUpdated"},
            ... )

        Note:
            Dispatchers are NOT automatically linked to routes. You must register
            routes separately that reference the dispatcher_id.
        """
        # Validate inputs before acquiring lock
        if not dispatcher_id or not dispatcher_id.strip():
            raise ModelOnexError(
                message="Dispatcher ID cannot be empty or whitespace.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        if dispatcher is None or not callable(dispatcher):
            raise ModelOnexError(
                message=f"Dispatcher for '{dispatcher_id}' must be callable. "
                f"Got {type(dispatcher).__name__}.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        if not isinstance(category, EnumMessageCategory):
            raise ModelOnexError(
                message=f"Category must be EnumMessageCategory, got {type(category).__name__}.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        with self._registration_lock:
            if self._frozen:
                raise ModelOnexError(
                    message="Cannot register dispatcher: MessageDispatchEngine is frozen. "
                    "Registration is not allowed after freeze() has been called.",
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                )

            if dispatcher_id in self._dispatchers:
                raise ModelOnexError(
                    message=f"Dispatcher with ID '{dispatcher_id}' is already registered. "
                    "Cannot register duplicate dispatcher ID.",
                    error_code=EnumCoreErrorCode.DUPLICATE_REGISTRATION,
                )

            # Store dispatcher entry
            entry = DispatchEntryInternal(
                dispatcher_id=dispatcher_id,
                dispatcher=dispatcher,
                category=category,
                message_types=message_types,
            )
            self._dispatchers[dispatcher_id] = entry

            # Update category index
            self._dispatchers_by_category[category].append(dispatcher_id)

            self._logger.debug(
                "Registered dispatcher '%s' for category %s (message_types=%s)",
                dispatcher_id,
                category,
                message_types if message_types else "all",
            )

    def freeze(self) -> None:
        """
        Freeze the engine to prevent further registration.

        Once frozen, any calls to register_route() or register_dispatcher()
        will raise ModelOnexError with INVALID_STATE. This enforces the
        read-only-after-init pattern for thread safety.

        The freeze operation validates route-to-dispatcher consistency:
        all routes must reference existing dispatchers.

        Raises:
            ModelOnexError: If any route references a non-existent dispatcher
                (ITEM_NOT_REGISTERED)

        Example:
            >>> engine = MessageDispatchEngine()
            >>> engine.register_dispatcher("d1", dispatcher, EnumMessageCategory.EVENT)
            >>> engine.register_route(route)
            >>> engine.freeze()  # Validates and freezes
            >>> assert engine.is_frozen

        Note:
            This is a one-way operation. There is no unfreeze() method
            by design, as unfreezing would defeat thread-safety guarantees.

        .. versionadded:: 0.4.0
        """
        with self._registration_lock:
            if self._frozen:
                # Idempotent - already frozen
                return

            # Validate all routes reference existing dispatchers
            for route in self._routes.values():
                if route.dispatcher_id not in self._dispatchers:
                    raise ModelOnexError(
                        message=f"Route '{route.route_id}' references dispatcher "
                        f"'{route.dispatcher_id}' which is not registered. "
                        "Register the dispatcher before freezing.",
                        error_code=EnumCoreErrorCode.ITEM_NOT_REGISTERED,
                    )

            self._frozen = True
            self._logger.info(
                "MessageDispatchEngine frozen with %d routes and %d dispatchers",
                len(self._routes),
                len(self._dispatchers),
            )

    @property
    def is_frozen(self) -> bool:
        """
        Check if the engine is frozen.

        Returns:
            True if frozen and registration is disabled, False otherwise

        .. versionadded:: 0.4.0
        """
        return self._frozen

    def _build_log_context(
        self,
        topic: str | None = None,
        category: EnumMessageCategory | None = None,
        message_type: str | None = None,
        dispatcher_id: str | None = None,
        dispatcher_count: int | None = None,
        duration_ms: float | None = None,
        correlation_id: str | None = None,
        trace_id: str | None = None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Build structured log context dictionary.

        Args:
            topic: The topic being dispatched to.
            category: The message category.
            message_type: The message type.
            dispatcher_id: Dispatcher ID (or comma-separated list).
            dispatcher_count: Number of dispatchers matched.
            duration_ms: Dispatch duration in milliseconds.
            correlation_id: Correlation ID from envelope.
            trace_id: Trace ID from envelope.
            error_code: Error code if dispatch failed.

        Returns:
            Dictionary with non-None values for structured logging.
        """
        context: dict[str, Any] = {}
        if topic is not None:
            context["topic"] = topic
        if category is not None:
            context["category"] = category.value
        if message_type is not None:
            context["message_type"] = message_type
        if dispatcher_id is not None:
            context["dispatcher_id"] = dispatcher_id
        if dispatcher_count is not None:
            context["dispatcher_count"] = dispatcher_count
        if duration_ms is not None:
            context["duration_ms"] = round(duration_ms, 3)
        if correlation_id is not None:
            context["correlation_id"] = correlation_id
        if trace_id is not None:
            context["trace_id"] = trace_id
        if error_code is not None:
            context["error_code"] = error_code
        return context

    async def dispatch(
        self,
        topic: str,
        envelope: ModelEventEnvelope[Any],
    ) -> ModelDispatchResult:
        """
        Dispatch a message to matching dispatchers.

        Routes the message based on topic category and message type, executes
        all matching dispatchers, and collects their outputs.

        Dispatch Process:
            1. Parse topic to extract message category
            2. Validate envelope category matches topic category
            3. Get message type from envelope payload
            4. Find all matching dispatchers (by category + message type)
            5. Execute dispatchers (fan-out)
            6. Collect outputs and return result

        Args:
            topic: The topic the message was received on (e.g., "dev.user.events.v1")
            envelope: The message envelope to dispatch

        Returns:
            ModelDispatchResult with dispatch status, metrics, and dispatcher outputs

        Raises:
            ModelOnexError: If engine is not frozen (INVALID_STATE)
            ModelOnexError: If topic is empty (INVALID_PARAMETER)
            ModelOnexError: If envelope is None (INVALID_PARAMETER)

        Example:
            >>> result = await engine.dispatch(
            ...     topic="dev.user.events.v1",
            ...     envelope=ModelEventEnvelope(payload=UserCreatedEvent(...)),
            ... )
            >>> if result.is_successful():
            ...     print(f"Dispatched to {result.output_count} dispatchers")

        Note:
            Dispatcher exceptions are caught and reported in the result.
            The dispatch continues to other dispatchers even if one fails.

        .. versionadded:: 0.4.0
        """
        # Enforce freeze contract
        if not self._frozen:
            raise ModelOnexError(
                message="dispatch() called before freeze(). "
                "Registration MUST complete and freeze() MUST be called before dispatch. "
                "This is required for thread safety.",
                error_code=EnumCoreErrorCode.INVALID_STATE,
            )

        # Validate inputs
        if not topic or not topic.strip():
            raise ModelOnexError(
                message="Topic cannot be empty or whitespace.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        if envelope is None:
            raise ModelOnexError(
                message="Cannot dispatch None envelope. ModelEventEnvelope is required.",
                error_code=EnumCoreErrorCode.INVALID_PARAMETER,
            )

        # Start timing
        start_time = time.perf_counter()
        dispatch_id = uuid4()
        started_at = datetime.now(UTC)

        # Extract correlation/trace IDs for logging
        correlation_id_str = (
            str(envelope.correlation_id) if envelope.correlation_id else None
        )
        trace_id_str = str(envelope.trace_id) if envelope.trace_id else None

        # Update dispatch count (atomic for simple int)
        self._metrics["dispatch_count"] += 1

        # Step 1: Parse topic to get category
        topic_category = EnumMessageCategory.from_topic(topic)
        if topic_category is None:
            self._metrics["dispatch_error_count"] += 1
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics["total_latency_ms"] += duration_ms

            # Update structured metrics
            self._structured_metrics = self._structured_metrics.record_dispatch(
                duration_ms=duration_ms,
                success=False,
                category=None,
                no_handler=False,
                category_mismatch=False,
                topic=topic,
            )

            # Log error
            self._logger.error(
                "Dispatch failed: invalid topic category",
                extra=self._build_log_context(
                    topic=topic,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id_str,
                    trace_id=trace_id_str,
                    error_code="INVALID_TOPIC_CATEGORY",
                ),
            )

            return ModelDispatchResult(
                dispatch_id=dispatch_id,
                status=EnumDispatchStatus.INVALID_MESSAGE,
                topic=topic,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                error_message=f"Cannot infer message category from topic '{topic}'. "
                "Topic must contain .events, .commands, or .intents segment.",
                error_code="INVALID_TOPIC_CATEGORY",
            )

        # Log dispatch start at INFO level
        self._logger.info(
            "Dispatch started",
            extra=self._build_log_context(
                topic=topic,
                category=topic_category,
                correlation_id=correlation_id_str,
                trace_id=trace_id_str,
            ),
        )

        # Step 2: Validate envelope category matches topic category
        envelope_category = envelope.infer_category()
        if envelope_category != topic_category:
            self._metrics["category_mismatch_count"] += 1
            self._metrics["dispatch_error_count"] += 1
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics["total_latency_ms"] += duration_ms

            # Update structured metrics
            self._structured_metrics = self._structured_metrics.record_dispatch(
                duration_ms=duration_ms,
                success=False,
                category=topic_category,
                category_mismatch=True,
                topic=topic,
            )

            # Log warning
            self._logger.warning(
                "Dispatch failed: category mismatch (envelope=%s, topic=%s)",
                envelope_category,
                topic_category,
                extra=self._build_log_context(
                    topic=topic,
                    category=topic_category,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id_str,
                    trace_id=trace_id_str,
                    error_code="CATEGORY_MISMATCH",
                ),
            )

            return ModelDispatchResult(
                dispatch_id=dispatch_id,
                status=EnumDispatchStatus.INVALID_MESSAGE,
                topic=topic,
                message_category=topic_category,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                error_message=f"Envelope category '{envelope_category}' does not match "
                f"topic category '{topic_category}'. Envelope payload type: "
                f"{type(envelope.payload).__name__}",
                error_code="CATEGORY_MISMATCH",
            )

        # Step 3: Get message type from payload
        message_type = type(envelope.payload).__name__

        # Step 4: Find matching dispatchers
        matching_dispatchers = self._find_matching_dispatchers(
            topic=topic,
            category=topic_category,
            message_type=message_type,
        )

        # Log routing decision at DEBUG level
        self._logger.debug(
            "Routing decision: %d dispatchers matched for message_type '%s'",
            len(matching_dispatchers),
            message_type,
            extra=self._build_log_context(
                topic=topic,
                category=topic_category,
                message_type=message_type,
                dispatcher_count=len(matching_dispatchers),
                correlation_id=correlation_id_str,
                trace_id=trace_id_str,
            ),
        )

        if not matching_dispatchers:
            self._metrics["no_dispatcher_count"] += 1
            self._metrics["dispatch_error_count"] += 1
            duration_ms = (time.perf_counter() - start_time) * 1000
            self._metrics["total_latency_ms"] += duration_ms

            # Update structured metrics
            self._structured_metrics = self._structured_metrics.record_dispatch(
                duration_ms=duration_ms,
                success=False,
                category=topic_category,
                no_handler=True,
                topic=topic,
            )

            # Log warning
            self._logger.warning(
                "No dispatcher found for category '%s' and message type '%s'",
                topic_category,
                message_type,
                extra=self._build_log_context(
                    topic=topic,
                    category=topic_category,
                    message_type=message_type,
                    dispatcher_count=0,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id_str,
                    trace_id=trace_id_str,
                    error_code="NO_DISPATCHER_FOUND",
                ),
            )

            return ModelDispatchResult(
                dispatch_id=dispatch_id,
                status=EnumDispatchStatus.NO_HANDLER,
                topic=topic,
                message_category=topic_category,
                message_type=message_type,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                error_message=f"No dispatcher registered for category '{topic_category}' "
                f"and message type '{message_type}' matching topic '{topic}'.",
                error_code="NO_DISPATCHER_FOUND",
            )

        # Step 5: Execute dispatchers and collect outputs
        outputs: list[str] = []
        dispatcher_errors: list[str] = []
        executed_dispatcher_ids: list[str] = []

        for dispatcher_entry in matching_dispatchers:
            self._metrics["dispatcher_execution_count"] += 1
            dispatcher_start_time = time.perf_counter()

            # Log dispatcher execution at DEBUG level
            self._logger.debug(
                "Executing dispatcher '%s'",
                dispatcher_entry.dispatcher_id,
                extra=self._build_log_context(
                    topic=topic,
                    category=topic_category,
                    message_type=message_type,
                    dispatcher_id=dispatcher_entry.dispatcher_id,
                    correlation_id=correlation_id_str,
                    trace_id=trace_id_str,
                ),
            )

            try:
                result = await self._execute_dispatcher(dispatcher_entry, envelope)
                dispatcher_duration_ms = (
                    time.perf_counter() - dispatcher_start_time
                ) * 1000
                executed_dispatcher_ids.append(dispatcher_entry.dispatcher_id)

                # Update per-dispatcher metrics
                existing_dispatcher_metrics = (
                    self._structured_metrics.dispatcher_metrics.get(
                        dispatcher_entry.dispatcher_id
                    )
                )
                if existing_dispatcher_metrics is None:
                    existing_dispatcher_metrics = ModelDispatcherMetrics(
                        dispatcher_id=dispatcher_entry.dispatcher_id
                    )
                new_dispatcher_metrics = existing_dispatcher_metrics.record_execution(
                    duration_ms=dispatcher_duration_ms,
                    success=True,
                    topic=topic,
                )
                new_dispatcher_metrics_dict = dict(
                    self._structured_metrics.dispatcher_metrics
                )
                new_dispatcher_metrics_dict[dispatcher_entry.dispatcher_id] = (
                    new_dispatcher_metrics
                )
                # Update structured metrics with new dispatcher metrics
                self._structured_metrics = ModelDispatchMetrics(
                    total_dispatches=self._structured_metrics.total_dispatches,
                    successful_dispatches=self._structured_metrics.successful_dispatches,
                    failed_dispatches=self._structured_metrics.failed_dispatches,
                    no_handler_count=self._structured_metrics.no_handler_count,
                    category_mismatch_count=self._structured_metrics.category_mismatch_count,
                    dispatcher_execution_count=self._structured_metrics.dispatcher_execution_count
                    + 1,
                    dispatcher_error_count=self._structured_metrics.dispatcher_error_count,
                    routes_matched_count=self._structured_metrics.routes_matched_count,
                    total_latency_ms=self._structured_metrics.total_latency_ms,
                    min_latency_ms=self._structured_metrics.min_latency_ms,
                    max_latency_ms=self._structured_metrics.max_latency_ms,
                    latency_histogram=self._structured_metrics.latency_histogram,
                    dispatcher_metrics=new_dispatcher_metrics_dict,
                    category_metrics=self._structured_metrics.category_metrics,
                )

                # Log dispatcher completion at DEBUG level
                self._logger.debug(
                    "Dispatcher '%s' completed successfully in %.2f ms",
                    dispatcher_entry.dispatcher_id,
                    dispatcher_duration_ms,
                    extra=self._build_log_context(
                        topic=topic,
                        category=topic_category,
                        message_type=message_type,
                        dispatcher_id=dispatcher_entry.dispatcher_id,
                        duration_ms=dispatcher_duration_ms,
                        correlation_id=correlation_id_str,
                        trace_id=trace_id_str,
                    ),
                )

                # If dispatcher returns a topic string or list of topics, add to outputs
                if isinstance(result, str) and result:
                    outputs.append(result)
                elif isinstance(result, list):
                    outputs.extend(str(r) for r in result if r)
            except (SystemExit, KeyboardInterrupt, GeneratorExit):
                # Never catch cancellation/exit signals
                raise
            except asyncio.CancelledError:
                # Never suppress async cancellation
                raise
            except Exception as e:
                dispatcher_duration_ms = (
                    time.perf_counter() - dispatcher_start_time
                ) * 1000
                self._metrics["dispatcher_error_count"] += 1
                error_msg = f"Dispatcher '{dispatcher_entry.dispatcher_id}' failed: {type(e).__name__}: {e}"
                dispatcher_errors.append(error_msg)

                # Update per-dispatcher metrics with error
                existing_dispatcher_metrics = (
                    self._structured_metrics.dispatcher_metrics.get(
                        dispatcher_entry.dispatcher_id
                    )
                )
                if existing_dispatcher_metrics is None:
                    existing_dispatcher_metrics = ModelDispatcherMetrics(
                        dispatcher_id=dispatcher_entry.dispatcher_id
                    )
                new_dispatcher_metrics = existing_dispatcher_metrics.record_execution(
                    duration_ms=dispatcher_duration_ms,
                    success=False,
                    topic=topic,
                    error_message=str(e),
                )
                new_dispatcher_metrics_dict = dict(
                    self._structured_metrics.dispatcher_metrics
                )
                new_dispatcher_metrics_dict[dispatcher_entry.dispatcher_id] = (
                    new_dispatcher_metrics
                )
                # Update structured metrics with new dispatcher metrics
                self._structured_metrics = ModelDispatchMetrics(
                    total_dispatches=self._structured_metrics.total_dispatches,
                    successful_dispatches=self._structured_metrics.successful_dispatches,
                    failed_dispatches=self._structured_metrics.failed_dispatches,
                    no_handler_count=self._structured_metrics.no_handler_count,
                    category_mismatch_count=self._structured_metrics.category_mismatch_count,
                    dispatcher_execution_count=self._structured_metrics.dispatcher_execution_count
                    + 1,
                    dispatcher_error_count=self._structured_metrics.dispatcher_error_count
                    + 1,
                    routes_matched_count=self._structured_metrics.routes_matched_count,
                    total_latency_ms=self._structured_metrics.total_latency_ms,
                    min_latency_ms=self._structured_metrics.min_latency_ms,
                    max_latency_ms=self._structured_metrics.max_latency_ms,
                    latency_histogram=self._structured_metrics.latency_histogram,
                    dispatcher_metrics=new_dispatcher_metrics_dict,
                    category_metrics=self._structured_metrics.category_metrics,
                )

                # Log error
                self._logger.exception(
                    "Dispatcher '%s' failed: %s",
                    dispatcher_entry.dispatcher_id,
                    e,
                    extra=self._build_log_context(
                        topic=topic,
                        category=topic_category,
                        message_type=message_type,
                        dispatcher_id=dispatcher_entry.dispatcher_id,
                        duration_ms=dispatcher_duration_ms,
                        correlation_id=correlation_id_str,
                        trace_id=trace_id_str,
                        error_code="DISPATCHER_EXCEPTION",
                    ),
                )

        # Step 6: Build result
        duration_ms = (time.perf_counter() - start_time) * 1000
        self._metrics["total_latency_ms"] += duration_ms
        self._metrics["routes_matched_count"] += len(matching_dispatchers)

        # Determine final status
        if dispatcher_errors:
            if executed_dispatcher_ids:
                # Partial success - some dispatchers executed
                status = EnumDispatchStatus.HANDLER_ERROR
                self._metrics["dispatch_error_count"] += 1
            else:
                # Total failure - no dispatchers executed
                status = EnumDispatchStatus.HANDLER_ERROR
                self._metrics["dispatch_error_count"] += 1
        else:
            status = EnumDispatchStatus.SUCCESS
            self._metrics["dispatch_success_count"] += 1

        # Update structured metrics with final dispatch result
        self._structured_metrics = self._structured_metrics.record_dispatch(
            duration_ms=duration_ms,
            success=status == EnumDispatchStatus.SUCCESS,
            category=topic_category,
            dispatcher_id=executed_dispatcher_ids[0]
            if executed_dispatcher_ids
            else None,
            handler_error=len(dispatcher_errors) > 0,
            routes_matched=len(matching_dispatchers),
            topic=topic,
            error_message=dispatcher_errors[0] if dispatcher_errors else None,
        )

        # Find route ID that matched (first matching route for logging)
        matched_route_id: str | None = None
        for route in self._routes.values():
            if route.matches(topic, topic_category, message_type):
                matched_route_id = route.route_id
                break

        # Log dispatch completion at INFO level
        dispatcher_ids_str = (
            ", ".join(executed_dispatcher_ids) if executed_dispatcher_ids else None
        )
        if status == EnumDispatchStatus.SUCCESS:
            self._logger.info(
                "Dispatch completed successfully",
                extra=self._build_log_context(
                    topic=topic,
                    category=topic_category,
                    message_type=message_type,
                    dispatcher_id=dispatcher_ids_str,
                    dispatcher_count=len(executed_dispatcher_ids),
                    duration_ms=duration_ms,
                    correlation_id=correlation_id_str,
                    trace_id=trace_id_str,
                ),
            )
        else:
            self._logger.error(
                "Dispatch completed with errors",
                extra=self._build_log_context(
                    topic=topic,
                    category=topic_category,
                    message_type=message_type,
                    dispatcher_id=dispatcher_ids_str,
                    dispatcher_count=len(matching_dispatchers),
                    duration_ms=duration_ms,
                    correlation_id=correlation_id_str,
                    trace_id=trace_id_str,
                    error_code="DISPATCHER_EXECUTION_ERROR",
                ),
            )

        return ModelDispatchResult(
            dispatch_id=dispatch_id,
            status=status,
            route_id=matched_route_id,
            dispatcher_id=dispatcher_ids_str,
            topic=topic,
            message_category=topic_category,
            message_type=message_type,
            duration_ms=duration_ms,
            started_at=started_at,
            completed_at=datetime.now(UTC),
            outputs=outputs if outputs else None,
            output_count=len(outputs),
            error_message="; ".join(dispatcher_errors) if dispatcher_errors else None,
            error_code="DISPATCHER_EXECUTION_ERROR" if dispatcher_errors else None,
            correlation_id=envelope.correlation_id,
            trace_id=envelope.trace_id,
            span_id=envelope.span_id,
        )

    def _find_matching_dispatchers(
        self,
        topic: str,
        category: EnumMessageCategory,
        message_type: str,
    ) -> list[DispatchEntryInternal]:
        """
        Find all dispatchers that match the given criteria.

        Matching is done in two phases:
        1. Find routes that match topic pattern and category
        2. Find dispatchers for those routes that accept the message type

        Args:
            topic: The topic to match
            category: The message category
            message_type: The specific message type

        Returns:
            List of matching dispatcher entries (may be empty)
        """
        matching_dispatchers: list[DispatchEntryInternal] = []
        seen_dispatcher_ids: set[str] = set()

        # Find all routes that match this topic and category
        for route in self._routes.values():
            if not route.enabled:
                continue
            if not route.matches_topic(topic):
                continue
            if route.message_category != category:
                continue
            # Route-level message type filter (if specified)
            if route.message_type is not None and route.message_type != message_type:
                continue

            # Get the dispatcher for this route
            dispatcher_id = route.dispatcher_id
            if dispatcher_id in seen_dispatcher_ids:
                # Avoid duplicate dispatcher execution
                continue

            entry = self._dispatchers.get(dispatcher_id)
            if entry is None:
                # Dispatcher not found (should have been caught at freeze)
                self._logger.warning(
                    "Route '%s' references missing dispatcher '%s'",
                    route.route_id,
                    dispatcher_id,
                )
                continue

            # Check dispatcher-level message type filter
            if (
                entry.message_types is not None
                and message_type not in entry.message_types
            ):
                continue

            matching_dispatchers.append(entry)
            seen_dispatcher_ids.add(dispatcher_id)

        return matching_dispatchers

    async def _execute_dispatcher(
        self,
        entry: DispatchEntryInternal,
        envelope: ModelEventEnvelope[Any],
    ) -> Any:
        """
        Execute a dispatcher (sync or async).

        Args:
            entry: The dispatcher entry containing the callable
            envelope: The message envelope to process

        Returns:
            Dispatcher result (any type)

        Raises:
            Any exception raised by the dispatcher
        """
        dispatcher = entry.dispatcher

        # Check if dispatcher is async
        if inspect.iscoroutinefunction(dispatcher):
            return await dispatcher(envelope)
        else:
            # Sync dispatcher - run in executor to avoid blocking
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, dispatcher, envelope)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get dispatch metrics for observability (legacy format).

        Returns a snapshot of current metrics including:
        - dispatch_count: Total number of dispatch calls
        - dispatch_success_count: Successful dispatches
        - dispatch_error_count: Failed dispatches
        - total_latency_ms: Cumulative latency in milliseconds
        - dispatcher_execution_count: Total dispatcher executions
        - dispatcher_error_count: Dispatcher execution failures
        - routes_matched_count: Total route matches
        - no_dispatcher_count: Dispatches with no matching dispatcher
        - category_mismatch_count: Category validation failures

        Returns:
            Dictionary with metrics (copy of internal state)

        Example:
            >>> metrics = engine.get_metrics()
            >>> print(f"Success rate: {metrics['dispatch_success_count'] / metrics['dispatch_count']:.1%}")

        Note:
            Returns a copy to prevent external modification.
            For high-frequency monitoring, consider caching the result.
            For structured metrics, use get_structured_metrics() instead.

        .. versionadded:: 0.4.0
        """
        # Return a copy to prevent external modification
        return dict(self._metrics)

    def get_structured_metrics(self) -> ModelDispatchMetrics:
        """
        Get structured dispatch metrics using Pydantic model.

        Returns a comprehensive metrics model including:
        - Dispatch counts and success/error rates
        - Latency statistics (average, min, max)
        - Latency histogram for distribution analysis
        - Per-dispatcher metrics breakdown
        - Per-category metrics breakdown

        Returns:
            ModelDispatchMetrics with all observability data

        Example:
            >>> metrics = engine.get_structured_metrics()
            >>> print(f"Success rate: {metrics.success_rate:.1%}")
            >>> print(f"Avg latency: {metrics.avg_latency_ms:.2f} ms")
            >>> for dispatcher_id, dispatcher_metrics in metrics.dispatcher_metrics.items():
            ...     print(f"Dispatcher {dispatcher_id}: {dispatcher_metrics.execution_count} executions")

        .. versionadded:: 0.4.0
        """
        return self._structured_metrics

    def reset_metrics(self) -> None:
        """
        Reset all metrics to initial state.

        Useful for testing or when starting a new monitoring period.
        Resets both legacy dict-based metrics and structured metrics.

        Example:
            >>> engine.reset_metrics()
            >>> assert engine.get_metrics()["dispatch_count"] == 0
            >>> assert engine.get_structured_metrics().total_dispatches == 0

        Warning:
            This method is NOT thread-safe. Call only during initialization
            or when no dispatch operations are in progress.

        .. versionadded:: 0.4.0
        """
        self._metrics = {
            "dispatch_count": 0,
            "dispatch_success_count": 0,
            "dispatch_error_count": 0,
            "total_latency_ms": 0.0,
            "dispatcher_execution_count": 0,
            "dispatcher_error_count": 0,
            "routes_matched_count": 0,
            "no_dispatcher_count": 0,
            "category_mismatch_count": 0,
        }
        self._structured_metrics = ModelDispatchMetrics()
        self._logger.debug("Metrics reset to initial state")

    def get_dispatcher_metrics(
        self, dispatcher_id: str
    ) -> ModelDispatcherMetrics | None:
        """
        Get metrics for a specific dispatcher.

        Args:
            dispatcher_id: The dispatcher's unique identifier.

        Returns:
            ModelDispatcherMetrics for the dispatcher, or None if no metrics recorded.

        Example:
            >>> metrics = engine.get_dispatcher_metrics("user-event-dispatcher")
            >>> if metrics:
            ...     print(f"Executions: {metrics.execution_count}")
            ...     print(f"Error rate: {metrics.error_rate:.1%}")

        .. versionadded:: 0.4.0
        """
        return self._structured_metrics.dispatcher_metrics.get(dispatcher_id)

    @property
    def route_count(self) -> int:
        """Get the number of registered routes."""
        return len(self._routes)

    @property
    def dispatcher_count(self) -> int:
        """Get the number of registered dispatchers."""
        return len(self._dispatchers)

    # Legacy property for backward compatibility
    @property
    def handler_count(self) -> int:
        """Get the number of registered dispatchers (legacy alias)."""
        return len(self._dispatchers)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"MessageDispatchEngine[routes={len(self._routes)}, "
            f"dispatchers={len(self._dispatchers)}, frozen={self._frozen}]"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        route_ids = list(self._routes.keys())[:10]
        dispatcher_ids = list(self._dispatchers.keys())[:10]

        route_repr = (
            repr(route_ids)
            if len(self._routes) <= 10
            else f"<{len(self._routes)} routes>"
        )
        dispatcher_repr = (
            repr(dispatcher_ids)
            if len(self._dispatchers) <= 10
            else f"<{len(self._dispatchers)} dispatchers>"
        )

        return (
            f"MessageDispatchEngine("
            f"routes={route_repr}, "
            f"dispatchers={dispatcher_repr}, "
            f"frozen={self._frozen})"
        )

    # Legacy method aliases for backward compatibility
    def register_handler(
        self,
        handler_id: str,
        handler: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None = None,
    ) -> None:
        """Register a message handler (legacy alias for register_dispatcher)."""
        return self.register_dispatcher(
            dispatcher_id=handler_id,
            dispatcher=handler,
            category=category,
            message_types=message_types,
        )

    def get_handler_metrics(self, handler_id: str) -> ModelDispatcherMetrics | None:
        """Get metrics for a specific handler (legacy alias for get_dispatcher_metrics)."""
        return self.get_dispatcher_metrics(handler_id)
