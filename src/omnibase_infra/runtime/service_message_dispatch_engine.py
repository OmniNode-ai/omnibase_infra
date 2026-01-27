# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: TRY400
# TRY400 disabled: logger.error is intentional to avoid leaking sensitive data in stack traces
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

    **Metrics Thread Safety (TOCTOU Prevention)**:
    A core design goal of the metrics system is preventing TOCTOU (time-of-check-
    to-time-of-use) race conditions. Without proper synchronization, concurrent
    dispatch operations could:

    1. Read current metrics state (check)
    2. Compute new values based on that state
    3. Write updated values (use)

    If another thread modifies the state between steps 1 and 3, the final write
    would clobber that concurrent update, causing lost increments or corrupted
    aggregations.

    **Solution**: All read-modify-write operations on ``_structured_metrics`` are
    performed atomically within a single ``_metrics_lock`` acquisition. This
    ensures that the sequence (read → compute → write) completes without
    interleaving from other threads.

    **Why holding the lock during computation is acceptable**:
    The computations within the lock (``record_execution()``, ``model_copy()``)
    are:

    - **Pure**: No I/O, no external calls, no blocking operations
    - **Fast**: Simple arithmetic and Pydantic model copying (~microseconds)
    - **Bounded**: Fixed computational complexity regardless of data size

    The lock is NEVER held during I/O operations (dispatcher execution), ensuring
    that slow dispatchers do not block metrics updates in other threads.

    For production monitoring, use ``get_structured_metrics()`` which returns
    a consistent snapshot.

Related:
    - OMN-934: Message dispatch engine implementation
    - EnvelopeRouter: Transport-agnostic orchestrator (reference for freeze pattern)

Category Support:
    The engine supports three ONEX message categories for routing:
    - EVENT: Domain events (e.g., UserCreatedEvent)
    - COMMAND: Action requests (e.g., CreateUserCommand)
    - INTENT: User intentions (e.g., ProvisionUserIntent)

    Topic naming constraints:
    - EVENT topics: Must contain ".events" segment
    - COMMAND topics: Must contain ".commands" segment
    - INTENT topics: Must contain ".intents" segment

    Note on PROJECTION:
        PROJECTION is NOT a message category for routing. Projections are
        node output types (EnumNodeOutputType.PROJECTION) produced by REDUCER
        nodes as local state outputs. Projections are:
        - NOT routed via Kafka topics
        - NOT part of EnumMessageCategory
        - Applied locally by the runtime to a projection sink

        See EnumNodeOutputType for projection semantics and CLAUDE.md
        "Enum Usage" section for the distinction between message categories
        and node output types.

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
from typing import TYPE_CHECKING, TypedDict, Unpack, cast, overload
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.types import JsonType, PrimitiveValue
from omnibase_infra.enums import (
    EnumDispatchStatus,
    EnumInfraTransportType,
    EnumMessageCategory,
)
from omnibase_infra.errors import (
    BindingResolutionError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.models.bindings import (
    ModelBindingResolutionResult,
    ModelOperationBindingsSubcontract,
)
from omnibase_infra.models.dispatch.model_debug_trace_snapshot import (
    ModelDebugTraceSnapshot,
)
from omnibase_infra.models.dispatch.model_dispatch_context import ModelDispatchContext
from omnibase_infra.models.dispatch.model_dispatch_log_context import (
    ModelDispatchLogContext,
)
from omnibase_infra.models.dispatch.model_dispatch_metrics import ModelDispatchMetrics
from omnibase_infra.models.dispatch.model_dispatch_outcome import ModelDispatchOutcome
from omnibase_infra.models.dispatch.model_dispatch_outputs import ModelDispatchOutputs
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_infra.models.dispatch.model_dispatcher_metrics import (
    ModelDispatcherMetrics,
)
from omnibase_infra.models.dispatch.model_materialized_dispatch import (
    ModelMaterializedDispatch,
)
from omnibase_infra.runtime.binding_resolver import OperationBindingResolver
from omnibase_infra.runtime.dispatch_context_enforcer import DispatchContextEnforcer
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_core.enums.enum_node_kind import EnumNodeKind


class ModelLogContextKwargs(TypedDict, total=False):
    """TypedDict for _build_log_context kwargs to ensure type safety.

    All fields are optional (total=False) since callers pass only the
    relevant subset. ModelDispatchLogContext validators handle None-to-sentinel
    conversion.

    .. versionadded:: 0.6.3
        Created as part of Union Reduction Phase 2 (OMN-1002) to eliminate
        type: ignore comment in _build_log_context.
    """

    topic: str | None
    category: EnumMessageCategory | None
    message_type: str | None
    dispatcher_id: str | None
    dispatcher_count: int | None
    duration_ms: float | None
    correlation_id: UUID | None
    trace_id: UUID | None
    error_code: EnumCoreErrorCode | None


# Type alias for dispatcher output topics
#
# Dispatchers can return:
# - str: A single output topic
# - list[str]: Multiple output topics
# - None: No output topics to publish
# - ModelDispatchResult: Protocol-based dispatchers return this for structured output
DispatcherOutput = str | list[str] | None | ModelDispatchResult

# Module-level logger for fallback when no custom logger is provided
_module_logger = logging.getLogger(__name__)

# Minimum number of parameters for a dispatcher to be considered context-aware.
# Context-aware dispatchers have signature: (envelope, context, ...)
# Non-context-aware dispatchers have signature: (envelope)
# We use >= MIN_PARAMS_FOR_CONTEXT (not ==) to support dispatchers with additional
# optional parameters (e.g., for testing, logging, or future extensibility).
MIN_PARAMS_FOR_CONTEXT = 2

# Type alias for dispatcher functions
#
# Design Note (PR #61 Review):
# ModelEventEnvelope[object] is used instead of Any to satisfy ONEX "no Any types" rule.
#
# Rationale:
# - Input: ModelEventEnvelope[object] is intentionally generic because dispatchers
#   must accept envelopes with any payload type. The dispatch engine routes based
#   on topic/category/message_type, not payload shape. Using a TypeVar would require
#   dispatchers to be generic, adding complexity without benefit since the engine
#   already performs type-based routing.
# - Output: DispatcherOutput | Awaitable[DispatcherOutput] defines the valid return
#   types: str (single topic), list[str] (multiple topics), or None (no output).
#   Dispatchers can be sync or async.
#
# Using `object` instead of `Any` provides:
# - Explicit "any object" semantics that are more informative to type checkers
# - Compliance with ONEX coding guidelines
# - Same runtime behavior as Any but with clearer intent
#
# See also: ProtocolMessageDispatcher in dispatcher_registry.py for protocol-based
# dispatchers that return ModelDispatchResult.
DispatcherFunc = Callable[
    [ModelEventEnvelope[object]], DispatcherOutput | Awaitable[DispatcherOutput]
]

# Context-aware dispatcher type (for dispatchers registered with node_kind)
# These dispatchers receive a ModelDispatchContext with time injection based on node_kind:
# - REDUCER/COMPUTE: now=None (deterministic)
# - ORCHESTRATOR/EFFECT/RUNTIME_HOST: now=datetime.now(UTC)
#
# This type is used when register_dispatcher() is called with node_kind parameter.
# The dispatch engine inspects the callable's signature to determine if it accepts context.
ContextAwareDispatcherFunc = Callable[
    [ModelEventEnvelope[object], ModelDispatchContext],
    DispatcherOutput | Awaitable[DispatcherOutput],
]

# Sync-only dispatcher type for use with run_in_executor
# Used internally after runtime type narrowing via inspect.iscoroutinefunction
# All envelopes are materialized to dict format with __bindings namespace
_SyncDispatcherFunc = Callable[[dict[str, object]], DispatcherOutput]

# Sync-only context-aware dispatcher type for use with run_in_executor
# All envelopes are materialized to dict format with __bindings namespace
_SyncContextAwareDispatcherFunc = Callable[
    [dict[str, object], ModelDispatchContext],
    DispatcherOutput,
]


class DispatchEntryInternal:
    """
    Internal storage for dispatcher registration metadata.

    This class is an implementation detail and not part of the public API.
    It stores the dispatcher callable and associated metadata for the
    MessageDispatchEngine's internal routing.

    Attributes:
        dispatcher_id: Unique identifier for this dispatcher.
        dispatcher: The callable that processes messages.
        category: Message category this dispatcher handles.
        message_types: Specific message types to handle (None = all types).
        node_kind: Optional ONEX node kind for time injection context.
            When set, the dispatcher receives a ModelDispatchContext with
            appropriate time injection based on ONEX rules:
            - REDUCER/COMPUTE: now=None (deterministic)
            - ORCHESTRATOR/EFFECT/RUNTIME_HOST: now=datetime.now(UTC)
        accepts_context: Cached result of signature inspection indicating
            whether the dispatcher accepts a context parameter (2+ params).
            Computed once at registration time for performance.
        operation_bindings: Optional declarative bindings for resolving
            handler parameters from envelope/payload/context. When set,
            bindings are resolved BEFORE handler execution and materialized
            into a new envelope with __bindings namespace.
    """

    __slots__ = (
        "accepts_context",
        "category",
        "dispatcher",
        "dispatcher_id",
        "message_types",
        "node_kind",
        "operation_bindings",
    )

    def __init__(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc | ContextAwareDispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None,
        node_kind: EnumNodeKind | None = None,
        accepts_context: bool = False,
        operation_bindings: ModelOperationBindingsSubcontract | None = None,
    ) -> None:
        self.dispatcher_id = dispatcher_id
        self.dispatcher = dispatcher
        self.category = category
        self.message_types = message_types  # None means "all types"
        self.node_kind = node_kind  # None means no context injection
        self.accepts_context = accepts_context  # Cached: dispatcher has 2+ params
        self.operation_bindings = (
            operation_bindings  # Declarative bindings for this dispatcher
        )


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

        **TOCTOU Prevention** (core design goal):
        Structured metrics use ``_metrics_lock`` to ensure atomic read-modify-write
        operations. Without this, concurrent dispatches could lose updates:

        - Thread A reads metrics, computes increment
        - Thread B reads (stale) metrics, computes increment
        - Thread A writes → Thread B writes → Thread A's update is lost

        By holding the lock during the entire read→compute→write sequence, we
        guarantee no interleaving occurs. The computations within the lock are
        pure and fast (~microseconds), so lock contention is minimal.

        - Structured metrics: Use ``_metrics_lock`` for atomic updates
        - Use ``get_structured_metrics()`` for production monitoring

        **METRICS CAVEAT**: While metrics updates are protected by a lock,
        get_structured_metrics() provides point-in-time snapshots. Under high
        concurrent load, metrics may be approximate between snapshot reads.
        For production monitoring, consider exporting metrics to a dedicated
        metrics backend (Prometheus, StatsD, etc.) for accurate aggregation
        across time windows.

    Logging Levels:
        - **INFO**: Dispatch start/complete with topic, category, dispatcher count
        - **DEBUG**: Dispatcher execution details, routing decisions
        - **WARNING**: No dispatchers found, category mismatches
        - **ERROR**: Dispatcher exceptions, validation failures

    Example:
        >>> from omnibase_infra.runtime import MessageDispatchEngine
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
        _metrics_lock: Lock protecting structured metrics updates
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
        # NOTE: Only routable message categories are indexed here.
        # PROJECTION is NOT included because projections are reducer outputs,
        # not routable messages. See CLAUDE.md "Enum Usage" section.
        self._dispatchers_by_category: dict[EnumMessageCategory, list[str]] = {
            EnumMessageCategory.EVENT: [],
            EnumMessageCategory.COMMAND: [],
            EnumMessageCategory.INTENT: [],
        }

        # Freeze state
        self._frozen: bool = False
        self._registration_lock: threading.Lock = threading.Lock()

        # Metrics lock for TOCTOU-safe structured metrics updates
        # This lock protects the entire read-modify-write sequence on _structured_metrics:
        #   1. Read current metrics state
        #   2. Compute new values (record_execution, model_copy)
        #   3. Write updated metrics back
        # Holding the lock during computation prevents lost updates from concurrent dispatches.
        # The computations are pure and fast (~microseconds), minimizing lock contention.
        self._metrics_lock: threading.Lock = threading.Lock()

        # Structured metrics (Pydantic model)
        self._structured_metrics: ModelDispatchMetrics = ModelDispatchMetrics()

        # Context enforcer for creating dispatch contexts based on node_kind.
        # Delegates time injection rule enforcement to a single source of truth.
        self._context_enforcer: DispatchContextEnforcer = DispatchContextEnforcer()

        # Binding resolver for declarative operation bindings.
        # Resolves ${source.path} expressions from envelope/payload/context.
        self._binding_resolver: OperationBindingResolver = OperationBindingResolver()

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

    # --- @overload stubs for static type safety ---
    #
    # NOTE: These are TYPE STUBS only - they provide no runtime behavior.
    # The actual implementation is in the non-overloaded register_dispatcher() below.
    #
    # Purpose: Enable type checkers (mypy, pyright) to validate that:
    #   - When node_kind=None (or omitted): dispatcher must be DispatcherFunc
    #   - When node_kind=EnumNodeKind: dispatcher must be ContextAwareDispatcherFunc
    #
    # This pattern enforces compile-time type safety for the relationship between
    # node_kind presence and expected dispatcher signature.
    #
    # See ADR_DISPATCHER_TYPE_SAFETY.md Option 4 for design rationale.

    @overload
    def register_dispatcher(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None = None,
        node_kind: None = None,
        operation_bindings: ModelOperationBindingsSubcontract | None = None,
    ) -> None: ...  # Stub: no node_kind -> DispatcherFunc (no context)

    @overload
    def register_dispatcher(
        self,
        dispatcher_id: str,
        dispatcher: ContextAwareDispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None = None,
        *,
        node_kind: EnumNodeKind,
        operation_bindings: ModelOperationBindingsSubcontract | None = None,
    ) -> None: ...  # Stub: with node_kind -> ContextAwareDispatcherFunc (gets context)

    def register_dispatcher(
        self,
        dispatcher_id: str,
        dispatcher: DispatcherFunc | ContextAwareDispatcherFunc,
        category: EnumMessageCategory,
        message_types: set[str] | None = None,
        node_kind: EnumNodeKind | None = None,
        operation_bindings: ModelOperationBindingsSubcontract | None = None,
    ) -> None:
        """
        Register a message dispatcher.

        Dispatchers process messages that match their category and (optionally)
        message type. Multiple dispatchers can register for the same category
        and message type (fan-out pattern).

        Args:
            dispatcher_id: Unique identifier for this dispatcher
            dispatcher: Callable that processes messages. Can be sync or async.
                Signature: (envelope: ModelEventEnvelope[object]) -> DispatcherOutput
                Or with context:
                (envelope: ModelEventEnvelope[object], context: ModelDispatchContext) -> DispatcherOutput
            category: Message category this dispatcher processes
            message_types: Optional set of specific message types to handle.
                When None, handles all message types in the category.
            node_kind: Optional ONEX node kind for time injection context.
                When provided, the dispatcher receives a ModelDispatchContext
                with appropriate time injection based on ONEX rules:
                - REDUCER/COMPUTE: now=None (deterministic execution)
                - ORCHESTRATOR/EFFECT/RUNTIME_HOST: now=datetime.now(UTC)
                When None, dispatcher is called without context.
            operation_bindings: Optional declarative bindings for resolving
                handler parameters from envelope/payload/context. When provided,
                bindings are resolved BEFORE handler execution and materialized
                into a new envelope with __bindings namespace. The original
                envelope is NEVER mutated.

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
            >>>
            >>> # With time injection context for orchestrator
            >>> async def process_with_context(envelope, context):
            ...     current_time = context.now  # Injected time
            ...     return "processed"
            >>>
            >>> engine.register_dispatcher(
            ...     dispatcher_id="orchestrator-dispatcher",
            ...     dispatcher=process_with_context,
            ...     category=EnumMessageCategory.COMMAND,
            ...     node_kind=EnumNodeKind.ORCHESTRATOR,
            ... )

        Note:
            Dispatchers are NOT automatically linked to routes. You must register
            routes separately that reference the dispatcher_id.

        .. versionchanged:: 0.5.0
            Added ``node_kind`` parameter for time injection context support.
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

        # Runtime validation for node_kind to catch dynamic dispatch issues
        # where type checkers can't help (e.g., dynamically constructed arguments)
        if node_kind is not None:
            # Import here to avoid circular import at module level
            # EnumNodeKind is only in TYPE_CHECKING block at top of file
            from omnibase_core.enums.enum_node_kind import EnumNodeKind

            if not isinstance(node_kind, EnumNodeKind):
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="register_dispatcher",
                )
                raise ProtocolConfigurationError(
                    f"node_kind must be EnumNodeKind or None, got {type(node_kind).__name__}",
                    context=context,
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

            # Compute accepts_context once at registration time (cached)
            # This avoids expensive inspect.signature() calls on every dispatch
            accepts_context = self._dispatcher_accepts_context(dispatcher)

            # Store dispatcher entry
            entry = DispatchEntryInternal(
                dispatcher_id=dispatcher_id,
                dispatcher=dispatcher,
                category=category,
                message_types=message_types,
                node_kind=node_kind,
                accepts_context=accepts_context,
                operation_bindings=operation_bindings,
            )
            self._dispatchers[dispatcher_id] = entry

            # Log requirement for operation_bindings users
            # NOTE: When operation_bindings is provided, envelopes dispatched to this
            # handler MUST have an 'operation' attribute/key. The operation field is
            # extracted at dispatch time via _extract_operation() and used to select
            # the appropriate binding configuration. Missing operation fields will
            # result in binding resolution failures at dispatch time.
            if operation_bindings is not None:
                self._logger.debug(
                    "Dispatcher '%s' registered with operation_bindings. "
                    "Envelopes MUST have an 'operation' attribute/key for binding resolution.",
                    dispatcher_id,
                )

            # Update category index
            self._dispatchers_by_category[category].append(dispatcher_id)

            self._logger.debug(
                "Registered dispatcher '%s' for category %s (message_types=%s, node_kind=%s)",
                dispatcher_id,
                category,
                message_types if message_types else "all",
                node_kind.value if node_kind else "none",
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
        self, **kwargs: Unpack[ModelLogContextKwargs]
    ) -> dict[str, PrimitiveValue]:
        """
        Build structured log context dictionary.

        .. versionchanged:: 0.6.0
            Now delegates to ModelDispatchLogContext.to_dict() for type-safe
            context construction.

        .. versionchanged:: 0.6.2
            Refactored to use ``**kwargs`` forwarding to eliminate 9 union
            parameters from method signature (OMN-1002 Union Reduction Phase 2).
            ModelDispatchLogContext validators handle None-to-sentinel conversion.

        .. versionchanged:: 0.6.3
            Updated to use ``Unpack[ModelLogContextKwargs]`` TypedDict for type-safe
            kwargs (OMN-1002). Eliminates need for ``type: ignore`` comment.

        Design Note (Union Reduction - OMN-1002):
            This private method uses typed ``**kwargs`` via ``ModelLogContextKwargs``
            TypedDict to forward parameters to ModelDispatchLogContext. The
            TypedDict provides compile-time type checking while the model's
            field validators handle None-to-sentinel conversion at runtime.

        Args:
            **kwargs: Keyword arguments forwarded to ModelDispatchLogContext.
                Typed via ``ModelLogContextKwargs`` TypedDict with supported keys:
                topic, category, message_type, dispatcher_id, dispatcher_count,
                duration_ms, correlation_id, trace_id, error_code.
                None values are automatically converted to sentinel values by
                the model's field validators.

        Returns:
            Dictionary with non-sentinel values for structured logging.
            UUID values are converted to strings at serialization time.
        """
        # Forward all kwargs to ModelDispatchLogContext which handles
        # None-to-sentinel conversion via field validators.
        # Use model_validate() to properly invoke "before" validators that
        # accept None via object type annotation.
        ctx = ModelDispatchLogContext.model_validate(kwargs)
        return ctx.to_dict()

    async def dispatch(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
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

        # Extract correlation/trace IDs for logging (kept as UUID, converted to string at serialization)
        # Per ONEX guidelines: auto-generate correlation_id if not provided (uuid4())
        correlation_id = envelope.correlation_id or uuid4()
        trace_id = envelope.trace_id

        # Step 1: Parse topic to get category
        topic_category = EnumMessageCategory.from_topic(topic)
        if topic_category is None:
            # Capture duration and completed_at together for consistency
            duration_ms = (time.perf_counter() - start_time) * 1000
            completed_at = datetime.now(UTC)

            # Update metrics (protected by lock for thread safety)
            with self._metrics_lock:
                self._structured_metrics = self._structured_metrics.record_dispatch(
                    duration_ms=duration_ms,
                    success=False,
                    category=None,
                    no_dispatcher=False,
                    category_mismatch=False,
                    topic=topic,
                )

            # Log error
            self._logger.error(
                "Dispatch failed: invalid topic category",
                extra=self._build_log_context(
                    topic=topic,
                    duration_ms=duration_ms,
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                ),
            )

            return ModelDispatchResult(
                dispatch_id=dispatch_id,
                status=EnumDispatchStatus.INVALID_MESSAGE,
                topic=topic,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=f"Cannot infer message category from topic '{topic}'. "
                "Topic must contain .events, .commands, .intents, or .projections segment.",
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                correlation_id=correlation_id,
                output_events=[],
            )

        # Log dispatch start at INFO level
        self._logger.info(
            "Dispatch started",
            extra=self._build_log_context(
                topic=topic,
                category=topic_category,
                correlation_id=correlation_id,
                trace_id=trace_id,
            ),
        )

        # Step 2: Validate envelope category matches topic category
        # NOTE: ModelEventEnvelope.infer_category() is not yet implemented in omnibase_core.
        # Until it is, we trust the topic category as the source of truth for routing.
        # This is safe because the topic defines the message category, and handlers
        # are registered for specific categories - any mismatch would be a caller error.
        # TODO(OMN-934): Re-enable envelope category validation when infer_category() is available
        #
        # The code below is disabled until infer_category() is available:
        # envelope_category = envelope.infer_category()
        # if envelope_category != topic_category:
        #     ... (category mismatch handling with structured metrics)

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
                correlation_id=correlation_id,
                trace_id=trace_id,
            ),
        )

        if not matching_dispatchers:
            # Capture duration and completed_at together for consistency
            duration_ms = (time.perf_counter() - start_time) * 1000
            completed_at = datetime.now(UTC)

            # Update metrics (protected by lock for thread safety)
            with self._metrics_lock:
                self._structured_metrics = self._structured_metrics.record_dispatch(
                    duration_ms=duration_ms,
                    success=False,
                    category=topic_category,
                    no_dispatcher=True,
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
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    error_code=EnumCoreErrorCode.ITEM_NOT_REGISTERED,
                ),
            )

            return ModelDispatchResult(
                dispatch_id=dispatch_id,
                status=EnumDispatchStatus.NO_DISPATCHER,
                topic=topic,
                message_category=topic_category,
                message_type=message_type,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=f"No dispatcher registered for category '{topic_category}' "
                f"and message type '{message_type}' matching topic '{topic}'.",
                error_code=EnumCoreErrorCode.ITEM_NOT_REGISTERED,
                correlation_id=correlation_id,
                output_events=[],
            )

        # Step 5: Execute dispatchers and collect outputs
        outputs: list[str] = []
        dispatcher_errors: list[str] = []
        executed_dispatcher_ids: list[str] = []

        for dispatcher_entry in matching_dispatchers:
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
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                ),
            )

            try:
                result = await self._execute_dispatcher(
                    dispatcher_entry, envelope, topic
                )
                dispatcher_duration_ms = (
                    time.perf_counter() - dispatcher_start_time
                ) * 1000
                executed_dispatcher_ids.append(dispatcher_entry.dispatcher_id)

                # TOCTOU Prevention: Update per-dispatcher metrics atomically
                # ---------------------------------------------------------
                # The entire read-modify-write sequence below MUST execute within
                # a single lock acquisition to prevent race conditions:
                #   1. Read: Get existing dispatcher metrics (or create default)
                #   2. Modify: Call record_execution() to compute new values
                #   3. Write: Update _structured_metrics with new dispatcher entry
                #
                # These operations are pure (no I/O) and fast (~microseconds),
                # so holding the lock during computation is acceptable.
                with self._metrics_lock:
                    existing_dispatcher_metrics = (
                        self._structured_metrics.dispatcher_metrics.get(
                            dispatcher_entry.dispatcher_id
                        )
                    )
                    if existing_dispatcher_metrics is None:
                        existing_dispatcher_metrics = ModelDispatcherMetrics(
                            dispatcher_id=dispatcher_entry.dispatcher_id
                        )
                    new_dispatcher_metrics = (
                        existing_dispatcher_metrics.record_execution(
                            duration_ms=dispatcher_duration_ms,
                            success=True,
                            topic=topic,
                        )
                    )
                    new_dispatcher_metrics_dict = {
                        **self._structured_metrics.dispatcher_metrics,
                        dispatcher_entry.dispatcher_id: new_dispatcher_metrics,
                    }
                    self._structured_metrics = self._structured_metrics.model_copy(
                        update={
                            "dispatcher_execution_count": (
                                self._structured_metrics.dispatcher_execution_count + 1
                            ),
                            "dispatcher_metrics": new_dispatcher_metrics_dict,
                        }
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
                        correlation_id=correlation_id,
                        trace_id=trace_id,
                    ),
                )

                # Normalize dispatcher output using ModelDispatchOutcome to avoid
                # manual isinstance checks on the 3-way union (str | list[str] | None).
                # This centralizes the union handling in the model's from_legacy_output().
                outcome = ModelDispatchOutcome.from_legacy_output(result)
                outputs.extend(outcome.topics)
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
                # Sanitize exception message to prevent credential leakage
                # (e.g., connection strings with passwords, API keys in URLs)
                sanitized_error = sanitize_error_message(e)
                error_msg = (
                    f"Dispatcher '{dispatcher_entry.dispatcher_id}' "
                    f"failed: {sanitized_error}"
                )
                dispatcher_errors.append(error_msg)

                # TOCTOU Prevention: Update per-dispatcher error metrics atomically
                # ----------------------------------------------------------------
                # The entire read-modify-write sequence below MUST execute within
                # a single lock acquisition to prevent race conditions:
                #   1. Read: Get existing dispatcher metrics (or create default)
                #   2. Modify: Call record_execution() to compute new error values
                #   3. Write: Update _structured_metrics with new dispatcher entry
                #
                # These operations are pure (no I/O) and fast (~microseconds),
                # so holding the lock during computation is acceptable.
                with self._metrics_lock:
                    existing_dispatcher_metrics = (
                        self._structured_metrics.dispatcher_metrics.get(
                            dispatcher_entry.dispatcher_id
                        )
                    )
                    if existing_dispatcher_metrics is None:
                        existing_dispatcher_metrics = ModelDispatcherMetrics(
                            dispatcher_id=dispatcher_entry.dispatcher_id
                        )
                    new_dispatcher_metrics = (
                        existing_dispatcher_metrics.record_execution(
                            duration_ms=dispatcher_duration_ms,
                            success=False,
                            topic=topic,
                            # Use sanitized error message for metrics as well
                            error_message=sanitized_error,
                        )
                    )
                    new_dispatcher_metrics_dict = {
                        **self._structured_metrics.dispatcher_metrics,
                        dispatcher_entry.dispatcher_id: new_dispatcher_metrics,
                    }
                    self._structured_metrics = self._structured_metrics.model_copy(
                        update={
                            "dispatcher_execution_count": (
                                self._structured_metrics.dispatcher_execution_count + 1
                            ),
                            "dispatcher_error_count": (
                                self._structured_metrics.dispatcher_error_count + 1
                            ),
                            "dispatcher_metrics": new_dispatcher_metrics_dict,
                        }
                    )

                # Log error with sanitized message
                # Note: Using logger.error() with sanitized message instead of
                # logger.exception() to avoid leaking sensitive data in stack traces.
                # The sanitized_error variable already contains safe error details.
                # TRY400: Intentionally using error() instead of exception() for security
                self._logger.error(
                    "Dispatcher '%s' failed: %s",
                    dispatcher_entry.dispatcher_id,
                    sanitized_error,
                    extra=self._build_log_context(
                        topic=topic,
                        category=topic_category,
                        message_type=message_type,
                        dispatcher_id=dispatcher_entry.dispatcher_id,
                        duration_ms=dispatcher_duration_ms,
                        correlation_id=correlation_id,
                        trace_id=trace_id,
                        error_code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR,
                    ),
                )

        # Step 6: Build result
        # Capture duration and completed_at together for consistency
        duration_ms = (time.perf_counter() - start_time) * 1000
        completed_at = datetime.now(UTC)

        # Determine final status
        if dispatcher_errors:
            # Either partial or total failure
            status = EnumDispatchStatus.HANDLER_ERROR
        else:
            status = EnumDispatchStatus.SUCCESS

        # Update all metrics atomically (protected by lock)
        with self._metrics_lock:
            # NOTE: dispatcher_id and handler_error are NOT passed here because
            # per-dispatcher metrics (including dispatcher_execution_count and
            # dispatcher_error_count) are already updated in the dispatcher loop
            # above. Passing them here would cause double-counting.
            self._structured_metrics = self._structured_metrics.record_dispatch(
                duration_ms=duration_ms,
                success=status == EnumDispatchStatus.SUCCESS,
                category=topic_category,
                dispatcher_id=None,  # Already tracked in dispatcher loop
                handler_error=False,  # Already tracked in dispatcher loop
                routes_matched=len(matching_dispatchers),
                topic=topic,
                error_message=dispatcher_errors[0] if dispatcher_errors else None,
            )

        # Find route ID that matched (first matching route for logging)
        # Use empty string sentinel internally to avoid str | None union
        matched_route_id: str = ""
        for route in self._routes.values():
            if route.matches(topic, topic_category, message_type):
                matched_route_id = route.route_id
                break

        # Log dispatch completion at INFO level
        # Use empty string sentinel to avoid str | None union in local scope
        dispatcher_ids_str: str = (
            ", ".join(executed_dispatcher_ids) if executed_dispatcher_ids else ""
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
                    correlation_id=correlation_id,
                    trace_id=trace_id,
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
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    error_code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR,
                ),
            )

        # Convert list of output topics to ModelDispatchOutputs
        # Handle Pydantic validation errors (e.g., invalid topic format)
        dispatch_outputs: ModelDispatchOutputs | None = None
        if outputs:
            try:
                dispatch_outputs = ModelDispatchOutputs(topics=outputs)
            except (ValueError, ValidationError) as validation_error:
                # Log validation failure with context (no secrets in topic names)
                # Note: Using sanitize_error_message for consistency, though topic
                # validation errors typically don't contain sensitive data
                sanitized_validation_error = sanitize_error_message(validation_error)
                # TRY400: Intentionally using error() instead of exception() for security
                # - exception() would log stack trace which may expose internal paths
                # - sanitized_validation_error already contains safe error details
                self._logger.error(
                    "Failed to validate dispatch outputs (%d topics): %s",
                    len(outputs),
                    sanitized_validation_error,
                    extra=self._build_log_context(
                        topic=topic,
                        category=topic_category,
                        message_type=message_type,
                        correlation_id=correlation_id,
                        trace_id=trace_id,
                        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    ),
                )
                # Add validation error to dispatcher_errors for result
                validation_error_msg = (
                    f"Output validation failed: {sanitized_validation_error}"
                )
                dispatcher_errors.append(validation_error_msg)
                # Update status to reflect validation error
                status = EnumDispatchStatus.HANDLER_ERROR

        # Construct final dispatch result with ValidationError protection
        # This ensures any Pydantic validation failure in ModelDispatchResult
        # is handled gracefully rather than propagating as an unhandled exception
        try:
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
                completed_at=completed_at,
                outputs=dispatch_outputs,
                output_count=len(outputs),
                error_message="; ".join(dispatcher_errors)
                if dispatcher_errors
                else None,
                error_code=EnumCoreErrorCode.HANDLER_EXECUTION_ERROR
                if dispatcher_errors
                else None,
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=envelope.span_id,
            )
        except ValidationError as result_validation_error:
            # Pydantic validation failed during result construction
            # This is a critical internal error - log and return a minimal error result
            sanitized_result_error = sanitize_error_message(result_validation_error)
            # TRY400: Intentionally using error() instead of exception() for security
            self._logger.error(
                "Failed to construct ModelDispatchResult: %s",
                sanitized_result_error,
                extra=self._build_log_context(
                    topic=topic,
                    category=topic_category,
                    message_type=message_type,
                    correlation_id=correlation_id,
                    trace_id=trace_id,
                    error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                ),
            )
            # Return a minimal fallback result that should always succeed
            return ModelDispatchResult(
                dispatch_id=dispatch_id,
                status=EnumDispatchStatus.INTERNAL_ERROR,
                topic=topic,
                started_at=started_at,
                completed_at=datetime.now(UTC),
                duration_ms=duration_ms,
                error_message=f"Internal error constructing dispatch result: {sanitized_result_error}",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
                correlation_id=correlation_id,
                output_events=[],
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
        envelope: ModelEventEnvelope[object],
        topic: str,
    ) -> DispatcherOutput:
        """
        Execute a dispatcher (sync or async).

        Sync dispatchers are executed via ``loop.run_in_executor()`` using the
        default ``ThreadPoolExecutor``. This allows sync code to run without
        blocking the event loop, but has important implications:

        Thread Pool Considerations:
            - The default executor uses a limited thread pool (typically
              ``min(32, os.cpu_count() + 4)`` threads in Python 3.8+)
            - Each sync dispatcher execution consumes one thread until completion
            - Blocking dispatchers can exhaust the thread pool, causing:
              - Starvation of other sync dispatchers waiting for threads
              - Delayed scheduling of new async tasks
              - Potential deadlocks under high concurrent load
              - Increased latency for all executor-based operations

        Best Practices:
            - Sync dispatchers SHOULD complete quickly (< 100ms recommended)
            - For blocking I/O (network, database, file), use async dispatchers
            - For CPU-bound work, consider using a dedicated ProcessPoolExecutor
            - Monitor ``dispatcher_execution_count`` metrics for bottlenecks

        Args:
            entry: The dispatcher entry containing the callable
            envelope: The message envelope to process

        Returns:
            DispatcherOutput: str (single topic), list[str] (multiple topics),
                or None (no output topics)

        Raises:
            Any exception raised by the dispatcher

        Warning:
            Sync dispatchers that block for extended periods (> 100ms) can
            severely degrade dispatch engine throughput. Prefer async dispatchers
            for any operation involving I/O or external service calls.

        .. versionchanged:: 0.5.0
            Added support for context-aware dispatchers via ``node_kind``.

        .. versionchanged:: 0.2.6
            Added binding resolution before handler execution (OMN-1518).
        """
        dispatcher = entry.dispatcher

        # =================================================================
        # Binding Resolution Phase (OMN-1518)
        # =================================================================
        # ALWAYS materialize envelope to dict format for consistent dispatcher API.
        # INVARIANT: Original envelope is NEVER mutated.
        resolved_bindings: dict[str, JsonType] = {}

        if entry.operation_bindings is not None:
            # Extract correlation_id FIRST for error context and tracing
            correlation_id = self._extract_correlation_id(envelope)

            # Extract operation name from envelope for binding lookup (fail-fast)
            operation = self._extract_operation_from_envelope(envelope, correlation_id)

            # Create context for binding resolution (see constants.VALID_CONTEXT_PATHS)
            dispatch_context: dict[str, object] = {
                "now_iso": datetime.now(UTC).isoformat(),
                "dispatcher_id": entry.dispatcher_id,
                "correlation_id": correlation_id,
            }

            # Check for declared additional_context_paths that are not provided
            # This is a CONTRACT: if handler declares additional_context_paths,
            # the dispatch engine should provide them. Log warning if missing.
            declared_additional_paths = (
                entry.operation_bindings.additional_context_paths
            )
            if declared_additional_paths:
                missing_paths = [
                    path
                    for path in declared_additional_paths
                    if path not in dispatch_context
                ]
                if missing_paths:
                    self._logger.warning(
                        "Dispatcher '%s' declares additional_context_paths %s "
                        "but these are not provided in dispatch context. "
                        "Bindings using these paths will resolve to None unless "
                        "they have defaults. correlation_id=%s",
                        entry.dispatcher_id,
                        missing_paths,
                        correlation_id,
                    )

            # Resolve all bindings for this operation
            resolution = self._binding_resolver.resolve(
                operation=operation,
                bindings_subcontract=entry.operation_bindings,
                envelope=envelope,
                context=dispatch_context,
                correlation_id=correlation_id,
            )

            if not resolution.success:
                # Fail fast on binding resolution failure
                raise BindingResolutionError(
                    f"Binding resolution failed: {resolution.error}",
                    operation_name=resolution.operation_name,
                    parameter_name="unknown",
                    expression="unknown",
                    correlation_id=correlation_id,
                )

            # dict() creates a shallow copy to avoid mutating the resolution result
            resolved_bindings = dict(resolution.resolved_parameters)

        # ALWAYS materialize envelope with bindings (never mutate original)
        # Empty dict for __bindings if no bindings configured
        envelope_for_handler: dict[str, JsonType] = (
            self._materialize_envelope_with_bindings(envelope, resolved_bindings, topic)
        )

        # =================================================================
        # Context Creation Phase
        # =================================================================
        # Create context ONLY if both conditions are met:
        # 1. node_kind is set (time injection rules apply)
        # 2. dispatcher accepts context (will actually use it)
        # This avoids unnecessary object creation on the dispatch hot path when
        # a dispatcher has node_kind set but doesn't accept a context parameter.
        context: ModelDispatchContext | None = None
        if entry.node_kind is not None and entry.accepts_context:
            context = self._create_context_for_entry(entry, envelope)

        # =================================================================
        # Dispatcher Execution Phase
        # =================================================================
        # Check if dispatcher is async
        # Note: context is only non-None when entry.accepts_context is True,
        # so checking `context is not None` is sufficient to determine whether
        # to pass context to the dispatcher.
        if inspect.iscoroutinefunction(dispatcher):
            if context is not None:
                # NOTE: Dispatcher signature varies - context param may be optional.
                # Return type depends on dispatcher implementation (dict or model).
                return await dispatcher(envelope_for_handler, context)  # type: ignore[call-arg,no-any-return]  # NOTE: dispatcher signature varies
            # NOTE: Return type depends on dispatcher implementation (dict or model).
            return await dispatcher(envelope_for_handler)  # type: ignore[no-any-return]  # NOTE: dispatcher return type varies
        else:
            # Sync dispatcher execution via ThreadPoolExecutor
            # -----------------------------------------------
            # WARNING: Sync dispatchers MUST be non-blocking (< 100ms execution).
            # Blocking dispatchers can exhaust the thread pool, causing:
            # - Starvation of other sync dispatchers
            # - Delayed async dispatcher scheduling
            # - Potential deadlocks under high load
            #
            # For blocking I/O operations, use async dispatchers instead.
            loop = asyncio.get_running_loop()

            if context is not None:
                # Context-aware sync dispatcher
                sync_ctx_dispatcher = cast(
                    "_SyncContextAwareDispatcherFunc", dispatcher
                )
                # NOTE: run_in_executor arg-type check fails because envelope_for_handler
                # is dict[str, JsonType] but dispatcher expects dict[str, object].
                # JsonType is a subset of object, so this is safe at runtime.
                return await loop.run_in_executor(
                    None,
                    sync_ctx_dispatcher,  # type: ignore[arg-type]
                    envelope_for_handler,
                    context,
                )
            else:
                # Cast to sync-only type - safe because iscoroutinefunction check above
                # guarantees this branch only executes for non-async callables
                sync_dispatcher = cast("_SyncDispatcherFunc", dispatcher)
                # NOTE: run_in_executor arg-type check fails because envelope_for_handler
                # is dict[str, JsonType] but dispatcher expects dict[str, object].
                # JsonType is a subset of object, so this is safe at runtime.
                return await loop.run_in_executor(
                    None,
                    sync_dispatcher,  # type: ignore[arg-type]
                    envelope_for_handler,
                )

    def _create_context_for_entry(
        self,
        entry: DispatchEntryInternal,
        envelope: ModelEventEnvelope[object],
    ) -> ModelDispatchContext:
        """
        Create dispatch context based on entry's node_kind.

        Delegates to DispatchContextEnforcer.create_context_for_node_kind() to
        ensure a single source of truth for time injection rules. This method
        is a thin wrapper that validates node_kind is not None before delegation.

        Creates a ModelDispatchContext with appropriate time injection based on
        the ONEX node kind:
        - REDUCER: now=None (deterministic state aggregation)
        - COMPUTE: now=None (pure transformation)
        - ORCHESTRATOR: now=datetime.now(UTC) (coordination)
        - EFFECT: now=datetime.now(UTC) (I/O operations)
        - RUNTIME_HOST: now=datetime.now(UTC) (infrastructure)

        Args:
            entry: The dispatcher entry containing node_kind.
            envelope: The event envelope containing correlation metadata.

        Returns:
            ModelDispatchContext configured appropriately for the node kind.

        Raises:
            ModelOnexError: If node_kind is None or unrecognized.

        Note:
            This is an internal method. Callers should ensure entry.node_kind
            is not None before calling.

        Time Semantics:
            The ``now`` field is captured at context creation time (dispatch time),
            NOT at handler execution time. For ORCHESTRATOR, EFFECT, and RUNTIME_HOST
            nodes, this means:

            - ``now`` represents when MessageDispatchEngine created the context
            - Handler execution may occur microseconds to milliseconds later
            - For most use cases, this drift is negligible
            - If sub-millisecond precision is required, handlers should capture
              their own time at the start of execution

        .. versionadded:: 0.5.0
        .. versionchanged:: 0.5.1
            Now delegates to DispatchContextEnforcer.create_context_for_node_kind()
            to eliminate code duplication.
        """
        node_kind = entry.node_kind
        if node_kind is None:
            raise ModelOnexError(
                message=f"Cannot create context for dispatcher '{entry.dispatcher_id}': "
                "node_kind is None. This is an internal error.",
                error_code=EnumCoreErrorCode.INTERNAL_ERROR,
            )

        # Delegate to the shared context enforcer for time injection rules.
        # This eliminates duplication between MessageDispatchEngine and any
        # other components that need to create contexts based on node_kind.
        return self._context_enforcer.create_context_for_node_kind(
            node_kind=node_kind,
            envelope=envelope,
            dispatcher_id=entry.dispatcher_id,
        )

    def _dispatcher_accepts_context(
        self,
        dispatcher: DispatcherFunc | ContextAwareDispatcherFunc,
    ) -> bool:
        """
        Check if a dispatcher callable accepts a context parameter.

        Uses inspect.signature to determine if the dispatcher has a second
        parameter for ModelDispatchContext. This enables backwards-compatible
        context injection - dispatchers without a context parameter will be
        called with just the envelope.

        This method is called once at registration time and the result is
        cached in DispatchEntryInternal.accepts_context for performance.
        No signature inspection occurs during dispatch execution.

        Type Safety Warnings:
            When a dispatcher has 2+ parameters but the second parameter doesn't
            follow conventional naming (containing 'context' or 'ctx'), a warning
            is logged to help developers identify potential signature mismatches.
            This is non-blocking - the method still returns True for backwards
            compatibility with existing dispatchers.

        Args:
            dispatcher: The dispatcher callable to inspect.

        Returns:
            True if dispatcher accepts a context parameter, False otherwise.

        .. versionadded:: 0.5.0
        .. versionchanged:: 0.5.1
            Added warning logging for unconventional parameter naming.
        """
        try:
            sig = inspect.signature(dispatcher)
            params = list(sig.parameters.values())
            # Dispatcher with context has 2+ parameters: (envelope, context, ...)
            # Dispatcher without context has 1 parameter: (envelope)
            #
            # Design Decision: We use >= MIN_PARAMS_FOR_CONTEXT (not ==) intentionally
            # to support:
            # - Future extensibility (e.g., envelope, context, **kwargs)
            # - Dispatchers with additional optional parameters for testing/logging
            # - Protocol compliance without strict arity enforcement
            #
            # Strict == MIN_PARAMS_FOR_CONTEXT would reject valid dispatchers that
            # happen to have extra optional parameters, which is unnecessarily restrictive.
            if len(params) < MIN_PARAMS_FOR_CONTEXT:
                return False

            # Type safety enhancement: Warn if second parameter doesn't follow
            # context naming convention. This helps developers identify potential
            # signature mismatches where a 2+ parameter dispatcher might not
            # actually expect a ModelDispatchContext.
            #
            # This is NON-BLOCKING - we still return True.
            # The warning is informational to help improve code quality.
            second_param = params[1]
            second_name = second_param.name.lower()
            if "context" not in second_name and "ctx" not in second_name:
                dispatcher_name = getattr(dispatcher, "__name__", str(dispatcher))
                self._logger.warning(
                    "Dispatcher '%s' has 2+ parameters but second parameter '%s' "
                    "doesn't follow context naming convention. "
                    "Expected parameter name containing 'context' or 'ctx'. "
                    "If this dispatcher expects a ModelDispatchContext, consider "
                    "renaming the parameter for clarity.",
                    dispatcher_name,
                    second_param.name,
                )

            return True
        except (ValueError, TypeError) as e:
            # If we can't inspect the signature, assume no context and log warning
            self._logger.warning(
                "Failed to inspect dispatcher signature: %s. "
                "Assuming no context parameter. Uninspectable dispatchers "
                "(C extensions, certain decorators) will receive envelope only.",
                e,
            )
            return False

    def _extract_operation_from_envelope(
        self, envelope: object, correlation_id: UUID | None = None
    ) -> str:
        """Extract operation name from envelope.

        Supports both dict-based envelopes and Pydantic model envelopes
        with an ``operation`` attribute.

        Args:
            envelope: Event envelope (dict or Pydantic model).
            correlation_id: Optional correlation ID for error context.

        Returns:
            Operation name string.

        Raises:
            BindingResolutionError: When operation cannot be extracted. This
                error is raised in the following cases:

                1. **Dict envelope without ``operation`` key**: The envelope is a dict
                   but does not contain an ``"operation"`` key.
                2. **Model envelope without ``operation`` attribute**: The envelope is
                   a Pydantic model or object but lacks an ``operation`` attribute.
                3. **Empty/None operation value**: The operation key/attribute exists
                   but has a None or falsy value.

        Fail-Fast Behavior:
            This method implements fail-fast semantics. If an operation cannot be
            extracted, it raises ``BindingResolutionError`` immediately rather than
            returning a fallback value. This ensures:

            - Early detection of misconfigured envelopes
            - Clear error messages with diagnostic context
            - No silent failures from attempting to resolve bindings for "unknown" operation

        Note:
            If ``entry.operation_bindings`` is ``None`` (no bindings configured),
            this method is not called at all, avoiding unnecessary extraction.

        .. versionadded:: 0.2.6
            Added as part of OMN-1518 - Declarative operation bindings.

        .. versionchanged:: 0.2.7
            Changed from fallback-to-unknown to fail-fast behavior.
        """
        # Dict-based envelopes (common for Kafka/JSON payloads)
        if isinstance(envelope, dict):
            operation = envelope.get("operation")
            if operation is not None:
                return str(operation)
            # NOTE: Do not log envelope.keys() - may contain sensitive field names
            raise BindingResolutionError(
                "Operation extraction failed: dict envelope missing 'operation' key. "
                f"Ensure envelope contains 'operation' key (found {len(envelope)} keys).",
                operation_name="extraction_failed",
                parameter_name="operation",
                expression="envelope['operation']",
                missing_segment="operation",
                correlation_id=correlation_id,
            )

        # Pydantic model or object with operation attribute
        if hasattr(envelope, "operation"):
            operation = getattr(envelope, "operation", None)
            if operation is not None:
                return str(operation)
            raise BindingResolutionError(
                "Operation extraction failed: envelope has 'operation' attribute "
                f"but value is None or falsy. Envelope type: {type(envelope).__name__}",
                operation_name="extraction_failed",
                parameter_name="operation",
                expression="envelope.operation",
                missing_segment="operation",
                correlation_id=correlation_id,
            )

        # No operation attribute at all
        raise BindingResolutionError(
            "Operation extraction failed: envelope has no 'operation' attribute. "
            f"Envelope type: {type(envelope).__name__}",
            operation_name="extraction_failed",
            parameter_name="operation",
            expression="envelope.operation or envelope['operation']",
            missing_segment="operation",
            correlation_id=correlation_id,
        )

    def _extract_correlation_id(self, envelope: object) -> UUID | None:
        """Extract correlation_id from envelope.

        Supports both dict-based envelopes and Pydantic model envelopes
        with a ``correlation_id`` attribute. Handles both UUID and string values.

        Args:
            envelope: Event envelope (dict or Pydantic model).

        Returns:
            UUID correlation_id if found and valid, None otherwise.

        .. versionadded:: 0.2.6
            Added as part of OMN-1518 - Declarative operation bindings.
        """
        cid: object = None
        if isinstance(envelope, dict):
            cid = envelope.get("correlation_id")
        elif hasattr(envelope, "correlation_id"):
            cid = getattr(envelope, "correlation_id", None)

        if isinstance(cid, UUID):
            return cid
        if isinstance(cid, str):
            try:
                return UUID(cid)
            except ValueError:
                return None
        return None

    def _materialize_envelope_with_bindings(
        self,
        original_envelope: object,
        resolved_bindings: dict[str, JsonType],
        topic: str,
    ) -> dict[str, JsonType]:
        """Create new JSON-safe envelope with __bindings namespace.

        INVARIANT: original_envelope is NEVER mutated.
        INVARIANT: All output values are JSON-serializable.

        This method ALWAYS creates a new dict containing:
        - ``payload``: Event payload as JSON-safe dict
        - ``__bindings``: Resolved binding parameters (empty dict if no bindings)
        - ``__debug_trace``: Serialized trace metadata snapshot (debug only)

        The dispatch boundary is a **serialization boundary**. All data crossing
        this layer must be transport-safe (JSON-serializable) to enable:
        - Event replay from logs or Kafka
        - Distributed dispatch across processes
        - Observability tooling (logging, tracing, dashboards)

        Warning:
            ``__debug_trace`` is provided ONLY for debugging and observability.
            It is a serialized snapshot of trace metadata, NOT the live envelope.

            **DO NOT**:
            - Use ``__debug_trace`` for business logic
            - Assume ``__debug_trace`` reflects complete envelope state
            - Depend on specific fields being present

        Args:
            original_envelope: Original event envelope (dict or model).
            resolved_bindings: JSON-safe resolved binding parameters (may be empty).
            topic: The topic this message was received on.

        Returns:
            New dict conforming to ModelMaterializedDispatch schema.
            All values are JSON-serializable (transport-safe).

        Example:
            >>> materialized = engine._materialize_envelope_with_bindings(
            ...     original_envelope={"payload": {"sql": "SELECT 1"}},
            ...     resolved_bindings={"sql": "SELECT 1", "limit": 100},
            ...     topic="dev.db.commands.v1",
            ... )
            >>> materialized["__bindings"]
            {'sql': 'SELECT 1', 'limit': 100}

        .. versionadded:: 0.2.6
            Added as part of OMN-1518 - Declarative operation bindings.

        .. versionchanged:: 0.2.8
            Changed to strict JSON-safe contract:
            - Payload is serialized to dict (Pydantic models call model_dump)
            - Bindings values are serialized (UUIDs/datetimes → strings)
            - __debug_original_envelope replaced with __debug_trace snapshot
            - Return type changed to dict[str, JsonType]
        """
        # Extract and serialize payload to JSON-safe dict
        payload_json = self._serialize_payload(original_envelope)

        # Serialize bindings to JSON-safe values
        bindings_json = self._serialize_bindings(resolved_bindings)

        # Create debug trace snapshot (serialized, non-authoritative)
        debug_trace = self._create_debug_trace_snapshot(original_envelope, topic)

        # Build materialized dict conforming to ModelMaterializedDispatch schema
        # NOTE: debug_trace is dict[str, str | None] which is a subset of JsonType
        materialized: dict[str, JsonType] = {
            "payload": payload_json,
            "__bindings": bindings_json,
            "__debug_trace": cast("JsonType", debug_trace),
        }

        # Validate against schema to enforce contract invariants
        # This catches shape drift and provides clear error messages
        ModelMaterializedDispatch.model_validate(materialized)

        return materialized

    def _serialize_payload(self, original_envelope: object) -> JsonType:
        """Extract and serialize payload to JSON-safe dict.

        Handlers that need typed Pydantic models should hydrate locally:
            ``ModelFoo.model_validate(dispatch["payload"])``

        Args:
            original_envelope: Original event envelope (dict or model).

        Returns:
            JSON-safe payload (dict for complex types, wrapped for primitives).
        """
        # Extract original payload
        original_payload: object
        if isinstance(original_envelope, dict):
            original_payload = original_envelope.get("payload", original_envelope)
        elif hasattr(original_envelope, "payload"):
            original_payload = getattr(original_envelope, "payload", original_envelope)
        else:
            original_payload = original_envelope

        # Serialize to JSON-safe format
        if hasattr(original_payload, "model_dump"):
            # Pydantic model - serialize to dict with JSON mode
            # NOTE: model_dump returns Any, but we know it's JSON-compatible
            return original_payload.model_dump(mode="json")  # type: ignore[union-attr, no-any-return]
        elif isinstance(original_payload, dict):
            # Dict - recursively serialize values
            return self._serialize_dict_values(original_payload)
        elif isinstance(original_payload, (str, int, float, bool, type(None))):
            # JSON primitive - wrap to maintain dict structure
            # NOTE: This is a last-resort escape hatch. Prefer dict payloads.
            return {"_raw": original_payload}
        elif isinstance(original_payload, list):
            # List - recursively serialize elements
            return [self._serialize_value(item) for item in original_payload]
        elif hasattr(original_payload, "__dict__"):
            # Plain Python object with attributes - serialize its __dict__
            # This handles domain objects that aren't Pydantic models
            self._logger.warning(
                "Serializing payload via __dict__ fallback for type %s. "
                "Consider using a Pydantic model for explicit serialization control.",
                type(original_payload).__name__,
            )
            return self._serialize_dict_values(vars(original_payload))
        else:
            # Unknown type - attempt string conversion
            self._logger.warning(
                "Serializing payload via string fallback for type %s. "
                "This may indicate a configuration error - payload types should be "
                "Pydantic models, dicts, or JSON primitives.",
                type(original_payload).__name__,
            )
            return {"_raw": str(original_payload)}

    def _serialize_bindings(self, bindings: dict[str, JsonType]) -> dict[str, JsonType]:
        """Ensure binding values are JSON-safe (idempotent serialization).

        When bindings are already JSON-safe (from OperationBindingResolver),
        this method is effectively a no-op but provides defensive serialization.

        Args:
            bindings: Dict of resolved binding parameters (already JSON-safe).

        Returns:
            Dict with JSON-safe values.
        """
        return {key: self._serialize_value(value) for key, value in bindings.items()}

    def _serialize_value(self, value: object) -> JsonType:
        """Serialize a single value to JSON-safe type.

        Args:
            value: Any value to serialize.

        Returns:
            JSON-safe representation.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value  # type: ignore[return-value]
        elif isinstance(value, UUID):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif hasattr(value, "model_dump"):
            # Pydantic model - model_dump returns Any, but we know it's JSON-compatible
            return value.model_dump(mode="json")  # type: ignore[union-attr, no-any-return]
        elif isinstance(value, dict):
            return self._serialize_dict_values(value)
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        else:
            # Unknown type - string conversion
            return str(value)

    def _serialize_dict_values(self, d: dict[str, object]) -> dict[str, JsonType]:
        """Recursively serialize dict values."""
        return {key: self._serialize_value(value) for key, value in d.items()}

    def _create_debug_trace_snapshot(
        self, original_envelope: object, topic: str
    ) -> dict[str, str | None]:
        """Create serialized trace metadata snapshot.

        This snapshot is for debugging and observability ONLY.
        It is NOT authoritative and should NOT be used for business logic.

        Args:
            original_envelope: Original event envelope.
            topic: The topic this message was received on.

        Returns:
            Dict with serialized trace metadata (all strings or None).
        """
        # Extract trace fields safely (any missing field → None)
        event_type: str | None = None
        correlation_id: str | None = None
        trace_id: str | None = None
        causation_id: str | None = None
        timestamp: str | None = None
        partition_key: str | None = None

        if isinstance(original_envelope, dict):
            event_type = self._safe_str(original_envelope.get("event_type"))
            correlation_id = self._safe_str(original_envelope.get("correlation_id"))
            trace_id = self._safe_str(original_envelope.get("trace_id"))
            causation_id = self._safe_str(original_envelope.get("causation_id"))
            timestamp = self._safe_str(original_envelope.get("timestamp"))
            partition_key = self._safe_str(original_envelope.get("partition_key"))
        else:
            # Pydantic model or other object - use getattr
            # All fields go through _safe_str to handle non-string types gracefully
            event_type = self._safe_str(getattr(original_envelope, "event_type", None))
            correlation_id = self._safe_str(
                getattr(original_envelope, "correlation_id", None)
            )
            trace_id = self._safe_str(getattr(original_envelope, "trace_id", None))
            causation_id = self._safe_str(
                getattr(original_envelope, "causation_id", None)
            )
            timestamp = self._safe_str(getattr(original_envelope, "timestamp", None))
            partition_key = self._safe_str(
                getattr(original_envelope, "partition_key", None)
            )

        # Build snapshot using the model for validation
        snapshot = ModelDebugTraceSnapshot(
            event_type=event_type,
            correlation_id=correlation_id,
            trace_id=trace_id,
            causation_id=causation_id,
            topic=topic,
            timestamp=timestamp,
            partition_key=partition_key,
        )

        return snapshot.model_dump()

    def _safe_str(self, value: object) -> str | None:
        """Safely convert value to string or None.

        This method is defensive - it only returns a string if the value
        can be meaningfully converted. Mock objects and other test artifacts
        are filtered out to avoid polluting trace snapshots.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        # For other types, only convert if it results in a meaningful string
        # Filter out mock objects and other test artifacts
        str_value = str(value)

        # Check for mock objects by module (most reliable)
        value_module = type(value).__module__
        if value_module.startswith("unittest.mock"):
            return None

        # Check for common mock patterns in string representation
        mock_patterns = ("Mock", "MagicMock", "AsyncMock")
        if any(pattern in str_value for pattern in mock_patterns):
            return None

        # Check for Python internal repr patterns (more specific than just <...>)
        # These patterns indicate non-serializable internal objects
        if str_value.startswith("<") and str_value.endswith(">"):
            internal_prefixes = (
                "<class ",
                "<function ",
                "<module ",
                "<bound method ",
                "<built-in ",
                "<coroutine ",
                "<generator ",
                "<async_generator ",
            )
            if any(str_value.startswith(prefix) for prefix in internal_prefixes):
                return None

        return str_value

    def get_structured_metrics(self) -> ModelDispatchMetrics:
        """
        Get structured dispatch metrics using Pydantic model.

        Returns a comprehensive metrics model including:
        - Dispatch counts and success/error rates
        - Latency statistics (average, min, max)
        - Latency histogram for distribution analysis
        - Per-dispatcher metrics breakdown
        - Per-category metrics breakdown

        Thread Safety:
            This method acquires ``_metrics_lock`` to return a consistent snapshot.
            The same lock protects all metrics updates, ensuring TOCTOU-safe
            read-modify-write operations during dispatch. The returned Pydantic
            model is immutable and safe to use after the lock is released.

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
        # Return under lock to ensure consistent snapshot
        with self._metrics_lock:
            return self._structured_metrics

    def reset_metrics(self) -> None:
        """
        Reset all metrics to initial state.

        Useful for testing or when starting a new monitoring period.

        Thread Safety:
            This method acquires ``_metrics_lock`` to ensure atomic reset
            of all metrics. Safe to call during concurrent dispatch operations,
            though the reset will briefly block in-flight metric updates.

        Example:
            >>> engine.reset_metrics()
            >>> assert engine.get_structured_metrics().total_dispatches == 0

        .. versionadded:: 0.4.0
        """
        with self._metrics_lock:
            self._structured_metrics = ModelDispatchMetrics()
        self._logger.debug("Metrics reset to initial state")

    def get_dispatcher_metrics(
        self, dispatcher_id: str
    ) -> ModelDispatcherMetrics | None:
        """
        Get metrics for a specific dispatcher.

        Thread Safety:
            This method acquires ``_metrics_lock`` to return a consistent snapshot.
            The returned Pydantic model is immutable and safe to use after the
            lock is released.

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
        with self._metrics_lock:
            return self._structured_metrics.dispatcher_metrics.get(dispatcher_id)

    @property
    def route_count(self) -> int:
        """Get the number of registered routes."""
        return len(self._routes)

    @property
    def dispatcher_count(self) -> int:
        """Get the number of registered dispatchers."""
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
