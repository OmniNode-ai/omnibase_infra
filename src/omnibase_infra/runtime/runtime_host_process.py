# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Host Process implementation for ONEX Infrastructure.

This module implements the RuntimeHostProcess class, which is responsible for:
- Owning and managing an event bus instance (InMemoryEventBus or KafkaEventBus)
- Registering handlers via the wiring module
- Subscribing to event bus topics and routing envelopes to handlers
- Handling errors by producing success=False response envelopes
- Processing envelopes sequentially (no parallelism in MVP)
- Basic shutdown (no graceful drain in MVP)

The RuntimeHostProcess is the central coordinator for infrastructure runtime,
bridging event-driven message routing with protocol handlers.

Event Bus Support:
    The RuntimeHostProcess supports two event bus implementations:
    - InMemoryEventBus: For local development and testing
    - KafkaEventBus: For production use with Kafka/Redpanda

    The event bus can be injected via constructor or auto-created based on config.

Example Usage:
    ```python
    from omnibase_infra.runtime import RuntimeHostProcess

    async def main() -> None:
        process = RuntimeHostProcess()
        await process.start()
        try:
            # Process handles messages via event bus subscription
            await asyncio.sleep(60)
        finally:
            await process.stop()
    ```

Integration with Handlers:
    Handlers are registered during start() via the wiring module. Each handler
    processes envelopes for a specific protocol type (e.g., "http", "db").
    The handler_type field in envelopes determines routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    EnvelopeValidationError,
    ModelInfraErrorContext,
    RuntimeHostError,
    UnknownHandlerTypeError,
)
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
from omnibase_infra.runtime.envelope_validator import (
    normalize_correlation_id,
    validate_envelope,
)
from omnibase_infra.runtime.handler_registry import ProtocolBindingRegistry
from omnibase_infra.runtime.models import ModelDuplicateResponse
from omnibase_infra.runtime.protocol_lifecycle_executor import ProtocolLifecycleExecutor
from omnibase_infra.runtime.wiring import wire_default_handlers
from omnibase_infra.utils.util_env_parsing import parse_env_float

if TYPE_CHECKING:
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

    from omnibase_infra.event_bus.models import ModelEventMessage
    from omnibase_infra.idempotency import ModelIdempotencyGuardConfig
    from omnibase_infra.idempotency.protocol_idempotency_store import (
        ProtocolIdempotencyStore,
    )
    from omnibase_infra.models.types import JsonValue

from omnibase_infra.models.types import JsonDict

# Expose wire_default_handlers as wire_handlers for test patching compatibility
# Tests patch "omnibase_infra.runtime.runtime_host_process.wire_handlers"
wire_handlers = wire_default_handlers

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_INPUT_TOPIC = "requests"
DEFAULT_OUTPUT_TOPIC = "responses"
DEFAULT_GROUP_ID = "runtime-host"

# Health check timeout bounds (per ModelLifecycleSubcontract)
MIN_HEALTH_CHECK_TIMEOUT = 1.0
MAX_HEALTH_CHECK_TIMEOUT = 60.0
DEFAULT_HEALTH_CHECK_TIMEOUT: float = parse_env_float(
    "ONEX_HEALTH_CHECK_TIMEOUT",
    5.0,
    min_value=MIN_HEALTH_CHECK_TIMEOUT,
    max_value=MAX_HEALTH_CHECK_TIMEOUT,
    transport_type=EnumInfraTransportType.RUNTIME,
    service_name="runtime_host_process",
)

# Drain timeout bounds for graceful shutdown (OMN-756)
# Controls how long to wait for in-flight messages to complete before shutdown
MIN_DRAIN_TIMEOUT_SECONDS = 1.0
MAX_DRAIN_TIMEOUT_SECONDS = 300.0
DEFAULT_DRAIN_TIMEOUT_SECONDS: float = parse_env_float(
    "ONEX_DRAIN_TIMEOUT",
    30.0,
    min_value=MIN_DRAIN_TIMEOUT_SECONDS,
    max_value=MAX_DRAIN_TIMEOUT_SECONDS,
    transport_type=EnumInfraTransportType.RUNTIME,
    service_name="runtime_host_process",
)


class RuntimeHostProcess:
    """Runtime host process that owns event bus and coordinates handlers.

    The RuntimeHostProcess is the central coordinator for ONEX infrastructure
    runtime. It owns an event bus instance (InMemoryEventBus or KafkaEventBus),
    registers handlers via the wiring module, and routes incoming envelopes to
    appropriate handlers.

    Container Integration:
        RuntimeHostProcess now accepts a ModelONEXContainer parameter for
        dependency injection. The container provides access to:
        - ProtocolBindingRegistry: Handler registry for protocol routing

        This follows ONEX container-based DI patterns for better testability
        and lifecycle management. The legacy singleton pattern is deprecated
        in favor of container resolution.

    Attributes:
        event_bus: The owned event bus instance (InMemoryEventBus or KafkaEventBus)
        is_running: Whether the process is currently running
        input_topic: Topic to subscribe to for incoming envelopes
        output_topic: Topic to publish responses to
        group_id: Consumer group identifier

    Example:
        ```python
        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.runtime.container_wiring import wire_infrastructure_services

        # Container-based initialization (preferred)
        container = ModelONEXContainer()
        wire_infrastructure_services(container)
        process = RuntimeHostProcess(container=container)
        await process.start()
        health = await process.health_check()
        await process.stop()

        # Legacy initialization (backwards compatible, no container)
        process = RuntimeHostProcess()  # Uses singleton registries
        ```

    Graceful Shutdown:
        The stop() method implements graceful shutdown with a configurable drain
        period. After unsubscribing from topics, it waits for in-flight messages
        to complete before shutting down handlers and closing the event bus.
        See stop() docstring for configuration details.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus | KafkaEventBus | None = None,
        input_topic: str = DEFAULT_INPUT_TOPIC,
        output_topic: str = DEFAULT_OUTPUT_TOPIC,
        config: JsonDict | None = None,
        handler_registry: ProtocolBindingRegistry | None = None,
    ) -> None:
        """Initialize the runtime host process.

        Args:
            event_bus: Optional event bus instance (InMemoryEventBus or KafkaEventBus).
                       If None, creates InMemoryEventBus.
            input_topic: Topic to subscribe to for incoming envelopes.
            output_topic: Topic to publish responses to.
            config: Optional configuration dict that can override topics and group_id.
                Supported keys:
                    - input_topic: Override input topic
                    - output_topic: Override output topic
                    - group_id: Override consumer group identifier
                    - health_check_timeout_seconds: Timeout for individual handler
                      health checks (default: 5.0 seconds, valid range: 1-60 per
                      ModelLifecycleSubcontract). Values outside this range are
                      clamped to the nearest bound with a warning logged.
                      Invalid string values fall back to the default with a warning.
                    - drain_timeout_seconds: Maximum time to wait for in-flight
                      messages to complete during graceful shutdown (default: 30.0
                      seconds, valid range: 1-300). Values outside this range are
                      clamped to the nearest bound with a warning logged.
            handler_registry: Optional ProtocolBindingRegistry instance for handler lookup.
                Type: ProtocolBindingRegistry | None

                Purpose:
                    Provides the registry that maps handler_type strings (e.g., "http", "db")
                    to their corresponding ProtocolHandler classes. The registry is queried
                    during start() to instantiate and initialize all registered handlers.

                Resolution Order:
                    1. If handler_registry is provided, uses this pre-resolved registry
                    2. If None, falls back to singleton via get_handler_registry()

                Container Integration:
                    When using container-based DI (recommended), resolve the registry from
                    the container and pass it to RuntimeHostProcess:

                    ```python
                    container = ModelONEXContainer()
                    wire_infrastructure_services(container)
                    registry = container.service_registry.resolve_service(ProtocolBindingRegistry)
                    process = RuntimeHostProcess(handler_registry=registry)
                    ```

                    This follows ONEX container-based DI patterns for better testability
                    and explicit dependency management.
        """
        # Handler registry (container-based DI or singleton fallback)
        self._handler_registry: ProtocolBindingRegistry | None = handler_registry

        # Create or use provided event bus
        self._event_bus: InMemoryEventBus | KafkaEventBus = (
            event_bus or InMemoryEventBus()
        )

        # Extract configuration with defaults
        config = config or {}

        # Topic configuration (config overrides constructor args)
        self._input_topic: str = str(config.get("input_topic", input_topic))
        self._output_topic: str = str(config.get("output_topic", output_topic))
        # Note: ModelRuntimeConfig uses field name "consumer_group" with alias "group_id".
        # When config.model_dump() is called, it outputs "consumer_group" by default.
        # We check both keys for backwards compatibility with existing configs.
        # Empty strings and whitespace-only strings fall through to the next option.
        consumer_group = config.get("consumer_group")
        group_id = config.get("group_id")
        self._group_id: str = str(
            (consumer_group if consumer_group and str(consumer_group).strip() else None)
            or (group_id if group_id and str(group_id).strip() else None)
            or DEFAULT_GROUP_ID
        )

        # Health check configuration (from lifecycle subcontract pattern)
        # Default: 5.0 seconds, valid range: 1-60 seconds per ModelLifecycleSubcontract
        # Values outside bounds are clamped with a warning
        _timeout_raw = config.get("health_check_timeout_seconds")
        timeout_value: float = DEFAULT_HEALTH_CHECK_TIMEOUT
        if isinstance(_timeout_raw, int | float):
            timeout_value = float(_timeout_raw)
        elif isinstance(_timeout_raw, str):
            try:
                timeout_value = float(_timeout_raw)
            except ValueError:
                logger.warning(
                    "Invalid health_check_timeout_seconds string value, using default",
                    extra={
                        "invalid_value": _timeout_raw,
                        "default_value": DEFAULT_HEALTH_CHECK_TIMEOUT,
                    },
                )
                timeout_value = DEFAULT_HEALTH_CHECK_TIMEOUT

        # Validate bounds and clamp if necessary
        if (
            timeout_value < MIN_HEALTH_CHECK_TIMEOUT
            or timeout_value > MAX_HEALTH_CHECK_TIMEOUT
        ):
            logger.warning(
                "health_check_timeout_seconds out of valid range, clamping",
                extra={
                    "original_value": timeout_value,
                    "min_value": MIN_HEALTH_CHECK_TIMEOUT,
                    "max_value": MAX_HEALTH_CHECK_TIMEOUT,
                    "clamped_value": max(
                        MIN_HEALTH_CHECK_TIMEOUT,
                        min(timeout_value, MAX_HEALTH_CHECK_TIMEOUT),
                    ),
                },
            )
            timeout_value = max(
                MIN_HEALTH_CHECK_TIMEOUT,
                min(timeout_value, MAX_HEALTH_CHECK_TIMEOUT),
            )

        self._health_check_timeout_seconds: float = timeout_value

        # Drain timeout configuration for graceful shutdown (OMN-756)
        # Default: 30.0 seconds, valid range: 1-300 seconds
        # Values outside bounds are clamped with a warning
        _drain_timeout_raw = config.get("drain_timeout_seconds")
        drain_timeout_value: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
        if isinstance(_drain_timeout_raw, int | float):
            drain_timeout_value = float(_drain_timeout_raw)
        elif isinstance(_drain_timeout_raw, str):
            try:
                drain_timeout_value = float(_drain_timeout_raw)
            except ValueError:
                logger.warning(
                    "Invalid drain_timeout_seconds string value, using default",
                    extra={
                        "invalid_value": _drain_timeout_raw,
                        "default_value": DEFAULT_DRAIN_TIMEOUT_SECONDS,
                    },
                )
                drain_timeout_value = DEFAULT_DRAIN_TIMEOUT_SECONDS

        # Validate drain timeout bounds and clamp if necessary
        if (
            drain_timeout_value < MIN_DRAIN_TIMEOUT_SECONDS
            or drain_timeout_value > MAX_DRAIN_TIMEOUT_SECONDS
        ):
            logger.warning(
                "drain_timeout_seconds out of valid range, clamping",
                extra={
                    "original_value": drain_timeout_value,
                    "min_value": MIN_DRAIN_TIMEOUT_SECONDS,
                    "max_value": MAX_DRAIN_TIMEOUT_SECONDS,
                    "clamped_value": max(
                        MIN_DRAIN_TIMEOUT_SECONDS,
                        min(drain_timeout_value, MAX_DRAIN_TIMEOUT_SECONDS),
                    ),
                },
            )
            drain_timeout_value = max(
                MIN_DRAIN_TIMEOUT_SECONDS,
                min(drain_timeout_value, MAX_DRAIN_TIMEOUT_SECONDS),
            )

        self._drain_timeout_seconds: float = drain_timeout_value

        # Handler executor for lifecycle operations (shutdown, health check)
        self._lifecycle_executor = ProtocolLifecycleExecutor(
            health_check_timeout_seconds=self._health_check_timeout_seconds
        )

        # Store full config for handler initialization
        self._config: JsonDict = config

        # Runtime state
        self._is_running: bool = False

        # Subscription handle (callable to unsubscribe)
        self._subscription: Callable[[], Awaitable[None]] | None = None

        # Handler registry (handler_type -> handler instance)
        # This will be populated from the singleton registry during start()
        self._handlers: dict[str, ProtocolHandler] = {}

        # Track failed handler instantiations (handler_type -> error message)
        # Used by health_check() to report degraded state
        self._failed_handlers: dict[str, str] = {}

        # Pending message tracking for graceful shutdown (OMN-756)
        # Tracks count of in-flight messages currently being processed
        self._pending_message_count: int = 0
        self._pending_lock: asyncio.Lock = asyncio.Lock()

        # Drain state tracking for graceful shutdown (OMN-756)
        # True when stop() has been called and we're waiting for messages to drain
        self._is_draining: bool = False

        # Idempotency guard for duplicate message detection (OMN-945)
        # None = disabled, otherwise points to configured store
        self._idempotency_store: ProtocolIdempotencyStore | None = None
        self._idempotency_config: ModelIdempotencyGuardConfig | None = None

        logger.debug(
            "RuntimeHostProcess initialized",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
                "health_check_timeout_seconds": self._health_check_timeout_seconds,
                "drain_timeout_seconds": self._drain_timeout_seconds,
            },
        )

    @property
    def event_bus(self) -> InMemoryEventBus | KafkaEventBus:
        """Return the owned event bus instance.

        Returns:
            The event bus instance managed by this process.
        """
        return self._event_bus

    @property
    def is_running(self) -> bool:
        """Return True if runtime is started.

        Returns:
            Boolean indicating whether the process is running.
        """
        return self._is_running

    @property
    def input_topic(self) -> str:
        """Return the input topic for envelope subscription.

        Returns:
            The topic name to subscribe to for incoming envelopes.
        """
        return self._input_topic

    @property
    def output_topic(self) -> str:
        """Return the output topic for response publishing.

        Returns:
            The topic name to publish responses to.
        """
        return self._output_topic

    @property
    def group_id(self) -> str:
        """Return the consumer group identifier.

        Returns:
            The consumer group ID for this process.
        """
        return self._group_id

    @property
    def is_draining(self) -> bool:
        """Return True if the process is draining pending messages during shutdown.

        This property indicates whether the runtime host is in the graceful shutdown
        drain period - the phase where stop() has been called, new messages are no
        longer being accepted, and the process is waiting for in-flight messages to
        complete before shutting down handlers and the event bus.

        Drain State Transitions:
            - False: Normal operation (accepting and processing messages)
            - True: Drain period active (stop() called, waiting for pending messages)
            - False: After drain completes and shutdown finishes

        Use Cases:
            - Health check reporting (indicate service is shutting down)
            - Load balancer integration (remove from rotation during drain)
            - Monitoring dashboards (show lifecycle state)
            - Debugging shutdown behavior

        Returns:
            True if currently in drain period during graceful shutdown, False otherwise.
        """
        return self._is_draining

    @property
    def pending_message_count(self) -> int:
        """Return the current count of in-flight messages being processed.

        This property provides visibility into how many messages are currently
        being processed by the runtime host. Used for graceful shutdown to
        determine when it's safe to complete the shutdown process.

        Atomicity Guarantees:
            This property returns the raw counter value WITHOUT acquiring the
            async lock (_pending_lock). This is safe because:

            1. Single int read is atomic under CPython's GIL - reading a single
               integer value cannot be interrupted mid-operation
            2. The value is only used for observability/monitoring purposes
               where exact precision is not required
            3. The slight possibility of reading a stale value during concurrent
               increment/decrement is acceptable for monitoring use cases

        Thread Safety Considerations:
            While the read itself is atomic, the value may be approximate if
            read occurs during concurrent message processing:
            - Another coroutine may be in the middle of incrementing/decrementing
            - The value represents a point-in-time snapshot, not a synchronized view
            - For observability, this approximation is acceptable and avoids
              lock contention that would impact performance

        Use Cases (appropriate for this property):
            - Logging current message count for debugging
            - Metrics/observability dashboards
            - Approximate health status reporting
            - Monitoring drain progress during shutdown

        When to use shutdown_ready() instead:
            For shutdown decisions requiring precise count, use the async
            shutdown_ready() method which acquires the lock to ensure no
            race condition with in-flight message processing. The stop()
            method uses shutdown_ready() internally for this reason.

        Returns:
            Current count of messages being processed. May be approximate
            if reads occur during concurrent increment/decrement operations.
        """
        return self._pending_message_count

    async def shutdown_ready(self) -> bool:
        """Check if process is ready for shutdown (no pending messages).

        This method acquires the pending message lock to ensure an accurate
        count of in-flight messages. Use this method during graceful shutdown
        to determine when all pending messages have been processed.

        Returns:
            True if no messages are currently being processed, False otherwise.
        """
        async with self._pending_lock:
            return self._pending_message_count == 0

    async def start(self) -> None:
        """Start the runtime host.

        Performs the following steps:
        1. Start event bus (if not already started)
        2. Wire handlers via wiring module (registers handler classes to singleton)
        3. Populate self._handlers from singleton registry (instantiate and initialize)
        4. Subscribe to input topic

        This method is idempotent - calling start() on an already started
        process is safe and has no effect.
        """
        if self._is_running:
            logger.debug("RuntimeHostProcess already started, skipping")
            return

        logger.info(
            "Starting RuntimeHostProcess",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
            },
        )

        # Step 1: Start event bus
        await self._event_bus.start()

        # Step 2: Wire handlers via wiring module
        # This registers default handler CLASSES with the singleton registry
        wire_handlers()

        # Step 3: Populate self._handlers from singleton registry
        # The wiring module registers handler classes, so we need to:
        # - Get each registered handler class from the singleton registry
        # - Instantiate the handler class
        # - Call initialize() on each handler instance with config
        # - Store the handler instance in self._handlers for routing
        await self._populate_handlers_from_registry()

        # Step 3.5: Initialize idempotency store if configured (OMN-945)
        await self._initialize_idempotency_store()

        # Step 4: Subscribe to input topic
        self._subscription = await self._event_bus.subscribe(
            topic=self._input_topic,
            group_id=self._group_id,
            on_message=self._on_message,
        )

        self._is_running = True

        logger.info(
            "RuntimeHostProcess started successfully",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
                "registered_handlers": list(self._handlers.keys()),
            },
        )

    async def stop(self) -> None:
        """Stop the runtime host with graceful drain period.

        Performs the following steps:
        1. Unsubscribe from topics (stop receiving new messages)
        2. Wait for in-flight messages to drain (up to drain_timeout_seconds)
        3. Shutdown all registered handlers by priority (release resources)
        4. Close event bus

        This method is idempotent - calling stop() on an already stopped
        process is safe and has no effect.

        Drain Period:
            After unsubscribing from topics, the process waits for in-flight
            messages to complete processing. The drain period is controlled by
            the drain_timeout_seconds configuration parameter (default: 30.0
            seconds, valid range: 1-300).

            During the drain period:
            - No new messages are received (unsubscribed from topics)
            - Messages currently being processed are allowed to complete
            - shutdown_ready() is polled every 100ms to check completion
            - If timeout is exceeded, shutdown proceeds with a warning

        Handler Shutdown Order:
            Handlers are shutdown in priority order, with higher priority handlers
            shutting down first. Within the same priority level, handlers are
            shutdown in parallel for performance.

            Priority is determined by the handler's shutdown_priority() method:
            - Higher values = shutdown first
            - Handlers without shutdown_priority() get default priority of 0

            Recommended Priority Scheme:
            - 100: Consumers (stop receiving before stopping producers)
            - 80: Active connections (close before closing pools)
            - 50: Producers (stop producing before closing pools)
            - 40: Connection pools (close last)
            - 0: Default for handlers without explicit priority

            This ensures dependency-based ordering:
            - Consumers shutdown before producers
            - Connections shutdown before connection pools
            - Downstream resources shutdown before upstream resources
        """
        if not self._is_running:
            logger.debug("RuntimeHostProcess already stopped, skipping")
            return

        logger.info("Stopping RuntimeHostProcess")

        # Step 1: Unsubscribe from topics (stop receiving new messages)
        if self._subscription is not None:
            await self._subscription()
            self._subscription = None

        # Step 1.5: Wait for in-flight messages to drain (OMN-756)
        # This allows messages currently being processed to complete
        loop = asyncio.get_running_loop()
        drain_start = loop.time()
        drain_deadline = drain_start + self._drain_timeout_seconds
        last_progress_log = drain_start

        # Mark drain state for health check visibility (OMN-756)
        self._is_draining = True

        # Log drain start for observability
        logger.info(
            "Starting drain period",
            extra={
                "pending_messages": self._pending_message_count,
                "drain_timeout_seconds": self._drain_timeout_seconds,
            },
        )

        while not await self.shutdown_ready():
            remaining = drain_deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Drain timeout exceeded, forcing shutdown",
                    extra={
                        "pending_messages": self._pending_message_count,
                        "drain_timeout_seconds": self._drain_timeout_seconds,
                        "metric.drain_timeout_exceeded": True,
                        "metric.pending_at_timeout": self._pending_message_count,
                    },
                )
                break

            # Wait a short interval before checking again
            await asyncio.sleep(min(0.1, remaining))

            # Log progress every 5 seconds during long drains for observability
            elapsed = loop.time() - drain_start
            if elapsed - (last_progress_log - drain_start) >= 5.0:
                logger.info(
                    "Drain in progress",
                    extra={
                        "pending_messages": self._pending_message_count,
                        "elapsed_seconds": round(elapsed, 2),
                        "remaining_seconds": round(remaining, 2),
                    },
                )
                last_progress_log = loop.time()

        # Clear drain state after drain period completes
        self._is_draining = False

        logger.info(
            "Drain period completed",
            extra={
                "drain_duration_seconds": loop.time() - drain_start,
                "pending_messages": self._pending_message_count,
                "metric.drain_duration": loop.time() - drain_start,
                "metric.forced_shutdown": self._pending_message_count > 0,
            },
        )

        # Step 2: Shutdown all handlers by priority (release resources like DB/Kafka connections)
        # Delegates to ProtocolLifecycleExecutor which handles:
        # - Grouping handlers by priority (higher priority first)
        # - Parallel shutdown within priority groups for performance
        if self._handlers:
            shutdown_result = (
                await self._lifecycle_executor.shutdown_handlers_by_priority(
                    self._handlers
                )
            )

            # Log summary (ProtocolLifecycleExecutor already logs detailed info)
            logger.info(
                "Handler shutdown completed",
                extra={
                    "succeeded_handlers": shutdown_result.succeeded_handlers,
                    "failed_handlers": [
                        f.handler_type for f in shutdown_result.failed_handlers
                    ],
                    "total_handlers": shutdown_result.total_count,
                    "success_count": shutdown_result.success_count,
                    "failure_count": shutdown_result.failure_count,
                },
            )

        # Step 2.5: Cleanup idempotency store if initialized (OMN-945)
        await self._cleanup_idempotency_store()

        # Step 3: Close event bus
        await self._event_bus.close()

        self._is_running = False

        logger.info("RuntimeHostProcess stopped successfully")

    async def _populate_handlers_from_registry(self) -> None:
        """Populate self._handlers from handler registry (container or singleton).

        This method bridges the gap between the wiring module (which registers
        handler CLASSES to the registry) and the RuntimeHostProcess
        (which needs handler INSTANCES in self._handlers for routing).

        Registry Resolution:
            - If handler_registry provided: Uses pre-resolved registry
            - If no handler_registry: Falls back to singleton get_handler_registry()

        For each registered handler type in the registry:
        1. Skip if handler type is already registered (e.g., by tests)
        2. Get the handler class from the registry
        3. Instantiate the handler class
        4. Call initialize() on the handler instance with self._config
        5. Store the handler instance in self._handlers

        This ensures that after start() is called, self._handlers contains
        fully initialized handler instances ready for envelope routing.

        Note: Handlers already in self._handlers (e.g., injected by tests via
        register_handler() or patch.object()) are preserved and not overwritten.
        """
        # Get handler registry (pre-resolved or singleton)
        handler_registry = self._get_handler_registry()
        registered_types = handler_registry.list_protocols()

        logger.debug(
            "Populating handlers from singleton registry",
            extra={
                "registered_types": registered_types,
                "existing_handlers": list(self._handlers.keys()),
            },
        )

        for handler_type in registered_types:
            # Skip if handler is already registered (e.g., by tests or explicit registration)
            if handler_type in self._handlers:
                logger.debug(
                    "Handler already registered, skipping",
                    extra={
                        "handler_type": handler_type,
                        "existing_handler_class": type(
                            self._handlers[handler_type]
                        ).__name__,
                    },
                )
                continue

            try:
                # Get handler class from singleton registry
                handler_cls: type[ProtocolHandler] = handler_registry.get(handler_type)

                # Instantiate the handler
                handler_instance: ProtocolHandler = handler_cls()

                # Call initialize() if the handler has this method
                # Handlers may require async initialization with config
                if hasattr(handler_instance, "initialize"):
                    await handler_instance.initialize(self._config)

                # Store the handler instance for routing
                self._handlers[handler_type] = handler_instance

                logger.debug(
                    "Handler instantiated and initialized",
                    extra={
                        "handler_type": handler_type,
                        "handler_class": handler_cls.__name__,
                    },
                )

            except Exception as e:
                # Track the failure for health_check() reporting
                self._failed_handlers[handler_type] = str(e)

                # Log error but continue with other handlers
                # This allows partial handler availability
                correlation_id = uuid4()
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="populate_handlers",
                    target_name=handler_type,
                    correlation_id=correlation_id,
                )
                infra_error = RuntimeHostError(
                    f"Failed to instantiate handler for type {handler_type}: {e}",
                    context=context,
                )
                infra_error.__cause__ = e

                logger.warning(
                    "Failed to instantiate handler, skipping",
                    extra={
                        "handler_type": handler_type,
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        logger.info(
            "Handlers populated from registry",
            extra={
                "populated_handlers": list(self._handlers.keys()),
                "total_count": len(self._handlers),
            },
        )

    def _get_handler_registry(self) -> ProtocolBindingRegistry:
        """Get handler registry (pre-resolved or singleton).

        Returns:
            ProtocolBindingRegistry instance (pre-resolved from container or singleton).
        """
        if self._handler_registry is not None:
            # Use pre-resolved registry from container
            return self._handler_registry
        else:
            # Backwards compatibility: fall back to singleton pattern
            from omnibase_infra.runtime.handler_registry import get_handler_registry

            return get_handler_registry()

    async def _on_message(self, message: ModelEventMessage) -> None:
        """Handle incoming message from event bus subscription.

        This is the callback invoked by the event bus when a message arrives
        on the input topic. It deserializes the envelope and routes it.

        The method tracks pending messages for graceful shutdown support (OMN-756).
        The pending message count is incremented at the start of processing and
        decremented when processing completes (success or failure).

        Args:
            message: The event message containing the envelope payload.
        """
        # Increment pending message count (OMN-756: graceful shutdown tracking)
        async with self._pending_lock:
            self._pending_message_count += 1

        try:
            # Deserialize envelope from message value
            envelope = json.loads(message.value.decode("utf-8"))
            await self._handle_envelope(envelope)
        except json.JSONDecodeError as e:
            # Create infrastructure error context for tracing
            correlation_id = uuid4()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="decode_envelope",
                target_name=message.topic,
                correlation_id=correlation_id,
            )
            # Chain the error with infrastructure context
            infra_error = RuntimeHostError(
                f"Failed to decode JSON envelope from message: {e}",
                context=context,
            )
            infra_error.__cause__ = e  # Proper error chaining

            logger.exception(
                "Failed to decode envelope from message",
                extra={
                    "error": str(e),
                    "topic": message.topic,
                    "offset": message.offset,
                    "correlation_id": str(correlation_id),
                },
            )
            # Publish error response for malformed messages
            error_response = self._create_error_response(
                error=f"Invalid JSON in message: {e}",
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
        finally:
            # Decrement pending message count (OMN-756: graceful shutdown tracking)
            async with self._pending_lock:
                self._pending_message_count -= 1

    async def _handle_envelope(self, envelope: JsonDict) -> None:
        """Route envelope to appropriate handler.

        Validates envelope before dispatch and routes it to the appropriate
        registered handler. Publishes the response to the output topic.

        Validation (performed before dispatch):
        1. Operation presence and type validation
        2. Handler prefix validation against registry
        3. Payload requirement validation for specific operations
        4. Correlation ID normalization to UUID

        Args:
            envelope: Dict with 'operation', 'payload', optional 'correlation_id',
                and 'handler_type'.
        """
        # Pre-validation: Get correlation_id for error responses if validation fails
        # This handles the case where validation itself throws before normalizing
        pre_validation_correlation_id = normalize_correlation_id(
            envelope.get("correlation_id")
        )

        # Step 1: Validate envelope BEFORE dispatch
        # This validates operation, prefix, payload requirements, and normalizes correlation_id
        try:
            validate_envelope(envelope, self._get_handler_registry())
        except EnvelopeValidationError as e:
            # Validation failed - missing operation or payload
            error_response = self._create_error_response(
                error=str(e),
                correlation_id=pre_validation_correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
            logger.warning(
                "Envelope validation failed",
                extra={
                    "error": str(e),
                    "correlation_id": str(pre_validation_correlation_id),
                    "error_type": "EnvelopeValidationError",
                },
            )
            return
        except UnknownHandlerTypeError as e:
            # Unknown handler prefix - hard failure
            error_response = self._create_error_response(
                error=str(e),
                correlation_id=pre_validation_correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
            logger.warning(
                "Unknown handler type in envelope",
                extra={
                    "error": str(e),
                    "correlation_id": str(pre_validation_correlation_id),
                    "error_type": "UnknownHandlerTypeError",
                },
            )
            return

        # After validation, correlation_id is guaranteed to be a UUID
        correlation_id = envelope.get("correlation_id")
        if not isinstance(correlation_id, UUID):
            correlation_id = pre_validation_correlation_id

        # Step 2: Check idempotency before handler dispatch (OMN-945)
        # This prevents duplicate processing under at-least-once delivery
        if not await self._check_idempotency(envelope, correlation_id):
            # Duplicate detected - response already published, return early
            return

        # Extract operation (validated to exist and be a string)
        operation = str(envelope.get("operation"))

        # Determine handler_type from envelope
        # If handler_type not explicit, extract from operation (e.g., "http.get" -> "http")
        handler_type = envelope.get("handler_type")
        if handler_type is None:
            handler_type = operation.split(".")[0]

        # Get handler from registry
        handler = self._handlers.get(str(handler_type))

        if handler is None:
            # Handler not instantiated (different from unknown prefix - validation already passed)
            # This can happen if handler registration failed during start()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation=str(operation),
                target_name=str(handler_type),
                correlation_id=correlation_id,
            )

            # Create structured error for logging and tracking
            routing_error = RuntimeHostError(
                f"Handler type {handler_type!r} is registered but not instantiated",
                context=context,
            )

            # Publish error response for envelope-based error handling
            error_response = self._create_error_response(
                error=str(routing_error),
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)

            # Log with structured error
            logger.warning(
                "Handler registered but not instantiated",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "registered_handlers": list(self._handlers.keys()),
                    "error": str(routing_error),
                },
            )
            return

        # Execute handler
        try:
            # Handler expected to have async execute(envelope) method
            # NOTE: MVP adapters use legacy execute(envelope: dict) signature.
            # TODO(OMN-40): Migrate handlers to new protocol signature execute(request, operation_config)
            response = await handler.execute(envelope)  # type: ignore[call-arg]

            # Ensure response has correlation_id
            # Make a copy to avoid mutating handler's internal state
            if isinstance(response, dict):
                response = dict(response)
                if "correlation_id" not in response:
                    response["correlation_id"] = correlation_id

            await self._publish_envelope_safe(response, self._output_topic)

            logger.debug(
                "Handler executed successfully",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                },
            )

        except Exception as e:
            # Create infrastructure error context for handler execution failure
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="handler_execution",
                target_name=str(handler_type),
                correlation_id=correlation_id,
            )
            # Chain the error with infrastructure context
            infra_error = RuntimeHostError(
                f"Handler execution failed for {handler_type}: {e}",
                context=context,
            )
            infra_error.__cause__ = e  # Proper error chaining

            # Handler execution failed - produce failure envelope
            error_response = self._create_error_response(
                error=str(e),
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)

            logger.exception(
                "Handler execution failed",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "error": str(e),
                    "infra_error": str(infra_error),
                },
            )

    def _create_error_response(
        self,
        error: str,
        correlation_id: UUID | None,
    ) -> JsonDict:
        """Create a standardized error response envelope.

        Args:
            error: Error message to include.
            correlation_id: Correlation ID to preserve for tracking.

        Returns:
            Error response dict with success=False and error details.
        """
        # Use correlation_id or generate a new one, keeping as UUID for internal use
        final_correlation_id = correlation_id or uuid4()
        return {
            "success": False,
            "status": "error",
            "error": error,
            "correlation_id": final_correlation_id,
        }

    def _serialize_envelope(self, envelope: JsonDict | BaseModel) -> JsonDict:
        """Recursively convert UUID objects to strings for JSON serialization.

        Handles both dict envelopes and Pydantic models (e.g., ModelDuplicateResponse).

        Args:
            envelope: Envelope dict or Pydantic model that may contain UUID objects.

        Returns:
            New dict with all UUIDs converted to strings.
        """
        # Convert Pydantic models to dict first, ensuring type safety
        envelope_dict: JsonDict = (
            envelope.model_dump() if isinstance(envelope, BaseModel) else envelope
        )

        def convert_value(value: object) -> object:
            if isinstance(value, UUID):
                return str(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            return value

        return {k: convert_value(v) for k, v in envelope_dict.items()}

    async def _publish_envelope_safe(
        self, envelope: JsonDict | BaseModel, topic: str
    ) -> None:
        """Publish envelope with UUID serialization support.

        Converts any UUID objects to strings before publishing to ensure
        JSON serialization works correctly.

        Args:
            envelope: Envelope dict or Pydantic model (may contain UUID objects).
            topic: Target topic to publish to.
        """
        # Always serialize UUIDs upfront - single code path
        json_safe_envelope = self._serialize_envelope(envelope)
        await self._event_bus.publish_envelope(json_safe_envelope, topic)

    async def health_check(self) -> JsonValue:
        """Return health check status.

        Returns:
            Dictionary with health status information:
                - healthy: Overall health status (True only if running,
                  event bus healthy, no handlers failed to instantiate,
                  and all registered handlers are healthy)
                - degraded: True when process is running but some handlers
                  failed to instantiate. Indicates partial functionality -
                  the system is operational but not at full capacity.
                - is_running: Whether the process is running
                - is_draining: Whether the process is in graceful shutdown drain
                  period, waiting for in-flight messages to complete (OMN-756).
                  Load balancers can use this to remove the service from rotation
                  before the container becomes unhealthy.
                - pending_message_count: Number of messages currently being
                  processed. Useful for monitoring drain progress and determining
                  when the service is ready for shutdown.
                - event_bus: Event bus health status (if running)
                - event_bus_healthy: Boolean indicating event bus health
                - failed_handlers: Dict of handler_type -> error message for
                  handlers that failed to instantiate during start()
                - registered_handlers: List of successfully registered handler types
                - handlers: Dict of handler_type -> health status for each
                  registered handler

        Health State Matrix:
            - healthy=True, degraded=False: Fully operational
            - healthy=False, degraded=True: Running with reduced functionality
            - healthy=False, degraded=False: Not running or event bus unhealthy

        Drain State:
            When is_draining=True, the service is shutting down gracefully:
            - New messages are no longer being accepted
            - In-flight messages are being allowed to complete
            - Health status may still show healthy during drain
            - Load balancers should remove the service from rotation

        Note:
            Handler health checks are performed concurrently using asyncio.gather()
            with individual timeouts (configurable via health_check_timeout_seconds
            config, default: 5.0 seconds) to prevent slow handlers from blocking.
        """
        # Get event bus health if available
        event_bus_health: JsonValue = {}
        event_bus_healthy = False

        try:
            event_bus_health = await self._event_bus.health_check()
            # Assert for type narrowing: health_check() returns dict per contract
            assert isinstance(event_bus_health, dict), (
                f"health_check() must return dict, got {type(event_bus_health).__name__}"
            )
            event_bus_healthy = bool(event_bus_health.get("healthy", False))
        except Exception as e:
            # Create infrastructure error context for health check failure
            correlation_id = uuid4()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="health_check",
                target_name="event_bus",
                correlation_id=correlation_id,
            )
            # Chain the error with infrastructure context
            infra_error = RuntimeHostError(
                f"Event bus health check failed: {e}",
                context=context,
            )
            infra_error.__cause__ = e  # Proper error chaining

            logger.warning(
                "Event bus health check failed",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                    "infra_error": str(infra_error),
                },
                exc_info=True,
            )
            event_bus_health = {"error": str(e), "correlation_id": str(correlation_id)}
            event_bus_healthy = False

        # Check handler health for all registered handlers concurrently
        # Delegates to ProtocolLifecycleExecutor with configured timeout to prevent blocking
        handler_health_results: dict[str, JsonValue] = {}
        handlers_all_healthy = True

        if self._handlers:
            # Run all handler health checks concurrently using asyncio.gather()
            health_check_tasks = [
                self._lifecycle_executor.check_handler_health(handler_type, handler)
                for handler_type, handler in self._handlers.items()
            ]
            results = await asyncio.gather(*health_check_tasks)

            # Process results and build the results dict
            for health_result in results:
                handler_health_results[health_result.handler_type] = (
                    health_result.details
                )
                if not health_result.healthy:
                    handlers_all_healthy = False

        # Check for failed handlers - any failures indicate degraded state
        has_failed_handlers = len(self._failed_handlers) > 0

        # Degraded state: process is running but some handlers failed to instantiate
        # This means the system is operational but with reduced functionality
        degraded = self._is_running and has_failed_handlers

        # Overall health is True only if running, event bus is healthy,
        # no handlers failed to instantiate, and all registered handlers are healthy
        healthy = (
            self._is_running
            and event_bus_healthy
            and not has_failed_handlers
            and handlers_all_healthy
        )

        return {
            "healthy": healthy,
            "degraded": degraded,
            "is_running": self._is_running,
            "is_draining": self._is_draining,
            "pending_message_count": self._pending_message_count,
            "event_bus": event_bus_health,
            "event_bus_healthy": event_bus_healthy,
            "failed_handlers": self._failed_handlers,
            "registered_handlers": list(self._handlers.keys()),
            "handlers": handler_health_results,
        }

    def register_handler(self, handler_type: str, handler: ProtocolHandler) -> None:
        """Register a handler for a specific type.

        Args:
            handler_type: Protocol type identifier (e.g., "http", "db").
            handler: Handler instance implementing the ProtocolHandler protocol.
        """
        self._handlers[handler_type] = handler
        logger.debug(
            "Handler registered",
            extra={
                "handler_type": handler_type,
                "handler_class": type(handler).__name__,
            },
        )

    def get_handler(self, handler_type: str) -> ProtocolHandler | None:
        """Get handler for type, returns None if not registered.

        Args:
            handler_type: Protocol type identifier.

        Returns:
            Handler instance if registered, None otherwise.
        """
        return self._handlers.get(handler_type)

    # =========================================================================
    # Idempotency Guard Methods (OMN-945)
    # =========================================================================

    async def _initialize_idempotency_store(self) -> None:
        """Initialize idempotency store from configuration.

        Reads idempotency configuration from the runtime config and wires
        the appropriate store implementation. If not configured or disabled,
        idempotency checking is skipped.

        Supported store types:
            - "postgres": PostgreSQL-backed durable store (production)
            - "memory": In-memory store (testing only)

        Configuration keys:
            - idempotency.enabled: bool (default: False)
            - idempotency.store_type: "postgres" | "memory" (default: "postgres")
            - idempotency.domain_from_operation: bool (default: True)
            - idempotency.skip_operations: list[str] (default: [])
            - idempotency_database: dict (PostgreSQL connection config)
        """
        # Check if config has idempotency section
        idempotency_raw = self._config.get("idempotency")
        if idempotency_raw is None:
            logger.debug("Idempotency guard not configured, skipping")
            return

        try:
            from omnibase_infra.idempotency import ModelIdempotencyGuardConfig

            if isinstance(idempotency_raw, dict):
                self._idempotency_config = ModelIdempotencyGuardConfig.model_validate(
                    idempotency_raw
                )
            elif isinstance(idempotency_raw, ModelIdempotencyGuardConfig):
                self._idempotency_config = idempotency_raw
            else:
                logger.warning(
                    "Invalid idempotency config type",
                    extra={"type": type(idempotency_raw).__name__},
                )
                return

            if not self._idempotency_config.enabled:
                logger.debug("Idempotency guard disabled in config")
                return

            # Create store based on store_type
            if self._idempotency_config.store_type == "postgres":
                from omnibase_infra.idempotency import (
                    ModelPostgresIdempotencyStoreConfig,
                    PostgresIdempotencyStore,
                )

                # Get database config from container or config
                db_config_raw = self._config.get("idempotency_database", {})
                if isinstance(db_config_raw, dict):
                    db_config = ModelPostgresIdempotencyStoreConfig.model_validate(
                        db_config_raw
                    )
                elif isinstance(db_config_raw, ModelPostgresIdempotencyStoreConfig):
                    db_config = db_config_raw
                else:
                    logger.warning(
                        "Invalid idempotency_database config type",
                        extra={"type": type(db_config_raw).__name__},
                    )
                    return

                self._idempotency_store = PostgresIdempotencyStore(config=db_config)
                await self._idempotency_store.initialize()

            elif self._idempotency_config.store_type == "memory":
                from omnibase_infra.idempotency import InMemoryIdempotencyStore

                self._idempotency_store = InMemoryIdempotencyStore()

            else:
                logger.warning(
                    "Unknown idempotency store type",
                    extra={"store_type": self._idempotency_config.store_type},
                )
                return

            logger.info(
                "Idempotency guard initialized",
                extra={
                    "store_type": self._idempotency_config.store_type,
                    "domain_from_operation": self._idempotency_config.domain_from_operation,
                    "skip_operations": self._idempotency_config.skip_operations,
                },
            )

        except Exception as e:
            logger.warning(
                "Failed to initialize idempotency store, proceeding without",
                extra={"error": str(e)},
            )
            self._idempotency_store = None
            self._idempotency_config = None

    # =========================================================================
    # WARNING: FAIL-OPEN BEHAVIOR
    # =========================================================================
    # This method implements FAIL-OPEN semantics: if the idempotency store
    # is unavailable or errors, messages are ALLOWED THROUGH for processing.
    #
    # This is an intentional design decision prioritizing availability over
    # exactly-once guarantees. See docstring below for full trade-off analysis.
    #
    # IMPORTANT: Downstream handlers MUST be designed for at-least-once delivery
    # and implement their own idempotency for critical operations.
    # =========================================================================
    async def _check_idempotency(
        self,
        envelope: dict[str, object],
        correlation_id: UUID,
    ) -> bool:
        """Check if envelope should be processed (idempotency guard).

        Extracts message_id from envelope headers and checks against the
        idempotency store. If duplicate detected, publishes a duplicate
        response and returns False.

        Fail-Open Semantics:
            This method implements **fail-open** error handling: if the
            idempotency store is unavailable or throws an error, the message
            is allowed through for processing (with a warning log).

            **Design Rationale**: In distributed event-driven systems, the
            idempotency store (e.g., Redis/Valkey) is a supporting service,
            not a critical path dependency. A temporary store outage should
            not halt message processing entirely, as this would cascade into
            broader system unavailability.

            **Trade-offs**:
            - Pro: High availability - processing continues during store outages
            - Pro: Graceful degradation - system remains functional
            - Con: May result in duplicate message processing during outages
            - Con: Downstream handlers must be designed for at-least-once delivery

            **Mitigation**: Handlers consuming messages should implement their
            own idempotency logic for critical operations (e.g., using database
            constraints or transaction guards) to ensure correctness even when
            duplicates slip through.

        Args:
            envelope: Validated envelope dict.
            correlation_id: Normalized correlation ID (UUID).

        Returns:
            True if message should be processed (new message).
            False if message is duplicate (skip processing).
        """
        # Skip check if idempotency not configured
        if self._idempotency_store is None or self._idempotency_config is None:
            return True

        if not self._idempotency_config.enabled:
            return True

        # Check if operation is in skip list
        operation = envelope.get("operation")
        if isinstance(operation, str):
            if not self._idempotency_config.should_check_idempotency(operation):
                logger.debug(
                    "Skipping idempotency check for operation",
                    extra={
                        "operation": operation,
                        "correlation_id": str(correlation_id),
                    },
                )
                return True

        # Extract message_id from envelope
        message_id = self._extract_message_id(envelope, correlation_id)

        # Extract domain from operation if configured
        domain = self._extract_idempotency_domain(envelope)

        # Check and record in store
        try:
            is_new = await self._idempotency_store.check_and_record(
                message_id=message_id,
                domain=domain,
                correlation_id=correlation_id,
            )

            if not is_new:
                # Duplicate detected - publish duplicate response (NOT an error)
                logger.info(
                    "Duplicate message detected, skipping processing",
                    extra={
                        "message_id": str(message_id),
                        "domain": domain,
                        "correlation_id": str(correlation_id),
                    },
                )

                duplicate_response = self._create_duplicate_response(
                    message_id=message_id,
                    correlation_id=correlation_id,
                )
                # duplicate_response is already a dict from _create_duplicate_response
                await self._publish_envelope_safe(
                    duplicate_response, self._output_topic
                )
                return False

            return True

        except Exception as e:
            # FAIL-OPEN: Allow message through on idempotency store errors.
            # Rationale: Availability over exactly-once. Store outages should not
            # halt processing. Downstream handlers must tolerate duplicates.
            # See docstring for full trade-off analysis.
            logger.warning(
                "Idempotency check failed, allowing message through (fail-open)",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "message_id": str(message_id),
                    "domain": domain,
                    "correlation_id": str(correlation_id),
                },
            )
            return True

    def _extract_message_id(
        self,
        envelope: dict[str, object],
        correlation_id: UUID,
    ) -> UUID:
        """Extract message_id from envelope, falling back to correlation_id.

        Priority:
            1. envelope["headers"]["message_id"]
            2. envelope["message_id"]
            3. Use correlation_id as message_id (fallback)

        Args:
            envelope: Envelope dict to extract message_id from.
            correlation_id: Fallback UUID if message_id not found.

        Returns:
            UUID representing the message_id.
        """
        # Try headers first
        headers = envelope.get("headers")
        if isinstance(headers, dict):
            header_msg_id = headers.get("message_id")
            if header_msg_id is not None:
                if isinstance(header_msg_id, UUID):
                    return header_msg_id
                if isinstance(header_msg_id, str):
                    try:
                        return UUID(header_msg_id)
                    except ValueError:
                        pass

        # Try top-level message_id
        top_level_msg_id = envelope.get("message_id")
        if top_level_msg_id is not None:
            if isinstance(top_level_msg_id, UUID):
                return top_level_msg_id
            if isinstance(top_level_msg_id, str):
                try:
                    return UUID(top_level_msg_id)
                except ValueError:
                    pass

        # Fallback: use correlation_id as message_id
        return correlation_id

    def _extract_idempotency_domain(
        self,
        envelope: dict[str, object],
    ) -> str | None:
        """Extract domain for idempotency key from envelope.

        If domain_from_operation is enabled in config, extracts domain
        from the operation prefix (e.g., "db.query" -> "db").

        Args:
            envelope: Envelope dict to extract domain from.

        Returns:
            Domain string if found and configured, None otherwise.
        """
        if self._idempotency_config is None:
            return None

        if not self._idempotency_config.domain_from_operation:
            return None

        operation = envelope.get("operation")
        if isinstance(operation, str):
            return self._idempotency_config.extract_domain(operation)

        return None

    def _create_duplicate_response(
        self,
        message_id: UUID,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Create response for duplicate message detection.

        This is NOT an error response - duplicates are expected under
        at-least-once delivery. The response indicates successful
        deduplication.

        Args:
            message_id: UUID of the duplicate message.
            correlation_id: Correlation ID for tracing.

        Returns:
            Dict representation of ModelDuplicateResponse for envelope publishing.
        """
        return ModelDuplicateResponse(
            message_id=message_id,
            correlation_id=correlation_id,
        ).model_dump()

    async def _cleanup_idempotency_store(self) -> None:
        """Cleanup idempotency store during shutdown.

        Closes the idempotency store connection if initialized.
        Called during stop() to release resources.
        """
        if self._idempotency_store is None:
            return

        try:
            if hasattr(self._idempotency_store, "shutdown"):
                await self._idempotency_store.shutdown()
            elif hasattr(self._idempotency_store, "close"):
                await self._idempotency_store.close()
            logger.debug("Idempotency store shutdown complete")
        except Exception as e:
            logger.warning(
                "Failed to shutdown idempotency store",
                extra={"error": str(e)},
            )
        finally:
            self._idempotency_store = None


__all__: list[str] = [
    "RuntimeHostProcess",
    "wire_handlers",
]
