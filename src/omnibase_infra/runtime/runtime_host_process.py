# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Host Process implementation for ONEX Infrastructure.

This module implements the RuntimeHostProcess class, which is responsible for:
- Owning and managing an InMemoryEventBus instance
- Registering handlers via the wiring module
- Subscribing to event bus topics and routing envelopes to handlers
- Handling errors by producing success=False response envelopes
- Processing envelopes sequentially (no parallelism in MVP)
- Basic shutdown (no graceful drain in MVP)

The RuntimeHostProcess is the central coordinator for infrastructure runtime,
bridging event-driven message routing with protocol handlers.

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
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    EnvelopeValidationError,
    ModelInfraErrorContext,
    RuntimeHostError,
    UnknownHandlerTypeError,
)
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.envelope_validator import validate_envelope
from omnibase_infra.runtime.handler_registry import get_handler_registry
from omnibase_infra.runtime.protocol_lifecycle_executor import ProtocolLifecycleExecutor
from omnibase_infra.runtime.wiring import wire_default_handlers

if TYPE_CHECKING:
    from omnibase_spi.protocols.handlers.protocol_handler import ProtocolHandler

    from omnibase_infra.event_bus.models import ModelEventMessage

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
DEFAULT_HEALTH_CHECK_TIMEOUT = 5.0


class RuntimeHostProcess:
    """Runtime host process that owns event bus and coordinates handlers.

    The RuntimeHostProcess is the central coordinator for ONEX infrastructure
    runtime. It owns an InMemoryEventBus instance, registers handlers via the
    wiring module, and routes incoming envelopes to appropriate handlers.

    Attributes:
        event_bus: The owned InMemoryEventBus instance
        is_running: Whether the process is currently running
        input_topic: Topic to subscribe to for incoming envelopes
        output_topic: Topic to publish responses to
        group_id: Consumer group identifier

    Example:
        ```python
        process = RuntimeHostProcess()
        await process.start()
        health = await process.health_check()
        await process.stop()
        ```

    Note:
        MVP Limitation: The stop() method performs immediate shutdown without
        draining in-flight messages. See stop() docstring for details.
    """

    def __init__(
        self,
        event_bus: Optional[InMemoryEventBus] = None,
        input_topic: str = DEFAULT_INPUT_TOPIC,
        output_topic: str = DEFAULT_OUTPUT_TOPIC,
        config: Optional[dict[str, object]] = None,
    ) -> None:
        """Initialize the runtime host process.

        Args:
            event_bus: Optional event bus instance. If None, creates InMemoryEventBus.
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
        """
        # Create or use provided event bus
        self._event_bus: InMemoryEventBus = event_bus or InMemoryEventBus()

        # Extract configuration with defaults
        config = config or {}

        # Topic configuration (config overrides constructor args)
        self._input_topic: str = str(config.get("input_topic", input_topic))
        self._output_topic: str = str(config.get("output_topic", output_topic))
        self._group_id: str = str(config.get("group_id", DEFAULT_GROUP_ID))

        # Health check configuration (from lifecycle subcontract pattern)
        # Default: 5.0 seconds, valid range: 1-60 seconds per ModelLifecycleSubcontract
        # Values outside bounds are clamped with a warning
        _timeout_raw = config.get("health_check_timeout_seconds")
        timeout_value: float = DEFAULT_HEALTH_CHECK_TIMEOUT
        if isinstance(_timeout_raw, (int, float)):
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

        # Handler executor for lifecycle operations (shutdown, health check)
        self._lifecycle_executor = ProtocolLifecycleExecutor(
            health_check_timeout_seconds=self._health_check_timeout_seconds
        )

        # Store full config for handler initialization
        self._config: dict[str, object] = config

        # Runtime state
        self._is_running: bool = False

        # Subscription handle (callable to unsubscribe)
        self._subscription: Optional[Callable[[], Awaitable[None]]] = None

        # Handler registry (handler_type -> handler instance)
        # This will be populated from the singleton registry during start()
        self._handlers: dict[str, ProtocolHandler] = {}

        # Track failed handler instantiations (handler_type -> error message)
        # Used by health_check() to report degraded state
        self._failed_handlers: dict[str, str] = {}

        logger.debug(
            "RuntimeHostProcess initialized",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
                "health_check_timeout_seconds": self._health_check_timeout_seconds,
            },
        )

    @property
    def event_bus(self) -> InMemoryEventBus:
        """Return the owned event bus instance.

        Returns:
            The InMemoryEventBus instance managed by this process.
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
        """Stop the runtime host.

        Performs the following steps:
        1. Unsubscribe from topics
        2. Shutdown all registered handlers by priority (release resources)
        3. Close event bus

        This method is idempotent - calling stop() on an already stopped
        process is safe and has no effect.

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

        Note:
            MVP Limitation: This implementation immediately unsubscribes without
            draining in-flight envelopes. Any messages currently being processed
            may be lost. For production use cases requiring graceful shutdown,
            a drain period should be implemented.

        TODO(OMN-XXX): Implement graceful shutdown with configurable drain period:
            - Add drain_timeout_seconds parameter (default: 30)
            - Wait for in-flight messages to complete before unsubscribing
            - Track pending message count for shutdown readiness
            - Add shutdown_ready() method to check drain status
        """
        if not self._is_running:
            logger.debug("RuntimeHostProcess already stopped, skipping")
            return

        logger.info("Stopping RuntimeHostProcess")

        # Step 1: Unsubscribe from topics
        if self._subscription is not None:
            await self._subscription()
            self._subscription = None

        # Step 2: Shutdown all handlers by priority (release resources like DB/Kafka connections)
        # Delegates to ProtocolLifecycleExecutor which handles:
        # - Grouping handlers by priority (higher priority first)
        # - Parallel shutdown within priority groups for performance
        if self._handlers:
            (
                all_succeeded,
                all_failed,
            ) = await self._lifecycle_executor.shutdown_handlers_by_priority(
                self._handlers
            )

            # Log summary (ProtocolLifecycleExecutor already logs detailed info)
            logger.info(
                "Handler shutdown completed",
                extra={
                    "succeeded_handlers": all_succeeded,
                    "failed_handlers": [f[0] for f in all_failed],
                    "total_handlers": len(all_succeeded) + len(all_failed),
                    "success_count": len(all_succeeded),
                    "failure_count": len(all_failed),
                },
            )

        # Step 3: Close event bus
        await self._event_bus.close()

        self._is_running = False

        logger.info("RuntimeHostProcess stopped successfully")

    async def _populate_handlers_from_registry(self) -> None:
        """Populate self._handlers from the singleton handler registry.

        This method bridges the gap between the wiring module (which registers
        handler CLASSES to the singleton registry) and the RuntimeHostProcess
        (which needs handler INSTANCES in self._handlers for routing).

        For each registered handler type in the singleton registry:
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
        handler_registry = get_handler_registry()
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

    async def _on_message(self, message: ModelEventMessage) -> None:
        """Handle incoming message from event bus subscription.

        This is the callback invoked by the event bus when a message arrives
        on the input topic. It deserializes the envelope and routes it.

        Args:
            message: The event message containing the envelope payload.
        """
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

    async def _handle_envelope(self, envelope: dict[str, object]) -> None:
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
        raw_correlation_id = envelope.get("correlation_id")
        pre_validation_correlation_id: Optional[UUID] = None
        if isinstance(raw_correlation_id, UUID):
            pre_validation_correlation_id = raw_correlation_id
        elif raw_correlation_id is not None:
            try:
                pre_validation_correlation_id = UUID(str(raw_correlation_id))
            except (ValueError, TypeError):
                pre_validation_correlation_id = uuid4()
        else:
            pre_validation_correlation_id = uuid4()

        # Step 1: Validate envelope BEFORE dispatch
        # This validates operation, prefix, payload requirements, and normalizes correlation_id
        try:
            validate_envelope(envelope, get_handler_registry())
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
            # Will migrate to execute(request, operation_config) in future.
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
        correlation_id: Optional[UUID],
    ) -> dict[str, object]:
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

    def _serialize_envelope(self, envelope: dict[str, object]) -> dict[str, object]:
        """Recursively convert UUID objects to strings for JSON serialization.

        Args:
            envelope: Envelope dict that may contain UUID objects.

        Returns:
            New dict with all UUIDs converted to strings.
        """

        def convert_value(value: object) -> object:
            if isinstance(value, UUID):
                return str(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            return value

        return {k: convert_value(v) for k, v in envelope.items()}

    async def _publish_envelope_safe(
        self, envelope: dict[str, object], topic: str
    ) -> None:
        """Publish envelope with UUID serialization support.

        Converts any UUID objects to strings before publishing to ensure
        JSON serialization works correctly.

        Args:
            envelope: Envelope dict (may contain UUID objects).
            topic: Target topic to publish to.
        """
        # Always serialize UUIDs upfront - single code path
        json_safe_envelope = self._serialize_envelope(envelope)
        await self._event_bus.publish_envelope(json_safe_envelope, topic)

    async def health_check(self) -> dict[str, object]:
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

        Note:
            Handler health checks are performed concurrently using asyncio.gather()
            with individual timeouts (configurable via health_check_timeout_seconds
            config, default: 5.0 seconds) to prevent slow handlers from blocking.
        """
        # Get event bus health if available
        event_bus_health: dict[str, object] = {}
        event_bus_healthy = False

        try:
            event_bus_health = await self._event_bus.health_check()
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
        handler_health_results: dict[str, dict[str, object]] = {}
        handlers_all_healthy = True

        if self._handlers:
            # Run all handler health checks concurrently using asyncio.gather()
            health_check_tasks = [
                self._lifecycle_executor.check_handler_health(handler_type, handler)
                for handler_type, handler in self._handlers.items()
            ]
            results = await asyncio.gather(*health_check_tasks)

            # Process results and build the results dict
            for handler_type, health_result in results:
                handler_health_results[handler_type] = health_result
                if not health_result.get("healthy", False):
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

    def get_handler(self, handler_type: str) -> Optional[ProtocolHandler]:
        """Get handler for type, returns None if not registered.

        Args:
            handler_type: Protocol type identifier.

        Returns:
            Handler instance if registered, None otherwise.
        """
        return self._handlers.get(handler_type)


__all__: list[str] = [
    "RuntimeHostProcess",
    "wire_handlers",
]
