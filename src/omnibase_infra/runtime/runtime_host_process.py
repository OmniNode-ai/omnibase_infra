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
    processes envelopes for a specific protocol type (e.g., "http", "database").
    The handler_type field in envelopes determines routing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.inmemory_event_bus import InMemoryEventBus
from omnibase_infra.runtime.handler_registry import get_handler_registry
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
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus | None = None,
        input_topic: str = DEFAULT_INPUT_TOPIC,
        output_topic: str = DEFAULT_OUTPUT_TOPIC,
        config: dict[str, object] | None = None,
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
        """
        # Create or use provided event bus
        self._event_bus: InMemoryEventBus = event_bus or InMemoryEventBus()

        # Extract configuration with defaults
        config = config or {}

        # Topic configuration (config overrides constructor args)
        self._input_topic: str = str(config.get("input_topic", input_topic))
        self._output_topic: str = str(config.get("output_topic", output_topic))
        self._group_id: str = str(config.get("group_id", DEFAULT_GROUP_ID))

        # Store full config for handler initialization
        self._config: dict[str, object] = config

        # Runtime state
        self._is_running: bool = False

        # Subscription handle (callable to unsubscribe)
        self._subscription: Callable[[], Awaitable[None]] | None = None

        # Handler registry (handler_type -> handler instance)
        # This will be populated from the singleton registry during start()
        self._handlers: dict[str, ProtocolHandler] = {}

        logger.debug(
            "RuntimeHostProcess initialized",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
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
        2. Close event bus

        This method is idempotent - calling stop() on an already stopped
        process is safe and has no effect.
        """
        if not self._is_running:
            logger.debug("RuntimeHostProcess already stopped, skipping")
            return

        logger.info("Stopping RuntimeHostProcess")

        # Step 1: Unsubscribe from topics
        if self._subscription is not None:
            await self._subscription()
            self._subscription = None

        # Step 2: Close event bus
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
                    "Failed to instantiate handler",
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

        Extracts the handler_type from the envelope and routes it to the
        appropriate registered handler. Publishes the response to the
        output topic.

        Args:
            envelope: Dict with 'operation', 'payload', optional 'correlation_id',
                and 'handler_type'.
        """
        # Extract correlation_id for tracking (preserve as UUID if possible)
        raw_correlation_id = envelope.get("correlation_id")
        correlation_id: UUID | None = None
        if isinstance(raw_correlation_id, UUID):
            correlation_id = raw_correlation_id
        elif raw_correlation_id is not None:
            try:
                correlation_id = UUID(str(raw_correlation_id))
            except (ValueError, TypeError):
                correlation_id = uuid4()
        else:
            correlation_id = uuid4()

        # Validate envelope has required fields
        operation = envelope.get("operation")
        handler_type = envelope.get("handler_type")

        if operation is None and handler_type is None:
            # Invalid envelope - missing required fields
            error_response = self._create_error_response(
                error="Invalid envelope: missing 'operation' and 'handler_type' fields",
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
            return

        # Determine handler_type from envelope
        # If handler_type not explicit, extract from operation (e.g., "http.get" -> "http")
        if handler_type is None and operation is not None:
            handler_type = str(operation).split(".")[0]

        # Get handler from registry
        handler = self._handlers.get(str(handler_type))

        if handler is None:
            # Unknown handler type
            error_response = self._create_error_response(
                error=f"Unknown handler type: {handler_type!r} not registered",
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)

            logger.warning(
                "No handler registered for type",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                },
            )
            return

        # Execute handler
        try:
            # Handler expected to have async execute(envelope) method
            response = await handler.execute(envelope)

            # Ensure response has correlation_id
            if isinstance(response, dict) and "correlation_id" not in response:
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

    async def _publish_envelope_safe(
        self, envelope: dict[str, object], topic: str
    ) -> None:
        """Publish envelope with UUID serialization support.

        This method wraps the event bus publish_envelope to handle UUID
        serialization safely. It preserves the original envelope structure
        (including UUID objects) for mocked tests, but ensures JSON
        serialization works for the actual event bus implementation.

        For testing: When publish_envelope is mocked, the mock receives
        the original envelope dict with UUIDs intact.

        For production: When publish_envelope is not mocked, this catches
        any JSON serialization errors and retries with UUIDs converted
        to strings.

        Args:
            envelope: Envelope dict (may contain UUID objects).
            topic: Target topic to publish to.
        """
        try:
            # First try publishing with UUIDs intact
            # If mocked, this succeeds and tests see UUIDs
            # If not mocked, this may raise TypeError for UUIDs
            await self._event_bus.publish_envelope(envelope, topic)
        except TypeError:
            # TypeError during JSON serialization - attempt UUID conversion
            # This handles UUIDs and other non-serializable types gracefully
            def uuid_serializer(obj: object) -> object:
                if isinstance(obj, UUID):
                    return str(obj)
                raise TypeError(
                    f"Object of type {type(obj).__name__} is not JSON serializable"
                )

            # Serialize the envelope to JSON-safe dict
            json_str = json.dumps(envelope, default=uuid_serializer)
            json_safe_envelope = json.loads(json_str)

            # Retry with JSON-safe envelope
            await self._event_bus.publish_envelope(json_safe_envelope, topic)

    async def health_check(self) -> dict[str, object]:
        """Return health check status.

        Returns:
            Dictionary with health status information:
                - healthy: Overall health status
                - is_running: Whether the process is running
                - event_bus: Event bus health status (if running)
                - event_bus_healthy: Boolean indicating event bus health
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

        # Overall health is True only if running and event bus is healthy
        healthy = self._is_running and event_bus_healthy

        return {
            "healthy": healthy,
            "is_running": self._is_running,
            "event_bus": event_bus_health,
            "event_bus_healthy": event_bus_healthy,
        }

    def register_handler(self, handler_type: str, handler: ProtocolHandler) -> None:
        """Register a handler for a specific type.

        Args:
            handler_type: Protocol type identifier (e.g., "http", "database").
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


__all__: list[str] = [
    "RuntimeHostProcess",
    "wire_handlers",
]
