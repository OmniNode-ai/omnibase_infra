# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Introspection event router for kernel event processing.

This module provides an extracted event router for routing introspection
events in the ONEX kernel. Extracted from kernel.py for better testability
and separation of concerns.

The router:
    - Parses incoming event messages as ModelEventEnvelope
    - Validates payload as ModelNodeIntrospectionEvent
    - Routes to the introspection dispatcher
    - Publishes output events to the configured output topic

Design:
    This class encapsulates the message routing logic that was previously
    a nested callback in kernel.py. By extracting it, we enable:
    - Unit testing without full kernel bootstrap
    - Mocking of dependencies for isolation
    - Clearer separation between bootstrap and event routing

    The router uses ProtocolEventBusLike for event publishing, enabling
    duck typing with any event bus implementation (Kafka, InMemory, etc.).

Related:
    - OMN-888: Registration Orchestrator
    - OMN-892: 2-way Registration E2E Integration Test
"""

from __future__ import annotations

__all__ = ["IntrospectionEventRouter"]

import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.nodes.node_registration_orchestrator.dispatchers import (
        DispatcherNodeIntrospected,
    )
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)


def _normalize_metadata(metadata: dict[str, object] | None) -> dict[str, object] | None:
    """Normalize metadata values, handling bytes to str conversion.

    Kafka headers and metadata may arrive as bytes. This function defensively
    converts bytes values to strings using UTF-8 decoding to ensure consistent
    string-based metadata handling downstream.

    Uses duck-typing (hasattr check for decode method) instead of isinstance
    to align with protocol-based design principles.

    Args:
        metadata: Optional metadata dictionary with potentially mixed value types.

    Returns:
        Normalized metadata dict with bytes converted to str, or None if input is None.
    """
    if metadata is None:
        return None

    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        # Duck-type: check for decode method (bytes-like) instead of isinstance
        if hasattr(value, "decode"):
            try:
                normalized[key] = value.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                # If decoding fails, use repr as fallback
                normalized[key] = repr(value)
        else:
            normalized[key] = value
    return normalized


class IntrospectionEventRouter:
    """Router for introspection event messages from event bus.

    This router handles incoming event messages, parses them as
    ModelNodeIntrospectionEvent payloads wrapped in ModelEventEnvelope,
    and routes them to the introspection dispatcher for registration
    orchestration.

    The router propagates correlation IDs from incoming messages for
    distributed tracing. If no correlation ID is present, it generates
    a new one to ensure all operations can be traced.

    This class follows the container-based dependency injection pattern,
    receiving a ModelONEXContainer for service resolution while also
    accepting explicit dependencies for router-specific configuration.

    Attributes:
        _container: ONEX service container for dependency resolution.
        _dispatcher: The DispatcherNodeIntrospected to route events to.
        _event_bus: Event bus implementing ProtocolEventBusLike for publishing.
        _output_topic: The topic to publish output events to.

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> router = IntrospectionEventRouter(
        ...     container=container,
        ...     dispatcher=introspection_dispatcher,
        ...     event_bus=event_bus,
        ...     output_topic="registration.output",
        ... )
        >>> # Use as callback for event bus subscription
        >>> await event_bus.subscribe(
        ...     topic="registration.input",
        ...     group_id="my-group",
        ...     on_message=router.handle_message,
        ... )

    See Also:
        - docs/patterns/container_dependency_injection.md for detailed DI patterns.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        dispatcher: DispatcherNodeIntrospected,
        event_bus: ProtocolEventBusLike,
        output_topic: str,
    ) -> None:
        """Initialize IntrospectionEventRouter with container-based dependency injection.

        Follows the ONEX container-based DI pattern where the container is passed
        as the first parameter for service resolution, with additional explicit
        parameters for router-specific configuration.

        Args:
            container: ONEX service container for dependency resolution. Provides
                access to service_registry for resolving shared services.
            dispatcher: The DispatcherNodeIntrospected to route events to.
            event_bus: Event bus implementing ProtocolEventBusLike for publishing.
            output_topic: The topic to publish output events to.

        Raises:
            ValueError: If output_topic is empty.

        Example:
            >>> from omnibase_core.container import ModelONEXContainer
            >>> container = ModelONEXContainer()
            >>> router = IntrospectionEventRouter(
            ...     container=container,
            ...     dispatcher=introspection_dispatcher,
            ...     event_bus=event_bus,
            ...     output_topic="registration.output",
            ... )

        See Also:
            - docs/patterns/container_dependency_injection.md for DI patterns.
        """
        if not output_topic:
            raise ValueError("output_topic cannot be empty")

        self._container = container
        self._dispatcher = dispatcher
        self._event_bus = event_bus
        self._output_topic = output_topic

        logger.debug(
            "IntrospectionEventRouter initialized",
            extra={
                "output_topic": output_topic,
                "dispatcher_type": type(self._dispatcher).__name__,
                "event_bus_type": type(self._event_bus).__name__,
            },
        )

    @property
    def container(self) -> ModelONEXContainer:
        """Return the ONEX service container.

        The ModelONEXContainer provides protocol-based service resolution:

        - get_service_async(protocol_type, service_name=None, correlation_id=None):
          Async service resolution with caching and logging.
        - get_service_sync(protocol_type, service_name=None):
          Sync service resolution with optional performance monitoring.
        - get_service(protocol_type, service_name=None):
          Compatibility alias for get_service_sync().
        - get_service_optional(protocol_type, service_name=None):
          Returns None if service not found (non-throwing).
        - service_registry property:
          Direct access to ServiceRegistry for service registration.

        Returns:
            The ModelONEXContainer instance passed during initialization.

        Example:
            >>> # Resolve a service from the container by protocol (async)
            >>> from omnibase_infra.runtime.protocol_policy import ProtocolPolicy
            >>> policy = await router.container.get_service_async(ProtocolPolicy)
        """
        return self._container

    @property
    def output_topic(self) -> str:
        """Return the configured output topic for event publishing."""
        return self._output_topic

    @property
    def dispatcher(self) -> DispatcherNodeIntrospected:
        """Return the dispatcher instance."""
        return self._dispatcher

    @property
    def event_bus(self) -> ProtocolEventBusLike:
        """Return the event bus instance."""
        return self._event_bus

    def _extract_correlation_id_from_message(self, msg: ModelEventMessage) -> UUID:
        """Extract correlation ID from message headers or generate new one.

        Attempts to extract the correlation_id from message headers to ensure
        proper propagation for distributed tracing. Falls back to generating
        a new UUID if no correlation ID is found.

        Uses duck-typing patterns for type detection instead of isinstance checks
        to align with protocol-based design principles.

        Args:
            msg: The incoming event message.

        Returns:
            UUID: The extracted or generated correlation ID.
        """
        # Try to extract from message headers if available
        # ModelEventHeaders is a Pydantic model with typed attributes, not a dict/list
        if hasattr(msg, "headers") and msg.headers is not None:
            headers = msg.headers
            if (
                hasattr(headers, "correlation_id")
                and headers.correlation_id is not None
            ):
                # Duck-type: normalize correlation_id to UUID
                # Works uniformly for UUID objects, strings, and bytes
                try:
                    correlation_id = headers.correlation_id
                    # Check for bytes-like (has decode method) - duck typing
                    if hasattr(correlation_id, "decode"):
                        correlation_id = correlation_id.decode("utf-8")
                    # Convert to UUID (handles UUID objects via str() and strings directly)
                    return UUID(str(correlation_id))
                except (ValueError, TypeError, UnicodeDecodeError, AttributeError):
                    pass  # Fall through to try payload extraction

        # If we can peek at the payload, try to extract correlation_id
        # This happens when we can parse the message but before full validation
        try:
            if msg.value is not None:
                # Duck-type: check for decode method (bytes-like) first
                if hasattr(msg.value, "decode"):
                    payload_dict = json.loads(msg.value.decode("utf-8"))
                else:
                    # Try JSON parsing (works for strings)
                    # Falls back to treating as dict-like if TypeError
                    try:
                        payload_dict = json.loads(msg.value)
                    except TypeError:
                        # Already a dict or dict-like object
                        payload_dict = msg.value

                if payload_dict:
                    # Check envelope-level correlation_id first
                    if "correlation_id" in payload_dict:
                        return UUID(str(payload_dict["correlation_id"]))
                    # Check payload-level correlation_id (duck-type dict check via 'in')
                    payload_content = payload_dict.get("payload")
                    if payload_content and hasattr(payload_content, "get"):
                        nested_corr_id = payload_content.get("correlation_id")
                        if nested_corr_id is not None:
                            return UUID(str(nested_corr_id))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError):
            pass  # Fall through to generate new ID

        # Generate new correlation ID as last resort
        return uuid4()

    async def handle_message(self, msg: ModelEventMessage) -> None:
        """Handle incoming introspection event message.

        This callback is invoked for each message received on the input topic.
        It parses the raw JSON payload as ModelNodeIntrospectionEvent and routes
        it to the introspection dispatcher.

        The method propagates the correlation_id from the incoming message
        for distributed tracing. If no correlation_id is present in the message,
        a new one is generated.

        Args:
            msg: The event message containing raw bytes in .value field.
        """
        # Extract correlation_id from message for proper propagation
        # This ensures distributed tracing continuity across service boundaries
        callback_correlation_id = self._extract_correlation_id_from_message(msg)
        callback_start_time = time.time()

        logger.debug(
            "Introspection message callback invoked (correlation_id=%s)",
            callback_correlation_id,
            extra={
                "message_offset": getattr(msg, "offset", None),
                "message_partition": getattr(msg, "partition", None),
                "message_topic": getattr(msg, "topic", None),
            },
        )

        try:
            # ModelEventMessage has .value as bytes
            if msg.value is None:
                logger.debug(
                    "Message value is None, skipping (correlation_id=%s)",
                    callback_correlation_id,
                )
                return

            # Parse message value using duck-typing patterns
            # Check for bytes-like (has decode method) first
            if hasattr(msg.value, "decode"):
                logger.debug(
                    "Parsing message value as bytes-like (correlation_id=%s)",
                    callback_correlation_id,
                    extra={"value_length": len(msg.value)},
                )
                payload_dict = json.loads(msg.value.decode("utf-8"))
            else:
                # Try JSON parsing (works for strings)
                try:
                    logger.debug(
                        "Parsing message value as string-like (correlation_id=%s)",
                        callback_correlation_id,
                        extra={
                            "value_length": len(msg.value)
                            if hasattr(msg.value, "__len__")
                            else None
                        },
                    )
                    payload_dict = json.loads(msg.value)
                except TypeError:
                    # Already a dict-like object (has keys/items)
                    if hasattr(msg.value, "keys"):
                        logger.debug(
                            "Message value already dict-like (correlation_id=%s)",
                            callback_correlation_id,
                        )
                        payload_dict = msg.value
                    else:
                        logger.debug(
                            "Unexpected message value type: %s (correlation_id=%s)",
                            type(msg.value).__name__,
                            callback_correlation_id,
                        )
                        return

            # Parse as ModelEventEnvelope containing ModelNodeIntrospectionEvent
            logger.debug(
                "Validating payload as ModelEventEnvelope (correlation_id=%s)",
                callback_correlation_id,
            )

            raw_envelope = ModelEventEnvelope[dict].model_validate(payload_dict)

            # Validate payload as ModelNodeIntrospectionEvent
            introspection_event = ModelNodeIntrospectionEvent.model_validate(
                raw_envelope.payload
            )
            # Create typed envelope with validated payload
            # Note: Defensively normalize metadata to handle bytes values from Kafka
            event_envelope = ModelEventEnvelope[ModelNodeIntrospectionEvent](
                payload=introspection_event,
                envelope_id=raw_envelope.envelope_id,
                envelope_timestamp=raw_envelope.envelope_timestamp,
                correlation_id=raw_envelope.correlation_id or callback_correlation_id,
                source_tool=raw_envelope.source_tool,
                target_tool=raw_envelope.target_tool,
                metadata=_normalize_metadata(raw_envelope.metadata),  # type: ignore[arg-type]
                priority=raw_envelope.priority,
                timeout_seconds=raw_envelope.timeout_seconds,
                trace_id=raw_envelope.trace_id,
                span_id=raw_envelope.span_id,
            )
            logger.info(
                "Envelope parsed successfully (correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "envelope_id": str(event_envelope.envelope_id),
                    "node_id": str(introspection_event.node_id),
                    "node_type": introspection_event.node_type,
                    "event_version": introspection_event.node_version,
                },
            )

            # Route to dispatcher
            logger.info(
                "Routing to introspection dispatcher (correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "envelope_correlation_id": str(event_envelope.correlation_id),
                    "node_id": introspection_event.node_id,
                },
            )
            dispatcher_start_time = time.time()
            result = await self._dispatcher.handle(event_envelope)  # type: ignore[arg-type]
            dispatcher_duration = time.time() - dispatcher_start_time

            if result.is_successful():
                logger.info(
                    "Introspection event processed successfully: node_id=%s in %.3fs "
                    "(correlation_id=%s)",
                    introspection_event.node_id,
                    dispatcher_duration,
                    callback_correlation_id,
                    extra={
                        "envelope_correlation_id": str(event_envelope.correlation_id),
                        "dispatcher_duration_seconds": dispatcher_duration,
                        "node_id": introspection_event.node_id,
                        "node_type": introspection_event.node_type,
                    },
                )

                # Publish output events to output_topic
                if result.output_events:
                    for output_event in result.output_events:
                        # Wrap output event in envelope
                        output_envelope = ModelEventEnvelope(  # type: ignore[var-annotated]
                            payload=output_event,
                            correlation_id=event_envelope.correlation_id,
                            envelope_timestamp=datetime.now(UTC),
                        )

                        # Publish to output topic
                        await self._event_bus.publish_envelope(
                            envelope=output_envelope,
                            topic=self._output_topic,
                        )

                        logger.info(
                            "Published output event to %s (correlation_id=%s)",
                            self._output_topic,
                            callback_correlation_id,
                            extra={
                                "output_event_type": type(output_event).__name__,
                                "envelope_id": str(output_envelope.envelope_id),
                                "node_id": str(introspection_event.node_id),
                            },
                        )

                    logger.debug(
                        "Published %d output events to %s (correlation_id=%s)",
                        len(result.output_events),
                        self._output_topic,
                        callback_correlation_id,
                    )
                else:
                    # No output events after successful processing - this may indicate
                    # the handler didn't produce expected completion events
                    logger.warning(
                        "Introspection event processed but no output events produced. "
                        "Handler may not have emitted completion events "
                        "(correlation_id=%s)",
                        callback_correlation_id,
                        extra={
                            "node_id": str(introspection_event.node_id),
                            "node_type": introspection_event.node_type,
                            "dispatcher_duration_seconds": dispatcher_duration,
                        },
                    )
            else:
                logger.warning(
                    "Introspection event processing failed: %s (correlation_id=%s)",
                    result.error_message,
                    callback_correlation_id,
                    extra={
                        "envelope_correlation_id": str(event_envelope.correlation_id),
                        "error_message": result.error_message,
                        "node_id": introspection_event.node_id,
                        "dispatcher_duration_seconds": dispatcher_duration,
                    },
                )

        except ValidationError as validation_error:
            # Not an introspection event - skip silently
            # (other message types on the topic are handled by RuntimeHostProcess)
            logger.debug(
                "Message is not a valid introspection event, skipping "
                "(correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "validation_error_count": validation_error.error_count(),
                },
            )

        except json.JSONDecodeError as json_error:
            logger.warning(
                "Failed to decode JSON from message: %s (correlation_id=%s)",
                sanitize_error_message(json_error),
                callback_correlation_id,
                extra={
                    "error_type": type(json_error).__name__,
                    "error_position": getattr(json_error, "pos", None),
                },
            )

        except Exception as msg_error:
            # Use warning instead of exception to avoid credential exposure
            # in tracebacks (connection errors may contain DSN with password)
            logger.warning(
                "Failed to process introspection message: %s (correlation_id=%s)",
                sanitize_error_message(msg_error),
                callback_correlation_id,
                extra={
                    "error_type": type(msg_error).__name__,
                },
            )

        finally:
            callback_duration = time.time() - callback_start_time
            logger.debug(
                "Introspection message callback completed in %.3fs (correlation_id=%s)",
                callback_duration,
                callback_correlation_id,
                extra={
                    "callback_duration_seconds": callback_duration,
                },
            )
