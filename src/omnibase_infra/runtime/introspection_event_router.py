# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Introspection event router for kernel event processing.

This module provides an extracted event router for routing introspection
events in the ONEX kernel. Extracted from kernel.py for better testability
and separation of concerns.

The router:
    - Parses incoming Kafka messages as ModelEventEnvelope or raw events
    - Validates payload as ModelNodeIntrospectionEvent
    - Routes to the introspection dispatcher
    - Publishes output events to the configured output topic

Design:
    This class encapsulates the message routing logic that was previously
    a nested callback in kernel.py. By extracting it, we enable:
    - Unit testing without full kernel bootstrap
    - Mocking of dependencies for isolation
    - Clearer separation between bootstrap and event routing

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

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from pydantic import ValidationError

from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)

if TYPE_CHECKING:
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus
    from omnibase_infra.runtime.dispatchers import DispatcherNodeIntrospected

logger = logging.getLogger(__name__)


class IntrospectionEventRouter:
    """Router for introspection event messages from Kafka.

    This router handles incoming Kafka messages, parses them as
    ModelNodeIntrospectionEvent payloads (wrapped in envelopes or raw),
    and routes them to the introspection dispatcher for registration
    orchestration.

    The router propagates correlation IDs from incoming messages for
    distributed tracing. If no correlation ID is present, it generates
    a new one to ensure all operations can be traced.

    Attributes:
        _dispatcher: The DispatcherNodeIntrospected to route events to.
        _event_bus: The KafkaEventBus for publishing output events.
        _output_topic: The topic to publish output events to.

    Example:
        >>> router = IntrospectionEventRouter(
        ...     dispatcher=introspection_dispatcher,
        ...     event_bus=kafka_bus,
        ...     output_topic="registration.output",
        ... )
        >>> # Use as callback for event bus subscription
        >>> await event_bus.subscribe(
        ...     topic="registration.input",
        ...     group_id="my-group",
        ...     on_message=router.handle_message,
        ... )
    """

    def __init__(
        self,
        dispatcher: DispatcherNodeIntrospected,
        event_bus: KafkaEventBus,
        output_topic: str,
    ) -> None:
        """Initialize the event router.

        Args:
            dispatcher: The DispatcherNodeIntrospected to route events to.
            event_bus: The KafkaEventBus for publishing output events.
            output_topic: The topic to publish output events to.
        """
        self._dispatcher = dispatcher
        self._event_bus = event_bus
        self._output_topic = output_topic

    def _extract_correlation_id_from_message(self, msg: ModelEventMessage) -> UUID:
        """Extract correlation ID from message headers or generate new one.

        Attempts to extract the correlation_id from message headers to ensure
        proper propagation for distributed tracing. Falls back to generating
        a new UUID if no correlation ID is found.

        Args:
            msg: The incoming event message.

        Returns:
            UUID: The extracted or generated correlation ID.
        """
        # Try to extract from message headers if available
        if hasattr(msg, "headers") and msg.headers:
            for key, value in msg.headers:
                if key == "correlation_id":
                    try:
                        if isinstance(value, bytes):
                            return UUID(value.decode("utf-8"))
                        elif isinstance(value, str):
                            return UUID(value)
                    except (ValueError, TypeError):
                        pass  # Fall through to generate new ID

        # If we can peek at the payload, try to extract correlation_id
        # This happens when we can parse the message but before full validation
        try:
            if msg.value is not None:
                if isinstance(msg.value, bytes):
                    payload_dict = json.loads(msg.value.decode("utf-8"))
                elif isinstance(msg.value, str):
                    payload_dict = json.loads(msg.value)
                elif isinstance(msg.value, dict):
                    payload_dict = msg.value
                else:
                    payload_dict = None

                if payload_dict:
                    # Check envelope-level correlation_id first
                    if "correlation_id" in payload_dict:
                        return UUID(str(payload_dict["correlation_id"]))
                    # Check payload-level correlation_id
                    if "payload" in payload_dict and isinstance(
                        payload_dict["payload"], dict
                    ):
                        if "correlation_id" in payload_dict["payload"]:
                            return UUID(str(payload_dict["payload"]["correlation_id"]))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError):
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

            # Parse raw bytes as JSON
            if isinstance(msg.value, bytes):
                logger.debug(
                    "Parsing message value as bytes (correlation_id=%s)",
                    callback_correlation_id,
                    extra={"value_length": len(msg.value)},
                )
                payload_dict = json.loads(msg.value.decode("utf-8"))
            elif isinstance(msg.value, str):
                logger.debug(
                    "Parsing message value as string (correlation_id=%s)",
                    callback_correlation_id,
                    extra={"value_length": len(msg.value)},
                )
                payload_dict = json.loads(msg.value)
            elif isinstance(msg.value, dict):
                logger.debug(
                    "Message value already dict (correlation_id=%s)",
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
            # Events MUST be wrapped in envelopes on the wire
            logger.debug(
                "Validating payload as ModelEventEnvelope (correlation_id=%s)",
                callback_correlation_id,
            )

            # First, parse as envelope to extract payload and metadata
            try:
                raw_envelope = ModelEventEnvelope[dict].model_validate(payload_dict)
            except Exception as envelope_error:
                # For backwards compatibility, try parsing as raw event
                logger.warning(
                    "Failed to parse as envelope, trying raw event format "
                    "(correlation_id=%s): %s",
                    callback_correlation_id,
                    str(envelope_error),
                )
                # Wrap raw event in envelope for processing
                introspection_event = ModelNodeIntrospectionEvent.model_validate(
                    payload_dict
                )
                event_envelope = ModelEventEnvelope(
                    payload=introspection_event,
                    correlation_id=introspection_event.correlation_id
                    or callback_correlation_id,
                    envelope_timestamp=datetime.now(UTC),
                )
                logger.info(
                    "Raw event wrapped in envelope (correlation_id=%s)",
                    callback_correlation_id,
                    extra={
                        "node_id": str(introspection_event.node_id),
                        "node_type": introspection_event.node_type,
                    },
                )
            else:
                # Validate payload as ModelNodeIntrospectionEvent
                introspection_event = ModelNodeIntrospectionEvent.model_validate(
                    raw_envelope.payload
                )
                # Create typed envelope with validated payload
                event_envelope = ModelEventEnvelope[ModelNodeIntrospectionEvent](
                    payload=introspection_event,
                    envelope_id=raw_envelope.envelope_id,
                    envelope_timestamp=raw_envelope.envelope_timestamp,
                    correlation_id=raw_envelope.correlation_id
                    or introspection_event.correlation_id
                    or callback_correlation_id,
                    source_tool=raw_envelope.source_tool,
                    target_tool=raw_envelope.target_tool,
                    metadata=raw_envelope.metadata,
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
            result = await self._dispatcher.handle(event_envelope)
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
                        output_envelope = ModelEventEnvelope(
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
                json_error,
                callback_correlation_id,
                extra={
                    "error_type": type(json_error).__name__,
                    "error_position": getattr(json_error, "pos", None),
                },
            )

        except Exception as msg_error:
            logger.exception(
                "Failed to process introspection message: %s (correlation_id=%s)",
                msg_error,
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
