# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract registration event router for kernel event processing.

This module provides an event router for routing contract registration events
in the ONEX kernel. Follows the same pattern as IntrospectionEventRouter.

The router:
    - Subscribes to contract registration Kafka topics
    - Parses incoming event messages based on topic suffix
    - Routes to ContractRegistryReducer.reduce() for state transitions
    - Emits intents for Effect layer execution (PostgreSQL operations)

Topics Handled:
    - {env}.onex.evt.platform.contract-registered.v1 -> ModelContractRegisteredEvent
    - {env}.onex.evt.platform.contract-deregistered.v1 -> ModelContractDeregisteredEvent
    - {env}.onex.evt.platform.node-heartbeat.v1 -> ModelNodeHeartbeatEvent

Design:
    This class encapsulates the message routing logic for contract registration
    events. The router uses topic suffix matching to determine the event type
    and routes to the appropriate reducer handler.

    The reducer returns ModelReducerOutput containing:
    - New state (ModelContractRegistryState)
    - Intents tuple for Effect layer execution

    Intents are NOT published back to Kafka by the router. They are returned
    for the caller (typically the kernel) to dispatch to the Effect layer.

Related:
    - OMN-1869: Contract Registration Event Router
    - OMN-1653: Contract Registry Reducer
    - IntrospectionEventRouter: Reference implementation pattern
"""

from __future__ import annotations

__all__ = ["ContractRegistrationEventRouter"]

import json
import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import ValidationError

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
    ModelNodeHeartbeatEvent,
)
from omnibase_core.nodes import ModelReducerOutput
from omnibase_core.types import JsonType
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.nodes.contract_registry_reducer.models.model_contract_registry_state import (
    ModelContractRegistryState,
)
from omnibase_infra.nodes.contract_registry_reducer.reducer import (
    ContractRegistryReducer,
)
from omnibase_infra.utils import sanitize_error_message

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolEventBusLike

logger = logging.getLogger(__name__)

# Topic suffix patterns for event type matching
# These match the topic suffix after the environment prefix
TOPIC_SUFFIX_CONTRACT_REGISTERED = "onex.evt.platform.contract-registered.v1"
TOPIC_SUFFIX_CONTRACT_DEREGISTERED = "onex.evt.platform.contract-deregistered.v1"
TOPIC_SUFFIX_NODE_HEARTBEAT = "onex.evt.platform.node-heartbeat.v1"


class ContractRegistrationEventRouter:
    """Router for contract registration event messages from event bus.

    This router handles incoming event messages for the contract registration
    domain. It parses events based on topic suffix and routes them to the
    ContractRegistryReducer for state machine processing.

    The router propagates correlation IDs from incoming messages for
    distributed tracing. If no correlation ID is present, it generates
    a new one to ensure all operations can be traced.

    This class follows the container-based dependency injection pattern,
    receiving a ModelONEXContainer for service resolution while also
    accepting explicit dependencies for router-specific configuration.

    Attributes:
        _container: ONEX service container for dependency resolution.
        _reducer: The ContractRegistryReducer to route events to.
        _event_bus: Event bus implementing ProtocolEventBusLike for publishing.
        _output_topic: The topic to publish intent completion events to.
        _state: Current contract registry state (maintained across messages).

    Example:
        >>> from omnibase_core.container import ModelONEXContainer
        >>> container = ModelONEXContainer()
        >>> reducer = ContractRegistryReducer()
        >>> router = ContractRegistrationEventRouter(
        ...     container=container,
        ...     reducer=reducer,
        ...     event_bus=event_bus,
        ...     output_topic="contract.intents.output",
        ... )
        >>> # Use as callback for event bus subscription
        >>> await event_bus.subscribe(
        ...     topic="dev.onex.evt.platform.contract-registered.v1",
        ...     group_id="contract-registry",
        ...     on_message=router.handle_message,
        ... )

    See Also:
        - docs/patterns/container_dependency_injection.md for DI patterns.
        - IntrospectionEventRouter for reference implementation.
    """

    def __init__(
        self,
        container: ModelONEXContainer,
        reducer: ContractRegistryReducer,
        event_bus: ProtocolEventBusLike,
        output_topic: str,
    ) -> None:
        """Initialize ContractRegistrationEventRouter with container-based DI.

        Follows the ONEX container-based DI pattern where the container is passed
        as the first parameter for service resolution, with additional explicit
        parameters for router-specific configuration.

        Args:
            container: ONEX service container for dependency resolution. Provides
                access to service_registry for resolving shared services.
            reducer: The ContractRegistryReducer to route events to.
            event_bus: Event bus implementing ProtocolEventBusLike for publishing.
            output_topic: The topic to publish intent completion events to.

        Raises:
            ValueError: If output_topic is empty.

        Example:
            >>> from omnibase_core.container import ModelONEXContainer
            >>> container = ModelONEXContainer()
            >>> reducer = ContractRegistryReducer()
            >>> router = ContractRegistrationEventRouter(
            ...     container=container,
            ...     reducer=reducer,
            ...     event_bus=event_bus,
            ...     output_topic="contract.intents.output",
            ... )

        See Also:
            - docs/patterns/container_dependency_injection.md for DI patterns.
        """
        if not output_topic:
            raise ValueError("output_topic cannot be empty")

        self._container = container
        self._reducer = reducer
        self._event_bus = event_bus
        self._output_topic = output_topic
        # Initialize empty state - maintained across message processing
        self._state = ModelContractRegistryState()

        logger.debug(
            "ContractRegistrationEventRouter initialized",
            extra={
                "output_topic": output_topic,
                "reducer_type": type(self._reducer).__name__,
                "event_bus_type": type(self._event_bus).__name__,
            },
        )

    @property
    def container(self) -> ModelONEXContainer:
        """Return the ONEX service container.

        Returns:
            The ModelONEXContainer instance passed during initialization.
        """
        return self._container

    @property
    def output_topic(self) -> str:
        """Return the configured output topic for event publishing."""
        return self._output_topic

    @property
    def reducer(self) -> ContractRegistryReducer:
        """Return the reducer instance."""
        return self._reducer

    @property
    def event_bus(self) -> ProtocolEventBusLike:
        """Return the event bus instance."""
        return self._event_bus

    @property
    def state(self) -> ModelContractRegistryState:
        """Return the current contract registry state."""
        return self._state

    def _extract_correlation_id_from_message(self, msg: ModelEventMessage) -> UUID:
        """Extract correlation ID from message headers or payload.

        Attempts to extract the correlation_id from message headers or payload
        to ensure proper propagation for distributed tracing. Falls back to
        generating a new UUID if no correlation ID is found.

        Args:
            msg: The incoming event message.

        Returns:
            UUID: The extracted or generated correlation ID.
        """
        # Try to extract from message headers if available
        if hasattr(msg, "headers") and msg.headers is not None:
            headers = msg.headers
            if (
                hasattr(headers, "correlation_id")
                and headers.correlation_id is not None
            ):
                try:
                    correlation_id = headers.correlation_id
                    # Handle bytes-like values (duck typing)
                    if hasattr(correlation_id, "decode"):
                        correlation_id = correlation_id.decode("utf-8")
                    return UUID(str(correlation_id))
                except (ValueError, TypeError, UnicodeDecodeError, AttributeError):
                    pass  # Fall through to try payload extraction

        # Try to extract from payload
        try:
            if msg.value is not None:
                if hasattr(msg.value, "decode"):
                    payload_dict = json.loads(msg.value.decode("utf-8"))
                else:
                    try:
                        payload_dict = json.loads(msg.value)
                    except TypeError:
                        payload_dict = msg.value

                if payload_dict and "correlation_id" in payload_dict:
                    return UUID(str(payload_dict["correlation_id"]))
        except (json.JSONDecodeError, ValueError, TypeError, KeyError, AttributeError):
            pass  # Fall through to generate new ID

        # Generate new correlation ID as last resort
        return uuid4()

    def _determine_event_type(
        self, topic: str
    ) -> (
        type[
            ModelContractRegisteredEvent
            | ModelContractDeregisteredEvent
            | ModelNodeHeartbeatEvent
        ]
        | None
    ):
        """Determine event type based on topic suffix.

        Args:
            topic: The Kafka topic name.

        Returns:
            The event model class to use for parsing, or None if topic not recognized.
        """
        if topic.endswith(TOPIC_SUFFIX_CONTRACT_REGISTERED):
            return ModelContractRegisteredEvent
        elif topic.endswith(TOPIC_SUFFIX_CONTRACT_DEREGISTERED):
            return ModelContractDeregisteredEvent
        elif topic.endswith(TOPIC_SUFFIX_NODE_HEARTBEAT):
            return ModelNodeHeartbeatEvent
        return None

    async def handle_message(
        self, msg: ModelEventMessage
    ) -> ModelReducerOutput[ModelContractRegistryState] | None:
        """Handle incoming contract registration event message.

        This callback is invoked for each message received on the subscribed topics.
        It parses the raw JSON payload based on topic suffix, routes to the
        ContractRegistryReducer, and updates internal state.

        The method propagates the correlation_id from the incoming message
        for distributed tracing. If no correlation_id is present in the message,
        a new one is generated.

        Args:
            msg: The event message containing raw bytes in .value field.

        Returns:
            ModelReducerOutput containing new state and intents, or None on error.
            The intents should be dispatched to the Effect layer by the caller.
        """
        # Extract correlation_id from message for proper propagation
        callback_correlation_id = self._extract_correlation_id_from_message(msg)
        callback_start_time = time.time()

        # Extract topic from message
        topic = getattr(msg, "topic", "") or ""
        partition = getattr(msg, "partition", 0) or 0
        offset = getattr(msg, "offset", 0) or 0

        logger.debug(
            "Contract registration message callback invoked (correlation_id=%s)",
            callback_correlation_id,
            extra={
                "message_offset": offset,
                "message_partition": partition,
                "message_topic": topic,
            },
        )

        try:
            # Determine event type from topic
            event_class = self._determine_event_type(topic)
            if event_class is None:
                logger.debug(
                    "Topic not recognized as contract registration event, skipping "
                    "(correlation_id=%s)",
                    callback_correlation_id,
                    extra={"topic": topic},
                )
                return None

            # Parse message value
            if msg.value is None:
                logger.debug(
                    "Message value is None, skipping (correlation_id=%s)",
                    callback_correlation_id,
                )
                return None

            # Parse message value using duck-typing patterns
            if hasattr(msg.value, "decode"):
                logger.debug(
                    "Parsing message value as bytes-like (correlation_id=%s)",
                    callback_correlation_id,
                    extra={"value_length": len(msg.value)},
                )
                payload_dict = json.loads(msg.value.decode("utf-8"))
            else:
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
                        return None

            # Parse as the appropriate event model
            logger.debug(
                "Validating payload as %s (correlation_id=%s)",
                event_class.__name__,
                callback_correlation_id,
            )

            event = event_class.model_validate(payload_dict)

            logger.info(
                "Event parsed successfully (correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "event_type": type(event).__name__,
                    "event_id": str(event.event_id),
                    "node_name": event.node_name,
                },
            )

            # Build event metadata for reducer
            event_metadata: dict[str, JsonType] = {
                "topic": topic,
                "partition": partition,
                "offset": offset,
                "correlation_id": str(callback_correlation_id),
            }

            # Route to reducer
            logger.info(
                "Routing to contract registry reducer (correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "event_type": type(event).__name__,
                    "node_name": event.node_name,
                },
            )
            reducer_start_time = time.time()
            result = self._reducer.reduce(self._state, event, event_metadata)
            reducer_duration = time.time() - reducer_start_time

            # Update internal state
            self._state = result.result

            logger.info(
                "Contract registration event processed successfully: node_name=%s "
                "in %.3fs (correlation_id=%s)",
                event.node_name,
                reducer_duration,
                callback_correlation_id,
                extra={
                    "reducer_duration_seconds": reducer_duration,
                    "event_type": type(event).__name__,
                    "node_name": event.node_name,
                    "intents_count": len(result.intents),
                    "items_processed": result.items_processed,
                },
            )

            # Log intent summary for debugging
            if result.intents:
                logger.debug(
                    "Reducer emitted %d intents (correlation_id=%s)",
                    len(result.intents),
                    callback_correlation_id,
                    extra={
                        "intent_targets": [intent.target for intent in result.intents],
                    },
                )

            return result

        except ValidationError as validation_error:
            # Not a valid event for this topic - skip
            logger.debug(
                "Message is not a valid contract registration event, skipping "
                "(correlation_id=%s)",
                callback_correlation_id,
                extra={
                    "validation_error_count": validation_error.error_count(),
                    "topic": topic,
                },
            )
            return None

        except json.JSONDecodeError as json_error:
            logger.warning(
                "Failed to decode JSON from message: %s (correlation_id=%s)",
                sanitize_error_message(json_error),
                callback_correlation_id,
                extra={
                    "error_type": type(json_error).__name__,
                    "error_position": getattr(json_error, "pos", None),
                    "topic": topic,
                },
            )
            return None

        except Exception as msg_error:
            # Use warning instead of exception to avoid credential exposure
            # in tracebacks (connection errors may contain DSN with password)
            logger.warning(
                "Failed to process contract registration message: %s (correlation_id=%s)",
                sanitize_error_message(msg_error),
                callback_correlation_id,
                extra={
                    "error_type": type(msg_error).__name__,
                    "topic": topic,
                },
            )
            return None

        finally:
            callback_duration = time.time() - callback_start_time
            logger.debug(
                "Contract registration message callback completed in %.3fs "
                "(correlation_id=%s)",
                callback_duration,
                callback_correlation_id,
                extra={
                    "callback_duration_seconds": callback_duration,
                    "topic": topic,
                },
            )
