# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event bus subcontract wiring for contract-driven Kafka subscriptions.

This module provides the bridge between contract-declared topics (from the
`event_bus` subcontract) and actual Kafka subscriptions. The runtime owns
all Kafka plumbing - nodes/handlers never create consumers or producers directly.

Architecture:
    The EventBusSubcontractWiring class is responsible for:
    1. Reading `subscribe_topics` from ModelEventBusSubcontract
    2. Resolving topic suffixes to full topic names with environment prefix
    3. Creating Kafka subscriptions with appropriate consumer groups
    4. Bridging received messages to the MessageDispatchEngine
    5. Managing subscription lifecycle (creation and cleanup)

    This follows the ARCH-002 principle: "Runtime owns all Kafka plumbing."
    Nodes and handlers declare their topic requirements in contracts, but
    never directly interact with Kafka consumers or producers.

Topic Resolution:
    Topic suffixes from contracts follow the ONEX naming convention:
        onex.{kind}.{producer}.{event-name}.v{n}

    The wiring resolves these to full topics by prepending the environment:
        {environment}.onex.{kind}.{producer}.{event-name}.v{n}

    Example:
        - Contract declares: "onex.evt.omniintelligence.intent-classified.v1"
        - Resolved (dev): "dev.onex.evt.omniintelligence.intent-classified.v1"
        - Resolved (prod): "prod.onex.evt.omniintelligence.intent-classified.v1"

Related:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - ModelEventBusSubcontract: Contract model defining subscribe/publish topics
    - MessageDispatchEngine: Dispatch engine that processes received messages
    - EventBusKafka: Kafka event bus implementation

.. versionadded:: 0.2.5
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.protocols.event_bus.protocol_event_bus_subscriber import (
    ProtocolEventBusSubscriber,
)
from omnibase_core.protocols.event_bus.protocol_event_message import (
    ProtocolEventMessage,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.protocols import ProtocolDispatchEngine

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )


class EventBusSubcontractWiring:
    """Wires event_bus subcontracts to Kafka subscriptions and publishers.

    This class bridges contract-declared topics to actual Kafka subscriptions,
    ensuring that nodes/handlers never directly interact with Kafka infrastructure.
    The runtime owns all Kafka plumbing per ARCH-002.

    Responsibilities:
        - Parse subscribe_topics from ModelEventBusSubcontract
        - Resolve topic suffixes to full topic names with environment prefix
        - Create Kafka subscriptions with appropriate consumer groups
        - Deserialize incoming messages to ModelEventEnvelope
        - Dispatch envelopes to MessageDispatchEngine
        - Manage subscription lifecycle (cleanup on shutdown)

    Thread Safety:
        This class is designed for single-threaded async use. All subscription
        operations should be performed from a single async context. The underlying
        event bus implementations (EventBusKafka, EventBusInmemory) handle their
        own thread safety for message delivery.

    Example:
        ```python
        from omnibase_infra.runtime import EventBusSubcontractWiring
        from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract

        # Create wiring with event bus and dispatch engine
        wiring = EventBusSubcontractWiring(
            event_bus=event_bus,
            dispatch_engine=dispatch_engine,
            environment="dev",
        )

        # Wire subscriptions from subcontract
        subcontract = ModelEventBusSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            subscribe_topics=["onex.evt.omniintelligence.intent-classified.v1"],
        )
        await wiring.wire_subscriptions(subcontract, node_name="my-handler")

        # Cleanup on shutdown
        await wiring.cleanup()
        ```

    Attributes:
        _event_bus: The event bus implementation (Kafka or in-memory)
        _dispatch_engine: Engine to dispatch received messages to handlers
        _environment: Environment prefix for topics (e.g., 'dev', 'prod')
        _unsubscribe_callables: List of callables to unsubscribe from topics
        _logger: Logger for debug and error messages

    .. versionadded:: 0.2.5
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusSubscriber,
        dispatch_engine: ProtocolDispatchEngine,
        environment: str,
    ) -> None:
        """Initialize event bus wiring.

        Args:
            event_bus: The event bus implementation (EventBusKafka or EventBusInmemory).
                Must implement subscribe(topic, group_id, on_message) -> unsubscribe callable.
                Duck typed per ONEX patterns.
            dispatch_engine: Engine to dispatch received messages to handlers.
                Must implement ProtocolDispatchEngine interface.
                Must be frozen (registrations complete) before wiring subscriptions.
            environment: Environment prefix for topics (e.g., 'dev', 'prod').
                Used to resolve topic suffixes to full topic names.

        Note:
            The dispatch_engine should be frozen before wiring subscriptions.
            Attempting to dispatch to an unfrozen engine will raise an error.

        Raises:
            ValueError: If environment is empty or whitespace-only.
        """
        if not environment or not environment.strip():
            raise ValueError("environment must be a non-empty string")

        self._event_bus = event_bus
        self._dispatch_engine = dispatch_engine
        self._environment = environment
        self._unsubscribe_callables: list[Callable[[], Awaitable[None]]] = []
        self._logger = logging.getLogger(__name__)

    def resolve_topic(self, topic_suffix: str) -> str:
        """Resolve topic suffix to full topic name with environment prefix.

        Topic suffixes from contracts follow the ONEX naming convention:
            onex.{kind}.{producer}.{event-name}.v{n}

        This method prepends the environment to create the full topic name:
            {environment}.onex.{kind}.{producer}.{event-name}.v{n}

        Args:
            topic_suffix: ONEX format topic suffix
                (e.g., 'onex.evt.omniintelligence.intent-classified.v1')

        Returns:
            Full topic name with environment prefix
                (e.g., 'dev.onex.evt.omniintelligence.intent-classified.v1')

        Example:
            >>> wiring = EventBusSubcontractWiring(bus, engine, "dev")
            >>> wiring.resolve_topic("onex.evt.user.created.v1")
            'dev.onex.evt.user.created.v1'
        """
        return f"{self._environment}.{topic_suffix}"

    async def wire_subscriptions(
        self,
        subcontract: ModelEventBusSubcontract,
        node_name: str,
    ) -> None:
        """Wire Kafka subscriptions from subcontract.subscribe_topics.

        Creates Kafka subscriptions for each topic declared in the subcontract's
        subscribe_topics list. Each subscription uses a consumer group ID based
        on the environment and node name for proper load balancing.

        Consumer Group Naming:
            Consumer groups are named as: {environment}.{node_name}
            Example: "dev.registration-handler"

            This ensures:
            - Each node instance in an environment shares the same consumer group
            - Multiple instances of the same node load-balance message processing
            - Different environments are completely isolated

        Args:
            subcontract: The event_bus subcontract from a handler's contract.
                Contains subscribe_topics list with topic suffixes.
            node_name: Name of the node/handler for consumer group identification.
                Should be unique per handler type (e.g., "registration-handler").

        Raises:
            InfraConnectionError: If Kafka connection fails during subscription.
            InfraTimeoutError: If subscription times out.

        Example:
            >>> subcontract = ModelEventBusSubcontract(
            ...     version=ModelSemVer(major=1, minor=0, patch=0),
            ...     subscribe_topics=["onex.evt.node.introspected.v1"],
            ... )
            >>> await wiring.wire_subscriptions(subcontract, "registration-handler")
        """
        if not subcontract.subscribe_topics:
            self._logger.debug(
                "No subscribe_topics in subcontract for node '%s'",
                node_name,
            )
            return

        for topic_suffix in subcontract.subscribe_topics:
            full_topic = self.resolve_topic(topic_suffix)
            group_id = f"{self._environment}.{node_name}"

            # Create dispatch callback for this topic
            callback = self._create_dispatch_callback(full_topic)

            # Subscribe and store unsubscribe callable
            unsubscribe = await self._event_bus.subscribe(
                topic=full_topic,
                group_id=group_id,
                on_message=callback,
            )
            self._unsubscribe_callables.append(unsubscribe)

            self._logger.info(
                "Wired subscription: topic=%s, group_id=%s, node=%s",
                full_topic,
                group_id,
                node_name,
            )

    def _create_dispatch_callback(
        self,
        topic: str,
    ) -> Callable[[ProtocolEventMessage], Awaitable[None]]:
        """Create callback that bridges Kafka consumer to dispatch engine.

        Creates an async callback function that:
        1. Receives ProtocolEventMessage from the Kafka consumer
        2. Deserializes the message value to ModelEventEnvelope
        3. Dispatches the envelope to the MessageDispatchEngine

        Error Handling:
            - Deserialization errors are logged and the message is skipped
            - Dispatch errors are propagated (handled by the event bus DLQ logic)

        Args:
            topic: The full topic name for routing context in logs.

        Returns:
            Async callback function compatible with event bus subscribe().
        """

        async def callback(message: ProtocolEventMessage) -> None:
            """Process incoming Kafka message and dispatch to engine."""
            try:
                envelope = self._deserialize_to_envelope(message)
                # Dispatch via ProtocolDispatchEngine interface
                await self._dispatch_engine.dispatch(topic, envelope)
            except json.JSONDecodeError as e:
                self._logger.exception(
                    "Failed to deserialize message from topic '%s': %s",
                    topic,
                    e,
                )
                # Wrap in OnexError per CLAUDE.md: "OnexError Only"
                raise RuntimeHostError(
                    f"Failed to deserialize message from topic '{topic}'",
                    context=ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.KAFKA,
                        operation="event_bus_deserialize",
                    ),
                ) from e
            except Exception as e:
                self._logger.exception(
                    "Failed to dispatch message from topic '%s': %s",
                    topic,
                    e,
                )
                # Wrap in OnexError per CLAUDE.md: "OnexError Only"
                raise RuntimeHostError(
                    f"Failed to dispatch message from topic '{topic}'",
                    context=ModelInfraErrorContext.with_correlation(
                        transport_type=EnumInfraTransportType.KAFKA,
                        operation="event_bus_dispatch",
                    ),
                ) from e

        return callback

    def _deserialize_to_envelope(
        self,
        message: ProtocolEventMessage,
    ) -> ModelEventEnvelope[object]:
        """Deserialize Kafka message to event envelope.

        Converts the raw bytes in ProtocolEventMessage.value to a ModelEventEnvelope
        that can be processed by the dispatch engine.

        Deserialization Strategy:
            1. Decode message.value from UTF-8 bytes to string
            2. Parse JSON string to dict
            3. Validate and construct ModelEventEnvelope

        Args:
            message: ProtocolEventMessage from Kafka consumer containing raw bytes.

        Returns:
            Deserialized ModelEventEnvelope for dispatch.

        Raises:
            json.JSONDecodeError: If message value is not valid JSON.
            ValidationError: If JSON does not match ModelEventEnvelope schema.
        """
        # Decode bytes to string
        json_str = message.value.decode("utf-8")

        # Parse JSON to dict
        data = json.loads(json_str)

        # Validate and construct envelope
        return ModelEventEnvelope[object].model_validate(data)

    async def cleanup(self) -> None:
        """Unsubscribe from all topics.

        Should be called during runtime shutdown to properly clean up
        Kafka consumer subscriptions. This ensures:
        - Consumer group offsets are committed
        - Connections are properly closed
        - Resources are released

        This method is safe to call multiple times - subsequent calls
        are no-ops after the first successful cleanup.

        Example:
            >>> # During shutdown
            >>> await wiring.cleanup()
        """
        cleanup_count = len(self._unsubscribe_callables)

        for unsubscribe in self._unsubscribe_callables:
            try:
                await unsubscribe()
            except Exception as e:
                self._logger.warning(
                    "Error during unsubscribe: %s",
                    e,
                )

        self._unsubscribe_callables.clear()

        if cleanup_count > 0:
            self._logger.info(
                "Cleaned up %d event bus subscription(s)",
                cleanup_count,
            )


def load_event_bus_subcontract(
    contract_path: Path,
    logger: logging.Logger | None = None,
) -> ModelEventBusSubcontract | None:
    """Load event_bus subcontract from contract YAML file.

    Reads a contract YAML file and extracts the event_bus section,
    returning a validated ModelEventBusSubcontract if present.

    File Format:
        The contract YAML should have an `event_bus` section:

        ```yaml
        event_bus:
          version:
            major: 1
            minor: 0
            patch: 0
          subscribe_topics:
            - onex.evt.node.introspected.v1
            - onex.evt.node.registered.v1
          publish_topics:
            - onex.cmd.node.register.v1
        ```

    Args:
        contract_path: Path to the contract YAML file.
        logger: Optional logger for warnings. If not provided, uses module logger.

    Returns:
        ModelEventBusSubcontract if event_bus section exists and is valid,
        None otherwise.

    Example:
        >>> subcontract = load_event_bus_subcontract(Path("contract.yaml"))
        >>> if subcontract:
        ...     print(f"Subscribe topics: {subcontract.subscribe_topics}")
    """
    _logger = logger or logging.getLogger(__name__)

    if not contract_path.exists():
        _logger.warning(
            "Contract file not found: %s",
            contract_path,
        )
        return None

    try:
        with contract_path.open() as f:
            contract_data = yaml.safe_load(f)

        if contract_data is None:
            _logger.warning(
                "Empty contract file: %s",
                contract_path,
            )
            return None

        event_bus_data = contract_data.get("event_bus")
        if not event_bus_data:
            _logger.debug(
                "No event_bus section in contract: %s",
                contract_path,
            )
            return None

        return ModelEventBusSubcontract.model_validate(event_bus_data)

    except yaml.YAMLError as e:
        _logger.warning(
            "Failed to parse YAML in contract %s: %s",
            contract_path,
            e,
        )
        return None
    except ValidationError as e:
        _logger.warning(
            "Invalid event_bus subcontract in %s: %s",
            contract_path,
            e,
        )
        return None


__all__: list[str] = [
    "EventBusSubcontractWiring",
    "load_event_bus_subcontract",
]
