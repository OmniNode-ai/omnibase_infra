# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic-scoped publisher that validates against contract-declared publish topics.

This module implements a publisher that enforces topic-level access control
based on the contract's `event_bus.publish_topics` section. Handlers can only
publish to topics explicitly declared in their contract, preventing unauthorized
event emission and maintaining clean architectural boundaries.

Design Principles:
    - **Contract-Driven Access Control**: Topics must be declared in contract
    - **Environment-Aware Routing**: Topic suffixes are prefixed with environment
    - **Fail-Fast Validation**: Invalid topics raise immediately, not at delivery
    - **Duck-Typed Protocol**: Implements publisher protocol without explicit inheritance

Architecture Context:
    In the ONEX handler architecture, each handler receives a topic-scoped
    publisher configured with only the topics from its contract. This ensures:

    1. Handlers cannot publish to arbitrary topics
    2. Topic dependencies are explicit and auditable
    3. Contract changes required to add new publish targets
    4. Clear separation between handler capabilities

Example Usage:
    ```python
    from omnibase_infra.runtime.publisher_topic_scoped import PublisherTopicScoped
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

    # Create topic-scoped publisher from contract
    publisher = PublisherTopicScoped(
        event_bus=kafka_event_bus,
        allowed_topics={"onex.fsm.state.transitions.v1", "onex.events.v1"},
        environment="dev",
    )

    # Publish to allowed topic (succeeds)
    await publisher.publish(
        event_type="state.transition",
        payload={"from": "pending", "to": "active"},
        topic="onex.fsm.state.transitions.v1",
        correlation_id="abc-123",
    )

    # Publish to disallowed topic (raises ProtocolConfigurationError)
    await publisher.publish(
        event_type="audit.log",
        payload={"action": "login"},
        topic="onex.audit.v1",  # Not in allowed_topics
        correlation_id="xyz-789",
    )
    # ProtocolConfigurationError: Topic 'onex.audit.v1' not in contract's publish_topics.
    #                             Allowed: ['onex.events.v1', 'onex.fsm.state.transitions.v1']
    ```

Thread Safety:
    This class is coroutine-safe for concurrent async publishing. The underlying
    event bus handles synchronization. No mutable state is shared between
    publish operations.

Related Tickets:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven topic wiring

See Also:
    - ProtocolEventBusLike: Event bus protocol
    - ModelKafkaEventBusConfig: Kafka configuration model
    - EventBusKafka: Production Kafka event bus implementation
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING
from uuid import UUID

from omnibase_core.types import JsonType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

logger = logging.getLogger(__name__)


class PublisherTopicScoped:
    """Publisher that validates against allowed topics from contract.

    This publisher ensures handlers can only publish to topics explicitly
    declared in their contract's event_bus.publish_topics section.
    Implements a publisher protocol via duck typing (no explicit inheritance
    required per ONEX conventions).

    Features:
        - Contract-driven topic access control
        - Environment-aware topic resolution
        - Fail-fast validation on disallowed topics
        - JSON serialization for payloads
        - Correlation ID propagation for distributed tracing

    Attributes:
        _event_bus: The underlying event bus for publishing
        _allowed_topics: Set of topic suffixes allowed by contract
        _environment: Environment prefix for topic resolution

    Example:
        >>> publisher = PublisherTopicScoped(
        ...     event_bus=kafka_bus,
        ...     allowed_topics={"events.v1", "commands.v1"},
        ...     environment="dev",
        ... )
        >>> await publisher.publish(
        ...     event_type="user.created",
        ...     payload={"user_id": "123"},
        ...     topic="events.v1",
        ... )
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusLike,
        allowed_topics: set[str],
        environment: str,
    ) -> None:
        """Initialize topic-scoped publisher.

        Args:
            event_bus: The event bus implementation for actual publishing
                (EventBusKafka or EventBusInmemory).
                Must implement publish(topic, key, value) method. Duck typed per ONEX.
            allowed_topics: Set of topic suffixes from contract's publish_topics.
                These are the ONLY topics this publisher can publish to.
            environment: Environment prefix (e.g., 'dev', 'staging', 'prod').
                Used to construct full topic names.

        Example:
            >>> publisher = PublisherTopicScoped(
            ...     event_bus=EventBusKafka.default(),
            ...     allowed_topics={"onex.events.v1"},
            ...     environment="dev",
            ... )

        Raises:
            ValueError: If environment is empty or whitespace-only.
        """
        if not environment or not environment.strip():
            raise ValueError("environment must be a non-empty string")

        self._event_bus = event_bus
        self._allowed_topics = frozenset(allowed_topics)
        self._environment = environment
        self._logger = logging.getLogger(__name__)

    def _normalize_correlation_id(
        self,
        correlation_id: str | UUID | None,
    ) -> bytes | None:
        """Normalize correlation ID to bytes for Kafka message key.

        Correlation IDs in ONEX can be either UUID objects (in-memory canonical)
        or strings (wire canonical). Both serialize deterministically to the
        same byte representation.

        Args:
            correlation_id: Correlation ID as string, UUID, or None.

        Returns:
            UTF-8 encoded bytes for use as Kafka message key, or None.
        """
        if correlation_id is None:
            return None
        return str(correlation_id).encode("utf-8")

    def resolve_topic(self, topic_suffix: str) -> str:
        """Resolve topic suffix to full topic name with environment prefix.

        The full topic name follows the ONEX convention:
        `{environment}.{topic_suffix}`

        Args:
            topic_suffix: ONEX format topic suffix (e.g., 'onex.events.v1')

        Returns:
            Full topic name with environment prefix (e.g., 'dev.onex.events.v1')

        Example:
            >>> publisher.resolve_topic("onex.events.v1")
            'dev.onex.events.v1'
        """
        return f"{self._environment}.{topic_suffix}"

    async def publish(
        self,
        event_type: str,
        payload: JsonType,
        topic: str | None = None,
        correlation_id: str | UUID | None = None,
        **kwargs: object,
    ) -> bool:
        """Publish to allowed topic. Raises if topic not in contract.

        Validates the topic against the contract's publish_topics whitelist,
        serializes the payload to JSON, and publishes via the underlying
        event bus.

        Args:
            event_type: Type of event being published (for logging/tracing).
            payload: Event payload (must be JSON-serializable).
                Accepts any JsonType: str, int, float, bool, None,
                list[JsonType], or dict[str, JsonType].
            topic: Topic suffix to publish to (required).
                Must be in the contract's publish_topics.
            correlation_id: Optional correlation ID for distributed tracing.
                If provided, used as the message key for partitioning.
            **kwargs: Additional keyword arguments (ignored, for protocol flexibility).

        Returns:
            True if publish succeeded.

        Raises:
            ProtocolConfigurationError: If topic is None or not in contract's publish_topics.

        Example:
            >>> await publisher.publish(
            ...     event_type="user.created",
            ...     payload={"user_id": "123", "name": "John"},
            ...     topic="onex.events.v1",
            ...     correlation_id="corr-abc-123",
            ... )
            True
        """
        if topic is None:
            raise ProtocolConfigurationError(
                "topic is required for PublisherTopicScoped"
            )

        if topic not in self._allowed_topics:
            raise ProtocolConfigurationError(
                f"Topic '{topic}' not in contract's publish_topics. "
                f"Allowed: {sorted(self._allowed_topics)}"
            )

        full_topic = self.resolve_topic(topic)

        # Serialize payload to JSON bytes
        value = json.dumps(payload).encode("utf-8")
        key = self._normalize_correlation_id(correlation_id)

        # Publish to event bus
        await self._event_bus.publish(
            topic=full_topic,
            key=key,
            value=value,
        )

        self._logger.debug(
            "Published to topic=%s, event_type=%s, correlation_id=%s",
            full_topic,
            event_type,
            correlation_id,
        )

        return True

    @property
    def allowed_topics(self) -> frozenset[str]:
        """Return immutable set of allowed topics.

        Returns:
            Frozen set of topic suffixes allowed by this publisher.

        Example:
            >>> publisher.allowed_topics
            frozenset({'onex.events.v1', 'onex.commands.v1'})
        """
        return self._allowed_topics

    @property
    def environment(self) -> str:
        """Return the environment prefix.

        Returns:
            Environment string used for topic resolution.

        Example:
            >>> publisher.environment
            'dev'
        """
        return self._environment


__all__: list[str] = ["PublisherTopicScoped"]
