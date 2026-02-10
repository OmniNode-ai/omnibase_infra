# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event bus subcontract wiring for contract-driven Kafka subscriptions.

This module provides the bridge between contract-declared topics (from the
`event_bus` subcontract) and actual Kafka subscriptions. The runtime owns
all Kafka plumbing - nodes/handlers never create consumers or producers directly.

Architecture:
    The EventBusSubcontractWiring class is responsible for:
    1. Reading `subscribe_topics` from ModelEventBusSubcontract
    2. Passing topic suffixes through unchanged (topics are realm-agnostic)
    3. Creating Kafka subscriptions with appropriate consumer groups
    4. Bridging received messages to the MessageDispatchEngine
    5. Managing subscription lifecycle (creation and cleanup)
    6. Classifying errors as content vs infrastructure for proper handling

    This follows the ARCH-002 principle: "Runtime owns all Kafka plumbing."
    Nodes and handlers declare their topic requirements in contracts, but
    never directly interact with Kafka consumers or producers.

Error Classification:
    The wiring distinguishes between two error categories:

    Content Errors (non-retryable):
        Schema validation failures, malformed payloads, missing required fields,
        type conversion errors. These will NOT fix themselves with retry.
        Default behavior: Send to DLQ and commit offset (dlq_and_commit).
        Identified by: ProtocolConfigurationError, json.JSONDecodeError,
        pydantic.ValidationError

    Infrastructure Errors (potentially retryable):
        Database timeouts, network failures, service unavailability.
        These errors MAY fix themselves after retry.
        Default behavior: Fail fast (fail_fast) to avoid hiding infrastructure
        fires in the DLQ.
        Identified by: RuntimeHostError and subclasses (InfraConnectionError,
        InfraTimeoutError, InfraUnavailableError, etc.)

DLQ Consumer Group Alignment:
    IMPORTANT: The consumer_group used for DLQ publishing MUST match the
    consumer_group used when subscribing to topics. This is critical for:
    - Traceability: DLQ messages can be correlated back to their source consumer
    - Replay operations: DLQ replay tools can identify which consumer group failed
    - Debugging: Operations teams can trace failures to specific consumer groups

    The wiring ensures this alignment by:
    1. Computing consumer_group as "{environment}.{node_name}" in wire_subscriptions
    2. Passing this same consumer_group to _create_dispatch_callback
    3. Using it in all _publish_to_dlq calls within the callback closure

Topic Resolution:
    Topics are realm-agnostic and do NOT include environment prefixes.
    The environment/realm is a routing boundary enforced via envelope identity,
    not a topic prefix. This enables cross-environment event routing when needed.

    Topic suffixes from contracts follow the ONEX naming convention:
        onex.{kind}.{producer}.{event-name}.v{n}

    The wiring passes these topic suffixes through unchanged:
        onex.{kind}.{producer}.{event-name}.v{n}

    Example:
        - Contract declares: "onex.evt.omniintelligence.intent-classified.v1"
        - Resolved: "onex.evt.omniintelligence.intent-classified.v1"

    Note: Consumer groups still include environment for isolation.

Related:
    - OMN-1621: Runtime consumes event_bus subcontract for contract-driven wiring
    - OMN-1740: Error classification (content vs infra) in wiring
    - OMN-2038: Propagate event_type from contract topic metadata into envelopes
    - ModelEventBusSubcontract: Contract model defining subscribe/publish topics
    - MessageDispatchEngine: Dispatch engine that processes received messages
    - EventBusKafka: Kafka event bus implementation

.. versionadded:: 0.2.5
.. versionchanged:: 0.2.9
    Added error classification (content vs infrastructure) with DLQ integration.
.. versionchanged:: 0.3.0
    Propagate event_type from ONEX topic naming convention into deserialized envelopes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

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
from omnibase_infra.enums import EnumConsumerGroupPurpose, EnumInfraTransportType
from omnibase_infra.errors import (
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
)
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.event_bus import (
    ModelConsumerRetryConfig,
    ModelDlqConfig,
    ModelIdempotencyConfig,
    ModelOffsetPolicyConfig,
)
from omnibase_infra.observability.wiring_health import MixinConsumptionCounter
from omnibase_infra.protocols import ProtocolDispatchEngine, ProtocolIdempotencyStore
from omnibase_infra.topics import TopicResolver
from omnibase_infra.utils import compute_consumer_group_id

if TYPE_CHECKING:
    from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
    from omnibase_infra.runtime.service_dispatch_result_applier import (
        DispatchResultApplier,
    )
    from omnibase_infra.runtime.service_message_dispatch_engine import (
        MessageDispatchEngine,
    )


class EventBusSubcontractWiring(MixinConsumptionCounter):
    """Wires event_bus subcontracts to Kafka subscriptions and publishers.

    This class bridges contract-declared topics to actual Kafka subscriptions,
    ensuring that nodes/handlers never directly interact with Kafka infrastructure.
    The runtime owns all Kafka plumbing per ARCH-002.

    Responsibilities:
        - Parse subscribe_topics from ModelEventBusSubcontract
        - Pass topic suffixes through unchanged (topics are realm-agnostic)
        - Create Kafka subscriptions with appropriate consumer groups
        - Deserialize incoming messages to ModelEventEnvelope
        - Check idempotency and skip duplicate messages (if enabled)
        - Classify errors as content (DLQ) vs infrastructure (fail-fast)
        - Dispatch envelopes to MessageDispatchEngine
        - Manage subscription lifecycle (cleanup on shutdown)

    Error Classification:
        Content Errors (non-retryable): ProtocolConfigurationError, ValidationError,
        json.JSONDecodeError. Default: DLQ and commit offset.

        Infrastructure Errors (retryable): RuntimeHostError and subclasses.
        Default: Fail-fast (no DLQ, no commit).

    Idempotency:
        When configured with an idempotency store and enabled config, the wiring
        deduplicates messages based on the `envelope_id` field from the envelope.
        Messages with the same envelope_id (within a topic domain) are processed
        only once - duplicates are logged and skipped.

        Requirements when idempotency is enabled:
        - All envelopes MUST have a non-None envelope_id field
        - Missing envelope_id raises ProtocolConfigurationError

    Thread Safety:
        This class is designed for single-threaded async use. All subscription
        operations should be performed from a single async context. The underlying
        event bus implementations (EventBusKafka, EventBusInmemory) handle their
        own thread safety for message delivery.

    Example:
        ```python
        from omnibase_infra.runtime import EventBusSubcontractWiring
        from omnibase_infra.models.event_bus import (
            ModelIdempotencyConfig,
            ModelDlqConfig,
            ModelConsumerRetryConfig,
            ModelOffsetPolicyConfig,
        )
        from omnibase_infra.idempotency import StoreIdempotencyInmemory
        from omnibase_core.models.contracts.subcontracts import ModelEventBusSubcontract

        # Create wiring with full error handling configuration
        wiring = EventBusSubcontractWiring(
            event_bus=event_bus,
            dispatch_engine=dispatch_engine,
            environment="dev",
            node_name="my-handler",
            idempotency_store=StoreIdempotencyInmemory(),
            idempotency_config=ModelIdempotencyConfig(enabled=True),
            dlq_config=ModelDlqConfig(enabled=True),
            retry_config=ModelConsumerRetryConfig.create_standard(),
            offset_policy=ModelOffsetPolicyConfig(),
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
        _environment: Environment identifier for consumer groups (e.g., 'dev', 'prod')
        _node_name: Name of the node/handler for consumer group and logging
        _result_applier: Optional applier for processing dispatch results
            (publishing output events and delegating intents)
        _idempotency_store: Optional store for tracking processed messages
        _idempotency_config: Configuration for idempotency behavior
        _dlq_config: Configuration for Dead Letter Queue behavior
        _retry_config: Configuration for consumer-side retry behavior
        _offset_policy: Configuration for offset commit strategy
        _unsubscribe_callables: List of callables to unsubscribe from topics
        _logger: Logger for debug and error messages
        _retry_counts: Tracks retry attempts per message (by correlation_id)

    .. versionadded:: 0.2.5
    .. versionchanged:: 0.2.9
        Added idempotency gate support via idempotency_store and idempotency_config.
        Added error classification (content vs infrastructure) with DLQ integration.
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusSubscriber,
        dispatch_engine: ProtocolDispatchEngine,
        environment: str,
        node_name: str,
        service: str,
        version: str,
        result_applier: DispatchResultApplier | None = None,
        idempotency_store: ProtocolIdempotencyStore | None = None,
        idempotency_config: ModelIdempotencyConfig | None = None,
        dlq_config: ModelDlqConfig | None = None,
        retry_config: ModelConsumerRetryConfig | None = None,
        offset_policy: ModelOffsetPolicyConfig | None = None,
    ) -> None:
        """Initialize event bus wiring.

        Args:
            event_bus: The event bus implementation (EventBusKafka or EventBusInmemory).
                Must implement subscribe(topic, node_identity, on_message) -> unsubscribe callable.
                Duck typed per ONEX patterns.
            dispatch_engine: Engine to dispatch received messages to handlers.
                Must implement ProtocolDispatchEngine interface.
                Must be frozen (registrations complete) before wiring subscriptions.
            environment: Environment identifier (e.g., 'dev', 'prod').
                Used for consumer group naming and node identity. Topics are
                realm-agnostic and do not include environment prefixes.
            node_name: Name of the node/handler for consumer group identification and logging.
            service: Service name for node identity (e.g., 'omniintelligence', 'omnibridge').
                Used to derive consumer group ID.
            version: Version string for node identity (e.g., 'v1', 'v1.0.0').
                Used to derive consumer group ID.
            result_applier: Optional DispatchResultApplier for processing
                dispatch results. When provided, the wiring captures the return value
                of dispatch_engine.dispatch() and passes it to the applier for output
                event publishing and intent delegation. Duck-typed: must have an async
                ``apply(result, correlation_id)`` method.
            idempotency_store: Optional idempotency store for message deduplication.
                If provided with enabled config, messages are deduplicated by envelope_id.
            idempotency_config: Optional configuration for idempotency behavior.
                If None, idempotency checking is disabled.
            dlq_config: Optional configuration for Dead Letter Queue behavior.
                Controls how content vs infrastructure errors are handled.
                If None, uses defaults (content -> DLQ, infra -> fail-fast).
            retry_config: Optional configuration for consumer-side retry behavior.
                Controls retry attempts and backoff for infrastructure errors.
                If None, uses standard defaults (3 attempts, exponential backoff).
            offset_policy: Optional configuration for offset commit strategy.
                Controls when offsets are committed relative to handler execution.
                If None, uses commit_after_handler (at-least-once delivery).

        Note:
            The dispatch_engine should be frozen before wiring subscriptions.
            Attempting to dispatch to an unfrozen engine will raise an error.

        Raises:
            ValueError: If environment, service, or version is empty or whitespace-only.
        """
        if not environment or not environment.strip():
            raise ValueError("environment must be a non-empty string")
        if not service or not service.strip():
            raise ValueError("service must be a non-empty string")
        if not version or not version.strip():
            raise ValueError("version must be a non-empty string")

        self._event_bus = event_bus
        self._dispatch_engine = dispatch_engine
        self._environment = environment
        self._node_name = node_name
        self._service = service
        self._version = version
        self._result_applier = result_applier
        self._idempotency_store = idempotency_store
        self._idempotency_config = idempotency_config or ModelIdempotencyConfig()
        self._dlq_config = dlq_config or ModelDlqConfig()
        self._retry_config = retry_config or ModelConsumerRetryConfig.create_standard()
        self._offset_policy = offset_policy or ModelOffsetPolicyConfig()
        self._unsubscribe_callables: list[Callable[[], Awaitable[None]]] = []
        self._logger = logging.getLogger(__name__)
        # Track retry attempts per correlation_id for infrastructure errors.
        # Bounded by _MAX_RETRY_ENTRIES to prevent unbounded growth from
        # orphaned entries (e.g., messages with None correlation_id generate
        # a new uuid4() each redelivery, never clearing the previous entry).
        self._retry_counts: dict[UUID, int] = {}
        self._MAX_RETRY_ENTRIES: int = 10_000

        # Initialize consumption counter mixin (wiring health monitoring)
        self._init_consumption_counter()

        # Canonical topic resolver - all topic resolution delegates here
        self._topic_resolver = TopicResolver()

    def resolve_topic(self, topic_suffix: str) -> str:
        """Resolve topic suffix to topic name (realm-agnostic, no environment prefix).

        Delegates to the canonical ``TopicResolver`` for centralized topic
        resolution logic. Topics are realm-agnostic in ONEX. The environment/realm
        is enforced via envelope identity, not topic naming. This enables
        cross-environment event routing when needed while maintaining proper
        isolation through identity.

        Topic suffixes from contracts follow the ONEX naming convention:
            onex.{kind}.{producer}.{event-name}.v{n}

        This method returns the topic suffix unchanged:
            onex.{kind}.{producer}.{event-name}.v{n}

        Args:
            topic_suffix: ONEX format topic suffix
                (e.g., 'onex.evt.omniintelligence.intent-classified.v1')

        Returns:
            Topic name (same as suffix, no environment prefix)
                (e.g., 'onex.evt.omniintelligence.intent-classified.v1')

        Example:
            >>> wiring = EventBusSubcontractWiring(bus, engine, "dev")
            >>> wiring.resolve_topic("onex.evt.user.created.v1")
            'onex.evt.user.created.v1'

        Note:
            Consumer groups still include environment for proper isolation.
            See wire_subscriptions() for consumer group naming.
        """
        return self._topic_resolver.resolve(topic_suffix)

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

            IMPORTANT: The same consumer_group is used for both subscriptions and
            DLQ publishing to maintain traceability. DLQ messages include the
            consumer_group that originally processed the message, enabling
            correlation during replay and debugging.

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

            # Create typed node identity for consumer group derivation
            # The event bus derives consumer group as: {env}.{service}.{node_name}.{purpose}.{version}
            node_identity = ModelNodeIdentity(
                env=self._environment,
                service=self._service,
                node_name=node_name,
                version=self._version,
            )

            # Consumer group for logging and DLQ traceability
            # Use shared helper for consistent derivation across codebase
            consumer_group = compute_consumer_group_id(
                node_identity, EnumConsumerGroupPurpose.CONSUME
            )

            # Create dispatch callback for this topic, capturing the consumer_group
            # used for this subscription to ensure DLQ messages have consistent
            # consumer_group metadata
            callback = self._create_dispatch_callback(full_topic, consumer_group)

            # Subscribe and store unsubscribe callable
            unsubscribe = await self._event_bus.subscribe(
                topic=full_topic,
                node_identity=node_identity,
                on_message=callback,
            )
            self._unsubscribe_callables.append(unsubscribe)

            self._logger.info(
                "Wired subscription: topic=%s, consumer_group=%s, node=%s",
                full_topic,
                consumer_group,
                node_name,
            )

    def _should_commit_after_handler(self) -> bool:
        """Check if offset should be committed after handler execution.

        Returns:
            True if offset_policy is commit_after_handler (at-least-once).
        """
        return self._offset_policy.commit_strategy == "commit_after_handler"

    async def _commit_offset(
        self,
        message: ProtocolEventMessage,
        correlation_id: UUID | None,
    ) -> None:
        """Commit Kafka offset for the processed message.

        Delegates to the event bus if it supports offset commits.
        This is a no-op for event buses that don't support explicit commits.

        Args:
            message: The message whose offset should be committed.
            correlation_id: Optional correlation ID for logging.
        """
        # Duck-type check for commit_offset method
        commit_fn = getattr(self._event_bus, "commit_offset", None)
        if commit_fn is not None and callable(commit_fn):
            try:
                await commit_fn(message)
                self._logger.debug(
                    "offset_committed topic=%s offset=%s correlation_id=%s",
                    getattr(message, "topic", "unknown"),
                    getattr(message, "offset", "unknown"),
                    str(correlation_id) if correlation_id else "none",
                )
            except Exception as e:
                self._logger.warning(
                    "offset_commit_failed topic=%s error=%s correlation_id=%s",
                    getattr(message, "topic", "unknown"),
                    str(e),
                    str(correlation_id) if correlation_id else "none",
                )

    async def _publish_to_dlq(
        self,
        topic: str,
        message: ProtocolEventMessage,
        error: Exception,
        correlation_id: UUID,
        error_category: str,
        consumer_group: str,
    ) -> None:
        """Publish failed message to Dead Letter Queue.

        Delegates to the event bus if it supports DLQ publishing.
        Falls back to logging if DLQ is not available.

        Args:
            topic: The original topic the message was consumed from.
            message: The message that failed processing.
            error: The exception that caused the failure.
            correlation_id: Correlation ID for tracing.
            error_category: Either "content" or "infra" for classification.
            consumer_group: The consumer group ID that was subscribed to this topic.
                This should match the group_id used in wire_subscriptions() for
                consistent traceability in DLQ messages.
        """
        if not self._dlq_config.enabled:
            self._logger.debug(
                "dlq_disabled topic=%s correlation_id=%s error_category=%s",
                topic,
                str(correlation_id),
                error_category,
            )
            return

        # Duck-type check for DLQ publish method
        publish_dlq_fn = getattr(self._event_bus, "_publish_raw_to_dlq", None)
        if publish_dlq_fn is not None and callable(publish_dlq_fn):
            try:
                await publish_dlq_fn(
                    original_topic=topic,
                    raw_msg=message,
                    error=error,
                    correlation_id=correlation_id,
                    failure_type=f"{error_category}_error",
                    consumer_group=consumer_group,
                )
                self._logger.warning(
                    "dlq_published topic=%s error_category=%s error_type=%s "
                    "correlation_id=%s",
                    topic,
                    error_category,
                    type(error).__name__,
                    str(correlation_id),
                )
            except Exception as dlq_error:
                self._logger.exception(
                    "dlq_publish_failed topic=%s error=%s correlation_id=%s",
                    topic,
                    str(dlq_error),
                    str(correlation_id),
                )
        else:
            # Fallback: log at ERROR level if DLQ not available
            self._logger.error(
                "dlq_not_available topic=%s error_category=%s error_type=%s "
                "error_message=%s correlation_id=%s",
                topic,
                error_category,
                type(error).__name__,
                str(error),
                str(correlation_id),
            )

    def _get_retry_count(self, correlation_id: UUID) -> int:
        """Get current retry count for a correlation ID.

        Args:
            correlation_id: The correlation ID to check.

        Returns:
            Current retry count (0 if not tracked).
        """
        return self._retry_counts.get(correlation_id, 0)

    def _increment_retry_count(self, correlation_id: UUID) -> int:
        """Increment retry count for a correlation ID.

        Prunes oldest entries when the dict exceeds ``_MAX_RETRY_ENTRIES``
        to prevent unbounded memory growth from orphaned entries.

        Args:
            correlation_id: The correlation ID to increment.

        Returns:
            New retry count after increment.
        """
        # Prune if over capacity — clear oldest half to amortize cost.
        # Orphaned entries have low retry counts; active retries will be
        # re-added on next failure with count reset to 1.
        if len(self._retry_counts) >= self._MAX_RETRY_ENTRIES:
            self._logger.warning(
                "retry_counts pruned: size=%d exceeded max=%d",
                len(self._retry_counts),
                self._MAX_RETRY_ENTRIES,
            )
            # Keep the most recent half (dict preserves insertion order in 3.7+)
            entries = list(self._retry_counts.items())
            half = len(entries) // 2
            self._retry_counts = dict(entries[half:])

        current = self._retry_counts.get(correlation_id, 0)
        self._retry_counts[correlation_id] = current + 1
        return current + 1

    def _clear_retry_count(self, correlation_id: UUID) -> None:
        """Clear retry count for a correlation ID after successful processing.

        Args:
            correlation_id: The correlation ID to clear.
        """
        self._retry_counts.pop(correlation_id, None)

    def _is_retry_exhausted(self, correlation_id: UUID) -> bool:
        """Check if retry budget is exhausted for a correlation ID.

        Args:
            correlation_id: The correlation ID to check.

        Returns:
            True if retry attempts exceed max_attempts from config.
        """
        return self._get_retry_count(correlation_id) >= self._retry_config.max_attempts

    def _create_dispatch_callback(
        self,
        topic: str,
        consumer_group: str,
    ) -> Callable[[ProtocolEventMessage], Awaitable[None]]:
        """Create callback that bridges Kafka consumer to dispatch engine.

        Creates an async callback function that:
        1. Receives ProtocolEventMessage from the Kafka consumer
        2. Deserializes the message value to ModelEventEnvelope, deriving
           ``event_type`` from the ONEX topic convention if not already
           present in the payload (see ``_deserialize_to_envelope``)
        3. Checks idempotency (if enabled) to skip duplicate messages
        4. Dispatches the envelope to the MessageDispatchEngine
        5. Classifies errors as content (DLQ) vs infrastructure (fail-fast)
        6. Manages offset commits based on policy

        Error Classification:
            Content Errors (ProtocolConfigurationError, ValidationError, JSONDecodeError):
                - Non-retryable (will never succeed with retry)
                - Default: DLQ and commit offset
                - Policy override via dlq_config.on_content_error

            Infrastructure Errors (RuntimeHostError and subclasses):
                - Potentially retryable (may succeed after service recovery)
                - Default: Fail-fast (no DLQ, no commit, re-raise)
                - If retry exhausted and policy allows: DLQ and commit
                - Policy override via dlq_config.on_infra_exhausted

        Args:
            topic: The full topic name for routing context in logs.
            consumer_group: The consumer group ID used for this topic subscription.
                This is passed to DLQ publishing to ensure consistent traceability
                between subscriptions and their associated DLQ messages.

        Returns:
            Async callback function compatible with event bus subscribe().
        """

        async def callback(message: ProtocolEventMessage) -> None:
            """Process incoming Kafka message and dispatch to engine."""
            envelope: ModelEventEnvelope[object] | None = None
            correlation_id: UUID = uuid4()  # Default if not in envelope

            try:
                envelope = self._deserialize_to_envelope(message, topic)
                correlation_id = envelope.correlation_id or uuid4()

                # Idempotency gate: check for duplicate messages
                if self._idempotency_store and self._idempotency_config.enabled:
                    envelope_id = envelope.envelope_id
                    if envelope_id is None:
                        # Missing envelope_id is a content error when idempotency is enabled
                        raise ProtocolConfigurationError(
                            "Envelope missing envelope_id for idempotency",
                            context=ModelInfraErrorContext.with_correlation(
                                correlation_id=correlation_id,
                                transport_type=EnumInfraTransportType.KAFKA,
                                operation="idempotency_check",
                            ),
                        )

                    is_new = await self._idempotency_store.check_and_record(
                        message_id=envelope_id,
                        domain=topic,  # Use topic as domain for namespace isolation
                        correlation_id=correlation_id,
                    )
                    if not is_new:
                        # Duplicate - skip processing but commit offset to prevent
                        # infinite redelivery. This is critical: even though we don't
                        # reprocess the message, we must advance the consumer offset.
                        self._logger.info(
                            "idempotency_skip envelope_id=%s topic=%s "
                            "correlation_id=%s node=%s reason=duplicate_message",
                            str(envelope_id),
                            topic,
                            str(correlation_id),
                            self._node_name,
                        )
                        # Commit offset for duplicate to prevent infinite redelivery
                        if self._should_commit_after_handler():
                            await self._commit_offset(message, correlation_id)
                        return  # Skip dispatch

                # Dispatch via ProtocolDispatchEngine interface
                result = await self._dispatch_engine.dispatch(topic, envelope)

                # Apply dispatch result (publish output events + delegate intents)
                if self._result_applier is not None and result is not None:
                    await self._result_applier.apply(result, correlation_id)

                # Success - commit offset if policy requires and clear retry count
                if self._should_commit_after_handler():
                    await self._commit_offset(message, correlation_id)
                self._clear_retry_count(correlation_id)

                # Record consumption for wiring health monitoring
                # Only called on SUCCESS - DLQ-routed messages are NOT counted
                await self._record_consumption(topic)

            except (json.JSONDecodeError, ValidationError) as e:
                # Content error: malformed JSON or schema validation failure
                # These are non-retryable - the message will never parse correctly
                self._logger.warning(
                    "content_error_deserialization topic=%s error_type=%s "
                    "error=%s correlation_id=%s",
                    topic,
                    type(e).__name__,
                    str(e),
                    str(correlation_id),
                )

                if self._dlq_config.on_content_error == "dlq_and_commit":
                    await self._publish_to_dlq(
                        topic, message, e, correlation_id, "content", consumer_group
                    )
                    await self._commit_offset(message, correlation_id)
                    return  # Handled - don't re-raise

                # fail_fast - wrap and re-raise
                raise ProtocolConfigurationError(
                    f"Content error: failed to deserialize message from topic '{topic}'",
                    context=ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.KAFKA,
                        operation="event_bus_deserialize",
                    ),
                ) from e

            except ProtocolConfigurationError as e:
                # Content error: already classified as non-retryable
                self._logger.warning(
                    "content_error_configuration topic=%s error=%s correlation_id=%s",
                    topic,
                    str(e),
                    str(correlation_id),
                )

                if self._dlq_config.on_content_error == "dlq_and_commit":
                    await self._publish_to_dlq(
                        topic, message, e, correlation_id, "content", consumer_group
                    )
                    await self._commit_offset(message, correlation_id)
                    return  # Handled - don't re-raise

                # fail_fast - re-raise without wrapping (already proper OnexError)
                raise

            except RuntimeHostError as e:
                # Infrastructure error: potentially retryable
                # Track retry attempts and check exhaustion
                retry_count = self._increment_retry_count(correlation_id)
                is_exhausted = self._is_retry_exhausted(correlation_id)

                # TRY400 disabled: logger.error intentional to avoid leaking stack traces
                self._logger.error(  # noqa: TRY400
                    "infra_error topic=%s error_type=%s error=%s "
                    "retry_count=%d max_attempts=%d exhausted=%s correlation_id=%s",
                    topic,
                    type(e).__name__,
                    str(e),
                    retry_count,
                    self._retry_config.max_attempts,
                    is_exhausted,
                    str(correlation_id),
                )

                if is_exhausted:
                    # Retry budget exhausted - check policy
                    if self._dlq_config.on_infra_exhausted == "dlq_and_commit":
                        await self._publish_to_dlq(
                            topic, message, e, correlation_id, "infra", consumer_group
                        )
                        await self._commit_offset(message, correlation_id)
                        self._clear_retry_count(correlation_id)
                        return  # Handled - don't re-raise

                # fail_fast (default) - re-raise without committing
                # Kafka will redeliver the message
                raise

            except Exception as e:
                # Unexpected error - classify as infrastructure error
                # This catches errors from handlers that aren't properly wrapped
                retry_count = self._increment_retry_count(correlation_id)
                is_exhausted = self._is_retry_exhausted(correlation_id)

                self._logger.exception(
                    "unexpected_error topic=%s error_type=%s error=%s "
                    "retry_count=%d exhausted=%s correlation_id=%s",
                    topic,
                    type(e).__name__,
                    str(e),
                    retry_count,
                    is_exhausted,
                    str(correlation_id),
                )

                if is_exhausted:
                    if self._dlq_config.on_infra_exhausted == "dlq_and_commit":
                        await self._publish_to_dlq(
                            topic, message, e, correlation_id, "infra", consumer_group
                        )
                        await self._commit_offset(message, correlation_id)
                        self._clear_retry_count(correlation_id)
                        return

                # Wrap in RuntimeHostError and re-raise
                raise RuntimeHostError(
                    f"Failed to dispatch message from topic '{topic}'",
                    context=ModelInfraErrorContext.with_correlation(
                        correlation_id=correlation_id,
                        transport_type=EnumInfraTransportType.KAFKA,
                        operation="event_bus_dispatch",
                    ),
                ) from e

        return callback

    @staticmethod
    def _derive_event_type_from_topic(topic: str) -> str | None:
        """Derive event_type routing key from ONEX topic naming convention.

        ONEX topics follow the convention::

            onex.{kind}.{producer}.{event-name}.v{n}

        This method extracts ``{producer}.{event-name}`` as a dot-path routing
        key suitable for the ``ModelEventEnvelope.event_type`` field.

        Args:
            topic: Full topic name following ONEX naming convention
                (e.g., ``'onex.evt.omniintelligence.intent-classified.v1'``).

        Returns:
            Derived event_type as ``'{producer}.{event-name}'``
            (e.g., ``'omniintelligence.intent-classified'``), or ``None`` if
            the topic does not follow the expected ONEX format (at least 5
            dot-separated segments starting with ``onex``).

        Example:
            >>> EventBusSubcontractWiring._derive_event_type_from_topic(
            ...     "onex.evt.omniintelligence.intent-classified.v1"
            ... )
            'omniintelligence.intent-classified'

        .. versionadded:: 0.3.0
        """
        parts = topic.split(".")
        if len(parts) >= 5 and parts[0] == "onex":
            # onex.{kind}.{producer}.{event-name}.v{n}
            producer = parts[2]
            event_name = parts[3]
            return f"{producer}.{event_name}"
        return None

    def _deserialize_to_envelope(
        self,
        message: ProtocolEventMessage,
        topic: str,
    ) -> ModelEventEnvelope[object]:
        """Deserialize Kafka message to event envelope.

        Converts the raw bytes in ProtocolEventMessage.value to a ModelEventEnvelope
        that can be processed by the dispatch engine. When the deserialized envelope
        does not already have an ``event_type`` set, derives it from the ONEX topic
        naming convention (``{producer}.{event-name}``).

        Deserialization Strategy:
            1. Decode message.value from UTF-8 bytes to string
            2. Parse JSON string to dict
            3. Validate and construct ModelEventEnvelope
            4. If ``event_type`` is None, derive from topic and set on envelope

        Args:
            message: ProtocolEventMessage from Kafka consumer containing raw bytes.
            topic: The full topic name used to derive event_type when not present
                in the envelope payload.

        Returns:
            Deserialized ModelEventEnvelope with event_type populated (either from
            the original payload or derived from the topic).

        Raises:
            json.JSONDecodeError: If message value is not valid JSON.
            ValidationError: If JSON does not match ModelEventEnvelope schema.

        .. versionchanged:: 0.3.0
            Added topic parameter and event_type derivation from ONEX topic convention.
        """
        # Decode bytes to string
        json_str = message.value.decode("utf-8")

        # Parse JSON to dict
        data = json.loads(json_str)

        # Validate and construct envelope
        envelope = ModelEventEnvelope[object].model_validate(data)

        # Propagate event_type: check the RAW data dict (not the envelope) to
        # preserve any explicit event_type from the producer. ModelEventEnvelope
        # doesn't define event_type as a field, so model_validate() strips it.
        # Use model_copy() to attach it post-deserialization.
        # NOTE: This relies on event_type NOT being a field on ModelEventEnvelope.
        # If upstream adds event_type as a model field, this raw-dict lookup must
        # be removed to avoid overriding the model's validated value.
        explicit_event_type = data.get("event_type")
        if explicit_event_type:
            envelope = envelope.model_copy(update={"event_type": explicit_event_type})
        else:
            derived_event_type = self._derive_event_type_from_topic(topic)
            if derived_event_type is not None:
                envelope = envelope.model_copy(
                    update={"event_type": derived_event_type}
                )
                self._logger.debug(
                    "event_type_derived topic=%s event_type=%s correlation_id=%s",
                    topic,
                    derived_event_type,
                    str(envelope.correlation_id) if envelope.correlation_id else "none",
                )

        return envelope

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
        self._retry_counts.clear()

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
