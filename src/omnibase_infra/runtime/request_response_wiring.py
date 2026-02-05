# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Request-response wiring for correlation-based RPC-style Kafka communication.

This module provides the RequestResponseWiring class for implementing request-response
patterns over Kafka. Unlike the standard EventBusSubcontractWiring (designed for 24/7
consumers), this wiring supports correlation-based request-response flows where a
publisher sends a request and awaits a correlated response.

Architecture:
    The RequestResponseWiring class is responsible for:
    1. Reading ModelRequestResponseConfig from contracts
    2. Creating dedicated consumers for reply topics (completed + failed)
    3. Managing correlation ID tracking with in-flight futures
    4. Injecting correlation IDs if not present in outgoing requests
    5. Matching incoming responses to pending requests via correlation ID
    6. Handling timeouts with InfraTimeoutError
    7. Circuit breaker protection for publish failures

    This follows ARCH-002: "Runtime owns all Kafka plumbing." Nodes and handlers
    declare request-response requirements in contracts but never directly interact
    with Kafka consumers or producers.

Boot Nonce:
    A per-process boot nonce (8-character hex string from UUID4) is generated once
    at module load time. This ensures consumer groups are unique per process instance,
    preventing message stealing between concurrent processes.

Consumer Group Naming:
    Consumer groups are named as: {environment}.rr.{instance_name}.{boot_nonce}
    Example: "dev.rr.code-analysis.a1b2c3d4"

    This ensures:
    - Each process instance has its own consumer group
    - Multiple instances don't steal each other's responses
    - Process restarts get new consumer groups

Correlation ID Handling:
    When sending requests, the wiring:
    1. Checks if correlation_id exists at the configured location (default: body.correlation_id)
    2. If missing, injects a new UUID4 correlation_id into the payload
    3. Returns the correlation_id in the response for tracing

Error Handling:
    - Timeout: Raises InfraTimeoutError (NOT InfraUnavailableError)
    - Circuit breaker open: Raises InfraUnavailableError
    - Publish failures: Recorded by circuit breaker, wrapped in appropriate error

Related:
    - OMN-1742: Request-response wiring for Kafka RPC patterns
    - ModelRequestResponseConfig: Contract model for request-response configuration
    - EventBusSubcontractWiring: Standard 24/7 consumer wiring (different pattern)

.. versionadded:: 0.3.1
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from aiokafka import AIOKafkaConsumer

from omnibase_core.models.contracts.subcontracts import (
    ModelCorrelationConfig,
    ModelRequestResponseConfig,
    ModelRequestResponseInstance,
)
from omnibase_core.protocols.event_bus.protocol_event_bus_publisher import (
    ProtocolEventBusPublisher,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker

if TYPE_CHECKING:
    from aiokafka import ConsumerRecord

logger = logging.getLogger(__name__)

# Boot nonce: Generated ONCE per process at module load time.
# Used to create unique consumer groups per process instance.
_BOOT_NONCE: str = uuid4().hex[:8]


@dataclass
class RequestResponseInstanceState:
    """Internal state for a single request-response instance.

    Tracks pending requests, consumer task, and consumer instance for
    a configured request-response pattern.
    """

    name: str
    request_topic: str
    completed_topic: str
    failed_topic: str
    timeout_seconds: int
    correlation_config: ModelCorrelationConfig
    consumer_group: str
    pending: dict[str, asyncio.Future[dict[str, object]]] = field(default_factory=dict)
    consumer: AIOKafkaConsumer | None = None
    consumer_task: asyncio.Task[None] | None = None


class RequestResponseWiring(MixinAsyncCircuitBreaker):
    """Wires request-response patterns to Kafka for correlation-based RPC.

    This class implements the request-response pattern over Kafka, where:
    1. A request is published to a request topic
    2. The wiring awaits a correlated response on reply topics
    3. Correlation is tracked via correlation_id in the message payload

    Unlike EventBusSubcontractWiring (designed for 24/7 consumers), this wiring
    creates ephemeral consumers that match responses to pending requests.

    Consumer Startup:
        Consumers are started eagerly when wire_request_response() is called.
        This ensures responses can be received immediately after the first request.

    Correlation ID Injection:
        If the outgoing payload lacks a correlation_id at the configured location,
        the wiring injects a new UUID4. The correlation_id is always returned
        in the response for tracing.

    Timeout Handling:
        If no response is received within the configured timeout (default: 30s),
        InfraTimeoutError is raised. Note: This is a timeout error, NOT
        InfraUnavailableError which is reserved for circuit breaker states.

    Circuit Breaker:
        Publish failures are tracked by the circuit breaker. When the circuit
        opens, InfraUnavailableError is raised immediately without attempting
        to publish.

    Thread Safety:
        This class is designed for single-threaded async use. All operations
        should be performed from a single async context.

    Example:
        ```python
        from omnibase_infra.runtime import RequestResponseWiring
        from omnibase_core.models.contracts.subcontracts import (
            ModelRequestResponseConfig,
            ModelRequestResponseInstance,
            ModelReplyTopics,
        )

        # Create wiring
        wiring = RequestResponseWiring(
            event_bus=event_bus,
            environment="dev",
            app_name="my-service",
        )

        # Wire from config
        config = ModelRequestResponseConfig(
            instances=[
                ModelRequestResponseInstance(
                    name="code-analysis",
                    request_topic="onex.cmd.intelligence.analyze-code.v1",
                    reply_topics=ModelReplyTopics(
                        completed="onex.evt.intelligence.code-analyzed.v1",
                        failed="onex.evt.intelligence.code-analysis-failed.v1",
                    ),
                    timeout_seconds=30,
                )
            ]
        )
        await wiring.wire_request_response(config)

        # Send request and await response
        response = await wiring.send_request(
            instance_name="code-analysis",
            payload={"code": "print('hello')"},
        )

        # Cleanup on shutdown
        await wiring.cleanup()
        ```

    Attributes:
        _event_bus: Event bus for publishing requests
        _environment: Environment identifier for consumer groups (e.g., 'dev', 'prod')
        _app_name: Application name for consumer group identification
        _instances: Dict mapping instance names to their state
        _bootstrap_servers: Kafka bootstrap servers from event bus

    .. versionadded:: 0.3.1
    """

    def __init__(
        self,
        event_bus: ProtocolEventBusPublisher,
        environment: str,
        app_name: str,
        bootstrap_servers: str | None = None,
    ) -> None:
        """Initialize request-response wiring.

        Args:
            event_bus: Event bus for publishing requests. Must implement
                ProtocolEventBusPublisher interface.
            environment: Environment identifier (e.g., 'dev', 'prod').
                Used for consumer group naming. Topics are realm-agnostic and
                do not include environment prefixes.
            app_name: Application name for logging and consumer group naming.
            bootstrap_servers: Kafka bootstrap servers. If not provided, attempts
                to read from event_bus._bootstrap_servers or environment variable.

        Raises:
            ValueError: If environment is empty or whitespace-only.
            ProtocolConfigurationError: If bootstrap_servers cannot be determined.
        """
        if not environment or not environment.strip():
            raise ValueError("environment must be a non-empty string")
        if not app_name or not app_name.strip():
            raise ValueError("app_name must be a non-empty string")

        self._event_bus = event_bus
        self._environment = environment
        self._app_name = app_name
        self._instances: dict[str, RequestResponseInstanceState] = {}
        self._logger = logging.getLogger(__name__)

        # Resolve bootstrap servers
        if bootstrap_servers:
            self._bootstrap_servers = bootstrap_servers
        elif hasattr(event_bus, "_bootstrap_servers"):
            self._bootstrap_servers = event_bus._bootstrap_servers  # type: ignore[union-attr]
        else:
            import os

            self._bootstrap_servers = os.environ.get(
                "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
            )

        # Initialize circuit breaker for publish protection
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name=f"request-response.{app_name}",
            transport_type=EnumInfraTransportType.KAFKA,
            half_open_successes=1,
        )

        self._logger.debug(
            "RequestResponseWiring initialized: environment=%s, app_name=%s, "
            "boot_nonce=%s, bootstrap_servers=%s",
            environment,
            app_name,
            _BOOT_NONCE,
            self._bootstrap_servers,
        )

    def resolve_topic(self, topic_suffix: str) -> str:
        """Resolve topic suffix to topic name (realm-agnostic, no environment prefix).

        Topics are realm-agnostic in ONEX. The environment/realm is enforced via
        envelope identity, not topic naming. This enables cross-environment event
        routing when needed while maintaining proper isolation through identity.

        Args:
            topic_suffix: ONEX format topic suffix
                (e.g., 'onex.cmd.intelligence.analyze-code.v1')

        Returns:
            Topic name (same as suffix, no environment prefix)
                (e.g., 'onex.cmd.intelligence.analyze-code.v1')

        Note:
            Consumer groups still include environment for proper isolation.
        """
        return topic_suffix

    async def wire_request_response(
        self,
        config: ModelRequestResponseConfig,
    ) -> None:
        """Wire request-response instances from configuration.

        Creates consumers for each instance's reply topics and starts them
        eagerly. Consumers run in background tasks, matching incoming responses
        to pending requests via correlation ID.

        Consumer Group Naming:
            Consumer groups are named as: {environment}.rr.{instance_name}.{boot_nonce}
            The boot_nonce ensures each process instance has unique consumer groups.

        Args:
            config: Request-response configuration with instance definitions.

        Raises:
            ProtocolConfigurationError: If instance name conflicts with existing.
            InfraConnectionError: If Kafka connection fails during consumer start.
        """
        for instance in config.instances:
            await self._wire_instance(instance)

    async def _wire_instance(self, instance: ModelRequestResponseInstance) -> None:
        """Wire a single request-response instance.

        Args:
            instance: Instance configuration to wire.

        Raises:
            ProtocolConfigurationError: If instance name already wired.
        """
        if instance.name in self._instances:
            raise ProtocolConfigurationError(
                f"Request-response instance '{instance.name}' already wired",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="wire_request_response",
                ),
                instance_name=instance.name,
            )

        # Build consumer group: {environment}.rr.{instance_name}.{boot_nonce}
        consumer_group = f"{self._environment}.rr.{instance.name}.{_BOOT_NONCE}"

        # Resolve topics with environment prefix
        request_topic = self.resolve_topic(instance.request_topic)
        completed_topic = self.resolve_topic(instance.reply_topics.completed)
        failed_topic = self.resolve_topic(instance.reply_topics.failed)

        # Default correlation config if not specified
        correlation_config = instance.correlation or ModelCorrelationConfig()

        # Create instance state
        rr_instance = RequestResponseInstanceState(
            name=instance.name,
            request_topic=request_topic,
            completed_topic=completed_topic,
            failed_topic=failed_topic,
            timeout_seconds=instance.timeout_seconds,
            correlation_config=correlation_config,
            consumer_group=consumer_group,
        )

        # Create consumer for reply topics
        consumer = AIOKafkaConsumer(
            completed_topic,
            failed_topic,
            bootstrap_servers=self._bootstrap_servers,
            group_id=consumer_group,
            auto_offset_reset=instance.auto_offset_reset,
            enable_auto_commit=True,
        )

        rr_instance.consumer = consumer

        # Start consumer eagerly
        await consumer.start()
        self._logger.info(
            "Started request-response consumer: instance=%s, "
            "consumer_group=%s, topics=[%s, %s]",
            instance.name,
            consumer_group,
            completed_topic,
            failed_topic,
        )

        # Start background task to process responses
        consumer_task = asyncio.create_task(
            self._consume_responses(rr_instance),
            name=f"rr-consumer-{instance.name}",
        )
        rr_instance.consumer_task = consumer_task

        # Store instance
        self._instances[instance.name] = rr_instance

    async def _consume_responses(self, instance: RequestResponseInstanceState) -> None:
        """Background task that consumes responses and resolves pending futures.

        Runs continuously until cleanup() is called. Matches incoming responses
        to pending requests via correlation ID.

        Args:
            instance: The request-response instance to consume for.
        """
        consumer = instance.consumer
        if consumer is None:
            return

        try:
            async for message in consumer:
                await self._handle_response_message(instance, message)
        except asyncio.CancelledError:
            self._logger.debug(
                "Consumer task cancelled for instance: %s",
                instance.name,
            )
            raise
        except Exception as e:
            self._logger.exception(
                "Unexpected error in consumer task for instance %s: %s",
                instance.name,
                e,
            )

    async def _handle_response_message(
        self,
        instance: RequestResponseInstanceState,
        message: ConsumerRecord,
    ) -> None:
        """Handle a single response message from Kafka.

        Extracts correlation ID and resolves the corresponding pending future.

        Args:
            instance: The request-response instance.
            message: The Kafka message received.
        """
        try:
            # Deserialize message value
            if message.value is None:
                self._logger.warning(
                    "Received empty message on topic %s, skipping",
                    message.topic,
                )
                return

            response_data: dict[str, object] = json.loads(message.value.decode("utf-8"))

            # Extract correlation ID based on config
            correlation_id = self._extract_correlation_id(
                response_data,
                instance.correlation_config,
            )

            if correlation_id is None:
                self._logger.warning(
                    "Response missing correlation_id: topic=%s, instance=%s",
                    message.topic,
                    instance.name,
                )
                return

            correlation_key = str(correlation_id)

            # Look up pending future
            future = instance.pending.pop(correlation_key, None)
            if future is None:
                self._logger.debug(
                    "Orphan response received (no pending request): "
                    "correlation_id=%s, topic=%s, instance=%s",
                    correlation_key,
                    message.topic,
                    instance.name,
                )
                return

            # Determine if this is a success or failure based on topic
            is_failure = message.topic == instance.failed_topic

            if is_failure:
                # Set exception for failed responses
                error_message = response_data.get("error", "Request failed")
                future.set_exception(RuntimeError(f"Request failed: {error_message}"))
            else:
                # Set result for successful responses
                # Include correlation_id in response for tracing
                response_data["_correlation_id"] = correlation_key
                future.set_result(response_data)

            self._logger.debug(
                "Resolved pending request: correlation_id=%s, topic=%s, "
                "is_failure=%s, instance=%s",
                correlation_key,
                message.topic,
                is_failure,
                instance.name,
            )

        except json.JSONDecodeError as e:
            self._logger.warning(
                "Failed to decode response JSON: topic=%s, error=%s",
                message.topic,
                e,
            )
        except Exception as e:
            self._logger.exception(
                "Error handling response message: topic=%s, error=%s",
                message.topic,
                e,
            )

    def _extract_correlation_id(
        self,
        data: dict[str, object],
        config: ModelCorrelationConfig,
    ) -> UUID | None:
        """Extract correlation ID from response data based on configuration.

        Args:
            data: Response data dictionary.
            config: Correlation configuration specifying location and field.

        Returns:
            Correlation ID as UUID if found, None otherwise.
        """
        value: object | None = None

        if config.location == "body":
            value = data.get(config.field)
        elif config.location == "headers":
            # Headers would be in message headers, not body
            # For now, we only support body location
            self._logger.warning(
                "Header-based correlation not implemented, falling back to body"
            )
            value = data.get(config.field)

        if value is None:
            return None

        # Parse to UUID - correlation IDs are always UUIDs
        try:
            return UUID(str(value))
        except ValueError:
            self._logger.warning(
                "Invalid correlation_id format (not a UUID): %s",
                value,
            )
            return None

    async def send_request(
        self,
        instance_name: str,
        payload: dict[str, object],
        timeout_seconds: int | None = None,
    ) -> dict[str, object]:
        """Send a request and await the correlated response.

        Publishes a request to the instance's request topic and waits for
        a response on the reply topics. If the payload lacks a correlation_id
        at the configured location, one is injected.

        Correlation ID Handling:
            - If correlation_id exists in payload: Use existing value
            - If missing: Inject new UUID4 into payload
            - Always: Return correlation_id in response (as _correlation_id)

        Args:
            instance_name: Name of the wired request-response instance.
            payload: Request payload dictionary. Modified in place to add
                correlation_id if not present.
            timeout_seconds: Override timeout for this request. If None,
                uses the instance's configured timeout (default: 30s).

        Returns:
            Response dictionary from the reply topic. Includes _correlation_id
            field for tracing.

        Raises:
            ProtocolConfigurationError: If instance_name is not wired.
            InfraTimeoutError: If no response received within timeout.
            InfraUnavailableError: If circuit breaker is open.
            RuntimeError: If request failed (response on failed topic).
        """
        # Get instance
        instance = self._instances.get(instance_name)
        if instance is None:
            raise ProtocolConfigurationError(
                f"Request-response instance '{instance_name}' not wired",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="send_request",
                ),
                instance_name=instance_name,
            )

        # Determine timeout
        timeout = (
            timeout_seconds if timeout_seconds is not None else instance.timeout_seconds
        )

        # Extract or inject correlation_id
        correlation_id = self._ensure_correlation_id(
            payload,
            instance.correlation_config,
        )
        correlation_key = str(correlation_id)

        # Create future for response
        future: asyncio.Future[dict[str, object]] = (
            asyncio.get_running_loop().create_future()
        )
        instance.pending[correlation_key] = future

        try:
            # Check circuit breaker before publish
            async with self._circuit_breaker_lock:
                await self._check_circuit_breaker(
                    operation="send_request",
                    correlation_id=correlation_id,
                )

            # Publish request
            await self._publish_request(instance, payload, correlation_id)

            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(future, timeout=timeout)

                # Record success in circuit breaker
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

                return response

            except TimeoutError:
                # Remove pending future on timeout
                instance.pending.pop(correlation_key, None)

                # Raise InfraTimeoutError (NOT InfraUnavailableError)
                timeout_context = ModelTimeoutErrorContext(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="send_request",
                    target_name=instance.request_topic,
                    correlation_id=correlation_id,
                    timeout_seconds=float(timeout),
                )
                raise InfraTimeoutError(
                    f"Request-response timeout after {timeout}s: "
                    f"instance={instance_name}, correlation_id={correlation_key}",
                    context=timeout_context,
                ) from None

        except InfraUnavailableError:
            # Circuit breaker open - re-raise without modification
            instance.pending.pop(correlation_key, None)
            raise

        except Exception:
            # Record failure in circuit breaker
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(
                    operation="send_request",
                    correlation_id=correlation_id,
                )

            # Clean up pending future
            instance.pending.pop(correlation_key, None)
            raise

    def _ensure_correlation_id(
        self,
        payload: dict[str, object],
        config: ModelCorrelationConfig,
    ) -> UUID:
        """Ensure correlation_id exists in payload, injecting if missing.

        Args:
            payload: Request payload dictionary. Modified in place.
            config: Correlation configuration.

        Returns:
            The correlation ID as UUID (existing parsed or newly generated).
        """
        existing = payload.get(config.field)

        if existing is not None:
            # Parse existing to UUID - correlation IDs are always UUIDs
            correlation_id = UUID(str(existing))
        else:
            # Generate new UUID
            correlation_id = uuid4()
            payload[config.field] = str(correlation_id)

        return correlation_id

    async def _publish_request(
        self,
        instance: RequestResponseInstanceState,
        payload: dict[str, object],
        correlation_id: UUID,
    ) -> None:
        """Publish request to the instance's request topic.

        Args:
            instance: Request-response instance.
            payload: Request payload.
            correlation_id: Correlation ID for logging and message key.
        """
        # Serialize payload
        value = json.dumps(payload).encode("utf-8")

        # Publish via event bus - convert UUID to string at serialization boundary
        await self._event_bus.publish(
            topic=instance.request_topic,
            key=str(correlation_id).encode("utf-8"),
            value=value,
        )

        self._logger.debug(
            "Published request: topic=%s, correlation_id=%s, instance=%s",
            instance.request_topic,
            correlation_id,
            instance.name,
        )

    async def cleanup(self) -> None:
        """Clean up all request-response instances.

        Cancels all consumer tasks, stops consumers, and clears pending futures
        with exceptions. Should be called during runtime shutdown.

        This method is safe to call multiple times - subsequent calls are no-ops.
        """
        cleanup_count = len(self._instances)
        if cleanup_count == 0:
            return

        for instance_name, instance in list(self._instances.items()):
            await self._cleanup_instance(instance)

        self._instances.clear()
        self._logger.info(
            "Cleaned up %d request-response instance(s)",
            cleanup_count,
        )

    async def _cleanup_instance(self, instance: RequestResponseInstanceState) -> None:
        """Clean up a single request-response instance.

        Args:
            instance: Instance to clean up.
        """
        # Cancel consumer task
        if instance.consumer_task is not None and not instance.consumer_task.done():
            instance.consumer_task.cancel()
            try:
                await instance.consumer_task
            except asyncio.CancelledError:
                pass

        # Stop consumer
        if instance.consumer is not None:
            try:
                await instance.consumer.stop()
            except Exception as e:
                self._logger.warning(
                    "Error stopping consumer for instance %s: %s",
                    instance.name,
                    e,
                )

        # Fail all pending futures
        cleanup_error = RuntimeError(
            f"Request-response instance '{instance.name}' was cleaned up"
        )
        for correlation_key, future in instance.pending.items():
            if not future.done():
                future.set_exception(cleanup_error)
                self._logger.debug(
                    "Failed pending request on cleanup: correlation_id=%s, instance=%s",
                    correlation_key,
                    instance.name,
                )

        instance.pending.clear()
        self._logger.debug(
            "Cleaned up instance: %s",
            instance.name,
        )

    def get_boot_nonce(self) -> str:
        """Return the boot nonce for this process.

        Useful for debugging and logging consumer group identification.

        Returns:
            8-character hex string unique to this process instance.
        """
        return _BOOT_NONCE


__all__: list[str] = [
    "RequestResponseWiring",
]
