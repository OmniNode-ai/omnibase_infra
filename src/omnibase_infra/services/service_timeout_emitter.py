# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Timeout Emitter for emitting timeout events and updating markers.

This emitter handles the emission of timeout decision events and updates
the projection's emission markers to ensure exactly-once semantics.

The pattern is:
1. Query for overdue entities (via TimeoutScanner)
2. For each overdue entity:
   a. Emit the appropriate timeout event
   b. Update the emission marker in projection
3. If restart occurs, only emit for entities without markers

This ensures restart-safe, exactly-once timeout event emission.

Coroutine Safety:
    This emitter is stateless and delegates coroutine safety to underlying
    components (event_bus, projector). Multiple coroutines may call
    process_timeouts concurrently as long as underlying components
    support concurrent access.

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
    - OMN-944 (F1): Implement Registration Projection Schema
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.services.service_timeout_scanner import ServiceTimeoutScanner

if TYPE_CHECKING:
    # Import protocols inside TYPE_CHECKING to avoid circular imports.
    # ProtocolEventBus is used only for type annotations.
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
    from omnibase_infra.runtime.projector_shell import ProjectorShell

logger = logging.getLogger(__name__)


class ModelTimeoutEmissionResult(BaseModel):
    """Result of timeout emission processing.

    Captures statistics about the timeout emission process for observability
    and monitoring. This model is returned by process_timeouts() to inform
    callers about what was emitted and any errors encountered.

    Counter Semantics (Important):
        All counters track **fully completed operations**, not partial successes.
        An operation is counted as successful only when BOTH the event publish
        AND the marker update succeed. This is intentional for exactly-once
        semantics from the system's perspective.

        If event publish succeeds but marker update fails:
        - The counter is NOT incremented (operation treated as failed)
        - The event WAS published to Kafka (at-least-once delivery)
        - The entity will be re-processed on next tick (marker still NULL)
        - Downstream consumers should deduplicate by event_id if needed

        This atomic counting ensures that `markers_updated` always equals
        `ack_timeouts_emitted + liveness_expirations_emitted` for successful
        operations, making monitoring and alerting straightforward.

    Attributes:
        ack_timeouts_emitted: Number of ack timeout operations fully completed
            (event published AND marker updated). Does not count partial
            successes where publish succeeded but marker update failed.
        liveness_expirations_emitted: Number of liveness expiry operations
            fully completed (event published AND marker updated).
        markers_updated: Number of projection markers successfully updated.
            Always equals ack_timeouts_emitted + liveness_expirations_emitted
            for successful operations.
        errors: Tuple of error messages for failed emissions (immutable for
            thread safety). Each error includes the node_id and reason for
            failure. Note: an error may indicate the event was published but
            marker update failed.
        processing_time_ms: Total processing time in milliseconds.
        tick_id: The RuntimeTick ID that triggered this processing.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> result = await emitter.process_timeouts(now=tick.now, tick_id=tick.tick_id)
        >>> print(f"Emitted {result.ack_timeouts_emitted} ack timeouts")
        >>> if result.errors:
        ...     print(f"Errors: {result.errors}")
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    ack_timeouts_emitted: int = Field(
        default=0,
        ge=0,
        description="Number of ack timeout operations fully completed (publish + marker update)",
    )
    liveness_expirations_emitted: int = Field(
        default=0,
        ge=0,
        description="Number of liveness expiry operations fully completed (publish + marker update)",
    )
    markers_updated: int = Field(
        default=0,
        ge=0,
        description="Number of projection markers successfully updated (equals sum of emitted counts)",
    )
    errors: tuple[str, ...] = Field(
        default=(),
        description="Tuple of error messages for failed emissions (immutable for thread safety)",
    )

    @field_validator("errors", mode="before")
    @classmethod
    def _coerce_errors_to_tuple(cls, v: object) -> tuple[str, ...]:
        """Convert list/sequence to tuple for immutability.

        Args:
            v: The input value to coerce.

        Returns:
            A tuple of error strings.

        Raises:
            ValueError: If input is not a valid sequence type.
        """
        # NOTE: isinstance checks validate runtime type, but mypy cannot narrow
        # the generic Sequence type to tuple[str, ...] in this validator context.
        if isinstance(v, tuple):
            return v  # type: ignore[return-value]  # NOTE: runtime type validated above
        if isinstance(v, Sequence) and not isinstance(v, str | bytes):
            return tuple(v)  # type: ignore[return-value]  # NOTE: runtime type validated above
        raise ValueError(
            f"errors must be a tuple or Sequence (excluding str/bytes), "
            f"got {type(v).__name__}"
        )

    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in milliseconds",
    )
    tick_id: UUID = Field(
        ...,
        description="The RuntimeTick ID that triggered this processing",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )

    @property
    def total_emitted(self) -> int:
        """Return total number of events emitted."""
        return self.ack_timeouts_emitted + self.liveness_expirations_emitted

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during processing."""
        return len(self.errors) > 0


class ModelTimeoutEmissionConfig(BaseModel):
    """Configuration for TimeoutEmitter.

    Encapsulates configuration parameters for timeout emission processing,
    including environment and namespace for topic routing.

    Attributes:
        environment: Environment identifier for topic routing (e.g., "local", "dev", "prod").
        namespace: Namespace for topic routing (e.g., "onex", "myapp").
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    environment: str = Field(
        default="local",
        description="Environment for topic routing",
    )
    namespace: str = Field(
        default="onex",
        description="Namespace for topic routing",
    )


class ServiceTimeoutEmitter:
    """Emitter for timeout events with emission marker updates.

    Ensures restart-safe, exactly-once timeout event emission by:
    1. Only processing entities without emission markers
    2. Updating markers atomically after event emission
    3. Using correlation_id from RuntimeTick for tracing

    The emission pattern guarantees:
    - Events are emitted BEFORE markers are updated (event-first semantics)
    - If emit succeeds but marker fails, event is duplicated on retry
        (at-least-once delivery, deduplicated by downstream consumers)
    - If emit fails, marker stays NULL and will be retried on next tick

    Design Note:
        This emitter does NOT implement circuit breaker - it delegates to
        the underlying event_bus and projector which have their own circuit
        breaker implementations.

    Usage:
        >>> emitter = ServiceTimeoutEmitter(
        ...     container=container,
        ...     timeout_query=timeout_scanner,
        ...     event_bus=event_bus,
        ...     projector=projector,
        ...     config=ModelTimeoutEmissionConfig(environment="dev", namespace="myapp"),
        ... )
        >>> result = await emitter.process_timeouts(
        ...     now=tick.now,
        ...     tick_id=tick.tick_id,
        ...     correlation_id=tick.correlation_id,
        ... )

    Raises:
        InfraConnectionError: If database or Kafka connection fails
        InfraTimeoutError: If operations time out
        InfraUnavailableError: If circuit breaker is open
    """

    # Default topic patterns following ONEX conventions
    DEFAULT_ACK_TIMEOUT_TOPIC = (
        "{env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1"
    )
    DEFAULT_LIVENESS_EXPIRED_TOPIC = (
        "{env}.{namespace}.onex.evt.node-liveness-expired.v1"
    )

    # Error rate threshold for systemic issue detection.
    # When more than 50% of emissions fail in a single batch, this indicates
    # a likely systemic issue (e.g., Kafka unavailable, database down) rather
    # than isolated entity-specific failures. This threshold was chosen because:
    #
    # 1. Low false positive rate: Individual entity failures (e.g., validation
    #    errors) typically affect <10% of entities, so 50% clearly indicates
    #    infrastructure problems.
    #
    # 2. Early detection: 50% is high enough to avoid noise but low enough to
    #    catch issues before the entire batch fails.
    #
    # 3. Operator actionability: At 50%+, operators should investigate the
    #    underlying infrastructure rather than individual entities.
    #
    # This threshold triggers an ERROR-level log for alerting and monitoring.
    ERROR_RATE_THRESHOLD: float = 0.5

    def __init__(
        self,
        container: ModelONEXContainer,
        timeout_query: ServiceTimeoutScanner,
        event_bus: ProtocolEventBus,
        projector: ProjectorShell,
        config: ModelTimeoutEmissionConfig | None = None,
    ) -> None:
        """Initialize with required dependencies.

        Args:
            container: ONEX container for dependency injection.
            timeout_query: Scanner for querying overdue entities.
                Must be initialized with a ProjectionReaderRegistration.
            event_bus: Event bus for publishing timeout events.
                Must implement ProtocolEventBus (publish_envelope method).
            projector: ProjectorShell for updating emission markers.
                Should be loaded from the registration projector contract.
            config: Configuration for environment and namespace.
                Defaults to ModelTimeoutEmissionConfig() if not provided.

        Example:
            >>> reader = ProjectionReaderRegistration(pool)
            >>> timeout_query = ServiceTimeoutScanner(container, reader)
            >>> bus = EventBusKafka.default()
            >>> projector = projector_loader.load("registration_projector")
            >>> emitter = ServiceTimeoutEmitter(
            ...     container=container,
            ...     timeout_query=timeout_query,
            ...     event_bus=bus,
            ...     projector=projector,
            ...     config=ModelTimeoutEmissionConfig(environment="dev"),
            ... )
        """
        self._container = container
        self._timeout_query = timeout_query
        self._event_bus = event_bus
        self._projector = projector
        self._config = config or ModelTimeoutEmissionConfig()

    @property
    def environment(self) -> str:
        """Return configured environment."""
        return self._config.environment

    @property
    def namespace(self) -> str:
        """Return configured namespace."""
        return self._config.namespace

    def _build_topic(self, topic_pattern: str) -> str:
        """Build topic name from pattern with environment and namespace.

        Args:
            topic_pattern: Topic pattern with {env} and {namespace} placeholders.

        Returns:
            Fully qualified topic name.
        """
        return topic_pattern.format(
            env=self._config.environment,
            namespace=self._config.namespace,
        )

    async def process_timeouts(
        self,
        now: datetime,
        tick_id: UUID,
        correlation_id: UUID,
        domain: str = "registration",
    ) -> ModelTimeoutEmissionResult:
        """Process all pending timeouts.

        Queries for overdue entities, emits timeout events for each,
        and updates emission markers to prevent duplicate emissions.

        The processing order is:
        1. Query for overdue ack and liveness entities
        2. For each ack timeout: emit event, then update marker
        3. For each liveness expiration: emit event, then update marker
        4. Capture any errors but continue processing remaining entities

        Args:
            now: Injected current time from RuntimeTick. This is the
                deterministic time used for detecting overdue entities.
            tick_id: RuntimeTick ID (becomes causation_id for emitted events).
                Links emitted events to the tick that triggered them.
            correlation_id: Correlation ID for distributed tracing.
                Propagated to all emitted events.
            domain: Domain namespace for topic routing. Defaults to "registration".

        Returns:
            ModelTimeoutEmissionResult with counts and any errors.
            Errors are captured but do not stop processing of remaining entities.

        Raises:
            InfraConnectionError: If database or Kafka connection fails during query
            InfraTimeoutError: If query or emit operations time out
            InfraUnavailableError: If circuit breaker is open

        Example:
            >>> result = await emitter.process_timeouts(
            ...     now=datetime.now(UTC),
            ...     tick_id=uuid4(),
            ...     correlation_id=uuid4(),
            ... )
            >>> print(f"Emitted {result.total_emitted} timeout events")
        """
        start_time = time.perf_counter()
        errors: list[str] = []
        ack_emitted = 0
        liveness_emitted = 0
        markers_updated = 0

        logger.debug(
            "Processing timeouts",
            extra={
                "now": now.isoformat(),
                "tick_id": str(tick_id),
                "correlation_id": str(correlation_id),
                "domain": domain,
            },
        )

        # Query for overdue entities
        query_result = await self._timeout_query.find_overdue_entities(
            now=now,
            domain=domain,
            correlation_id=correlation_id,
        )

        logger.debug(
            "Found overdue entities",
            extra={
                "ack_timeout_count": len(query_result.ack_timeouts),
                "liveness_expiration_count": len(query_result.liveness_expirations),
                "correlation_id": str(correlation_id),
            },
        )

        # Process ack timeouts
        for projection in query_result.ack_timeouts:
            try:
                await self._emit_ack_timeout(
                    projection=projection,
                    detected_at=now,
                    tick_id=tick_id,
                    correlation_id=correlation_id,
                )
                # Counters increment ONLY after full operation succeeds (publish + marker).
                # If marker update fails, _emit_ack_timeout raises and we skip these lines.
                # See ModelTimeoutEmissionResult docstring for counter semantics.
                ack_emitted += 1
                markers_updated += 1
            except Exception as e:
                error_msg = (
                    f"ack_timeout failed for node {projection.entity_id}: "
                    f"{type(e).__name__}"
                )
                errors.append(error_msg)
                logger.warning(
                    error_msg,
                    extra={
                        "node_id": str(projection.entity_id),
                        "correlation_id": str(correlation_id),
                        "error_type": type(e).__name__,
                    },
                )

        # Process liveness expirations
        for projection in query_result.liveness_expirations:
            try:
                await self._emit_liveness_expiration(
                    projection=projection,
                    detected_at=now,
                    tick_id=tick_id,
                    correlation_id=correlation_id,
                )
                # Counters increment ONLY after full operation succeeds (publish + marker).
                # If marker update fails, _emit_liveness_expiration raises and we skip these lines.
                # See ModelTimeoutEmissionResult docstring for counter semantics.
                liveness_emitted += 1
                markers_updated += 1
            except Exception as e:
                error_msg = (
                    f"liveness_expiration failed for node {projection.entity_id}: "
                    f"{type(e).__name__}"
                )
                errors.append(error_msg)
                logger.warning(
                    error_msg,
                    extra={
                        "node_id": str(projection.entity_id),
                        "correlation_id": str(correlation_id),
                        "error_type": type(e).__name__,
                    },
                )

        # Check error rate for systemic issue detection.
        # See ERROR_RATE_THRESHOLD documentation for threshold rationale.
        total_attempted = len(query_result.ack_timeouts) + len(
            query_result.liveness_expirations
        )
        if total_attempted > 0:
            error_rate = len(errors) / total_attempted
            if error_rate > self.ERROR_RATE_THRESHOLD:
                logger.error(
                    "High timeout emission failure rate - possible systemic issue",
                    extra={
                        "error_rate": error_rate,
                        "error_rate_threshold": self.ERROR_RATE_THRESHOLD,
                        "total_attempted": total_attempted,
                        "error_count": len(errors),
                        "correlation_id": str(correlation_id),
                    },
                )

        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000.0

        logger.info(
            "Timeout processing completed",
            extra={
                "ack_timeouts_emitted": ack_emitted,
                "liveness_expirations_emitted": liveness_emitted,
                "markers_updated": markers_updated,
                "error_count": len(errors),
                "processing_time_ms": duration_ms,
                "correlation_id": str(correlation_id),
            },
        )

        return ModelTimeoutEmissionResult(
            ack_timeouts_emitted=ack_emitted,
            liveness_expirations_emitted=liveness_emitted,
            markers_updated=markers_updated,
            errors=errors,
            processing_time_ms=duration_ms,
            tick_id=tick_id,
            correlation_id=correlation_id,
        )

    async def _emit_ack_timeout(
        self,
        projection: ModelRegistrationProjection,
        detected_at: datetime,
        tick_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Emit ack timeout event and update marker.

        This method follows the event-first pattern:
        1. Create and emit the timeout event
        2. Update the emission marker in projection atomically

        If the emit succeeds but marker update fails, the event will be
        duplicated on retry (at-least-once semantics). Downstream consumers
        should deduplicate by event_id if needed.

        Args:
            projection: The projection for the node that timed out.
            detected_at: When the timeout was detected (from RuntimeTick.now).
            tick_id: RuntimeTick ID (becomes causation_id for the event).
            correlation_id: Correlation ID for distributed tracing.

        Raises:
            ProtocolConfigurationError: If ack_deadline is None (invalid projection state)
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors
        """
        # Validate ack_deadline exists (should always be present for timeout candidates)
        if projection.ack_deadline is None:
            raise ProtocolConfigurationError(
                f"Cannot emit ack timeout for node {projection.entity_id}: "
                "ack_deadline is None",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="emit_ack_timeout",
                    target_name="registration_projection",
                    correlation_id=correlation_id,
                ),
            )

        # Runtime import to avoid circular import (see TYPE_CHECKING block)
        from omnibase_infra.models.registration.events.model_node_registration_ack_timed_out import (
            ModelNodeRegistrationAckTimedOut,
        )

        # 1. Create event
        event = ModelNodeRegistrationAckTimedOut(
            entity_id=projection.entity_id,
            node_id=projection.entity_id,
            correlation_id=correlation_id,
            causation_id=tick_id,
            emitted_at=detected_at,
            deadline_at=projection.ack_deadline,
            previous_state=projection.current_state,
        )

        # 2. Build topic and publish event
        topic = self._build_topic(self.DEFAULT_ACK_TIMEOUT_TOPIC)

        logger.debug(
            "Emitting ack timeout event",
            extra={
                "node_id": str(projection.entity_id),
                "topic": topic,
                "correlation_id": str(correlation_id),
            },
        )

        # Wrap event in ModelEventEnvelope for protocol compliance
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=event,
            correlation_id=correlation_id,
        )
        await self._event_bus.publish_envelope(
            envelope=envelope,  # type: ignore[arg-type]
            topic=topic,
        )

        # 3. Update emission marker atomically
        # This MUST happen AFTER successful publish to ensure exactly-once semantics
        # Uses atomic marker update to avoid race conditions with concurrent updates
        await self._projector.partial_update(
            aggregate_id=projection.entity_id,
            updates={
                "ack_timeout_emitted_at": detected_at,
                "updated_at": detected_at,
            },
            correlation_id=correlation_id,
        )

    async def _emit_liveness_expiration(
        self,
        projection: ModelRegistrationProjection,
        detected_at: datetime,
        tick_id: UUID,
        correlation_id: UUID,
    ) -> None:
        """Emit liveness expiration event and update marker.

        This method follows the event-first pattern:
        1. Create and emit the expiration event
        2. Update the emission marker in projection atomically

        If the emit succeeds but marker update fails, the event will be
        duplicated on retry (at-least-once semantics). Downstream consumers
        should deduplicate by event_id if needed.

        Args:
            projection: The projection for the node whose liveness expired.
            detected_at: When the expiration was detected (from RuntimeTick.now).
            tick_id: RuntimeTick ID (becomes causation_id for the event).
            correlation_id: Correlation ID for distributed tracing.

        Raises:
            ProtocolConfigurationError: If liveness_deadline is None (invalid projection state)
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors
        """
        # Validate liveness_deadline exists
        if projection.liveness_deadline is None:
            raise ProtocolConfigurationError(
                f"Cannot emit liveness expiration for node {projection.entity_id}: "
                "liveness_deadline is None",
                context=ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.DATABASE,
                    operation="emit_liveness_expiration",
                    target_name="registration_projection",
                    correlation_id=correlation_id,
                ),
            )

        # Runtime import to avoid circular import (see TYPE_CHECKING block)
        from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_liveness_expired import (
            ModelNodeLivenessExpired,
        )

        # 1. Create event
        # last_heartbeat_at: None if no heartbeats were ever received.
        # The projection tracks this field explicitly.
        event = ModelNodeLivenessExpired(
            node_id=projection.entity_id,
            liveness_deadline=projection.liveness_deadline,
            detected_at=detected_at,
            last_heartbeat_at=projection.last_heartbeat_at,
            correlation_id=correlation_id,
            causation_id=tick_id,
        )

        # 2. Build topic and publish event
        topic = self._build_topic(self.DEFAULT_LIVENESS_EXPIRED_TOPIC)

        logger.debug(
            "Emitting liveness expiration event",
            extra={
                "node_id": str(projection.entity_id),
                "topic": topic,
                "correlation_id": str(correlation_id),
            },
        )

        # Wrap event in ModelEventEnvelope for protocol compliance
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=event,
            correlation_id=correlation_id,
        )
        await self._event_bus.publish_envelope(
            envelope=envelope,  # type: ignore[arg-type]
            topic=topic,
        )

        # 3. Update emission marker atomically
        # This MUST happen AFTER successful publish to ensure exactly-once semantics
        # Uses atomic marker update to avoid race conditions with concurrent updates
        await self._projector.partial_update(
            aggregate_id=projection.entity_id,
            updates={
                "liveness_timeout_emitted_at": detected_at,
                "updated_at": detected_at,
            },
            correlation_id=correlation_id,
        )


__all__: list[str] = [
    "ModelTimeoutEmissionConfig",
    "ModelTimeoutEmissionResult",
    "ServiceTimeoutEmitter",
]
