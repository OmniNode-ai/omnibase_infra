# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Timeout Emission Service for emitting timeout events and updating markers.

This service handles the emission of timeout decision events and updates
the projection's emission markers to ensure exactly-once semantics.

The pattern is:
1. Query for overdue entities (via ServiceTimeoutQuery)
2. For each overdue entity:
   a. Emit the appropriate timeout event
   b. Update the emission marker in projection
3. If restart occurs, only emit for entities without markers

This ensures restart-safe, exactly-once timeout event emission.

Thread Safety:
    This service is stateless and delegates thread safety to underlying
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
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.mixins import ProtocolEventBusLike
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.projectors.projector_registration import ProjectorRegistration
from omnibase_infra.services.service_timeout_query import ServiceTimeoutQuery

if TYPE_CHECKING:
    # Import models inside TYPE_CHECKING to avoid circular import.
    # The circular import occurs because:
    # 1. services/__init__.py imports service_timeout_emission
    # 2. service_timeout_emission imports from node_registration_orchestrator.models
    # 3. node_registration_orchestrator/__init__.py imports handler_timeout
    # 4. handler_timeout imports from services (which is partially initialized)
    #
    # Using TYPE_CHECKING defers the import until type-checking time only.
    # The actual model classes are imported at runtime inside the methods.
    from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_liveness_expired import (
        ModelNodeLivenessExpired as ModelNodeLivenessExpiredType,
    )
    from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_registration_ack_timed_out import (
        ModelNodeRegistrationAckTimedOut as ModelNodeRegistrationAckTimedOutType,
    )

logger = logging.getLogger(__name__)


class ModelTimeoutEmissionResult(BaseModel):
    """Result of timeout emission processing.

    Captures statistics about the timeout emission process for observability
    and monitoring. This model is returned by process_timeouts() to inform
    callers about what was emitted and any errors encountered.

    Attributes:
        ack_timeouts_emitted: Number of ack timeout events successfully emitted.
        liveness_expirations_emitted: Number of liveness expiry events emitted.
        markers_updated: Number of projection markers successfully updated.
        errors: List of error messages for failed emissions. Each error
            includes the node_id and reason for failure.
        processing_time_ms: Total processing time in milliseconds.
        tick_id: The RuntimeTick ID that triggered this processing.
        correlation_id: Correlation ID for distributed tracing.

    Example:
        >>> result = await service.process_timeouts(now=tick.now, tick_id=tick.tick_id)
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
        description="Number of ack timeout events successfully emitted",
    )
    liveness_expirations_emitted: int = Field(
        default=0,
        ge=0,
        description="Number of liveness expiry events successfully emitted",
    )
    markers_updated: int = Field(
        default=0,
        ge=0,
        description="Number of projection markers successfully updated",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Error messages for failed emissions",
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


class ServiceTimeoutEmission:
    """Service for emitting timeout events and updating emission markers.

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
        This service does NOT implement circuit breaker - it delegates to
        the underlying event_bus and projector which have their own circuit
        breaker implementations.

    Usage:
        >>> service = ServiceTimeoutEmission(
        ...     timeout_query=timeout_query_service,
        ...     event_bus=event_bus,
        ...     projector=projector,
        ...     environment="dev",
        ...     namespace="myapp",
        ... )
        >>> result = await service.process_timeouts(
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
    DEFAULT_ACK_TIMEOUT_TOPIC = "{env}.{namespace}.onex.evt.node-registration-ack-timed-out.v1"
    DEFAULT_LIVENESS_EXPIRED_TOPIC = "{env}.{namespace}.onex.evt.node-liveness-expired.v1"

    def __init__(
        self,
        timeout_query: ServiceTimeoutQuery,
        event_bus: ProtocolEventBusLike,
        projector: ProjectorRegistration,
        environment: str = "local",
        namespace: str = "onex",
    ) -> None:
        """Initialize with required dependencies.

        Args:
            timeout_query: Service for querying overdue entities.
                Must be initialized with a ProjectionReaderRegistration.
            event_bus: Event bus for publishing timeout events.
                Must implement ProtocolEventBusLike (publish_envelope method).
            projector: Projector for updating emission markers.
                Must be initialized with an asyncpg connection pool.
            environment: Environment identifier for topic routing.
                Defaults to "local".
            namespace: Namespace for topic routing. Defaults to "onex".

        Example:
            >>> reader = ProjectionReaderRegistration(pool)
            >>> timeout_query = ServiceTimeoutQuery(reader)
            >>> bus = KafkaEventBus.default()
            >>> projector = ProjectorRegistration(pool)
            >>> service = ServiceTimeoutEmission(
            ...     timeout_query=timeout_query,
            ...     event_bus=bus,
            ...     projector=projector,
            ...     environment="dev",
            ... )
        """
        self._timeout_query = timeout_query
        self._event_bus = event_bus
        self._projector = projector
        self._environment = environment
        self._namespace = namespace

    @property
    def environment(self) -> str:
        """Return configured environment."""
        return self._environment

    @property
    def namespace(self) -> str:
        """Return configured namespace."""
        return self._namespace

    def _build_topic(self, topic_pattern: str) -> str:
        """Build topic name from pattern with environment and namespace.

        Args:
            topic_pattern: Topic pattern with {env} and {namespace} placeholders.

        Returns:
            Fully qualified topic name.
        """
        return topic_pattern.format(
            env=self._environment,
            namespace=self._namespace,
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
            >>> result = await service.process_timeouts(
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
                ack_emitted += 1
                markers_updated += 1
            except Exception as e:
                error_msg = f"ack_timeout failed for node {projection.entity_id}: {type(e).__name__}"
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
                liveness_emitted += 1
                markers_updated += 1
            except Exception as e:
                error_msg = f"liveness_expiration failed for node {projection.entity_id}: {type(e).__name__}"
                errors.append(error_msg)
                logger.warning(
                    error_msg,
                    extra={
                        "node_id": str(projection.entity_id),
                        "correlation_id": str(correlation_id),
                        "error_type": type(e).__name__,
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
        2. Update the emission marker in projection

        If the emit succeeds but marker update fails, the event will be
        duplicated on retry (at-least-once semantics). Downstream consumers
        should deduplicate by event_id if needed.

        Args:
            projection: The projection for the node that timed out.
            detected_at: When the timeout was detected (from RuntimeTick.now).
            tick_id: RuntimeTick ID (becomes causation_id for the event).
            correlation_id: Correlation ID for distributed tracing.

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors
        """
        # Validate ack_deadline exists (should always be present for timeout candidates)
        if projection.ack_deadline is None:
            raise ValueError(
                f"Cannot emit ack timeout for node {projection.entity_id}: ack_deadline is None"
            )

        # Runtime import to avoid circular import (see TYPE_CHECKING block for explanation)
        from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_registration_ack_timed_out import (
            ModelNodeRegistrationAckTimedOut,
        )

        # 1. Create event
        event = ModelNodeRegistrationAckTimedOut(
            node_id=projection.entity_id,
            ack_deadline=projection.ack_deadline,
            detected_at=detected_at,
            previous_state=projection.current_state,
            correlation_id=correlation_id,
            causation_id=tick_id,
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

        await self._event_bus.publish_envelope(
            envelope=event,
            topic=topic,
        )

        # 3. Update emission marker in projection via persist with updated model
        # This MUST happen AFTER successful publish to ensure exactly-once semantics
        updated_projection = ModelRegistrationProjection(
            entity_id=projection.entity_id,
            domain=projection.domain,
            current_state=projection.current_state,
            node_type=projection.node_type,
            node_version=projection.node_version,
            capabilities=projection.capabilities,
            ack_deadline=projection.ack_deadline,
            liveness_deadline=projection.liveness_deadline,
            ack_timeout_emitted_at=detected_at,  # Set the marker
            liveness_timeout_emitted_at=projection.liveness_timeout_emitted_at,
            last_applied_event_id=tick_id,  # Use tick_id as event_id
            last_applied_offset=projection.last_applied_offset + 1,
            last_applied_sequence=projection.last_applied_sequence,
            last_applied_partition=projection.last_applied_partition,
            registered_at=projection.registered_at,
            updated_at=detected_at,
            correlation_id=correlation_id,
        )

        sequence_info = ModelSequenceInfo(
            sequence=updated_projection.last_applied_offset,
            offset=updated_projection.last_applied_offset,
            partition=updated_projection.last_applied_partition,
        )

        await self._projector.persist(
            projection=updated_projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence_info,
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
        2. Update the emission marker in projection

        If the emit succeeds but marker update fails, the event will be
        duplicated on retry (at-least-once semantics). Downstream consumers
        should deduplicate by event_id if needed.

        Args:
            projection: The projection for the node whose liveness expired.
            detected_at: When the expiration was detected (from RuntimeTick.now).
            tick_id: RuntimeTick ID (becomes causation_id for the event).
            correlation_id: Correlation ID for distributed tracing.

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open
            RuntimeHostError: For other database errors
        """
        # Validate liveness_deadline exists
        if projection.liveness_deadline is None:
            raise ValueError(
                f"Cannot emit liveness expiration for node {projection.entity_id}: liveness_deadline is None"
            )

        # Runtime import to avoid circular import (see TYPE_CHECKING block for explanation)
        from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_liveness_expired import (
            ModelNodeLivenessExpired,
        )

        # 1. Create event
        # Note: last_heartbeat_at is not currently tracked in projection,
        # so we pass None. Future enhancement could track this.
        event = ModelNodeLivenessExpired(
            node_id=projection.entity_id,
            liveness_deadline=projection.liveness_deadline,
            detected_at=detected_at,
            last_heartbeat_at=None,
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

        await self._event_bus.publish_envelope(
            envelope=event,
            topic=topic,
        )

        # 3. Update emission marker in projection via persist with updated model
        # This MUST happen AFTER successful publish to ensure exactly-once semantics
        updated_projection = ModelRegistrationProjection(
            entity_id=projection.entity_id,
            domain=projection.domain,
            current_state=projection.current_state,
            node_type=projection.node_type,
            node_version=projection.node_version,
            capabilities=projection.capabilities,
            ack_deadline=projection.ack_deadline,
            liveness_deadline=projection.liveness_deadline,
            ack_timeout_emitted_at=projection.ack_timeout_emitted_at,
            liveness_timeout_emitted_at=detected_at,  # Set the marker
            last_applied_event_id=tick_id,  # Use tick_id as event_id
            last_applied_offset=projection.last_applied_offset + 1,
            last_applied_sequence=projection.last_applied_sequence,
            last_applied_partition=projection.last_applied_partition,
            registered_at=projection.registered_at,
            updated_at=detected_at,
            correlation_id=correlation_id,
        )

        sequence_info = ModelSequenceInfo(
            sequence=updated_projection.last_applied_offset,
            offset=updated_projection.last_applied_offset,
            partition=updated_projection.last_applied_partition,
        )

        await self._projector.persist(
            projection=updated_projection,
            entity_id=projection.entity_id,
            domain=projection.domain,
            sequence_info=sequence_info,
            correlation_id=correlation_id,
        )


__all__: list[str] = ["ModelTimeoutEmissionResult", "ServiceTimeoutEmission"]
