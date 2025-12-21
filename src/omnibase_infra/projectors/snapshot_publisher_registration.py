# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot Publisher for Registration Projections.

Publishes compacted snapshots to Kafka for read optimization. Snapshots are
derived from projections and NEVER replace the event log. The event log
remains the absolute source of truth.

Architecture Overview:
    This service implements F2 (Snapshot Publishing) of the ONEX registration
    projection pipeline:

    1. Projectors (F1) persist projections to PostgreSQL via ProjectorRegistration
    2. Snapshot Publisher (F2) reads projections and publishes compacted snapshots
    3. Consumers read snapshots for fast O(1) state queries

    ```
    Events -> Projector -> PostgreSQL -> Snapshot Publisher -> Kafka (compacted)
                                              |
                                              v
                                    Orchestrators/Readers
    ```

Design Principles:
    - **Read Optimization Only**: Snapshots are for fast reads, not data integrity
    - **Kafka Compaction**: Only latest snapshot per entity_id retained
    - **Tombstone Support**: Null values delete snapshots during compaction
    - **Version Tracking**: Monotonic versions for conflict resolution
    - **Circuit Breaker**: Resilience against Kafka failures

Thread Safety:
    This implementation is thread-safe for concurrent publishing.
    Uses asyncio locks for circuit breaker state management.

Error Handling:
    All methods raise ONEX error types:
    - InfraConnectionError: Kafka unavailable or connection failed
    - InfraTimeoutError: Publish operation timed out
    - InfraUnavailableError: Circuit breaker open

Example Usage:
    ```python
    from aiokafka import AIOKafkaProducer
    from omnibase_infra.projectors import SnapshotPublisherRegistration
    from omnibase_infra.models.projection import ModelSnapshotTopicConfig

    # Create producer and config
    producer = AIOKafkaProducer(bootstrap_servers="localhost:9092")
    config = ModelSnapshotTopicConfig.default()

    # Initialize publisher
    publisher = SnapshotPublisherRegistration(producer, config)
    await publisher.start()

    try:
        # Publish snapshot from projection
        snapshot = await publisher.publish_from_projection(projection)
        print(f"Published snapshot version {snapshot.snapshot_version}")

        # Or publish pre-built snapshot
        await publisher.publish_snapshot(snapshot)

        # Batch publish
        count = await publisher.publish_batch(snapshots)
        print(f"Published {count} snapshots")

        # Delete snapshot (tombstone)
        await publisher.delete_snapshot("entity-123", "registration")
    finally:
        await publisher.stop()
    ```

Performance Considerations:
    - Use publish_batch for bulk operations (e.g., periodic snapshot jobs)
    - Consider publish_from_projection for single updates (handles versioning)
    - Tombstones are cheap - use delete_snapshot for permanent removals
    - Monitor circuit breaker state for Kafka health

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model

See Also:
    - ProtocolSnapshotPublisher: Protocol definition for snapshot publishers
    - ModelRegistrationSnapshot: Snapshot model definition
    - ModelSnapshotTopicConfig: Topic configuration for compacted topics
    - ProjectorRegistration: Projection persistence (source for snapshots)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelRegistrationSnapshot,
    ModelSnapshotTopicConfig,
)

if TYPE_CHECKING:
    from aiokafka import AIOKafkaProducer

logger = logging.getLogger(__name__)


class SnapshotPublisherRegistration(MixinAsyncCircuitBreaker):
    """Publishes registration snapshots to a compacted Kafka topic.

    This service reads registration projections and publishes them as
    optimized snapshots to a Kafka compacted topic. Kafka compaction
    ensures only the latest snapshot per entity is retained, enabling
    fast state reconstruction without replaying events.

    The publisher implements ProtocolSnapshotPublisher for structural
    typing compatibility, allowing it to be used wherever the protocol
    is expected.

    Compaction Semantics:
        - Key: "{domain}:{entity_id}" (e.g., "registration:uuid-here")
        - Value: JSON-serialized ModelRegistrationSnapshot
        - Tombstone: null value deletes the key during compaction
        - After compaction: only latest snapshot per key survives

    Circuit Breaker:
        Uses MixinAsyncCircuitBreaker for resilience:
        - Opens after 5 consecutive failures
        - Resets after 60 seconds
        - Raises InfraUnavailableError when open

    Version Tracking:
        The publisher maintains a version tracker per entity to ensure
        monotonically increasing snapshot versions. This enables conflict
        resolution and ordering guarantees during compaction.

    NOTE: Snapshots are for READ OPTIMIZATION only. The immutable event
    log remains the authoritative source of truth. Snapshots can be
    regenerated from the event log at any time.

    Attributes:
        _producer: Kafka producer for publishing snapshots
        _config: Snapshot topic configuration
        _version_tracker: Dict tracking versions per entity
        _started: Whether the publisher has been started

    Example:
        >>> config = ModelSnapshotTopicConfig.default()
        >>> publisher = SnapshotPublisherRegistration(producer, config)
        >>> await publisher.start()
        >>>
        >>> # Publish snapshot from projection
        >>> snapshot = await publisher.publish_from_projection(projection)
        >>>
        >>> # Or publish existing snapshot
        >>> await publisher.publish_snapshot(snapshot)
        >>>
        >>> await publisher.stop()
    """

    def __init__(
        self,
        producer: AIOKafkaProducer,
        config: ModelSnapshotTopicConfig,
        *,
        snapshot_version_tracker: dict[str, int] | None = None,
    ) -> None:
        """Initialize snapshot publisher.

        Args:
            producer: AIOKafka producer for publishing snapshots. The producer
                should be configured for the target Kafka cluster but NOT
                started - the publisher will manage its lifecycle.
            config: Snapshot topic configuration defining the target topic
                and compaction settings.
            snapshot_version_tracker: Optional dict to track versions per entity.
                If not provided, a new dict is created internally. Useful for
                sharing version state across multiple publishers or for testing.

        Example:
            >>> producer = AIOKafkaProducer(
            ...     bootstrap_servers="localhost:9092",
            ...     value_serializer=lambda v: v,  # Publisher handles serialization
            ... )
            >>> config = ModelSnapshotTopicConfig.default()
            >>> publisher = SnapshotPublisherRegistration(producer, config)
        """
        self._producer = producer
        self._config = config
        self._version_tracker = snapshot_version_tracker or {}
        self._started = False

        # Initialize circuit breaker with Kafka-appropriate settings
        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name=f"snapshot-publisher.{config.topic}",
            transport_type=EnumInfraTransportType.KAFKA,
        )

    @property
    def topic(self) -> str:
        """Get the configured topic."""
        return self._config.topic

    @property
    def is_started(self) -> bool:
        """Check if the publisher has been started."""
        return self._started

    async def start(self) -> None:
        """Start the snapshot publisher.

        Starts the underlying Kafka producer. Must be called before
        publishing any snapshots.

        Raises:
            InfraConnectionError: If Kafka connection fails

        Example:
            >>> publisher = SnapshotPublisherRegistration(producer, config)
            >>> await publisher.start()
            >>> # Now ready to publish
        """
        if self._started:
            logger.debug("Snapshot publisher already started")
            return

        correlation_id = uuid4()
        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="start",
            target_name=self._config.topic,
            correlation_id=correlation_id,
        )

        try:
            await self._producer.start()
            self._started = True
            logger.info(
                "Snapshot publisher started for topic %s",
                self._config.topic,
                extra={"correlation_id": str(correlation_id)},
            )
        except Exception as e:
            raise InfraConnectionError(
                f"Failed to start Kafka producer for topic {self._config.topic}",
                context=ctx,
            ) from e

    async def stop(self) -> None:
        """Stop the snapshot publisher.

        Stops the underlying Kafka producer and cleans up resources.
        Safe to call multiple times.

        Example:
            >>> await publisher.stop()
            >>> # Publisher is now stopped
        """
        if not self._started:
            logger.debug("Snapshot publisher already stopped")
            return

        try:
            await self._producer.stop()
            self._started = False
            logger.info("Snapshot publisher stopped for topic %s", self._config.topic)
        except Exception as e:
            # Log but don't raise - stop should be best-effort
            logger.warning(
                "Error stopping Kafka producer: %s",
                str(e),
                extra={"topic": self._config.topic},
            )
            self._started = False

    def _get_next_version(self, entity_id: str, domain: str) -> int:
        """Get the next snapshot version for an entity.

        Increments and returns the version counter for the given entity.
        Versions are monotonically increasing within the lifetime of
        this publisher instance.

        Args:
            entity_id: The entity identifier
            domain: The domain namespace

        Returns:
            Next version number (starting from 1)
        """
        key = f"{domain}:{entity_id}"
        current = self._version_tracker.get(key, 0)
        next_version = current + 1
        self._version_tracker[key] = next_version
        return next_version

    async def publish_snapshot(
        self,
        snapshot: ModelRegistrationProjection,
    ) -> None:
        """Publish a single snapshot to the snapshot topic.

        Publishes the projection as a snapshot to the compacted Kafka topic.
        The key is derived from (entity_id, domain) for proper compaction.

        NOTE: This is a READ OPTIMIZATION. The event log remains source of truth.

        This method implements ProtocolSnapshotPublisher.publish_snapshot using
        ModelRegistrationProjection as the input type. For publishing pre-built
        ModelRegistrationSnapshot objects, use _publish_snapshot_model.

        Args:
            snapshot: The projection to publish as a snapshot. Must contain
                valid entity_id and domain for key construction.

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open

        Example:
            >>> projection = await reader.get_entity_state(entity_id)
            >>> await publisher.publish_snapshot(projection)
        """
        # Delegate to publish_from_projection which handles version tracking
        # and actual publishing. This method exists to satisfy the
        # ProtocolSnapshotPublisher interface which expects ModelRegistrationProjection
        # as input, while our internal implementation uses ModelRegistrationSnapshot.
        snapshot_model = await self.publish_from_projection(
            projection=snapshot,
            node_name=None,  # Not available from projection alone
        )
        logger.debug(
            "Published projection as snapshot version %d for %s:%s",
            snapshot_model.snapshot_version,
            snapshot.domain,
            str(snapshot.entity_id),
        )

    async def _publish_snapshot_model(
        self,
        snapshot: ModelRegistrationSnapshot,
    ) -> None:
        """Publish a pre-built snapshot model to Kafka.

        Internal method for publishing ModelRegistrationSnapshot objects.
        Use publish_snapshot for protocol compliance or publish_from_projection
        for automatic version tracking.

        Args:
            snapshot: The snapshot model to publish

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open
        """
        correlation_id = uuid4()

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("publish_snapshot", correlation_id)

        ctx = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="publish_snapshot",
            target_name=self._config.topic,
            correlation_id=correlation_id,
        )

        try:
            # Build key and value for Kafka
            key = snapshot.to_kafka_key().encode("utf-8")
            value = snapshot.model_dump_json().encode("utf-8")

            # Send and wait for acknowledgment
            await self._producer.send_and_wait(
                self._config.topic,
                key=key,
                value=value,
            )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            logger.debug(
                "Published snapshot for %s version %d",
                snapshot.to_kafka_key(),
                snapshot.snapshot_version,
                extra={"correlation_id": str(correlation_id)},
            )

        except TimeoutError as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("publish_snapshot", correlation_id)
            raise InfraTimeoutError(
                f"Timeout publishing snapshot: {snapshot.to_kafka_key()}",
                context=ctx,
            ) from e

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("publish_snapshot", correlation_id)
            raise InfraConnectionError(
                f"Failed to publish snapshot: {snapshot.to_kafka_key()}",
                context=ctx,
            ) from e

    async def publish_batch(
        self,
        snapshots: list[ModelRegistrationProjection],
    ) -> int:
        """Publish multiple snapshots in a batch operation.

        Publishes each projection as a snapshot, continuing on individual
        failures. This is the recommended method for bulk snapshot jobs.

        NOTE: This is a READ OPTIMIZATION. The event log remains source of truth.

        Args:
            snapshots: List of projections to publish as snapshots

        Returns:
            Count of successfully published snapshots.
            May be less than len(snapshots) if some fail.

        Raises:
            InfraConnectionError: Only if connection fails before any publishing

        Example:
            >>> projections = await reader.get_all()
            >>> count = await publisher.publish_batch(projections)
            >>> print(f"Published {count}/{len(projections)} snapshots")
        """
        if not snapshots:
            return 0

        success_count = 0
        for projection in snapshots:
            try:
                await self.publish_snapshot(projection)
                success_count += 1
            except (
                InfraConnectionError,
                InfraTimeoutError,
                InfraUnavailableError,
            ) as e:
                logger.warning(
                    "Failed to publish snapshot %s:%s: %s",
                    projection.domain,
                    str(projection.entity_id),
                    str(e),
                    extra={
                        "entity_id": str(projection.entity_id),
                        "domain": projection.domain,
                    },
                )
                # Continue with remaining snapshots (best-effort)

        logger.info(
            "Batch publish completed: %d/%d snapshots published",
            success_count,
            len(snapshots),
            extra={"topic": self._config.topic},
        )
        return success_count

    async def get_latest_snapshot(
        self,
        entity_id: str,
        domain: str,
    ) -> ModelRegistrationProjection | None:
        """Retrieve the latest snapshot for an entity.

        NOTE: This is a consumer operation. For production use, consider
        using a dedicated Kafka consumer or cache layer for reading snapshots.
        This publisher is optimized for writes, not reads.

        IMPORTANT: Snapshot may be slightly stale. For guaranteed freshness,
        combine with event log events since snapshot.updated_at.

        Args:
            entity_id: The entity identifier (UUID as string)
            domain: The domain namespace (e.g., "registration")

        Returns:
            The latest projection if found, None otherwise.
            Returns None because reading requires a consumer, which
            is beyond this publisher's scope.

        Example:
            >>> snapshot = await publisher.get_latest_snapshot("uuid", "registration")
            >>> if snapshot is None:
            ...     print("Entity not found or use dedicated consumer")
        """
        # Reading from compacted topics requires a consumer with seek-to-end
        # or a key-value store built from consumer. This is beyond the scope
        # of a producer-focused publisher.
        logger.debug(
            "get_latest_snapshot not fully implemented - requires dedicated consumer. "
            "Consider using a Kafka consumer or cache layer for snapshot reads.",
            extra={
                "entity_id": entity_id,
                "domain": domain,
                "topic": self._config.topic,
            },
        )
        return None

    async def delete_snapshot(
        self,
        entity_id: str,
        domain: str,
    ) -> bool:
        """Publish a tombstone to remove a snapshot.

        In Kafka compaction, a message with null value acts as a tombstone,
        causing the key to be removed during compaction. This effectively
        deletes the snapshot for the given entity.

        NOTE: This does NOT delete events from the event log. The event log
        is immutable and retains full history. Tombstones only affect the
        snapshot read path.

        Use Cases:
            - Node deregistration (permanent removal)
            - Entity lifecycle completion
            - Data retention cleanup

        Args:
            entity_id: The entity identifier (UUID as string)
            domain: The domain namespace (e.g., "registration")

        Returns:
            True if tombstone was published successfully.
            False if publish failed (caller should retry or handle).

        Raises:
            Does not raise - returns False on failure for caller to handle.

        Example:
            >>> # Handle node deregistration
            >>> deleted = await publisher.delete_snapshot(str(node_id), "registration")
            >>> if not deleted:
            ...     logger.warning(f"Failed to delete snapshot for {node_id}")
        """
        correlation_id = uuid4()

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            try:
                await self._check_circuit_breaker("delete_snapshot", correlation_id)
            except Exception as e:
                logger.warning(
                    "Circuit breaker prevented delete_snapshot: %s",
                    str(e),
                    extra={
                        "entity_id": entity_id,
                        "domain": domain,
                        "correlation_id": str(correlation_id),
                    },
                )
                return False

        try:
            # Build key for tombstone
            key = f"{domain}:{entity_id}".encode()

            # Publish tombstone (null value)
            await self._producer.send_and_wait(
                self._config.topic,
                key=key,
                value=None,  # Tombstone - null value triggers deletion on compaction
            )

            # Record success
            async with self._circuit_breaker_lock:
                await self._reset_circuit_breaker()

            # Clear version tracker for this entity
            tracker_key = f"{domain}:{entity_id}"
            self._version_tracker.pop(tracker_key, None)

            logger.info(
                "Published tombstone for %s:%s",
                domain,
                entity_id,
                extra={"correlation_id": str(correlation_id)},
            )
            return True

        except Exception as e:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure("delete_snapshot", correlation_id)

            logger.exception(
                "Failed to publish tombstone for %s:%s",
                domain,
                entity_id,
                extra={"correlation_id": str(correlation_id)},
            )
            return False

    async def publish_from_projection(
        self,
        projection: ModelRegistrationProjection,
        *,
        node_name: str | None = None,
    ) -> ModelRegistrationSnapshot:
        """Create and publish a snapshot from a projection.

        Convenience method that handles version tracking automatically.
        Converts the projection to a snapshot model, assigns the next
        version number, and publishes to Kafka.

        This is the recommended method for publishing snapshots when you
        have a projection and want automatic version management.

        Args:
            projection: The projection to convert and publish
            node_name: Optional node name to include in snapshot.
                Not stored in projection, must be provided externally
                (e.g., from introspection data).

        Returns:
            The published snapshot model with assigned version

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If publish times out
            InfraUnavailableError: If circuit breaker is open

        Example:
            >>> # Automatic versioning
            >>> snapshot1 = await publisher.publish_from_projection(proj)
            >>> print(snapshot1.snapshot_version)  # 1
            >>>
            >>> # Next snapshot for same entity increments version
            >>> snapshot2 = await publisher.publish_from_projection(proj)
            >>> print(snapshot2.snapshot_version)  # 2
            >>>
            >>> # Include node name for service discovery
            >>> snapshot = await publisher.publish_from_projection(
            ...     projection,
            ...     node_name="PostgresAdapter",
            ... )
        """
        entity_id_str = str(projection.entity_id)
        version = self._get_next_version(entity_id_str, projection.domain)

        # Create snapshot from projection
        snapshot = ModelRegistrationSnapshot.from_projection(
            projection=projection,
            snapshot_version=version,
            snapshot_created_at=datetime.now(UTC),
            node_name=node_name,
        )

        # Publish the snapshot model
        await self._publish_snapshot_model(snapshot)

        return snapshot

    async def publish_snapshot_batch(
        self,
        snapshots: list[ModelRegistrationSnapshot],
    ) -> int:
        """Publish multiple pre-built snapshots in a batch.

        Similar to publish_batch but for pre-built ModelRegistrationSnapshot
        objects instead of projections. Use this when you have already
        constructed snapshot models (e.g., from a different source).

        Args:
            snapshots: List of snapshot models to publish

        Returns:
            Count of successfully published snapshots

        Example:
            >>> snapshots = [
            ...     ModelRegistrationSnapshot.from_projection(p, version=1, ...)
            ...     for p in projections
            ... ]
            >>> count = await publisher.publish_snapshot_batch(snapshots)
        """
        if not snapshots:
            return 0

        success_count = 0
        for snapshot in snapshots:
            try:
                await self._publish_snapshot_model(snapshot)
                success_count += 1
            except (
                InfraConnectionError,
                InfraTimeoutError,
                InfraUnavailableError,
            ) as e:
                logger.warning(
                    "Failed to publish snapshot %s version %d: %s",
                    snapshot.to_kafka_key(),
                    snapshot.snapshot_version,
                    str(e),
                )
                # Continue with remaining snapshots (best-effort)

        logger.info(
            "Batch publish completed: %d/%d snapshots published",
            success_count,
            len(snapshots),
            extra={"topic": self._config.topic},
        )
        return success_count


__all__: list[str] = ["SnapshotPublisherRegistration"]
