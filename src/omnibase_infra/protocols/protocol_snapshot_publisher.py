# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol definition for snapshot publishing services.

This module defines the ProtocolSnapshotPublisher interface for services that
publish materialized projection snapshots to Kafka compacted topics. Snapshots
provide an optimized read path for entity state queries.

CRITICAL DESIGN PRINCIPLE - Snapshots vs Event Log:
    Snapshots are an optional READ OPTIMIZATION and NEVER replace the immutable
    event log which remains the absolute source of truth.

    Event Log (Source of Truth):
        - Immutable append-only log of all events
        - Complete audit trail with full history
        - Replayed to rebuild any state from scratch
        - Never modified, only appended
        - Used for: Compliance, debugging, state recovery

    Snapshots (Read Optimization):
        - Point-in-time materialized views of projections
        - Optimized for fast reads without log replay
        - Published to Kafka compacted topics (latest by key wins)
        - Can be regenerated from event log at any time
        - Used for: Query optimization, fast state access

Architecture Context:
    In the ONEX registration domain:
    - Projectors (F1) maintain real-time materialized views in PostgreSQL
    - Snapshot publishers (F2) periodically compact projections to Kafka
    - Orchestrators read snapshots for fast state queries
    - Full event log is preserved for replay, audit, and recovery

Compaction Semantics:
    Kafka compacted topics retain only the latest value per key:
    - Key: (entity_id, domain) composite key as string
    - Value: Serialized snapshot (JSON or Avro)
    - Tombstones: null value deletes the key (entity deletion)

    This means consumers reading the compacted topic get:
    - Latest snapshot for each entity
    - No historical versions (use event log for history)
    - Efficient bootstrap for new consumers

Example Usage:
    ```python
    from omnibase_infra.protocols import ProtocolSnapshotPublisher
    from omnibase_infra.models.projection import ModelRegistrationProjection

    class KafkaSnapshotPublisher:
        '''Concrete implementation publishing to Kafka compacted topic.'''

        async def publish_snapshot(
            self,
            snapshot: ModelRegistrationProjection,
        ) -> None:
            '''Publish snapshot to Kafka compacted topic.'''
            key = f"{snapshot.domain}:{snapshot.entity_id}"
            await self._kafka_producer.send(
                topic="registration.snapshots",
                key=key,
                value=snapshot.model_dump_json(),
            )

        async def delete_snapshot(
            self,
            entity_id: str,
            domain: str,
        ) -> bool:
            '''Publish tombstone to delete snapshot.'''
            key = f"{domain}:{entity_id}"
            await self._kafka_producer.send(
                topic="registration.snapshots",
                key=key,
                value=None,  # Tombstone
            )
            return True

    # Protocol conformance check via duck typing (per ONEX conventions)
    publisher = KafkaSnapshotPublisher()

    # Verify required methods exist and are callable
    assert hasattr(publisher, 'publish_snapshot') and callable(publisher.publish_snapshot)
    assert hasattr(publisher, 'delete_snapshot') and callable(publisher.delete_snapshot)
    ```

Performance Considerations:
    - Batch publishing recommended for high-volume updates
    - Consider snapshot frequency vs freshness trade-offs
    - Compacted topics have lower storage than full event log
    - Use get_latest_snapshot for read optimization, not reconstruction

See Also:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - omnibase_infra.models.projection for model definitions
    - ONEX event sourcing architecture documentation
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.models.projection import ModelRegistrationProjection

__all__ = [
    "ProtocolSnapshotPublisher",
]


@runtime_checkable
class ProtocolSnapshotPublisher(Protocol):
    """Protocol for snapshot publishing services.

    Snapshot publishers periodically compact projections into optimized read
    snapshots. Snapshots are published to compacted Kafka topics for efficient
    state queries by orchestrators and other consumers.

    IMPORTANT: Snapshots are for READ OPTIMIZATION ONLY and NEVER replace the
    immutable event log which remains the source of truth.

    Why Snapshots?
        Without snapshots, querying current entity state requires:
        1. Reading the entire event log from the beginning
        2. Replaying all events to compute current state
        3. O(n) complexity where n = number of events

        With snapshots:
        1. Read latest snapshot from compacted topic
        2. O(1) lookup for current state
        3. Optional: apply any events since snapshot for freshness

    Compaction Semantics:
        - Key: Composite of (entity_id, domain) as string
        - Value: Serialized snapshot (latest wins per key)
        - Tombstone: null value indicates entity deletion
        - Consumers see: only latest snapshot per entity

    Concurrency Safety:
        Implementations must be coroutine-safe for concurrent async publishing.
        Multiple coroutines may invoke publish_snapshot concurrently. Use
        asyncio.Lock for shared mutable state (coroutine-safe, not thread-safe).

    Error Handling:
        All methods should raise OnexError subclasses on failure:
        - InfraConnectionError: Kafka/transport unavailable
        - InfraTimeoutError: Publish operation timed out
        - ProtocolConfigurationError: Invalid topic/serialization config

    Example Implementation:
        ```python
        class KafkaSnapshotPublisher:
            def __init__(self, producer: KafkaProducer, topic: str):
                self._producer = producer
                self._topic = topic

            async def publish_snapshot(
                self,
                snapshot: ModelRegistrationProjection,
            ) -> None:
                key = self._build_key(snapshot.entity_id, snapshot.domain)
                value = snapshot.model_dump_json()
                await self._producer.send(self._topic, key=key, value=value)

            async def publish_batch(
                self,
                snapshots: list[ModelRegistrationProjection],
            ) -> int:
                count = 0
                for snapshot in snapshots:
                    try:
                        await self.publish_snapshot(snapshot)
                        count += 1
                    except Exception:  # noqa: BLE001 - batch continues on single failure
                        # Log and continue with batch
                        pass
                return count

            async def get_latest_snapshot(
                self,
                entity_id: str,
                domain: str,
            ) -> ModelRegistrationProjection | None:
                key = self._build_key(entity_id, domain)
                value = await self._consumer.get_latest(self._topic, key)
                if value is None:
                    return None
                return ModelRegistrationProjection.model_validate_json(value)

            async def delete_snapshot(
                self,
                entity_id: str,
                domain: str,
            ) -> bool:
                key = self._build_key(entity_id, domain)
                await self._producer.send(self._topic, key=key, value=None)
                return True

            def _build_key(self, entity_id: str, domain: str) -> str:
                return f"{domain}:{entity_id}"
        ```
    """

    async def publish_snapshot(
        self,
        snapshot: ModelRegistrationProjection,
    ) -> None:
        """Publish a single snapshot to the snapshot topic.

        Publishes a materialized projection snapshot to the compacted Kafka topic.
        The snapshot key is derived from (entity_id, domain) for proper compaction.

        NOTE: This is a READ OPTIMIZATION. The event log remains source of truth.
        Snapshots can be regenerated from the event log at any time.

        Args:
            snapshot: The projection snapshot to publish. Must contain valid
                entity_id and domain for key construction.

        Raises:
            InfraConnectionError: If Kafka/transport is unavailable
            InfraTimeoutError: If publish operation times out
            OnexError: For serialization or configuration errors

        Example:
            ```python
            from omnibase_core.enums import EnumNodeKind

            snapshot = ModelRegistrationProjection(
                entity_id=uuid4(),
                domain="registration",
                current_state=EnumRegistrationState.ACTIVE,
                node_type=EnumNodeKind.EFFECT,
                last_applied_event_id=uuid4(),
                registered_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
            await publisher.publish_snapshot(snapshot)
            ```

        Implementation Notes:
            - Key should be deterministic: f"{domain}:{entity_id}"
            - Value should be serialized consistently (JSON or Avro)
            - Include correlation_id in publish context for tracing
            - Consider batching for high-volume scenarios
        """
        ...

    async def publish_batch(
        self,
        snapshots: list[ModelRegistrationProjection],
    ) -> int:
        """Publish multiple snapshots in a batch operation.

        Batch publishing is recommended for periodic snapshot jobs that
        publish many entities at once. Implementations should optimize
        for throughput while handling individual failures gracefully.

        NOTE: This is a READ OPTIMIZATION. The event log remains source of truth.

        Args:
            snapshots: List of projection snapshots to publish.
                Each must contain valid entity_id and domain.

        Returns:
            Count of successfully published snapshots.
            May be less than len(snapshots) if some fail.

        Raises:
            InfraConnectionError: If Kafka/transport is unavailable
                (before any publishing starts)
            OnexError: For configuration errors

        Example:
            ```python
            # Periodic snapshot job
            projections = await projection_store.get_all()
            published = await publisher.publish_batch(projections)
            logger.info(f"Published {published}/{len(projections)} snapshots")
            ```

        Implementation Notes:
            - Continue publishing on individual failures (best-effort)
            - Log failures for operational visibility
            - Consider parallel publishing for large batches
            - Return accurate count of successes
            - Use transactional producers if exactly-once needed
        """
        ...

    async def get_latest_snapshot(
        self,
        entity_id: str,
        domain: str,
    ) -> ModelRegistrationProjection | None:
        """Retrieve the latest snapshot for an entity.

        Reads the latest snapshot from the compacted topic. This is the
        primary read path for state queries, providing O(1) access to
        current entity state without replaying the event log.

        IMPORTANT: Snapshot may be slightly stale. For guaranteed freshness,
        combine with event log events since snapshot.updated_at.

        NOTE: This is a READ OPTIMIZATION. For state recovery or audit,
        replay from the immutable event log instead.

        Args:
            entity_id: The entity identifier (UUID as string)
            domain: The domain namespace (e.g., "registration")

        Returns:
            The latest snapshot if exists, None if no snapshot found
            (entity never existed or was deleted via tombstone).

        Raises:
            InfraConnectionError: If Kafka/transport is unavailable
            InfraTimeoutError: If read operation times out
            OnexError: For deserialization errors

        Example:
            ```python
            snapshot = await publisher.get_latest_snapshot(
                entity_id=str(node_id),
                domain="registration",
            )
            if snapshot is None:
                # Entity not found - check event log or treat as new
                return None

            if snapshot.current_state.is_active():
                # Node is registered and active
                return snapshot.capabilities
            ```

        Freshness Considerations:
            - Snapshot represents state at snapshot.updated_at
            - Events may have occurred since snapshot was published
            - For real-time accuracy, query projection store directly
            - Snapshots are best for read-heavy, eventually-consistent queries
        """
        ...

    async def delete_snapshot(
        self,
        entity_id: str,
        domain: str,
    ) -> bool:
        """Publish a tombstone to remove a snapshot.

        Publishes a null value (tombstone) to the compacted topic for the
        given entity key. After compaction, consumers will no longer see
        this entity's snapshot.

        Use Case: When an entity is permanently deleted and should no longer
        appear in snapshot queries. Common scenarios:
        - Node deregistration (permanent removal)
        - Entity lifecycle completion
        - Data retention cleanup

        NOTE: This does NOT delete events from the event log. The event log
        is immutable and retains full history. Tombstones only affect the
        snapshot read path.

        Args:
            entity_id: The entity identifier (UUID as string)
            domain: The domain namespace (e.g., "registration")

        Returns:
            True if tombstone was published successfully.
            False if publish failed (caller should retry or handle).

        Raises:
            InfraConnectionError: If Kafka/transport is unavailable
            InfraTimeoutError: If publish operation times out

        Example:
            ```python
            # Handle node deregistration
            async def handle_node_deregistered(event: NodeDeregisteredEvent):
                # Publish tombstone to remove from snapshot queries
                deleted = await publisher.delete_snapshot(
                    entity_id=str(event.node_id),
                    domain="registration",
                )
                if not deleted:
                    logger.warning(f"Failed to delete snapshot for {event.node_id}")
            ```

        Compaction Behavior:
            - Tombstone is published immediately
            - Compaction runs periodically (Kafka config)
            - Until compaction, consumers may still see old snapshot
            - After compaction, key is removed from topic
        """
        ...
