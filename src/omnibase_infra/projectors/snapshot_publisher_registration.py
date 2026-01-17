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
    - **Lazy Consumer**: Consumer for reads is created on-demand

Concurrency Safety:
    This implementation is coroutine-safe for concurrent async publishing.
    Uses asyncio locks for circuit breaker state management and
    version tracker synchronization. Note: This is coroutine-safe, not
    thread-safe. For multi-threaded access, additional synchronization
    would be required.

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

        # Read snapshot (uses lazy consumer and in-memory cache)
        snapshot = await publisher.get_latest_snapshot("entity-123", "registration")
        if snapshot:
            print(f"Entity state: {snapshot.current_state}")

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
    - First read triggers cache loading (may take a few seconds for large topics)
    - Subsequent reads are O(1) from in-memory cache

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-1059: Implement snapshot read functionality

See Also:
    - ProtocolSnapshotPublisher: Protocol definition for snapshot publishers
    - ModelRegistrationSnapshot: Snapshot model definition
    - ModelSnapshotTopicConfig: Topic configuration for compacted topics
    - ProjectorRegistration: Projection persistence (source for snapshots)
"""

from __future__ import annotations

import asyncio
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
    ModelTimeoutErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelRegistrationSnapshot,
    ModelSnapshotTopicConfig,
)
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

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

        Version Tracker Semantics:
            - Versions start at 1 for each new entity
            - Versions increment monotonically per entity within publisher lifetime
            - Version tracker resets when publisher is recreated (new instance)
            - delete_snapshot clears the version tracker entry for that entity
            - For persistent version tracking across restarts, inject a shared
              snapshot_version_tracker dict in __init__
            - Coroutine-safe: Uses asyncio.Lock for concurrent access

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
        bootstrap_servers: str | None = None,
        consumer_timeout_ms: int = 5000,
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
            bootstrap_servers: Kafka bootstrap servers for the consumer (for reads).
                Required if you intend to use get_latest_snapshot(). If not provided,
                reads will attempt to extract from the producer configuration.
            consumer_timeout_ms: Timeout in milliseconds for consumer poll operations.
                Default is 5000ms (5 seconds). Used when loading the snapshot cache.

        Example:
            >>> producer = AIOKafkaProducer(
            ...     bootstrap_servers="localhost:9092",
            ...     value_serializer=lambda v: v,  # Publisher handles serialization
            ... )
            >>> config = ModelSnapshotTopicConfig.default()
            >>> publisher = SnapshotPublisherRegistration(
            ...     producer,
            ...     config,
            ...     bootstrap_servers="localhost:9092",
            ... )
        """
        self._producer = producer
        self._config = config
        self._version_tracker = snapshot_version_tracker or {}
        self._version_tracker_lock = asyncio.Lock()
        self._started = False

        # Consumer configuration for read operations
        self._bootstrap_servers = bootstrap_servers
        self._consumer_timeout_ms = consumer_timeout_ms
        self._consumer: AIOKafkaConsumer | None = None
        self._consumer_started = False

        # In-memory cache for O(1) snapshot lookups
        # Key: "{domain}:{entity_id}", Value: ModelRegistrationSnapshot
        #
        # Cache Size Expectations:
        #   - Typical deployment: 100-1000 registered nodes
        #   - Large deployment: 5000-10000 nodes
        #   - Maximum practical: ~50000 nodes (memory ~100MB with full snapshots)
        #   - Each snapshot is approximately 2KB serialized
        #
        # Memory Footprint Estimation:
        #   - 1000 nodes * 2KB = ~2MB
        #   - 10000 nodes * 2KB = ~20MB
        #   - 50000 nodes * 2KB = ~100MB
        self._snapshot_cache: dict[str, ModelRegistrationSnapshot] = {}
        self._cache_lock = asyncio.Lock()
        self._cache_loaded = False
        self._cache_warming_in_progress = False

        # Initialize circuit breaker with Kafka-appropriate settings
        cb_config = ModelCircuitBreakerConfig.from_env(
            service_name=f"snapshot-publisher.{config.topic}",
            transport_type=EnumInfraTransportType.KAFKA,
        )
        self._init_circuit_breaker_from_config(cb_config)

    @property
    def topic(self) -> str:
        """Get the configured topic."""
        return self._config.topic

    @property
    def is_started(self) -> bool:
        """Check if the publisher has been started."""
        return self._started

    async def start(self, *, warm_cache: bool = False) -> None:
        """Start the snapshot publisher.

        Starts the underlying Kafka producer. Must be called before
        publishing any snapshots.

        Args:
            warm_cache: If True, pre-load the snapshot cache from Kafka
                during startup. This is useful for read-heavy workloads
                where you want the first read to be fast. The warming
                is performed asynchronously and does not block start().
                Default is False for backward compatibility.

        Raises:
            InfraConnectionError: If Kafka connection fails

        Example:
            >>> publisher = SnapshotPublisherRegistration(producer, config)
            >>> await publisher.start()
            >>> # Now ready to publish
            >>>
            >>> # With cache warming for read-heavy workloads
            >>> await publisher.start(warm_cache=True)
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

            # Optionally warm the cache in the background
            if warm_cache and self._bootstrap_servers:
                await self._warm_cache_async(correlation_id)

        except Exception as e:
            raise InfraConnectionError(
                f"Failed to start Kafka producer for topic {self._config.topic}",
                context=ctx,
            ) from e

    async def _warm_cache_async(self, correlation_id: UUID) -> None:
        """Warm the snapshot cache asynchronously.

        Pre-loads all snapshots from the Kafka topic into the in-memory
        cache. This is called during start() when warm_cache=True.

        Cache warming is performed inline (not in background task) to ensure
        the cache is populated before start() returns. This provides
        predictable behavior for read-heavy workloads.

        Args:
            correlation_id: Correlation ID for tracing

        Note:
            Errors during cache warming are logged but do not fail startup.
            The cache will be loaded lazily on the first read if warming fails.
        """
        if self._cache_warming_in_progress:
            logger.debug("Cache warming already in progress, skipping")
            return

        self._cache_warming_in_progress = True

        try:
            logger.info(
                "Warming snapshot cache for topic %s",
                self._config.topic,
                extra={"correlation_id": str(correlation_id)},
            )

            await self._load_cache_from_topic(correlation_id)

            async with self._cache_lock:
                cache_size = len(self._snapshot_cache)

            logger.info(
                "Cache warming completed: %d snapshots loaded for topic %s",
                cache_size,
                self._config.topic,
                extra={
                    "correlation_id": str(correlation_id),
                    "cache_size": cache_size,
                    "topic": self._config.topic,
                },
            )
        except Exception as e:
            # Log but don't fail startup - cache can be loaded lazily
            logger.warning(
                "Cache warming failed for topic %s: %s. "
                "Cache will be loaded lazily on first read.",
                self._config.topic,
                str(e),
                extra={
                    "correlation_id": str(correlation_id),
                    "error_type": type(e).__name__,
                },
            )
        finally:
            self._cache_warming_in_progress = False

    async def stop(self) -> None:
        """Stop the snapshot publisher.

        Stops the underlying Kafka producer, consumer (if started), and
        cleans up resources. Safe to call multiple times.

        Example:
            >>> await publisher.stop()
            >>> # Publisher is now stopped
        """
        # Stop consumer if it was started
        if self._consumer_started and self._consumer is not None:
            try:
                await self._consumer.stop()
                self._consumer_started = False
                self._consumer = None
                logger.debug(
                    "Snapshot consumer stopped for topic %s", self._config.topic
                )
            except Exception as e:
                # Log but don't raise - stop should be best-effort
                logger.warning(
                    "Error stopping Kafka consumer: %s",
                    str(e),
                    extra={"topic": self._config.topic},
                )
                self._consumer_started = False
                self._consumer = None

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

        # Clear the cache on stop
        async with self._cache_lock:
            self._snapshot_cache.clear()
            self._cache_loaded = False

    async def _get_next_version(self, entity_id: str, domain: str) -> int:
        """Get the next snapshot version for an entity.

        Increments and returns the version counter for the given entity.
        Versions are monotonically increasing within the lifetime of
        this publisher instance.

        Concurrency Safety:
            Uses _version_tracker_lock (asyncio.Lock) to ensure atomic
            read-modify-write operations in concurrent coroutine contexts.

        Args:
            entity_id: The entity identifier
            domain: The domain namespace

        Returns:
            Next version number (starting from 1)
        """
        key = f"{domain}:{entity_id}"
        async with self._version_tracker_lock:
            current = self._version_tracker.get(key, 0)
            next_version = current + 1
            self._version_tracker[key] = next_version
            return next_version

    async def _cleanup_consumer(self) -> None:
        """Clean up Kafka consumer after cache load operations.

        Stops the consumer, resets the started flag, and clears the reference.
        This method is idempotent and safe to call even if no consumer exists.
        Used for cleanup after both successful and failed cache load operations.
        """
        if self._consumer_started:
            try:
                if self._consumer is not None:
                    await self._consumer.stop()
            except Exception:
                pass
            self._consumer_started = False
            self._consumer = None

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
        # Delegate to publish_from_projection for versioning and publishing
        await self.publish_from_projection(
            projection=snapshot,
            node_name=None,
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

            # Update cache if loaded (for read-after-write consistency)
            if self._cache_loaded:
                cache_key = snapshot.to_kafka_key()
                async with self._cache_lock:
                    self._snapshot_cache[cache_key] = snapshot

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
                context=ModelTimeoutErrorContext(
                    transport_type=ctx.transport_type,
                    operation=ctx.operation,
                    target_name=ctx.target_name,
                    correlation_id=ctx.correlation_id,
                    # timeout_seconds omitted - value not available in this context (defaults to None)
                ),
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
        *,
        parallel: bool = True,
    ) -> int:
        """Publish multiple snapshots in a batch operation.

        Publishes each projection as a snapshot, continuing on individual
        failures. This is the recommended method for bulk snapshot jobs.

        NOTE: This is a READ OPTIMIZATION. The event log remains source of truth.

        Args:
            snapshots: List of projections to publish as snapshots
            parallel: If True (default), publish concurrently using asyncio.gather.
                Set to False for sequential publishing (useful for debugging
                or rate-limited scenarios).

        Returns:
            Count of successfully published snapshots.
            May be less than len(snapshots) if some fail.

        Raises:
            InfraConnectionError: Only if connection fails before any publishing

        Example:
            >>> projections = await reader.get_all()
            >>> count = await publisher.publish_batch(projections)
            >>> print(f"Published {count}/{len(projections)} snapshots")
            >>>
            >>> # Sequential publishing for debugging
            >>> count = await publisher.publish_batch(projections, parallel=False)
        """
        if not snapshots:
            return 0

        if parallel:
            # Parallel publishing using asyncio.gather with return_exceptions=True
            results = await asyncio.gather(
                *[self.publish_snapshot(projection) for projection in snapshots],
                return_exceptions=True,
            )

            success_count = 0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    projection = snapshots[i]
                    logger.warning(
                        "Failed to publish snapshot %s:%s: %s",
                        projection.domain,
                        str(projection.entity_id),
                        str(result),
                        extra={
                            "entity_id": str(projection.entity_id),
                            "domain": projection.domain,
                        },
                    )
                else:
                    success_count += 1
        else:
            # Sequential publishing (original behavior)
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
            "Batch publish completed: %d/%d snapshots published (parallel=%s)",
            success_count,
            len(snapshots),
            parallel,
            extra={"topic": self._config.topic},
        )
        return success_count

    async def get_latest_snapshot(
        self,
        entity_id: str,
        domain: str,
    ) -> ModelRegistrationSnapshot | None:
        """Retrieve the latest snapshot for an entity.

        Reads the snapshot from an in-memory cache that is built from the
        compacted Kafka topic. The cache is loaded lazily on first read.

        IMPORTANT: Snapshot may be slightly stale. For guaranteed freshness,
        combine with event log events since snapshot.updated_at. Call
        refresh_cache() to reload the cache from Kafka.

        Args:
            entity_id: The entity identifier (UUID as string)
            domain: The domain namespace (e.g., "registration")

        Returns:
            The latest snapshot if found, None otherwise.

        Raises:
            InfraConnectionError: If Kafka connection fails during cache load
            InfraTimeoutError: If cache loading times out
            InfraUnavailableError: If circuit breaker is open

        Example:
            >>> snapshot = await publisher.get_latest_snapshot("uuid", "registration")
            >>> if snapshot is not None:
            ...     print(f"Entity state: {snapshot.current_state}")
            ... else:
            ...     print("Entity not found")
        """
        correlation_id = uuid4()

        # Load cache if not already loaded
        # Circuit breaker check is now inside _load_cache_from_topic()
        if not self._cache_loaded:
            await self._load_cache_from_topic(correlation_id)

        # Lookup in cache (O(1))
        key = f"{domain}:{entity_id}"
        async with self._cache_lock:
            snapshot = self._snapshot_cache.get(key)

        if snapshot is None:
            logger.debug(
                "Snapshot not found in cache for %s:%s",
                domain,
                entity_id,
                extra={
                    "entity_id": entity_id,
                    "domain": domain,
                    "topic": self._config.topic,
                    "correlation_id": str(correlation_id),
                },
            )
        else:
            logger.debug(
                "Snapshot retrieved from cache for %s:%s version %d",
                domain,
                entity_id,
                snapshot.snapshot_version,
                extra={
                    "entity_id": entity_id,
                    "domain": domain,
                    "snapshot_version": snapshot.snapshot_version,
                    "correlation_id": str(correlation_id),
                },
            )

        return snapshot

    async def _load_cache_from_topic(self, correlation_id: UUID) -> None:
        """Load the snapshot cache from the compacted Kafka topic.

        Reads all snapshots from the topic and populates the in-memory cache.
        Uses getmany() with timeout to avoid blocking indefinitely.

        This method is called lazily on the first read operation. It includes
        circuit breaker protection to ensure consistent protection regardless
        of the call site.

        Performance Notes:
            - Uses model_validate_json() for ~30% faster JSON parsing vs
              json.loads() + model_validate()
            - Logs progress every 1000 messages for observability during
              large topic scans (5000+ messages)

        Args:
            correlation_id: Correlation ID for tracing

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If consumer startup times out
            InfraUnavailableError: If circuit breaker is open
        """
        # Progress logging interval (log every N messages)
        progress_log_interval = 1000

        # Check circuit breaker before operation - moved inside this method
        # to ensure consistent protection regardless of call site
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("load_cache", correlation_id)

        async with self._cache_lock:
            # Double-check after acquiring lock
            if self._cache_loaded:
                return

            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="load_cache",
                target_name=self._config.topic,
                correlation_id=correlation_id,
            )

            # Get bootstrap servers - must be explicitly configured
            # We don't try to extract from producer because:
            # 1. The producer may use internal/private attributes that vary by version
            # 2. Mock producers don't have real bootstrap_servers
            # 3. It's cleaner to require explicit configuration for reads
            bootstrap_servers = self._bootstrap_servers

            # Validate bootstrap_servers is non-empty string with proper format
            if not bootstrap_servers or not bootstrap_servers.strip():
                raise InfraConnectionError(
                    "bootstrap_servers not configured or empty. Provide bootstrap_servers "
                    "in constructor to enable snapshot reads "
                    "(e.g., 'localhost:9092' or 'kafka1:9092,kafka2:9092').",
                    context=ctx,
                )

            # Validate host:port format for each server
            stripped_servers = bootstrap_servers.strip()
            for server in stripped_servers.split(","):
                server = server.strip()
                if not server:
                    raise InfraConnectionError(
                        f"bootstrap_servers contains empty entries: '{bootstrap_servers}'. "
                        "Each entry must be in 'host:port' format.",
                        context=ctx,
                    )
                if ":" not in server:
                    raise InfraConnectionError(
                        f"Invalid bootstrap server format '{server}'. "
                        "Expected 'host:port' (e.g., 'localhost:9092').",
                        context=ctx,
                    )
                host, port_str = server.rsplit(":", 1)
                if not host:
                    raise InfraConnectionError(
                        f"Invalid bootstrap server format '{server}'. "
                        "Host cannot be empty.",
                        context=ctx,
                    )
                try:
                    port = int(port_str)
                    if port < 1 or port > 65535:
                        raise InfraConnectionError(
                            f"Invalid port {port} in '{server}'. "
                            "Port must be between 1 and 65535.",
                            context=ctx,
                        )
                except ValueError:
                    raise InfraConnectionError(
                        f"Invalid port '{port_str}' in '{server}'. "
                        "Port must be a valid integer.",
                        context=ctx,
                    ) from None

            # Use the stripped and validated version
            bootstrap_servers = stripped_servers

            # Import consumer here to avoid circular imports
            from aiokafka import AIOKafkaConsumer
            from pydantic import ValidationError

            # Create consumer with unique group ID for this publisher instance
            # Using a unique group ensures we get our own offset tracking
            consumer_group = f"snapshot-reader-{self._config.topic}-{uuid4()!s}"
            consumer = AIOKafkaConsumer(
                self._config.topic,
                bootstrap_servers=bootstrap_servers,
                group_id=consumer_group,
                auto_offset_reset="earliest",
                enable_auto_commit=False,
            )

            try:
                await consumer.start()
                self._consumer = consumer
                self._consumer_started = True

                # Seek to beginning to read all snapshots
                await consumer.seek_to_beginning()

                # Read all messages from the topic until no more messages
                messages_read = 0
                tombstones_applied = 0
                parse_errors = 0
                last_progress_log = 0

                while True:
                    # Poll with timeout - returns empty dict when no more messages
                    messages = await consumer.getmany(
                        timeout_ms=self._consumer_timeout_ms
                    )
                    if not messages:
                        break  # No more messages within timeout

                    for _tp, msgs in messages.items():
                        for message in msgs:
                            key = message.key.decode("utf-8") if message.key else None

                            if key is None:
                                # Skip messages without keys
                                continue

                            if message.value is None:
                                # Tombstone - remove from cache
                                self._snapshot_cache.pop(key, None)
                                tombstones_applied += 1
                            else:
                                # Parse snapshot using model_validate_json for
                                # ~30% faster parsing (Pydantic v2 optimization)
                                try:
                                    snapshot = (
                                        ModelRegistrationSnapshot.model_validate_json(
                                            message.value
                                        )
                                    )
                                    self._snapshot_cache[key] = snapshot
                                    messages_read += 1
                                except (ValidationError, ValueError) as e:
                                    parse_errors += 1
                                    logger.warning(
                                        "Failed to parse snapshot for key %s: %s",
                                        key,
                                        str(e),
                                        extra={
                                            "key": key,
                                            "correlation_id": str(correlation_id),
                                        },
                                    )

                            # Log progress for large topic scans
                            total_processed = messages_read + tombstones_applied
                            if (
                                total_processed - last_progress_log
                                >= progress_log_interval
                            ):
                                logger.info(
                                    "Cache loading progress: %d messages processed "
                                    "(%d snapshots, %d tombstones, %d errors)",
                                    total_processed,
                                    messages_read,
                                    tombstones_applied,
                                    parse_errors,
                                    extra={
                                        "topic": self._config.topic,
                                        "messages_processed": total_processed,
                                        "snapshots": messages_read,
                                        "tombstones": tombstones_applied,
                                        "parse_errors": parse_errors,
                                        "correlation_id": str(correlation_id),
                                    },
                                )
                                last_progress_log = total_processed

                self._cache_loaded = True

                # Reset circuit breaker on success
                async with self._circuit_breaker_lock:
                    await self._reset_circuit_breaker()

                # Calculate cache memory estimate (approx 2KB per snapshot)
                cache_size = len(self._snapshot_cache)
                estimated_memory_kb = cache_size * 2

                logger.info(
                    "Snapshot cache loaded: %d snapshots, %d tombstones applied, "
                    "%d parse errors, cache size: %d entries (~%dKB)",
                    messages_read,
                    tombstones_applied,
                    parse_errors,
                    cache_size,
                    estimated_memory_kb,
                    extra={
                        "topic": self._config.topic,
                        "snapshots_loaded": messages_read,
                        "tombstones_applied": tombstones_applied,
                        "parse_errors": parse_errors,
                        "cache_size": cache_size,
                        "estimated_memory_kb": estimated_memory_kb,
                        "correlation_id": str(correlation_id),
                    },
                )

                # Stop consumer after successful cache load - consumer is only
                # needed during the cache loading phase, not for ongoing reads
                await self._cleanup_consumer()

            except TimeoutError as e:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure("load_cache", correlation_id)
                await self._cleanup_consumer()
                raise InfraTimeoutError(
                    f"Timeout loading snapshot cache from topic {self._config.topic}",
                    context=ModelTimeoutErrorContext(
                        transport_type=ctx.transport_type,
                        operation=ctx.operation,
                        target_name=ctx.target_name,
                        correlation_id=ctx.correlation_id,
                        timeout_seconds=float(self._consumer_timeout_ms) / 1000.0,
                    ),
                ) from e

            except Exception as e:
                async with self._circuit_breaker_lock:
                    await self._record_circuit_failure("load_cache", correlation_id)
                await self._cleanup_consumer()
                raise InfraConnectionError(
                    f"Failed to load snapshot cache from topic {self._config.topic}: {e}",
                    context=ctx,
                ) from e

    async def refresh_cache(self) -> int:
        """Refresh the snapshot cache by reloading from the Kafka topic.

        Reloads all snapshots from the compacted topic. Use this to ensure
        the cache reflects the latest published state.

        Error Recovery:
            If cache loading fails, the existing cache is preserved to avoid
            leaving the system in a broken state. This follows the principle
            of graceful degradation - stale data is better than no data.

        Returns:
            Number of snapshots loaded into the cache.

        Raises:
            InfraConnectionError: If Kafka connection fails
            InfraTimeoutError: If cache loading times out
            InfraUnavailableError: If circuit breaker is open

        Example:
            >>> count = await publisher.refresh_cache()
            >>> print(f"Loaded {count} snapshots")
        """
        correlation_id = uuid4()

        # Check circuit breaker before operation
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("refresh_cache", correlation_id)

        # Stop existing consumer if running
        if self._consumer_started and self._consumer is not None:
            try:
                await self._consumer.stop()
            except Exception:
                pass
            self._consumer_started = False
            self._consumer = None

        # Preserve existing cache state before attempting reload
        # This allows rollback on failure (graceful degradation)
        async with self._cache_lock:
            old_cache = self._snapshot_cache.copy()
            old_cache_loaded = self._cache_loaded
            self._snapshot_cache.clear()
            self._cache_loaded = False

        try:
            await self._load_cache_from_topic(correlation_id)

            async with self._cache_lock:
                count = len(self._snapshot_cache)

            logger.info(
                "Snapshot cache refreshed with %d snapshots",
                count,
                extra={
                    "topic": self._config.topic,
                    "snapshot_count": count,
                    "correlation_id": str(correlation_id),
                },
            )

            return count

        except Exception as e:
            # Restore previous cache on failure (graceful degradation)
            async with self._cache_lock:
                self._snapshot_cache = old_cache
                self._cache_loaded = old_cache_loaded

            logger.warning(
                "Cache refresh failed, preserving existing cache with %d snapshots",
                len(old_cache),
                extra={
                    "topic": self._config.topic,
                    "preserved_count": len(old_cache),
                    "error_type": type(e).__name__,
                    "correlation_id": str(correlation_id),
                },
            )
            raise

    @property
    def cache_size(self) -> int:
        """Get the number of snapshots in the cache.

        Returns:
            Number of snapshots currently in the cache.

        Note:
            This is a synchronous property that does not trigger cache loading.
            Call get_latest_snapshot() or refresh_cache() to load the cache first.
        """
        return len(self._snapshot_cache)

    @property
    def is_cache_loaded(self) -> bool:
        """Check if the cache has been loaded.

        Returns:
            True if the cache has been loaded from Kafka, False otherwise.
        """
        return self._cache_loaded

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
            InfraUnavailableError: If circuit breaker is open (fail-fast).

        Example:
            >>> # Handle node deregistration
            >>> deleted = await publisher.delete_snapshot(str(node_id), "registration")
            >>> if not deleted:
            ...     logger.warning(f"Failed to delete snapshot for {node_id}")
        """
        correlation_id = uuid4()

        # Check circuit breaker before operation - let InfraUnavailableError propagate
        # per ONEX fail-fast principles (callers need to know service is unavailable)
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker("delete_snapshot", correlation_id)

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

            # Clear version tracker for this entity (thread-safe)
            tracker_key = f"{domain}:{entity_id}"
            async with self._version_tracker_lock:
                self._version_tracker.pop(tracker_key, None)

            # Also remove from cache if loaded (for consistency)
            if self._cache_loaded:
                async with self._cache_lock:
                    self._snapshot_cache.pop(tracker_key, None)

            logger.info(
                "Published tombstone for %s:%s",
                domain,
                entity_id,
                extra={"correlation_id": str(correlation_id)},
            )
            return True

        except Exception:
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
        version = await self._get_next_version(entity_id_str, projection.domain)

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
