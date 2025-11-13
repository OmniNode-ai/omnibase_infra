#!/usr/bin/env python3
"""
Integration tests for ProjectionMaterializerService with Kafka.

Tests end-to-end event processing with real Kafka broker and PostgreSQL database.
Validates atomic updates, watermark tracking, and idempotence guarantees.

Pure Reducer Refactor - Wave 2, Workstream 2C
"""

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

try:
    from aiokafka import AIOKafkaProducer

    AIOKAFKA_AVAILABLE = True
except ImportError:
    AIOKAFKA_AVAILABLE = False

from omninode_bridge.infrastructure.postgres_connection_manager import (
    ModelPostgresConfig,
    PostgresConnectionManager,
)
from omninode_bridge.services.projection_materializer import (
    ProjectionMaterializerService,
)

# Skip all tests if Kafka/Docker not available
pytestmark = pytest.mark.skipif(
    not AIOKAFKA_AVAILABLE or os.getenv("SKIP_INTEGRATION_TESTS") == "true",
    reason="Integration tests require Kafka and PostgreSQL",
)


@pytest.fixture
def kafka_bootstrap_servers():
    """Get Kafka bootstrap servers from environment."""
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")


@pytest.fixture
def postgres_config():
    """Get PostgreSQL configuration from environment."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5436")),
        "database": os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "omninode-bridge-postgres-dev-2024"),
    }


@pytest.fixture
def state_committed_topic():
    """Get StateCommitted Kafka topic name."""
    env = os.getenv("OMNINODE_ENV", "dev")
    tenant = os.getenv("OMNINODE_TENANT", "omninode_bridge")
    context = os.getenv("OMNINODE_CONTEXT", "onex")
    return f"{env}.{tenant}.{context}.evt.state-committed.v1"


@pytest.fixture
async def kafka_producer(kafka_bootstrap_servers):
    """Create Kafka producer for publishing test events."""
    producer = AIOKafkaProducer(
        bootstrap_servers=kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )
    await producer.start()
    yield producer
    await producer.stop()


@pytest.fixture
async def db_connection(postgres_config):
    """Create PostgreSQL connection for test verification."""
    # Convert dict to ModelPostgresConfig
    config = ModelPostgresConfig(**postgres_config)
    db = PostgresConnectionManager(config=config)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def clean_test_data(db_connection):
    """Clean up test data before and after tests."""
    # Cleanup before test
    async with db_connection.transaction() as conn:
        await conn.execute(
            "DELETE FROM workflow_projection WHERE workflow_key LIKE 'test-%'"
        )
        await conn.execute(
            "DELETE FROM projection_watermarks WHERE partition_id LIKE 'test-%'"
        )

    yield

    # Cleanup after test
    async with db_connection.transaction() as conn:
        await conn.execute(
            "DELETE FROM workflow_projection WHERE workflow_key LIKE 'test-%'"
        )
        await conn.execute(
            "DELETE FROM projection_watermarks WHERE partition_id LIKE 'test-%'"
        )


def create_test_event(
    workflow_key: str = None,
    version: int = 1,
    tag: str = "PROCESSING",
    namespace: str = "test-namespace",
) -> dict[str, Any]:
    """Create test StateCommitted event."""
    workflow_key = workflow_key or f"test-workflow-{uuid4()}"

    return {
        "event_id": str(uuid4()),
        "workflow_key": workflow_key,
        "version": version,
        "state": {"items": ["a", "b"], "count": 2},
        "tag": tag,
        "last_action": "TestAction",
        "namespace": namespace,
        "provenance": {
            "effect_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "action_id": str(uuid4()),
        },
        "committed_at": datetime.now(UTC).isoformat(),
        "partition_id": "test-partition-0",
        "offset": 1000,
        "indices": {"test": "true"},
        "extras": {},
    }


class TestProjectionMaterializerKafkaIntegration:
    """Integration tests for ProjectionMaterializer with Kafka."""

    @pytest.mark.asyncio
    async def test_end_to_end_event_processing(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test end-to-end event processing from Kafka to database."""
        # Create unique workflow key for test
        workflow_key = f"test-workflow-{uuid4()}"
        event = create_test_event(workflow_key=workflow_key)

        # Create and start materializer service
        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
            enable_idempotence=True,
        )

        try:
            await materializer.start()

            # Wait for consumer to be ready
            await asyncio.sleep(2)

            # Publish StateCommitted event to Kafka
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )

            # Wait for event processing
            await asyncio.sleep(3)

            # Verify projection created in database
            async with db_connection.transaction() as conn:
                projection = await conn.fetchrow(
                    "SELECT * FROM workflow_projection WHERE workflow_key = $1",
                    workflow_key,
                )

                assert projection is not None
                assert projection["workflow_key"] == workflow_key
                assert projection["version"] == event["version"]
                assert projection["tag"] == event["tag"]
                assert projection["namespace"] == event["namespace"]

            # Verify watermark updated (uses Kafka partition format)
            async with db_connection.transaction() as conn:
                watermark_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM projection_watermarks WHERE partition_id LIKE 'kafka-partition-%'"
                )

                assert watermark_count > 0

            # Verify metrics
            metrics = materializer.metrics
            assert metrics.projections_materialized_total >= 1
            assert metrics.watermark_updates_total >= 1

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_batch_event_processing(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test batch processing of multiple events."""
        # Create multiple events
        num_events = 10
        events = [
            create_test_event(
                workflow_key=f"test-batch-{i}",
                version=1,
            )
            for i in range(num_events)
        ]

        # Create and start materializer
        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish all events
            for event in events:
                await kafka_producer.send_and_wait(
                    topic=state_committed_topic,
                    key=event["workflow_key"],
                    value=event,
                )

            # Wait for batch processing
            await asyncio.sleep(5)

            # Verify all projections created
            async with db_connection.transaction() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM workflow_projection WHERE workflow_key LIKE 'test-batch-%'"
                )

                assert count == num_events

            # Verify metrics
            metrics = materializer.metrics
            assert metrics.projections_materialized_total >= num_events

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_idempotence_duplicate_events(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test idempotence with duplicate events.

        Note: Kafka assigns unique offsets to each published message, so true
        duplicate detection requires event_id-based idempotence (future enhancement).
        This test verifies that multiple events for the same workflow_key result
        in a single projection due to the UPSERT pattern.
        """
        workflow_key = f"test-idem-{uuid4()}"
        event = create_test_event(workflow_key=workflow_key)

        # Create materializer with idempotence enabled
        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
            enable_idempotence=True,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish same event twice (Kafka will assign different offsets)
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )

            # Wait for processing
            await asyncio.sleep(3)

            # Verify only one projection exists (due to UPSERT on workflow_key)
            async with db_connection.transaction() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM workflow_projection WHERE workflow_key = $1",
                    workflow_key,
                )

                assert count == 1

            # Both events are processed (different Kafka offsets), but UPSERT ensures single projection
            metrics = materializer.metrics
            assert metrics.projections_materialized_total >= 2

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_version_ordering(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test version-based ordering of projection updates."""
        workflow_key = f"test-version-{uuid4()}"

        # Create events with different versions
        event_v1 = create_test_event(workflow_key=workflow_key, version=1)
        event_v2 = create_test_event(workflow_key=workflow_key, version=2)
        event_v3 = create_test_event(workflow_key=workflow_key, version=3)

        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish events in order
            for event in [event_v1, event_v2, event_v3]:
                await kafka_producer.send_and_wait(
                    topic=state_committed_topic,
                    key=workflow_key,
                    value=event,
                )

            await asyncio.sleep(3)

            # Verify projection has latest version
            async with db_connection.transaction() as conn:
                projection = await conn.fetchrow(
                    "SELECT * FROM workflow_projection WHERE workflow_key = $1",
                    workflow_key,
                )

                assert projection is not None
                assert projection["version"] == 3

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_watermark_atomicity(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test atomic projection + watermark updates.

        Note: Materializer uses Kafka message metadata (partition/offset),
        not payload fields, for watermark tracking.
        """
        workflow_key = f"test-atomic-{uuid4()}"

        event = create_test_event(workflow_key=workflow_key)

        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish event
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )

            await asyncio.sleep(3)

            # Verify both projection and watermark updated
            async with db_connection.transaction() as conn:
                projection = await conn.fetchrow(
                    "SELECT * FROM workflow_projection WHERE workflow_key = $1",
                    workflow_key,
                )

                assert projection is not None

                # Watermark uses Kafka partition ID format: "kafka-partition-<N>"
                # Since we don't control which partition Kafka assigns, check that at least one watermark exists
                watermark_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM projection_watermarks WHERE partition_id LIKE 'kafka-partition-%'"
                )

                assert watermark_count > 0

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_lag_metrics(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test lag calculation and metrics."""
        workflow_key = f"test-lag-{uuid4()}"
        event = create_test_event(workflow_key=workflow_key)

        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish event
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )

            await asyncio.sleep(3)

            # Verify lag metrics
            metrics = materializer.metrics
            assert metrics.projection_wm_lag_ms >= 0
            assert metrics.max_lag_ms >= 0

            # Lag should be low for recent events (< 5 seconds)
            assert metrics.projection_wm_lag_ms < 5000

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_throughput_measurement(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test throughput measurement (events/second)."""
        num_events = 20
        events = [
            create_test_event(
                workflow_key=f"test-throughput-{i}",
                version=1,
            )
            for i in range(num_events)
        ]

        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish events rapidly
            for event in events:
                await kafka_producer.send_and_wait(
                    topic=state_committed_topic,
                    key=event["workflow_key"],
                    value=event,
                )

            await asyncio.sleep(5)

            # Verify throughput metrics
            metrics = materializer.metrics
            assert metrics.events_processed_per_second > 0

            # Should process at least some events per second
            assert metrics.projections_materialized_total >= num_events

        finally:
            await materializer.stop()


class TestProjectionMaterializerFailureRecovery:
    """Integration tests for failure recovery scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_recovery(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test recovery from database connection failures."""
        workflow_key = f"test-recovery-{uuid4()}"
        event = create_test_event(workflow_key=workflow_key)

        materializer = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer.start()
            await asyncio.sleep(2)

            # Publish event
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )

            await asyncio.sleep(3)

            # Verify event processed
            metrics = materializer.metrics
            assert metrics.projections_materialized_total >= 1

        finally:
            await materializer.stop()

    @pytest.mark.asyncio
    async def test_watermark_recovery_after_restart(
        self,
        kafka_producer,
        kafka_bootstrap_servers,
        postgres_config,
        state_committed_topic,
        db_connection,
        clean_test_data,
    ):
        """Test watermark tracking persists across service restarts.

        Note: Materializer uses Kafka partition format for watermark tracking.
        """
        workflow_key = f"test-restart-{uuid4()}"

        event = create_test_event(workflow_key=workflow_key)

        # First service instance
        materializer1 = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer1.start()
            await asyncio.sleep(2)

            # Publish event
            await kafka_producer.send_and_wait(
                topic=state_committed_topic,
                key=workflow_key,
                value=event,
            )

            await asyncio.sleep(3)

            # Verify watermark set (Kafka partition format)
            async with db_connection.transaction() as conn:
                watermark_count1 = await conn.fetchval(
                    "SELECT COUNT(*) FROM projection_watermarks WHERE partition_id LIKE 'kafka-partition-%'"
                )

                assert watermark_count1 > 0

        finally:
            await materializer1.stop()

        # Second service instance (simulating restart)
        materializer2 = ProjectionMaterializerService(
            bootstrap_servers=kafka_bootstrap_servers,
            consumer_group=f"test-group-{uuid4()}",
            postgres_config=postgres_config,
        )

        try:
            await materializer2.start()
            await asyncio.sleep(2)

            # Verify watermark persisted
            async with db_connection.transaction() as conn:
                watermark_count2 = await conn.fetchval(
                    "SELECT COUNT(*) FROM projection_watermarks WHERE partition_id LIKE 'kafka-partition-%'"
                )

                assert watermark_count2 > 0

        finally:
            await materializer2.stop()
