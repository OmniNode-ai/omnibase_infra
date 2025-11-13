"""
Integration tests for CodegenEventTracer.

Tests real event tracing operations using testcontainers for isolated
testing environment. Validates session event tracing, performance metrics,
and correlation-based event discovery with actual PostgreSQL database.

Requirements:
- Docker must be running for testcontainers
- PostgreSQL container will be automatically started and stopped
- Tests use isolated database to prevent interference

Test Coverage:
- Session event tracing with timing calculations
- Performance metrics calculation with percentiles
- Correlation ID-based event discovery
- Edge cases (no events, single event, many events)
- Timestamp ordering and filtering
- Status determination and aggregation
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

try:
    from testcontainers.postgres import PostgresContainer

    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    PostgresContainer = None  # type: ignore

from omninode_bridge.dashboard.codegen_event_tracer import CodegenEventTracer
from omninode_bridge.infrastructure.postgres_connection_manager import (
    ModelPostgresConfig,
    PostgresConnectionManager,
)

# Skip all tests if testcontainers not available
pytestmark = [
    pytest.mark.skipif(
        not TESTCONTAINERS_AVAILABLE,
        reason="testcontainers not installed - required for integration tests",
    ),
    pytest.mark.integration,
    pytest.mark.requires_infrastructure,
]


@pytest.fixture(scope="module")
def postgres_container():
    """
    Start PostgreSQL container for integration tests.

    Yields:
        PostgresContainer instance with running database
    """
    if not TESTCONTAINERS_AVAILABLE:
        pytest.skip("testcontainers not available")

    container = PostgresContainer("postgres:16-alpine")
    container.start()

    yield container

    container.stop()


@pytest.fixture
async def postgres_config(postgres_container) -> ModelPostgresConfig:
    """
    Create PostgreSQL configuration from container.

    Args:
        postgres_container: Running PostgreSQL container

    Returns:
        ModelPostgresConfig with container connection details
    """
    return ModelPostgresConfig(
        host=postgres_container.get_container_host_ip(),
        port=int(postgres_container.get_exposed_port(5432)),
        database=postgres_container.dbname,
        user=postgres_container.username,
        password=postgres_container.password,
        schema="public",
        min_connections=2,
        max_connections=10,
    )


@pytest.fixture
async def connection_manager(postgres_config) -> PostgresConnectionManager:
    """
    Create and initialize connection manager with real database.

    Args:
        postgres_config: PostgreSQL configuration from container

    Yields:
        Initialized PostgresConnectionManager instance
    """
    manager = PostgresConnectionManager(postgres_config)
    await manager.initialize()

    yield manager

    await manager.close()


@pytest.fixture
async def event_logs_table(connection_manager):
    """
    Create event_logs table for integration tests.

    Args:
        connection_manager: Initialized connection manager

    Yields:
        Table name for testing
    """
    table_name = "event_logs"

    # Create event_logs table with schema matching CodegenEventTracer expectations
    create_table_query = f"""
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";

    CREATE TABLE IF NOT EXISTS {table_name} (
        event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID NOT NULL,
        correlation_id UUID NOT NULL,
        event_type TEXT NOT NULL CHECK (event_type IN ('request', 'response', 'status', 'error')),
        topic TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        status TEXT NOT NULL CHECK (status IN ('sent', 'received', 'failed', 'processing', 'completed')),
        payload JSONB DEFAULT '{{}}'::jsonb,
        processing_time_ms INTEGER,
        metadata JSONB DEFAULT '{{}}'::jsonb,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    -- Create indexes for query performance
    CREATE INDEX IF NOT EXISTS idx_event_logs_session_id ON {table_name}(session_id);
    CREATE INDEX IF NOT EXISTS idx_event_logs_correlation_id ON {table_name}(correlation_id);
    CREATE INDEX IF NOT EXISTS idx_event_logs_timestamp ON {table_name}(timestamp);
    CREATE INDEX IF NOT EXISTS idx_event_logs_session_timestamp ON {table_name}(session_id, timestamp);
    """

    await connection_manager.execute_query(create_table_query)

    yield table_name

    # Cleanup
    await connection_manager.execute_query(f"DROP TABLE IF EXISTS {table_name} CASCADE")


@pytest.fixture
async def event_tracer(connection_manager, event_logs_table):
    """
    Create CodegenEventTracer instance for testing.

    Args:
        connection_manager: Initialized connection manager
        event_logs_table: Event logs table name

    Returns:
        CodegenEventTracer instance
    """
    return CodegenEventTracer(connection_manager)


@pytest.fixture
async def sample_session_id():
    """Generate sample session ID for testing."""
    return uuid4()


@pytest.fixture
async def sample_correlation_id():
    """Generate sample correlation ID for testing."""
    return uuid4()


@pytest.fixture
async def insert_sample_events(connection_manager, event_logs_table, sample_session_id):
    """
    Insert sample events for testing.

    Creates a realistic event chain:
    - Request event: omniclaude → omniarchon
    - Status event: Processing started
    - Response event: omniarchon → omniclaude
    - Status event: Completed

    Args:
        connection_manager: Database connection manager
        event_logs_table: Event logs table name
        sample_session_id: Session ID to use for events

    Returns:
        Dictionary with inserted event details
    """
    base_time = datetime.now(UTC) - timedelta(hours=2)
    correlation_id = uuid4()

    events = [
        {
            "event_id": uuid4(),
            "session_id": sample_session_id,
            "correlation_id": correlation_id,
            "event_type": "request",
            "topic": "omninode_codegen_request_analyze_v1",
            "timestamp": base_time,
            "status": "sent",
            "payload": {"query": "analyze code quality", "language": "python"},
            "processing_time_ms": None,
            "metadata": {"source": "omniclaude", "version": "1.0"},
        },
        {
            "event_id": uuid4(),
            "session_id": sample_session_id,
            "correlation_id": correlation_id,
            "event_type": "status",
            "topic": "omninode_codegen_status_v1",
            "timestamp": base_time + timedelta(milliseconds=100),
            "status": "processing",
            "payload": {"message": "Analysis started"},
            "processing_time_ms": None,
            "metadata": {"source": "omniarchon"},
        },
        {
            "event_id": uuid4(),
            "session_id": sample_session_id,
            "correlation_id": correlation_id,
            "event_type": "response",
            "topic": "omninode_codegen_response_analyze_v1",
            "timestamp": base_time + timedelta(milliseconds=1500),
            "status": "received",
            "payload": {
                "results": {"quality_score": 0.85, "issues": []},
                "analysis_complete": True,
            },
            "processing_time_ms": 1400,
            "metadata": {"source": "omniarchon", "version": "1.0"},
        },
        {
            "event_id": uuid4(),
            "session_id": sample_session_id,
            "correlation_id": correlation_id,
            "event_type": "status",
            "topic": "omninode_codegen_status_v1",
            "timestamp": base_time + timedelta(milliseconds=1600),
            "status": "completed",
            "payload": {"message": "Analysis completed successfully"},
            "processing_time_ms": None,
            "metadata": {"source": "omniclaude"},
        },
    ]

    # Insert events
    for event in events:
        insert_query = f"""
        INSERT INTO {event_logs_table} (
            event_id, session_id, correlation_id, event_type,
            topic, timestamp, status, payload, processing_time_ms, metadata
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10::jsonb)
        """
        await connection_manager.execute_query(
            insert_query,
            event["event_id"],
            event["session_id"],
            event["correlation_id"],
            event["event_type"],
            event["topic"],
            event["timestamp"],
            event["status"],
            json.dumps(event["payload"]),  # Convert to JSON string for ::jsonb cast
            event["processing_time_ms"],
            json.dumps(event["metadata"]),  # Convert to JSON string for ::jsonb cast
        )

    return {
        "session_id": sample_session_id,
        "correlation_id": correlation_id,
        "events": events,
        "base_time": base_time,
    }


# Integration Tests


@pytest.mark.integration
@pytest.mark.asyncio
class TestCodegenEventTracerIntegration:
    """Integration test suite for CodegenEventTracer."""

    async def test_trace_session_events_returns_complete_event_chain(
        self, event_tracer, insert_sample_events
    ):
        """Test trace_session_events returns complete event chain for a session."""
        session_id = insert_sample_events["session_id"]

        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        assert trace.session_id == session_id
        assert trace.total_events == 4  # 4 events inserted by fixture
        assert len(trace.events) == 4
        assert trace.status == "completed"  # Latest event status is "completed"
        assert trace.time_range_hours == 24

        # Verify events are ordered by timestamp
        timestamps = [event.timestamp for event in trace.events]
        assert timestamps == sorted(timestamps)

        # Verify event types
        event_types = [event.event_type for event in trace.events]
        assert event_types == ["request", "status", "response", "status"]

    async def test_trace_session_events_calculates_correct_duration(
        self, event_tracer, insert_sample_events
    ):
        """Test trace_session_events calculates correct session duration."""
        session_id = insert_sample_events["session_id"]

        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        # Duration should be ~1600ms (first to last event: 0ms to 1600ms)
        assert trace.session_duration_ms == 1600
        assert trace.start_time is not None
        assert trace.end_time is not None
        assert trace.start_time < trace.end_time

    async def test_trace_session_events_with_time_range_filtering(
        self, event_tracer, insert_sample_events, connection_manager, event_logs_table
    ):
        """Test trace_session_events filters events by time range."""
        session_id = insert_sample_events["session_id"]

        # Insert old event (25 hours ago)
        old_event_id = uuid4()
        old_timestamp = datetime.now(UTC) - timedelta(hours=25)

        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            old_event_id,
            session_id,
            uuid4(),
            "status",
            "omninode_codegen_status_v1",
            old_timestamp,
            "completed",
        )

        # Query with 24-hour time range
        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        # Should only return recent events (4 from fixture, not the old one)
        assert trace.total_events == 4
        assert all(event.timestamp > old_timestamp for event in trace.events)

    async def test_trace_session_events_with_no_events(self, event_tracer):
        """Test trace_session_events handles session with no events."""
        non_existent_session = uuid4()

        trace = await event_tracer.trace_session_events(
            non_existent_session, time_range_hours=24
        )

        assert trace.session_id == non_existent_session
        assert trace.total_events == 0
        assert trace.events == []
        assert trace.session_duration_ms == 0
        assert trace.status == "unknown"
        assert trace.start_time is None
        assert trace.end_time is None

    async def test_trace_session_events_with_single_event(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test trace_session_events handles session with single event."""
        session_id = uuid4()
        event_id = uuid4()
        correlation_id = uuid4()

        # Insert single event
        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            event_id,
            session_id,
            correlation_id,
            "request",
            "omninode_codegen_request_analyze_v1",
            datetime.now(UTC),
            "sent",
        )

        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        # Should return 1 event with 0 duration
        assert trace.total_events == 1
        assert trace.session_duration_ms == 0
        assert trace.status == "in_progress"  # Single request event status is "sent"

    async def test_trace_session_events_with_many_events(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test trace_session_events handles session with many events (>100)."""
        session_id = uuid4()
        correlation_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert 150 events
        for i in range(150):
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                uuid4(),
                session_id,
                correlation_id,
                "status",
                "omninode_codegen_status_v1",
                base_time + timedelta(milliseconds=i * 10),
                "processing",
            )

        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        # Should return 150 events
        assert trace.total_events == 150
        assert len(trace.events) == 150

        # Verify events are ordered by timestamp
        timestamps = [event.timestamp for event in trace.events]
        assert timestamps == sorted(timestamps)

        # Duration should be ~1490ms (0ms to 1490ms, 150 events * 10ms spacing)
        assert trace.session_duration_ms == 1490

    async def test_get_session_metrics_calculates_correct_performance_metrics(
        self, event_tracer, insert_sample_events
    ):
        """Test get_session_metrics calculates correct performance metrics."""
        session_id = insert_sample_events["session_id"]

        metrics = await event_tracer.get_session_metrics(session_id)

        assert metrics.session_id == str(session_id)
        assert metrics.total_events == 4  # 4 events from fixture
        assert (
            metrics.successful_events == 3
        )  # 3 successful (sent, received, completed), 1 processing
        assert metrics.failed_events == 0
        assert metrics.success_rate == 0.75  # 75% success rate (3/4)

        # Verify response time metrics from single response event (1400ms)
        assert metrics.avg_response_time_ms == 1400.0
        assert metrics.min_response_time_ms == 1400
        assert metrics.max_response_time_ms == 1400
        assert metrics.p50_response_time_ms == 1400
        assert metrics.p95_response_time_ms == 1400
        assert metrics.p99_response_time_ms == 1400
        assert metrics.total_processing_time_ms == 1400

    async def test_get_session_metrics_event_type_breakdown(
        self, event_tracer, insert_sample_events
    ):
        """Test get_session_metrics provides event type breakdown."""
        session_id = insert_sample_events["session_id"]

        metrics = await event_tracer.get_session_metrics(session_id)

        assert isinstance(metrics.event_type_breakdown, dict)
        # Expected: 1 request, 1 response, 2 status
        assert metrics.event_type_breakdown["request"] == 1
        assert metrics.event_type_breakdown["response"] == 1
        assert metrics.event_type_breakdown["status"] == 2

    async def test_get_session_metrics_topic_breakdown(
        self, event_tracer, insert_sample_events
    ):
        """Test get_session_metrics provides topic breakdown."""
        session_id = insert_sample_events["session_id"]

        metrics = await event_tracer.get_session_metrics(session_id)

        assert isinstance(metrics.topic_breakdown, dict)
        # Expected breakdown by topics
        assert "omninode_codegen_request_analyze_v1" in metrics.topic_breakdown
        assert "omninode_codegen_response_analyze_v1" in metrics.topic_breakdown
        assert "omninode_codegen_status_v1" in metrics.topic_breakdown
        assert metrics.topic_breakdown["omninode_codegen_request_analyze_v1"] == 1
        assert metrics.topic_breakdown["omninode_codegen_response_analyze_v1"] == 1
        assert metrics.topic_breakdown["omninode_codegen_status_v1"] == 2

    async def test_get_session_metrics_identifies_bottlenecks(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test get_session_metrics identifies performance bottlenecks."""
        session_id = uuid4()
        correlation_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert slow request/response pair (>5000ms = bottleneck)
        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status, processing_time_ms
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            uuid4(),
            session_id,
            correlation_id,
            "response",
            "omninode_codegen_response_validate_v1",
            base_time,
            "received",
            6500,  # Slow response (>5000ms = high severity bottleneck)
        )

        metrics = await event_tracer.get_session_metrics(session_id)

        assert isinstance(metrics.bottlenecks, list)
        # Should identify bottleneck for validate topic (6500ms > 5000ms = high severity)
        assert len(metrics.bottlenecks) == 1
        bottleneck = metrics.bottlenecks[0]
        assert bottleneck.topic == "omninode_codegen_response_validate_v1"
        assert bottleneck.avg_response_time_ms == 6500.0
        assert bottleneck.count == 1
        assert bottleneck.severity == "high"

    async def test_get_session_metrics_timeline_calculation(
        self, event_tracer, insert_sample_events
    ):
        """Test get_session_metrics calculates timeline correctly."""
        session_id = insert_sample_events["session_id"]

        metrics = await event_tracer.get_session_metrics(session_id)

        assert metrics.timeline is not None
        assert metrics.timeline.start_time is not None
        assert metrics.timeline.end_time is not None
        assert metrics.timeline.start_time < metrics.timeline.end_time
        assert metrics.timeline.duration_ms == 1600  # First to last event: 1600ms

    async def test_get_session_metrics_with_no_events(self, event_tracer):
        """Test get_session_metrics handles session with no events."""
        non_existent_session = uuid4()

        metrics = await event_tracer.get_session_metrics(non_existent_session)

        assert metrics.total_events == 0
        assert metrics.successful_events == 0
        assert metrics.failed_events == 0
        assert metrics.success_rate == 0.0

    async def test_find_correlated_events_returns_complete_chain(
        self, event_tracer, insert_sample_events
    ):
        """Test find_correlated_events returns complete event chain."""
        correlation_id = insert_sample_events["correlation_id"]

        events = await event_tracer.find_correlated_events(correlation_id)

        # Should return 4 events with matching correlation_id
        assert isinstance(events, list)
        assert len(events) == 4
        assert all(event.correlation_id == correlation_id for event in events)

        # Verify event chain
        event_types = [event.event_type for event in events]
        assert event_types == ["request", "status", "response", "status"]

    async def test_find_correlated_events_ordered_by_timestamp(
        self, event_tracer, insert_sample_events
    ):
        """Test find_correlated_events returns events ordered by timestamp."""
        correlation_id = insert_sample_events["correlation_id"]

        events = await event_tracer.find_correlated_events(correlation_id)

        # Verify chronological ordering
        assert isinstance(events, list)
        timestamps = [event.timestamp for event in events]
        assert timestamps == sorted(timestamps)

    async def test_find_correlated_events_with_no_matches(self, event_tracer):
        """Test find_correlated_events handles no matching events."""
        non_existent_correlation = uuid4()

        events = await event_tracer.find_correlated_events(non_existent_correlation)

        assert events == []

    async def test_find_correlated_events_with_single_event(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test find_correlated_events handles single event."""
        session_id = uuid4()
        correlation_id = uuid4()

        # Insert single event
        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            uuid4(),
            session_id,
            correlation_id,
            "request",
            "omninode_codegen_request_analyze_v1",
            datetime.now(UTC),
            "sent",
        )

        events = await event_tracer.find_correlated_events(correlation_id)

        # Should return list with single event
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0].event_type == "request"
        assert events[0].correlation_id == correlation_id

    async def test_find_correlated_events_cross_session(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test find_correlated_events finds events across multiple sessions."""
        correlation_id = uuid4()
        session_id_1 = uuid4()
        session_id_2 = uuid4()
        base_time = datetime.now(UTC)

        # Insert events with same correlation_id but different sessions
        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status, payload, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            uuid4(),
            session_id_1,
            correlation_id,
            "request",
            "omninode_codegen_request_analyze_v1",
            base_time,
            "sent",
            json.dumps({}),
            json.dumps({}),
        )

        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status, payload, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            uuid4(),
            session_id_2,
            correlation_id,
            "response",
            "omninode_codegen_response_analyze_v1",
            base_time + timedelta(milliseconds=100),
            "received",
            json.dumps({}),
            json.dumps({}),
        )

        events = await event_tracer.find_correlated_events(correlation_id)

        # Should return 2 events from different sessions
        assert isinstance(events, list)
        assert len(events) == 2
        assert events[0].session_id == session_id_1
        assert events[1].session_id == session_id_2
        assert events[0].correlation_id == correlation_id
        assert events[1].correlation_id == correlation_id

    async def test_concurrent_trace_operations(
        self, event_tracer, insert_sample_events, connection_manager, event_logs_table
    ):
        """Test concurrent event tracing operations."""
        session_id_1 = insert_sample_events["session_id"]
        session_id_2 = uuid4()

        # Insert events for second session
        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            uuid4(),
            session_id_2,
            uuid4(),
            "request",
            "omninode_codegen_request_validate_v1",
            datetime.now(UTC),
            "sent",
        )

        # Execute concurrent trace operations
        results = await asyncio.gather(
            event_tracer.trace_session_events(session_id_1, time_range_hours=24),
            event_tracer.trace_session_events(session_id_2, time_range_hours=24),
            event_tracer.get_session_metrics(session_id_1),
        )

        # All operations should complete successfully
        assert len(results) == 3
        assert results[0].session_id == session_id_1
        assert results[1].session_id == session_id_2
        assert results[2].session_id == str(session_id_1)

    async def test_event_logs_index_performance(
        self, connection_manager, event_logs_table
    ):
        """Test that event_logs indexes improve query performance."""
        # Verify indexes were created
        index_query = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE tablename = $1
        ORDER BY indexname
        """

        indexes = await connection_manager.execute_query(index_query, event_logs_table)

        index_names = [idx["indexname"] for idx in indexes]

        # Verify expected indexes exist
        assert any("session_id" in name for name in index_names)
        assert any("correlation_id" in name for name in index_names)
        assert any("timestamp" in name for name in index_names)

    # === NEW TEST CASES FOR COMPREHENSIVE COVERAGE ===

    async def test_trace_session_events_performance_with_1000_events(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test trace_session_events with 1000+ events for performance validation."""
        session_id = uuid4()
        correlation_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert 1000 events
        for i in range(1000):
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status, payload, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                uuid4(),
                session_id,
                correlation_id,
                "status" if i % 2 == 0 else "request",
                "omninode_codegen_status_v1",
                base_time + timedelta(milliseconds=i),
                "processing",
                json.dumps({"iteration": i}),
                json.dumps({"batch": i // 100}),
            )

        # Trace events and validate performance
        import time

        start = time.time()
        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)
        elapsed_ms = (time.time() - start) * 1000

        # Verify results
        assert trace.total_events == 1000
        assert len(trace.events) == 1000
        assert trace.session_duration_ms == 999  # 0ms to 999ms

        # Performance validation: Should complete in < 500ms
        assert elapsed_ms < 500, f"Query took {elapsed_ms:.2f}ms, expected < 500ms"

    async def test_get_session_metrics_percentile_accuracy(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test percentile calculations with known distribution of processing times."""
        session_id = uuid4()
        base_time = datetime.now(UTC)

        # Create known distribution of processing times: 1ms, 2ms, 3ms, ..., 100ms
        processing_times = list(range(1, 101))

        for i, processing_time in enumerate(processing_times):
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status, processing_time_ms, payload, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                uuid4(),
                session_id,
                uuid4(),
                "response",
                "omninode_codegen_response_test_v1",
                base_time + timedelta(milliseconds=i),
                "received",
                processing_time,
                json.dumps({}),
                json.dumps({}),
            )

        metrics = await event_tracer.get_session_metrics(session_id)

        # Verify percentile calculations
        # P50 (median) should be around 50ms
        assert 49 <= metrics.p50_response_time_ms <= 51
        # P95 should be around 95ms
        assert 94 <= metrics.p95_response_time_ms <= 96
        # P99 should be around 99ms
        assert 98 <= metrics.p99_response_time_ms <= 100

        # Verify min/max
        assert metrics.min_response_time_ms == 1
        assert metrics.max_response_time_ms == 100

        # Verify average
        expected_avg = sum(processing_times) / len(processing_times)
        assert abs(metrics.avg_response_time_ms - expected_avg) < 0.1

    async def test_trace_session_events_with_mixed_statuses(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test session status determination with mixed event statuses."""
        session_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert events with various statuses, ending with failed
        event_data = [
            ("request", "sent"),
            ("status", "processing"),
            ("status", "processing"),
            ("response", "received"),
            ("status", "failed"),  # Last event is failed
        ]

        for i, (event_type, status) in enumerate(event_data):
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status, payload, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                uuid4(),
                session_id,
                uuid4(),
                event_type,
                f"omninode_codegen_{event_type}_v1",
                base_time + timedelta(milliseconds=i * 100),
                status,
                json.dumps({}),
                json.dumps({}),
            )

        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        # Session should be marked as failed due to last event
        assert trace.status == "failed"
        assert trace.total_events == 5

    async def test_get_session_metrics_with_multiple_bottlenecks(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test bottleneck identification with multiple slow topics."""
        session_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert events with various processing times
        bottleneck_data = [
            ("omninode_codegen_response_slow_v1", 6000, "high"),  # High severity
            ("omninode_codegen_response_medium_v1", 3000, "medium"),  # Medium severity
            ("omninode_codegen_response_low_v1", 1500, "low"),  # Low severity
            ("omninode_codegen_response_fast_v1", 500, None),  # No bottleneck
        ]

        for event_id, (topic, processing_time, _) in enumerate(bottleneck_data):
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status, processing_time_ms, payload, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                uuid4(),
                session_id,
                uuid4(),
                "response",
                topic,
                base_time + timedelta(milliseconds=event_id * 100),
                "received",
                processing_time,
                json.dumps({}),
                json.dumps({}),
            )

        metrics = await event_tracer.get_session_metrics(session_id)

        # Verify all bottlenecks identified (should be 3, fast_v1 not included)
        assert len(metrics.bottlenecks) == 3

        # Verify sorting by severity
        assert metrics.bottlenecks[0].severity == "high"
        assert metrics.bottlenecks[1].severity == "medium"
        assert metrics.bottlenecks[2].severity == "low"

        # Verify details
        assert metrics.bottlenecks[0].topic == "omninode_codegen_response_slow_v1"
        assert metrics.bottlenecks[0].avg_response_time_ms == 6000.0

    async def test_find_correlated_events_with_large_correlation_chain(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test finding correlated events with large chain (100+ events)."""
        correlation_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert 100 events with same correlation_id across multiple sessions
        for i in range(100):
            session_id = uuid4()
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status, payload, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                uuid4(),
                session_id,
                correlation_id,
                "status",
                "omninode_codegen_status_v1",
                base_time + timedelta(milliseconds=i),
                "processing",
                json.dumps({"step": i}),
                json.dumps({}),
            )

        events = await event_tracer.find_correlated_events(correlation_id)

        assert len(events) == 100
        assert all(event.correlation_id == correlation_id for event in events)

        # Verify chronological ordering
        timestamps = [event.timestamp for event in events]
        assert timestamps == sorted(timestamps)

    async def test_get_session_metrics_with_no_processing_times(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test metrics calculation when no events have processing_time_ms."""
        session_id = uuid4()
        base_time = datetime.now(UTC)

        # Insert events without processing times (status events)
        for i in range(5):
            await connection_manager.execute_query(
                f"""
                INSERT INTO {event_logs_table} (
                    event_id, session_id, correlation_id, event_type,
                    topic, timestamp, status, payload, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """,
                uuid4(),
                session_id,
                uuid4(),
                "status",
                "omninode_codegen_status_v1",
                base_time + timedelta(milliseconds=i * 100),
                "processing",
                json.dumps({}),
                json.dumps({}),
            )

        metrics = await event_tracer.get_session_metrics(session_id)

        # All response time metrics should be 0
        assert metrics.avg_response_time_ms == 0.0
        assert metrics.min_response_time_ms == 0
        assert metrics.max_response_time_ms == 0
        assert metrics.p50_response_time_ms == 0
        assert metrics.p95_response_time_ms == 0
        assert metrics.p99_response_time_ms == 0
        assert metrics.total_processing_time_ms == 0

        # No bottlenecks should be identified
        assert len(metrics.bottlenecks) == 0

    async def test_trace_session_events_timezone_handling(
        self, event_tracer, connection_manager, event_logs_table
    ):
        """Test timestamp timezone handling for session events."""
        session_id = uuid4()

        # Insert event with explicit timezone
        event_timestamp = datetime.now(UTC)

        await connection_manager.execute_query(
            f"""
            INSERT INTO {event_logs_table} (
                event_id, session_id, correlation_id, event_type,
                topic, timestamp, status, payload, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            uuid4(),
            session_id,
            uuid4(),
            "request",
            "omninode_codegen_request_test_v1",
            event_timestamp,
            "sent",
            json.dumps({}),
            json.dumps({}),
        )

        trace = await event_tracer.trace_session_events(session_id, time_range_hours=24)

        assert trace.total_events == 1
        # Verify timestamp is timezone-aware
        assert trace.start_time is not None
        assert trace.start_time.tzinfo is not None
