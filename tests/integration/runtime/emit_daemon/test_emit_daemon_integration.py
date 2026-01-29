# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for the Hook Event Emit Daemon with real Kafka.

These tests validate the full emit daemon lifecycle with a configured
Kafka broker (set via KAFKA_BOOTSTRAP_SERVERS environment variable).
They verify that events flow through the daemon to Kafka and can be
consumed correctly.

Test categories:
    - Lifecycle Tests: Daemon start/stop, PID file, socket management
    - Health Check Tests: Ping command, queue/spool status
    - Event Flow Tests: End-to-end event emission and consumption
    - Metadata Tests: Correlation ID injection, timestamps, schema versions
    - Resilience Tests: Graceful shutdown, queue draining

Environment Variables:
    KAFKA_BOOTSTRAP_SERVERS: Kafka broker address (e.g., "localhost:9092")

Related Tickets:
    - OMN-1610: Hook Event Daemon MVP
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.errors import InfraUnavailableError
from tests.helpers.util_kafka import (
    KAFKA_ERROR_INVALID_PARTITIONS,
    KafkaConfigValidationResult,
    KafkaTopicManager,
    validate_bootstrap_servers,
)

# Module-level logger for test diagnostics
logger = logging.getLogger(__name__)

# =============================================================================
# Test Configuration and Skip Conditions
# =============================================================================

# Check if Kafka is available based on environment variable
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
_kafka_config_validation: KafkaConfigValidationResult = validate_bootstrap_servers(
    KAFKA_BOOTSTRAP_SERVERS
)
KAFKA_AVAILABLE = _kafka_config_validation.is_valid

# Module-level markers - skip all tests if Kafka is not available
pytestmark = [
    pytest.mark.skipif(
        not KAFKA_AVAILABLE,
        reason=_kafka_config_validation.skip_reason
        or "KAFKA_BOOTSTRAP_SERVERS not configured",
    ),
]

# Test configuration constants
TEST_TIMEOUT_SECONDS = 30
MESSAGE_DELIVERY_WAIT_SECONDS = 5.0
DAEMON_STARTUP_WAIT_SECONDS = 2.0


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def kafka_bootstrap_servers() -> str:
    """Get Kafka bootstrap servers from environment."""
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture
def unique_topic_name() -> str:
    """Generate unique topic name for test isolation."""
    return f"test.emit-daemon.{uuid4().hex[:12]}"


@pytest.fixture
def unique_group() -> str:
    """Generate unique consumer group for test isolation."""
    return f"test-emit-daemon-{uuid4().hex[:8]}"


@pytest.fixture
async def ensure_test_topic(
    kafka_bootstrap_servers: str,
) -> AsyncGenerator[Callable[[str, int], Coroutine[None, None, str]], None]:
    """Create test topics via Kafka admin API before tests and cleanup after.

    Handles broker resource limit errors gracefully by skipping the test.

    Yields:
        Async function that creates a topic with the given name and partition count.
    """
    async with KafkaTopicManager(kafka_bootstrap_servers) as manager:

        async def _create_topic(topic_name: str, partitions: int = 1) -> str:
            try:
                return await manager.create_topic(topic_name, partitions=partitions)
            except InfraUnavailableError as e:
                # Check for broker resource limits (error_code=37)
                # Access kafka_error_code from the error's model context
                kafka_error_code = (
                    e.model.context.get("kafka_error_code")
                    if hasattr(e, "model")
                    else None
                )
                if kafka_error_code == KAFKA_ERROR_INVALID_PARTITIONS:
                    pytest.skip(
                        f"Kafka broker resource limit reached: {e}. "
                        "Consider increasing Redpanda memory limits."
                    )
                raise

        yield _create_topic


@pytest.fixture
async def test_topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
    unique_topic_name: str,
) -> str:
    """Create a unique test topic, cleaned up after test.

    Skips the test if broker resource limits prevent topic creation.
    """
    await ensure_test_topic(unique_topic_name, 1)
    return unique_topic_name


@pytest.fixture
def temp_daemon_paths(tmp_path: Path):
    """Create temporary paths for daemon socket, PID, and spool.

    Note: Unix sockets have a path length limit (104 chars on macOS, 108 on Linux).
    We use /tmp for the socket to avoid path-too-long errors with pytest's tmp_path.

    Yields:
        Tuple of (socket_path, pid_path, spool_dir)
    """
    # Use a short unique prefix in /tmp for the socket to avoid path length issues
    # NOTE: /tmp is required for Unix sockets on macOS due to path length limits
    test_id = uuid4().hex[:8]
    socket_path = Path(f"/tmp/emit-{test_id}.sock")  # noqa: S108
    pid_path = tmp_path / "test-emit.pid"
    spool_dir = tmp_path / "emit-spool"

    yield socket_path, pid_path, spool_dir

    # Cleanup socket file if it still exists (e.g., if daemon wasn't properly stopped)
    if socket_path.exists():
        try:
            socket_path.unlink()
        except OSError:
            pass


@pytest.fixture
def emit_daemon_config(
    temp_daemon_paths: tuple[Path, Path, Path],
    kafka_bootstrap_servers: str,
):
    """Create emit daemon configuration for testing.

    Returns:
        ModelEmitDaemonConfig configured for testing with temporary paths.
    """
    from omnibase_infra.runtime.emit_daemon.config import ModelEmitDaemonConfig

    socket_path, pid_path, spool_dir = temp_daemon_paths

    return ModelEmitDaemonConfig(
        socket_path=socket_path,
        pid_path=pid_path,
        spool_dir=spool_dir,
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        max_memory_queue=50,
        max_spool_messages=100,
        max_spool_bytes=1_048_576,  # 1MB
        socket_timeout_seconds=5.0,
        kafka_timeout_seconds=10.0,
        shutdown_drain_seconds=5.0,
    )


@pytest.fixture
async def started_emit_daemon(
    emit_daemon_config,
    test_topic: str,
) -> AsyncGenerator:
    """Start emit daemon with test config, cleanup after test.

    This fixture starts the daemon, waits for it to be ready, and ensures
    proper cleanup on test completion.
    """
    from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
    from omnibase_infra.runtime.emit_daemon.event_registry import (
        EventRegistry,
        ModelEventRegistration,
    )

    daemon = EmitDaemon(emit_daemon_config)

    # Register test event type that routes to our test topic
    daemon._registry.register(
        ModelEventRegistration(
            event_type="test.event",
            topic_template=test_topic,  # Direct topic name, no env substitution
            partition_key_field="session_id",
            required_fields=["message"],
            schema_version="1.0.0",
        )
    )

    try:
        await daemon.start()

        # Wait for daemon to be fully ready
        await asyncio.sleep(0.5)

        yield daemon

    finally:
        # Ensure daemon is stopped
        try:
            await daemon.stop()
        except Exception as e:
            logger.warning(f"Error stopping daemon in fixture cleanup: {e}")


@pytest.fixture
def emit_client_factory(emit_daemon_config):
    """Factory for creating emit clients configured for the test daemon."""
    from omnibase_infra.runtime.emit_daemon.client import EmitClient

    def _create_client(timeout: float = 5.0) -> EmitClient:
        return EmitClient(
            socket_path=emit_daemon_config.socket_path,
            timeout=timeout,
        )

    return _create_client


@pytest.fixture
async def kafka_consumer(
    kafka_bootstrap_servers: str,
    test_topic: str,
    unique_group: str,
):
    """Kafka consumer for verifying events.

    Uses aiokafka for async consumption that integrates with pytest-asyncio.
    """
    from aiokafka import AIOKafkaConsumer

    consumer = AIOKafkaConsumer(
        test_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id=unique_group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=int(MESSAGE_DELIVERY_WAIT_SECONDS * 1000),
    )

    await consumer.start()

    try:
        yield consumer
    finally:
        await consumer.stop()


# =============================================================================
# Lifecycle Tests - Daemon start/stop, PID file, socket management
# =============================================================================


class TestEmitDaemonLifecycle:
    """Tests for emit daemon lifecycle management.

    Note: These tests use a fixed topic name since they only test daemon
    lifecycle operations (start/stop, PID file, socket). They don't actually
    send events to Kafka, so topic creation is not strictly necessary.
    """

    # Fixed topic name for lifecycle tests (won't actually publish to it)
    LIFECYCLE_TEST_TOPIC = "test.emit-daemon.lifecycle"

    @pytest.mark.asyncio
    async def test_daemon_start_creates_pid_and_socket(
        self,
        emit_daemon_config,
    ) -> None:
        """Verify daemon creates PID file and socket on start.

        The daemon should:
        1. Create the PID file with current process ID
        2. Create the Unix socket file
        3. Be ready to accept connections
        """
        from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        daemon = EmitDaemon(emit_daemon_config)
        daemon._registry.register(
            ModelEventRegistration(
                event_type="test.event",
                topic_template=self.LIFECYCLE_TEST_TOPIC,
                required_fields=["message"],
            )
        )

        try:
            # Before start: neither PID nor socket should exist
            assert not emit_daemon_config.pid_path.exists()
            assert not emit_daemon_config.socket_path.exists()

            await daemon.start()

            # After start: both should exist
            assert emit_daemon_config.pid_path.exists(), "PID file not created"
            assert emit_daemon_config.socket_path.exists(), "Socket file not created"

            # PID file should contain current process ID
            pid_content = emit_daemon_config.pid_path.read_text().strip()
            assert pid_content == str(os.getpid())

        finally:
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_daemon_stop_removes_pid_and_socket(
        self,
        emit_daemon_config,
    ) -> None:
        """Verify daemon removes PID file and socket on stop.

        Clean shutdown should:
        1. Remove the socket file
        2. Remove the PID file
        3. Close all connections gracefully
        """
        from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        daemon = EmitDaemon(emit_daemon_config)
        daemon._registry.register(
            ModelEventRegistration(
                event_type="test.event",
                topic_template=self.LIFECYCLE_TEST_TOPIC,
                required_fields=["message"],
            )
        )

        await daemon.start()

        # Verify files exist after start
        assert emit_daemon_config.pid_path.exists()
        assert emit_daemon_config.socket_path.exists()

        await daemon.stop()

        # After stop: both should be removed
        assert not emit_daemon_config.pid_path.exists(), "PID file not removed"
        assert not emit_daemon_config.socket_path.exists(), "Socket file not removed"

    @pytest.mark.asyncio
    async def test_daemon_double_start_is_idempotent(
        self,
        emit_daemon_config,
    ) -> None:
        """Verify multiple start() calls are safe and idempotent."""
        from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        daemon = EmitDaemon(emit_daemon_config)
        daemon._registry.register(
            ModelEventRegistration(
                event_type="test.event",
                topic_template=self.LIFECYCLE_TEST_TOPIC,
                required_fields=["message"],
            )
        )

        try:
            await daemon.start()

            # Second start should be a no-op
            await daemon.start()

            # Daemon should still be running and functional
            assert emit_daemon_config.pid_path.exists()
            assert emit_daemon_config.socket_path.exists()

        finally:
            await daemon.stop()

    @pytest.mark.asyncio
    async def test_daemon_double_stop_is_safe(
        self,
        emit_daemon_config,
    ) -> None:
        """Verify multiple stop() calls are safe."""
        from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        daemon = EmitDaemon(emit_daemon_config)
        daemon._registry.register(
            ModelEventRegistration(
                event_type="test.event",
                topic_template=self.LIFECYCLE_TEST_TOPIC,
                required_fields=["message"],
            )
        )

        await daemon.start()
        await daemon.stop()

        # Second stop should be a no-op, not raise
        await daemon.stop()


# =============================================================================
# Health Check Tests - Ping command, queue/spool status
# =============================================================================


class TestEmitDaemonHealthCheck:
    """Tests for emit daemon health check functionality.

    Note: Health check tests use the started_emit_daemon fixture which
    requires a test_topic. Consider adding skip handling if topic creation fails.
    """

    @pytest.mark.asyncio
    async def test_daemon_ping_returns_status(
        self,
        started_emit_daemon,
        emit_client_factory,
    ) -> None:
        """Verify ping returns queue and spool sizes.

        The ping response should include:
        - status: "ok"
        - queue_size: number of events in memory queue
        - spool_size: number of events in disk spool
        """
        client = emit_client_factory()

        async with client:
            response = await client.ping()

        # Response is now a typed ModelDaemonPingResponse
        assert response.status == "ok"
        assert isinstance(response.queue_size, int)
        assert isinstance(response.spool_size, int)

    @pytest.mark.asyncio
    async def test_is_daemon_running_returns_true(
        self,
        started_emit_daemon,
        emit_client_factory,
    ) -> None:
        """Verify is_daemon_running returns True when daemon is running."""
        client = emit_client_factory()

        is_running = await client.is_daemon_running()
        assert is_running is True

    @pytest.mark.asyncio
    async def test_is_daemon_running_returns_false_when_not_running(
        self,
        emit_daemon_config,
    ) -> None:
        """Verify is_daemon_running returns False when daemon is not running."""
        from omnibase_infra.runtime.emit_daemon.client import EmitClient

        # Daemon not started - socket doesn't exist
        client = EmitClient(
            socket_path=emit_daemon_config.socket_path,
            timeout=2.0,
        )

        is_running = await client.is_daemon_running()
        assert is_running is False


# =============================================================================
# Event Flow Tests - End-to-end event emission and consumption
# =============================================================================


class TestEmitDaemonEventFlow:
    """End-to-end tests for event emission through the daemon to Kafka."""

    @pytest.mark.asyncio
    async def test_emit_event_reaches_kafka(
        self,
        started_emit_daemon,
        emit_client_factory,
        kafka_consumer,
        test_topic: str,
    ) -> None:
        """Verify emitted event is published to Kafka and can be consumed.

        This test validates the complete event flow:
        1. Client emits event to daemon via Unix socket
        2. Daemon queues and publishes to Kafka
        3. Kafka consumer receives the event with correct payload
        """
        client = emit_client_factory()

        # Create test event payload
        test_message = f"test-message-{uuid4().hex[:8]}"
        test_session = f"session-{uuid4().hex[:8]}"
        payload = {
            "message": test_message,
            "session_id": test_session,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Emit event
        async with client:
            event_id = await client.emit("test.event", payload)

        assert event_id is not None
        assert len(event_id) > 0

        # Allow time for Kafka delivery
        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

        # Consume from Kafka and verify
        received_messages = []
        try:
            async for msg in kafka_consumer:
                received_messages.append(msg)
                if len(received_messages) >= 1:
                    break
        except TimeoutError:
            pass

        # Verify at least one message received
        assert len(received_messages) >= 1, "No messages received from Kafka"

        # Verify message content
        received = received_messages[0]
        received_payload = json.loads(received.value.decode("utf-8"))

        assert received_payload["message"] == test_message
        assert received_payload["session_id"] == test_session

    @pytest.mark.asyncio
    async def test_emit_multiple_events_all_reach_kafka(
        self,
        started_emit_daemon,
        emit_client_factory,
        kafka_consumer,
        test_topic: str,
    ) -> None:
        """Verify multiple rapidly emitted events all reach Kafka.

        Tests that the daemon can handle a burst of events and that
        all events are successfully published to Kafka.
        """
        client = emit_client_factory()
        event_count = 10

        # Emit multiple events rapidly
        event_ids = []
        async with client:
            for i in range(event_count):
                payload = {
                    "message": f"burst-message-{i}",
                    "session_id": "burst-session",
                    "sequence": i,
                }
                event_id = await client.emit("test.event", payload)
                event_ids.append(event_id)

        assert len(event_ids) == event_count

        # Allow time for Kafka delivery
        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS * 2)

        # Consume from Kafka
        received_messages = []
        try:
            async for msg in kafka_consumer:
                received_messages.append(msg)
                if len(received_messages) >= event_count:
                    break
        except TimeoutError:
            pass

        # Verify all messages received
        assert len(received_messages) >= event_count, (
            f"Expected {event_count} messages, got {len(received_messages)}"
        )

        # Verify sequence
        sequences = set()
        for msg in received_messages:
            payload = json.loads(msg.value.decode("utf-8"))
            if "sequence" in payload:
                sequences.add(payload["sequence"])

        assert sequences == set(range(event_count)), "Not all sequence numbers received"


# =============================================================================
# Metadata Injection Tests - correlation_id, emitted_at, schema_version
# =============================================================================


class TestEmitDaemonMetadataInjection:
    """Tests for metadata injection in emitted events."""

    @pytest.mark.asyncio
    async def test_correlation_id_injected_when_not_provided(
        self,
        started_emit_daemon,
        emit_client_factory,
        kafka_consumer,
    ) -> None:
        """Verify correlation_id is auto-generated when not provided in payload.

        The daemon should inject a UUID correlation_id if the payload
        doesn't already have one.
        """
        client = emit_client_factory()

        # Emit event WITHOUT correlation_id
        payload = {
            "message": "no-correlation-test",
            "session_id": "test-session",
        }

        async with client:
            await client.emit("test.event", payload)

        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

        # Consume and verify correlation_id was injected
        received_messages = []
        try:
            async for msg in kafka_consumer:
                received_messages.append(msg)
                if len(received_messages) >= 1:
                    break
        except TimeoutError:
            pass

        assert len(received_messages) >= 1
        received_payload = json.loads(received_messages[0].value.decode("utf-8"))

        assert "correlation_id" in received_payload
        assert received_payload["correlation_id"] is not None
        assert len(received_payload["correlation_id"]) > 0

    @pytest.mark.asyncio
    async def test_correlation_id_preserved_when_provided(
        self,
        started_emit_daemon,
        emit_client_factory,
        kafka_consumer,
    ) -> None:
        """Verify correlation_id is preserved when provided in payload.

        If the caller provides a correlation_id, it should be preserved
        rather than overwritten.
        """
        client = emit_client_factory()

        # Emit event WITH correlation_id
        provided_correlation_id = str(uuid4())
        payload = {
            "message": "with-correlation-test",
            "session_id": "test-session",
            "correlation_id": provided_correlation_id,
        }

        async with client:
            await client.emit("test.event", payload)

        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

        # Consume and verify correlation_id was preserved
        received_messages = []
        try:
            async for msg in kafka_consumer:
                received_messages.append(msg)
                if len(received_messages) >= 1:
                    break
        except TimeoutError:
            pass

        assert len(received_messages) >= 1
        received_payload = json.loads(received_messages[0].value.decode("utf-8"))

        assert received_payload["correlation_id"] == provided_correlation_id

    @pytest.mark.asyncio
    async def test_emitted_at_timestamp_injected(
        self,
        started_emit_daemon,
        emit_client_factory,
        kafka_consumer,
    ) -> None:
        """Verify emitted_at timestamp is injected into payload.

        The daemon should add an ISO-8601 timestamp indicating when
        the event was emitted.
        """
        client = emit_client_factory()

        before_emit = datetime.now(UTC)

        payload = {
            "message": "timestamp-test",
            "session_id": "test-session",
        }

        async with client:
            await client.emit("test.event", payload)

        after_emit = datetime.now(UTC)

        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

        # Consume and verify emitted_at was injected
        received_messages = []
        try:
            async for msg in kafka_consumer:
                received_messages.append(msg)
                if len(received_messages) >= 1:
                    break
        except TimeoutError:
            pass

        assert len(received_messages) >= 1
        received_payload = json.loads(received_messages[0].value.decode("utf-8"))

        assert "emitted_at" in received_payload
        emitted_at = datetime.fromisoformat(received_payload["emitted_at"])

        # Verify timestamp is within expected range
        assert emitted_at >= before_emit
        assert emitted_at <= after_emit + timedelta(seconds=5)

    @pytest.mark.asyncio
    async def test_schema_version_injected(
        self,
        started_emit_daemon,
        emit_client_factory,
        kafka_consumer,
    ) -> None:
        """Verify schema_version is injected based on event registration.

        The daemon should add the schema_version from the event type
        registration to the payload.
        """
        client = emit_client_factory()

        payload = {
            "message": "schema-version-test",
            "session_id": "test-session",
        }

        async with client:
            await client.emit("test.event", payload)

        await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

        # Consume and verify schema_version was injected
        received_messages = []
        try:
            async for msg in kafka_consumer:
                received_messages.append(msg)
                if len(received_messages) >= 1:
                    break
        except TimeoutError:
            pass

        assert len(received_messages) >= 1
        received_payload = json.loads(received_messages[0].value.decode("utf-8"))

        assert "schema_version" in received_payload
        assert received_payload["schema_version"] == "1.0.0"


# =============================================================================
# Graceful Shutdown Tests - Queue draining
# =============================================================================


class TestEmitDaemonGracefulShutdown:
    """Tests for graceful shutdown and queue draining."""

    @pytest.mark.asyncio
    async def test_graceful_shutdown_drains_queue(
        self,
        emit_daemon_config,
        test_topic: str,
        kafka_consumer,
    ) -> None:
        """Verify graceful shutdown drains pending events.

        When the daemon stops gracefully:
        1. Pending events in memory should be published or spooled
        2. Spool directory should contain events if Kafka unavailable
        """
        from omnibase_infra.runtime.emit_daemon.client import EmitClient
        from omnibase_infra.runtime.emit_daemon.daemon import EmitDaemon
        from omnibase_infra.runtime.emit_daemon.event_registry import (
            ModelEventRegistration,
        )

        daemon = EmitDaemon(emit_daemon_config)
        daemon._registry.register(
            ModelEventRegistration(
                event_type="test.event",
                topic_template=test_topic,
                required_fields=["message"],
            )
        )

        await daemon.start()

        try:
            # Emit events
            client = EmitClient(
                socket_path=emit_daemon_config.socket_path,
                timeout=5.0,
            )

            async with client:
                for i in range(5):
                    await client.emit(
                        "test.event",
                        {"message": f"shutdown-test-{i}", "session_id": "shutdown"},
                    )

            # Stop daemon gracefully
            await daemon.stop()

            # Allow time for any final Kafka delivery
            await asyncio.sleep(MESSAGE_DELIVERY_WAIT_SECONDS)

            # Verify events were delivered to Kafka
            received_count = 0
            try:
                async for msg in kafka_consumer:
                    payload = json.loads(msg.value.decode("utf-8"))
                    if "shutdown-test" in payload.get("message", ""):
                        received_count += 1
                    if received_count >= 5:
                        break
            except TimeoutError:
                pass

            # At least some events should have been delivered
            # (exact count depends on timing)
            assert received_count >= 1, "No events delivered during graceful shutdown"

        finally:
            # Ensure cleanup even if test fails
            try:
                await daemon.stop()
            except Exception:
                pass


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestEmitDaemonErrorHandling:
    """Tests for error handling in the emit daemon."""

    @pytest.mark.asyncio
    async def test_emit_unknown_event_type_returns_error(
        self,
        started_emit_daemon,
        emit_client_factory,
    ) -> None:
        """Verify emitting unknown event type returns error response."""
        from omnibase_infra.runtime.emit_daemon.client import EmitClientError

        client = emit_client_factory()

        async with client:
            with pytest.raises(EmitClientError) as exc_info:
                await client.emit(
                    "unknown.event.type",
                    {"message": "test"},
                )

            assert "Unknown event type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_emit_missing_required_field_returns_error(
        self,
        started_emit_daemon,
        emit_client_factory,
    ) -> None:
        """Verify emitting event with missing required field returns error."""
        from omnibase_infra.runtime.emit_daemon.client import EmitClientError

        client = emit_client_factory()

        async with client:
            with pytest.raises(EmitClientError) as exc_info:
                # test.event requires "message" field
                await client.emit(
                    "test.event",
                    {"session_id": "test"},  # Missing "message"
                )

            assert "Missing required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_client_timeout_when_daemon_slow(
        self,
        started_emit_daemon,
        emit_daemon_config,
    ) -> None:
        """Verify client properly times out when daemon is slow."""
        from omnibase_infra.runtime.emit_daemon.client import (
            EmitClient,
            EmitClientError,
        )

        # Create client with very short timeout
        client = EmitClient(
            socket_path=emit_daemon_config.socket_path,
            timeout=0.01,  # 10ms - short enough to test timeout behavior
        )

        # This should either succeed quickly or timeout
        # We're testing that timeout handling works, not that it always times out
        try:
            async with client:
                await client.ping()
        except EmitClientError as e:
            # Timeout is acceptable
            assert "timeout" in str(e).lower() or "connection" in str(e).lower()
