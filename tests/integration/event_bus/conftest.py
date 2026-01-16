# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for Kafka event bus integration tests.

This module provides fixtures for managing Kafka topics in integration tests.
The Redpanda broker (configured via KAFKA_BOOTSTRAP_SERVERS env var) has topic
auto-creation disabled, so topics must be created explicitly before use.

Fixtures:
    ensure_test_topic: Creates topics via admin API before tests, cleans up after
    topic_factory: Factory fixture for creating multiple topics with custom settings
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import TYPE_CHECKING

import pytest

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)

# =============================================================================
# Module-Level Markers
# =============================================================================

pytestmark = [
    pytest.mark.kafka,
]

if TYPE_CHECKING:
    from aiokafka.admin import AIOKafkaAdminClient

# =============================================================================
# Configuration
# =============================================================================

# KAFKA_BOOTSTRAP_SERVERS must be set via environment variable.
# No hardcoded default to ensure portability across CI/CD environments.
# Tests will skip via fixture if not set. Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:29092
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")


# =============================================================================
# Kafka Error Code Remediation Hints
# =============================================================================
# Common Kafka/Redpanda error codes with actionable remediation hints.
# Reference: https://kafka.apache.org/protocol.html#protocol_error_codes
#
# These hints help developers quickly diagnose and fix common issues when
# running integration tests against Kafka/Redpanda brokers.
# =============================================================================

KAFKA_ERROR_REMEDIATION_HINTS: dict[int, str] = {
    # Topic management errors
    36: (
        "Topic already exists. This is usually harmless in test environments. "
        "If you need a fresh topic, use a unique name with UUID suffix."
    ),
    37: (
        "Invalid number of partitions. "
        "Hint: Ensure partitions >= 1. For Redpanda, check that the broker has "
        "sufficient memory allocated (see Docker memory limits or "
        "'redpanda.developer_mode' setting)."
    ),
    38: (
        "Invalid replication factor. "
        "Hint: Replication factor cannot exceed the number of brokers. "
        "For single-node test setups, use replication_factor=1."
    ),
    39: (
        "Invalid replica assignment. "
        "Hint: Check that all specified broker IDs exist in the cluster."
    ),
    40: (
        "Invalid topic configuration. "
        "Hint: Check topic config parameters (retention.ms, segment.bytes, etc.). "
        "Some Redpanda/Kafka versions have different config key names."
    ),
    # Cluster state errors
    41: (
        "Not the cluster controller. This is retriable. "
        "Hint: The cluster may be electing a new controller. Retry after a brief delay."
    ),
    # Authorization errors
    29: (
        "Cluster authorization failed. "
        "Hint: Check that your client has ClusterAction permission. "
        "For Redpanda, verify ACL configuration or disable authorization for tests."
    ),
    30: (
        "Group authorization failed. "
        "Hint: Check that your client has access to the consumer group. "
        "Verify KAFKA_SASL_* environment variables if using SASL authentication."
    ),
    # Resource errors
    89: (
        "Out of memory or resource exhausted on broker. "
        "Hint: For Redpanda in Docker, increase container memory limit "
        "(e.g., 'docker update --memory 2g <container>'). "
        "Check 'docker stats' for current memory usage."
    ),
}


def get_kafka_error_hint(error_code: int, error_message: str = "") -> str:
    """Get a remediation hint for a Kafka error code.

    Args:
        error_code: The Kafka protocol error code.
        error_message: Optional error message from the broker (for context).

    Returns:
        A formatted error message with remediation hints if available.
    """
    base_msg = f"Kafka error_code={error_code}"
    if error_message:
        base_msg += f", message='{error_message}'"

    hint = KAFKA_ERROR_REMEDIATION_HINTS.get(error_code)
    if hint:
        return f"{base_msg}. {hint}"

    # Generic hint for unknown errors
    return (
        f"{base_msg}. "
        "Hint: Check Kafka/Redpanda broker logs for details. "
        "Verify broker is running: 'docker ps | grep redpanda' or "
        "'curl -s http://<host>:9644/v1/status/ready' for Redpanda health."
    )


# =============================================================================
# Kafka Helpers (shared implementations)
# =============================================================================
# Re-exported from tests.helpers.kafka_utils for convenience.
# See tests/helpers/kafka_utils.py for the canonical implementations.
from tests.helpers.kafka_utils import wait_for_consumer_ready, wait_for_topic_metadata

__all__ = ["wait_for_consumer_ready", "wait_for_topic_metadata"]


# =============================================================================
# Topic Management Fixtures
# =============================================================================


@pytest.fixture
async def ensure_test_topic() -> AsyncGenerator[
    Callable[[str, int], Coroutine[None, None, str]], None
]:
    """Create test topics via Kafka admin API before tests and cleanup after.

    This fixture handles explicit topic creation for Redpanda/Kafka brokers
    that have topic auto-creation disabled. Topics are created before test
    execution and deleted during cleanup.

    After creating a topic, this fixture waits for the broker metadata to
    propagate to ensure the topic is ready for use.

    Yields:
        Async function that creates a topic with the given name and partition count.
        Returns the topic name for convenience.

    Example:
        async def test_publish_subscribe(ensure_test_topic):
            topic = await ensure_test_topic(f"test.integration.{uuid4().hex[:12]}")
            # Topic now exists and can be used for produce/consume
    """

    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    admin: AIOKafkaAdminClient | None = None
    created_topics: list[str] = []

    async def _create_topic(topic_name: str, partitions: int = 1) -> str:
        """Create a topic with the given name and partition count.

        Args:
            topic_name: Name of the topic to create.
            partitions: Number of partitions (default: 1).

        Returns:
            The topic name (for chaining convenience).
        """
        nonlocal admin, created_topics

        # Lazy initialization of admin client
        if admin is None:
            admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
            try:
                await admin.start()
            except Exception as conn_err:
                admin = None  # Reset to allow retry
                raise RuntimeError(
                    f"Failed to connect to Kafka broker at {KAFKA_BOOTSTRAP_SERVERS}. "
                    f"Hint: Verify the broker is running and accessible:\n"
                    f"  1. Check container status: 'docker ps | grep redpanda'\n"
                    f"  2. Test connectivity: 'nc -zv {KAFKA_BOOTSTRAP_SERVERS.split(':')[0]} "
                    f"{KAFKA_BOOTSTRAP_SERVERS.split(':')[1] if ':' in KAFKA_BOOTSTRAP_SERVERS else '29092'}'\n"
                    f"  3. For Redpanda, check health: 'curl -s http://<host>:9644/v1/status/ready'\n"
                    f"  4. Verify KAFKA_BOOTSTRAP_SERVERS env var is correct\n"
                    f"  5. If using Docker, ensure network connectivity to {KAFKA_BOOTSTRAP_SERVERS}\n"
                    f"Original error: {conn_err}"
                ) from conn_err

        try:
            response = await admin.create_topics(
                [
                    NewTopic(
                        name=topic_name,
                        num_partitions=partitions,
                        replication_factor=1,
                    )
                ]
            )
            # Check for errors in the response (e.g., memory limit, invalid config)
            # Response format: CreateTopicsResponse_v3(topic_errors=[(topic, error_code, error_message)])
            if hasattr(response, "topic_errors") and response.topic_errors:
                for topic_error in response.topic_errors:
                    # topic_error is a tuple: (topic_name, error_code, error_message)
                    _, error_code, error_message = topic_error
                    if error_code != 0:
                        # Error code 36 = TopicAlreadyExistsError (handled below)
                        if error_code == 36:
                            raise TopicAlreadyExistsError
                        # Provide actionable remediation hints for common errors
                        raise RuntimeError(
                            f"Failed to create topic '{topic_name}': "
                            f"{get_kafka_error_hint(error_code, error_message)}"
                        )

            created_topics.append(topic_name)

            # Wait for topic metadata to propagate with expected partition count
            await wait_for_topic_metadata(
                admin, topic_name, expected_partitions=partitions
            )
        except TopicAlreadyExistsError:
            # Topic already exists - this is acceptable in test environments
            # Still wait for metadata in case topic was just created by another process
            if admin is not None:
                await wait_for_topic_metadata(
                    admin, topic_name, timeout=5.0, expected_partitions=partitions
                )

        return topic_name

    yield _create_topic

    # Cleanup: delete created topics
    if admin is not None:
        if created_topics:
            try:
                await admin.delete_topics(created_topics)
            except Exception as e:
                logger.warning(
                    "Cleanup failed for Kafka topics %s: %s",
                    created_topics,
                    e,
                    exc_info=True,
                )
        try:
            await admin.close()
        except Exception as e:
            logger.warning(
                "Failed to close Kafka admin client: %s",
                e,
                exc_info=True,
            )


@pytest.fixture
async def created_unique_topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
) -> str:
    """Generate and pre-create a unique topic for test isolation.

    Combines topic name generation with automatic topic creation.
    Use this fixture when you need a topic that's ready to use immediately.

    Returns:
        The created topic name.

    Example:
        async def test_publish(started_kafka_bus, created_unique_topic):
            await started_kafka_bus.publish(created_unique_topic, None, b"hello")
    """
    import uuid

    topic_name = f"test.integration.{uuid.uuid4().hex[:12]}"
    await ensure_test_topic(topic_name)
    return topic_name


@pytest.fixture
async def created_unique_dlq_topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
) -> str:
    """Generate and pre-create a unique DLQ topic for test isolation.

    Similar to created_unique_topic but uses DLQ naming convention.

    Returns:
        The created DLQ topic name.
    """
    import uuid

    topic_name = f"test-dlq.dlq.intents.{uuid.uuid4().hex[:8]}"
    await ensure_test_topic(topic_name)
    return topic_name


@pytest.fixture
async def created_broadcast_topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
) -> str:
    """Pre-create the broadcast topic used by broadcast tests.

    Returns:
        The created broadcast topic name.
    """
    topic_name = "integration-test.broadcast"
    await ensure_test_topic(topic_name)
    return topic_name


@pytest.fixture
async def topic_factory() -> AsyncGenerator[
    Callable[[str, int, int], Coroutine[None, None, str]], None
]:
    """Factory fixture for creating topics with custom configurations.

    Similar to ensure_test_topic but allows specifying replication factor.
    Useful for testing with different topic configurations.

    Yields:
        Async function that creates a topic with custom settings.

    Example:
        async def test_replicated_topic(topic_factory):
            topic = await topic_factory("my.topic", partitions=3, replication=1)
    """
    from aiokafka.admin import AIOKafkaAdminClient, NewTopic
    from aiokafka.errors import TopicAlreadyExistsError

    admin: AIOKafkaAdminClient | None = None
    created_topics: list[str] = []

    async def _create_topic(
        topic_name: str,
        partitions: int = 1,
        replication_factor: int = 1,
    ) -> str:
        """Create a topic with custom configuration.

        Args:
            topic_name: Name of the topic to create.
            partitions: Number of partitions.
            replication_factor: Replication factor (usually 1 for testing).

        Returns:
            The topic name.
        """
        nonlocal admin, created_topics

        if admin is None:
            admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
            try:
                await admin.start()
            except Exception as conn_err:
                admin = None  # Reset to allow retry
                raise RuntimeError(
                    f"Failed to connect to Kafka broker at {KAFKA_BOOTSTRAP_SERVERS}. "
                    f"Hint: Verify the broker is running and accessible:\n"
                    f"  1. Check container status: 'docker ps | grep redpanda'\n"
                    f"  2. Test connectivity: 'nc -zv {KAFKA_BOOTSTRAP_SERVERS.split(':')[0]} "
                    f"{KAFKA_BOOTSTRAP_SERVERS.split(':')[1] if ':' in KAFKA_BOOTSTRAP_SERVERS else '29092'}'\n"
                    f"  3. For Redpanda, check health: 'curl -s http://<host>:9644/v1/status/ready'\n"
                    f"  4. Verify KAFKA_BOOTSTRAP_SERVERS env var is correct\n"
                    f"  5. If using Docker, ensure network connectivity to {KAFKA_BOOTSTRAP_SERVERS}\n"
                    f"Original error: {conn_err}"
                ) from conn_err

        try:
            response = await admin.create_topics(
                [
                    NewTopic(
                        name=topic_name,
                        num_partitions=partitions,
                        replication_factor=replication_factor,
                    )
                ]
            )
            # Check for errors in the response (e.g., memory limit, invalid config)
            if hasattr(response, "topic_errors") and response.topic_errors:
                for topic_error in response.topic_errors:
                    _, error_code, error_message = topic_error
                    if error_code != 0:
                        if error_code == 36:  # TopicAlreadyExistsError
                            raise TopicAlreadyExistsError
                        # Provide actionable remediation hints for common errors
                        raise RuntimeError(
                            f"Failed to create topic '{topic_name}': "
                            f"{get_kafka_error_hint(error_code, error_message)}"
                        )

            created_topics.append(topic_name)
            # Wait for topic metadata to propagate with expected partition count
            await wait_for_topic_metadata(
                admin, topic_name, expected_partitions=partitions
            )
        except TopicAlreadyExistsError:
            # Topic already exists - still wait for metadata
            if admin is not None:
                await wait_for_topic_metadata(
                    admin, topic_name, timeout=5.0, expected_partitions=partitions
                )

        return topic_name

    yield _create_topic

    # Cleanup
    if admin is not None:
        if created_topics:
            try:
                await admin.delete_topics(created_topics)
            except Exception as e:
                logger.warning(
                    "Cleanup failed for Kafka topics %s: %s",
                    created_topics,
                    e,
                    exc_info=True,
                )
        try:
            await admin.close()
        except Exception as e:
            logger.warning(
                "Failed to close Kafka admin client: %s",
                e,
                exc_info=True,
            )
