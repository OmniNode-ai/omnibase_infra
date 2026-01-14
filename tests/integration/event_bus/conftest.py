# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for Kafka event bus integration tests.

This module provides fixtures for managing Kafka topics in integration tests.
The remote Redpanda broker at 192.168.86.200:29092 has topic auto-creation
disabled, so topics must be created explicitly before use.

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

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "192.168.86.200:29092")


# =============================================================================
# Consumer Readiness Helper (shared implementation)
# =============================================================================
# Re-exported from tests.helpers.kafka_utils for convenience.
# See tests/helpers/kafka_utils.py for the canonical implementation.
from tests.helpers.kafka_utils import wait_for_consumer_ready

__all__ = ["wait_for_consumer_ready"]


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

    async def _wait_for_topic_metadata(
        admin_client: AIOKafkaAdminClient,
        topic_name: str,
        timeout: float = 10.0,
    ) -> bool:
        """Wait for topic metadata to be available in the broker.

        After topic creation, there's a delay before the broker metadata
        is updated. This function polls until the topic appears.

        Args:
            admin_client: The admin client to use for metadata checks.
            topic_name: The topic to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if topic was found, False if timed out.
        """
        start_time = asyncio.get_running_loop().time()
        while (asyncio.get_running_loop().time() - start_time) < timeout:
            try:
                # Describe topics to check if metadata is available
                # This forces a metadata refresh
                description = await admin_client.describe_topics([topic_name])
                if description:
                    return True
            except Exception:
                pass  # Topic not yet available
            await asyncio.sleep(0.5)  # Poll every 500ms
        return False

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
            await admin.start()

        try:
            await admin.create_topics(
                [
                    NewTopic(
                        name=topic_name,
                        num_partitions=partitions,
                        replication_factor=1,
                    )
                ]
            )
            created_topics.append(topic_name)

            # Wait for topic metadata to propagate
            await _wait_for_topic_metadata(admin, topic_name)
        except TopicAlreadyExistsError:
            # Topic already exists - this is acceptable in test environments
            # Still wait for metadata in case topic was just created by another process
            if admin is not None:
                await _wait_for_topic_metadata(admin, topic_name, timeout=5.0)

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
            await admin.start()

        try:
            await admin.create_topics(
                [
                    NewTopic(
                        name=topic_name,
                        num_partitions=partitions,
                        replication_factor=replication_factor,
                    )
                ]
            )
            created_topics.append(topic_name)
        except TopicAlreadyExistsError:
            pass  # Topic already exists - acceptable in test environments

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
