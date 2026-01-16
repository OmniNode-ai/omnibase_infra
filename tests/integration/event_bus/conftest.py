# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for Kafka event bus integration tests.

This module provides fixtures for managing Kafka topics in integration tests.
The Redpanda broker (configured via KAFKA_BOOTSTRAP_SERVERS env var) has topic
auto-creation disabled, so topics must be created explicitly before use.

Fixtures:
    ensure_test_topic: Creates topics via admin API before tests, cleans up after
    topic_factory: Factory fixture for creating multiple topics with custom settings

Implementation Note:
    This module uses shared helpers from tests.helpers.util_kafka to avoid code
    duplication. The KafkaTopicManager class provides the core topic lifecycle
    management functionality used by multiple fixtures.
"""

from __future__ import annotations

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
# Kafka Helpers (shared implementations)
# =============================================================================
# Imported from tests.helpers.util_kafka for centralized implementation.
# See tests/helpers/util_kafka.py for the canonical implementations.
from tests.helpers.util_kafka import (
    KAFKA_ERROR_INVALID_PARTITIONS,
    KAFKA_ERROR_REMEDIATION_HINTS,
    KAFKA_ERROR_TOPIC_ALREADY_EXISTS,
    KafkaTopicManager,
    get_kafka_error_hint,
    parse_bootstrap_servers,
    wait_for_consumer_ready,
    wait_for_topic_metadata,
)

# Re-export for backwards compatibility with any code importing from this module
__all__ = [
    "wait_for_consumer_ready",
    "wait_for_topic_metadata",
    "KafkaTopicManager",
    "parse_bootstrap_servers",
    "get_kafka_error_hint",
    "KAFKA_ERROR_REMEDIATION_HINTS",
    "KAFKA_ERROR_TOPIC_ALREADY_EXISTS",
    "KAFKA_ERROR_INVALID_PARTITIONS",
]


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

    Implementation:
        Uses KafkaTopicManager from tests.helpers.util_kafka for centralized
        topic lifecycle management and error handling.

    Yields:
        Async function that creates a topic with the given name and partition count.
        Returns the topic name for convenience.

    Example:
        async def test_publish_subscribe(ensure_test_topic):
            topic = await ensure_test_topic(f"test.integration.{uuid4().hex[:12]}")
            # Topic now exists and can be used for produce/consume
    """
    # Use the shared KafkaTopicManager for topic lifecycle management
    async with KafkaTopicManager(KAFKA_BOOTSTRAP_SERVERS) as manager:

        async def _create_topic(topic_name: str, partitions: int = 1) -> str:
            """Create a topic with the given name and partition count.

            Args:
                topic_name: Name of the topic to create.
                partitions: Number of partitions (default: 1).

            Returns:
                The topic name (for chaining convenience).
            """
            return await manager.create_topic(topic_name, partitions=partitions)

        yield _create_topic
        # Cleanup is handled automatically by KafkaTopicManager context exit


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

    Implementation:
        Uses KafkaTopicManager from tests.helpers.util_kafka for centralized
        topic lifecycle management and error handling.

    Yields:
        Async function that creates a topic with custom settings.

    Example:
        async def test_replicated_topic(topic_factory):
            topic = await topic_factory("my.topic", partitions=3, replication=1)
    """
    # Use the shared KafkaTopicManager for topic lifecycle management
    async with KafkaTopicManager(KAFKA_BOOTSTRAP_SERVERS) as manager:

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
            return await manager.create_topic(
                topic_name,
                partitions=partitions,
                replication_factor=replication_factor,
            )

        yield _create_topic
        # Cleanup is handled automatically by KafkaTopicManager context exit
