#!/usr/bin/env python3
"""
Integration Test Configuration and Fixtures.

This conftest provides:
- Auto-loading of .env file for test configuration
- Real service fixtures (Kafka, PostgreSQL, Consul)
- Service availability checks
- Test data cleanup
- Performance and reliability utilities
"""

import asyncio
import json
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✅ Loaded environment from {env_path}")
else:
    print(f"⚠️  No .env file found at {env_path}")

# Import real service clients
from omninode_bridge.services.kafka_client import KafkaClient
from omninode_bridge.services.metadata_stamping.registry.consul_client import (
    RegistryConsulClient,
)
from omninode_bridge.services.postgres_client import PostgresClient

# ============================================================================
# Service Availability Checks
# ============================================================================


def check_service_available(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP service is available."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def verify_services_available():
    """Verify required services are running before tests.

    This fixture runs once per test session and checks that all required
    infrastructure services are available. If services are unavailable,
    tests will be skipped with a clear error message.
    """
    services = {
        "Redpanda/Kafka": (
            os.getenv("KAFKA_BOOTSTRAP_SERVERS", "192.168.86.200:29092").split(":")[0],
            29092,  # External port for host scripts
        ),
        "PostgreSQL": (
            os.getenv("POSTGRES_HOST", "192.168.86.200"),
            int(os.getenv("POSTGRES_PORT", "5436")),
        ),
        "Consul": (
            os.getenv("CONSUL_HOST", "192.168.86.200"),
            int(os.getenv("CONSUL_PORT", "28500")),
        ),
    }

    unavailable = []
    for name, (host, port) in services.items():
        if not check_service_available(host, port):
            unavailable.append(f"{name} ({host}:{port})")

    if unavailable:
        pytest.skip(
            f"Integration tests require running services. "
            f"Unavailable: {', '.join(unavailable)}"
        )


# ============================================================================
# Real Service Fixtures
# ============================================================================


@pytest.fixture
async def kafka_client() -> AsyncIterator[KafkaClient]:
    """Real Kafka/Redpanda client for integration tests.

    Provides a fully functional Kafka client connected to the real
    Redpanda service at 192.168.86.200:29092.

    Yields:
        KafkaClient: Connected Kafka client instance
    """
    client = KafkaClient(
        bootstrap_servers=os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "192.168.86.200:29092",  # Remote Redpanda external port
        ),
        enable_dead_letter_queue=True,
        max_retry_attempts=3,
        timeout_seconds=30,
    )
    await client.connect()

    yield client

    await client.disconnect()


@pytest.fixture
async def postgres_client() -> AsyncIterator[PostgresClient]:
    """Real PostgreSQL client for integration tests.

    Provides a fully functional PostgreSQL client connected to the real
    PostgreSQL service at 192.168.86.200:5436.

    Yields:
        PostgresClient: Connected PostgreSQL client instance
    """
    client = PostgresClient(
        host=os.getenv("POSTGRES_HOST", "omninode-bridge-postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5436")),
        database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    await client.connect()

    yield client

    await client.disconnect()


@pytest.fixture
async def consul_client() -> AsyncIterator[RegistryConsulClient]:
    """Real Consul client for integration tests.

    Provides a fully functional Consul client connected to the
    Consul service at 192.168.86.200:28500.

    Yields:
        RegistryConsulClient: Connected Consul client instance
    """
    client = RegistryConsulClient(
        consul_host=os.getenv("CONSUL_HOST", "omninode-bridge-consul"),
        consul_port=int(os.getenv("CONSUL_PORT", "28500")),
    )

    # Note: RegistryConsulClient doesn't have async connect/disconnect
    # It initializes synchronously in __init__

    yield client

    # Cleanup: No async disconnect needed


# ============================================================================
# Test Data Cleanup Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
async def cleanup_test_data(postgres_client: PostgresClient):
    """Clean up test data before and after each test.

    This fixture automatically runs for every test, ensuring a clean
    database state by removing all test-related records.

    Args:
        postgres_client: PostgreSQL client for cleanup operations
    """
    # Cleanup before test
    await postgres_client.execute_query(
        "DELETE FROM node_registrations WHERE node_id LIKE 'test-%' OR node_id LIKE 'node-bridge-%'"
    )

    yield  # Run test

    # Cleanup after test
    await postgres_client.execute_query(
        "DELETE FROM node_registrations WHERE node_id LIKE 'test-%' OR node_id LIKE 'node-bridge-%'"
    )


# ============================================================================
# Kafka Consumer Utility Functions
# ============================================================================


async def get_latest_message_from_topic(
    topic: str, timeout: float = 5.0, bootstrap_servers: str = None
) -> dict | None:
    """Consume latest message from real Kafka topic.

    This utility function polls a Kafka topic for new messages, useful for
    verifying that events were published correctly in integration tests.

    Args:
        topic: Kafka topic name to consume from
        timeout: Maximum time to wait for message (seconds)
        bootstrap_servers: Kafka bootstrap servers (default from env)

    Returns:
        dict: Parsed message content, or None if no message received

    Example:
        event = await get_latest_message_from_topic("test.events.v1", timeout=5.0)
        assert event is not None, "No event received"
        assert event["event_type"] == "node-introspection"
    """
    from aiokafka import AIOKafkaConsumer

    bootstrap_servers = bootstrap_servers or os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "192.168.86.200:29092",  # Remote Redpanda external port
    )

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="latest",  # Start from latest
        group_id=f"test-consumer-{uuid4()}",  # Unique group ID
        enable_auto_commit=False,
    )

    await consumer.start()
    try:
        # Seek to end to get only new messages
        partitions = consumer.assignment()
        if not partitions:
            # Wait for assignment
            await asyncio.sleep(0.5)
            partitions = consumer.assignment()

        for partition in partitions:
            await consumer.seek_to_end(partition)

        # Wait for new messages with timeout
        try:
            async with asyncio.timeout(timeout):
                async for message in consumer:
                    # Parse message value
                    try:
                        value = json.loads(message.value.decode("utf-8"))
                        return value
                    except json.JSONDecodeError:
                        # Try returning as string if not JSON
                        return message.value.decode("utf-8")
        except TimeoutError:
            return None

    finally:
        await consumer.stop()


async def get_all_messages_from_topic(
    topic: str,
    max_messages: int = 100,
    timeout: float = 2.0,
    bootstrap_servers: str = None,
) -> list[dict]:
    """Consume all messages from real Kafka topic.

    This utility function consumes all available messages from a Kafka topic,
    useful for verifying batch operations or multiple events.

    Args:
        topic: Kafka topic name to consume from
        max_messages: Maximum number of messages to consume
        timeout: Maximum time to wait for messages (seconds)
        bootstrap_servers: Kafka bootstrap servers (default from env)

    Returns:
        list[dict]: List of parsed message contents

    Example:
        events = await get_all_messages_from_topic("test.events.v1", max_messages=10)
        assert len(events) >= 2, "Expected at least 2 events"
    """
    from aiokafka import AIOKafkaConsumer

    bootstrap_servers = bootstrap_servers or os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "192.168.86.200:29092",  # Remote Redpanda external port
    )

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",  # Start from beginning
        group_id=f"test-consumer-{uuid4()}",  # Unique group ID
        enable_auto_commit=False,
    )

    messages = []
    await consumer.start()
    try:
        try:
            async with asyncio.timeout(timeout):
                async for message in consumer:
                    # Parse message value
                    try:
                        value = json.loads(message.value.decode("utf-8"))
                        messages.append(value)
                    except json.JSONDecodeError:
                        # Try returning as string if not JSON
                        messages.append(message.value.decode("utf-8"))

                    if len(messages) >= max_messages:
                        break
        except TimeoutError:
            pass  # Return what we have

    finally:
        await consumer.stop()

    return messages


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest for integration tests.

    Note: integration and e2e markers are registered in pyproject.toml
    """
    # Print configuration info
    print("\n" + "=" * 80)
    print("Integration Test Configuration")
    print("=" * 80)
    print(f"Kafka Bootstrap Servers: {os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'NOT SET')}")
    print(f"PostgreSQL Host: {os.getenv('POSTGRES_HOST', 'NOT SET')}")
    print(f"PostgreSQL Port: {os.getenv('POSTGRES_PORT', 'NOT SET')}")
    print(f"Consul Host: {os.getenv('CONSUL_HOST', 'NOT SET')}")
    print(f"Consul Port: {os.getenv('CONSUL_PORT', 'NOT SET')}")
    print("=" * 80 + "\n")
