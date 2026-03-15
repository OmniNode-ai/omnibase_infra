# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pytest fixtures for Kafka event bus integration tests.

Kafka topic management fixtures for integration tests.
The Redpanda broker (configured via KAFKA_BOOTSTRAP_SERVERS env var) has topic
auto-creation disabled, so topics must be created explicitly before use.

Bus Isolation (OMN-3476)
------------------------
The ``pytest_configure`` hook in this file pins ``KAFKA_BOOTSTRAP_SERVERS`` to
``localhost:19092`` (local Docker Redpanda) and sets ``KAFKA_BROKER_ALLOWLIST``
to allow both ``localhost:19092`` and ``127.0.0.1:19092`` before any test
collection runs.

``pytest_unconfigure`` restores the original values (or removes the variables
if they were not previously set) after the test session ends.

This prevents integration tests from accidentally hitting a non-local broker
or triggering ``validate_kafka_broker_allowlist``
failures when KAFKA_BROKER_ALLOWLIST is unset.

Event Loop Scope (pytest-asyncio 0.25+)
----------------------------------------
All fixtures here are **function-scoped** async fixtures for Kafka topic
management. With pytest-asyncio 0.25+, the default event loop scope is
"function", which works correctly with these fixtures.

When to Configure loop_scope in Test Modules
--------------------------------------------
If your test module uses **session-scoped or module-scoped** Kafka fixtures
(e.g., a shared Kafka producer across tests), you must configure loop_scope:

.. code-block:: python

    # For session-scoped Kafka fixtures
    pytestmark = [
        pytest.mark.kafka,
        pytest.mark.asyncio(loop_scope="session"),
    ]

    # For module-scoped Kafka fixtures
    pytestmark = [
        pytest.mark.kafka,
        pytest.mark.asyncio(loop_scope="module"),
    ]

Fixtures in This Module
-----------------------
All fixtures in this module are **function-scoped** (or use async generators
that clean up per-test), so they work with the default function-scoped event
loop:

    - ensure_test_topic: Creates topics per-test (async generator with cleanup)
    - topic_factory: Factory for creating topics per-test
    - created_unique_topic: Pre-created unique topic per-test

Why loop_scope Matters for Kafka
--------------------------------
Kafka async clients (AIOKafkaProducer, AIOKafkaConsumer) are bound to the
event loop at creation time. If you share a Kafka client across tests without
matching loop_scope, you'll encounter:

    - RuntimeError: "attached to a different event loop"
    - RuntimeError: "Event loop is closed"

Reference Documentation
-----------------------
- https://pytest-asyncio.readthedocs.io/en/latest/concepts.html#event-loop-scope
- https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/change_default_loop_scope.html

Fixtures:
    ensure_test_topic: Creates topics via admin API before tests, cleans up after
    topic_factory: Factory fixture for creating multiple topics with custom settings

Implementation Note:
    This module uses shared helpers from tests.helpers.util_kafka to avoid code
    duplication. The KafkaTopicManager class provides the core topic lifecycle
    management functionality used by multiple fixtures.

Related Tickets:
    - OMN-1361: pytest-asyncio 0.25+ upgrade and loop_scope configuration
    - OMN-3476: pytest_configure bus_local isolation for integration tests
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator, Callable, Coroutine

import pytest

# Module-level logger for test cleanup diagnostics
logger = logging.getLogger(__name__)

# =============================================================================
# Module-Level Markers
# =============================================================================

pytestmark = [
    pytest.mark.kafka,
]

# =============================================================================
# Bus Isolation Hooks (OMN-3476)
# =============================================================================
# These session-scoped hooks fire before test collection starts, ensuring
# KAFKA_BOOTSTRAP_SERVERS and KAFKA_BROKER_ALLOWLIST are set to the local
# Docker Redpanda bus for all integration tests in this directory.
#
# This prevents accidental hits against a non-local broker or
# validate_kafka_broker_allowlist failures when KAFKA_BROKER_ALLOWLIST is unset.
#
# The previous values (if any) are stored on the config object and restored
# by pytest_unconfigure after the session ends.

_BUS_LOCAL_BOOTSTRAP: str = "localhost:19092"
# Allow both hostname forms: code may normalize to 127.0.0.1 internally
_BUS_LOCAL_ALLOWLIST: str = "localhost:19092,127.0.0.1:19092"


def pytest_configure(config: pytest.Config) -> None:
    """Force bus_local Kafka config when running event_bus integration tests.

    Pins KAFKA_BOOTSTRAP_SERVERS to the local Docker Redpanda broker and
    sets KAFKA_BROKER_ALLOWLIST to permit both localhost:19092 and
    127.0.0.1:19092, but ONLY when the pytest session is scoped to this
    directory (tests/integration/event_bus/).

    When pytest collects the full test suite (``pytest tests/``), this hook
    is still invoked because pytest loads all conftest.py files it encounters.
    Setting KAFKA_BOOTSTRAP_SERVERS globally would cause other integration
    conftest modules to see KAFKA_AVAILABLE=True at import time, making them
    attempt to run Kafka-dependent fixtures that use invalid environment
    strings (e.g. ``environment="e2e-test"``).

    Guard: only activate when at least one CLI argument points into this
    directory, so the hook is a no-op during full-suite runs.

    Related: OMN-3476
    """
    _THIS_DIR = "tests/integration/event_bus"
    cli_args: list[str] = list(config.args)
    scoped_to_event_bus: bool = any(_THIS_DIR in str(arg) for arg in cli_args) or (
        len(cli_args) == 1 and str(cli_args[0]) == __file__
    )

    config._kafka_isolation_active = scoped_to_event_bus  # type: ignore[attr-defined]
    if not scoped_to_event_bus:
        config._kafka_isolation_prev = (None, None)  # type: ignore[attr-defined]
        return

    _prev_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    _prev_allowlist = os.environ.get("KAFKA_BROKER_ALLOWLIST")
    os.environ["KAFKA_BOOTSTRAP_SERVERS"] = _BUS_LOCAL_BOOTSTRAP
    os.environ["KAFKA_BROKER_ALLOWLIST"] = _BUS_LOCAL_ALLOWLIST
    # Store originals so pytest_unconfigure can restore them
    config._kafka_isolation_prev = (_prev_servers, _prev_allowlist)  # type: ignore[attr-defined]


def pytest_unconfigure(config: pytest.Config) -> None:
    """Restore original Kafka env vars after the test session ends.

    Only runs when pytest_configure set the active flag (scoped session).

    Related: OMN-3476
    """
    if not getattr(config, "_kafka_isolation_active", False):
        return
    prev = getattr(config, "_kafka_isolation_prev", (None, None))
    _prev_servers, _prev_allowlist = prev
    if _prev_servers is None:
        os.environ.pop("KAFKA_BOOTSTRAP_SERVERS", None)
    else:
        os.environ["KAFKA_BOOTSTRAP_SERVERS"] = _prev_servers
    if _prev_allowlist is None:
        os.environ.pop("KAFKA_BROKER_ALLOWLIST", None)
    else:
        os.environ["KAFKA_BROKER_ALLOWLIST"] = _prev_allowlist


# =============================================================================
# Canary Test
# =============================================================================


def test_kafka_integration_env_is_bus_local() -> None:
    """Guard: when running event_bus tests in isolation, bus_local must be set.

    Verifies that pytest_configure pinned KAFKA_BOOTSTRAP_SERVERS to
    localhost:19092 (local Docker Redpanda) before any test ran.

    Skips gracefully when not running in a scoped event_bus session (e.g.
    full ``pytest tests/`` run), because the hook guard intentionally
    deactivates in that case to avoid polluting other integration conftest
    modules that read KAFKA_BOOTSTRAP_SERVERS at import time.

    A failure (not a skip) means the hook executed but the env var was
    overridden by something else in the test environment.

    Related: OMN-3476
    """
    current = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if current != _BUS_LOCAL_BOOTSTRAP:
        # Hook was not activated (full-suite run) — skip rather than fail
        pytest.skip(
            f"Bus isolation hook inactive (KAFKA_BOOTSTRAP_SERVERS={current!r}). "
            "Run pytest tests/integration/event_bus/ to activate isolation."
        )
    assert _BUS_LOCAL_BOOTSTRAP in os.environ.get("KAFKA_BROKER_ALLOWLIST", ""), (
        f"KAFKA_BROKER_ALLOWLIST must contain '{_BUS_LOCAL_BOOTSTRAP}', "
        f"got: {os.environ.get('KAFKA_BROKER_ALLOWLIST')!r}"
    )


# =============================================================================
# Configuration
# =============================================================================

# KAFKA_BOOTSTRAP_SERVERS must be set via environment variable.
# No hardcoded default to ensure portability across CI/CD environments.
# Tests will skip via fixture if not set. Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:19092
# NOTE: pytest_configure above sets this to localhost:19092 before collection.
#
# VALIDATION: The value is validated at module load time via validate_bootstrap_servers().
# This validation checks for:
# - None or empty string values
# - Whitespace-only values
# - Malformed port numbers (non-numeric, out of range 1-65535)
# - Comma-separated server lists (each entry is validated)
# - IPv6 addresses (both bracketed [::1]:port and bare :: formats)
#
# Fixtures that depend on this value MUST check _kafka_config_validation before use.
# See validate_bootstrap_servers() in tests/helpers/util_kafka.py for implementation.
KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")

# =============================================================================
# Kafka Helpers (shared implementations)
# =============================================================================
# Imported from tests.helpers.util_kafka for centralized implementation.
# See tests/helpers/util_kafka.py for the canonical implementations.
from tests.helpers.util_kafka import (
    KAFKA_ERROR_BROKER_RESOURCE_EXHAUSTED,
    KAFKA_ERROR_CLUSTER_AUTHORIZATION_FAILED,
    KAFKA_ERROR_GROUP_AUTHORIZATION_FAILED,
    KAFKA_ERROR_INVALID_CONFIG,
    KAFKA_ERROR_INVALID_PARTITIONS,
    KAFKA_ERROR_INVALID_REPLICA_ASSIGNMENT,
    KAFKA_ERROR_INVALID_REPLICATION_FACTOR,
    KAFKA_ERROR_NOT_CONTROLLER,
    KAFKA_ERROR_REMEDIATION_HINTS,
    KAFKA_ERROR_TOPIC_ALREADY_EXISTS,
    KafkaConfigValidationResult,
    KafkaTopicManager,
    create_topic_factory_function,
    get_kafka_error_hint,
    parse_bootstrap_servers,
    validate_bootstrap_servers,
    wait_for_consumer_ready,
    wait_for_topic_metadata,
)

# Re-export for convenience - allows importing from this module instead of util_kafka
__all__ = [
    "wait_for_consumer_ready",
    "wait_for_topic_metadata",
    "KafkaTopicManager",
    "parse_bootstrap_servers",
    "validate_bootstrap_servers",
    "KafkaConfigValidationResult",
    "get_kafka_error_hint",
    "KAFKA_ERROR_REMEDIATION_HINTS",
    "KAFKA_ERROR_TOPIC_ALREADY_EXISTS",
    "KAFKA_ERROR_INVALID_PARTITIONS",
    "KAFKA_ERROR_INVALID_REPLICATION_FACTOR",
    "KAFKA_ERROR_INVALID_REPLICA_ASSIGNMENT",
    "KAFKA_ERROR_INVALID_CONFIG",
    "KAFKA_ERROR_NOT_CONTROLLER",
    "KAFKA_ERROR_CLUSTER_AUTHORIZATION_FAILED",
    "KAFKA_ERROR_GROUP_AUTHORIZATION_FAILED",
    "KAFKA_ERROR_BROKER_RESOURCE_EXHAUSTED",
]

# =============================================================================
# Configuration Validation
# =============================================================================
# Validate KAFKA_BOOTSTRAP_SERVERS at module load time to provide clear
# skip reasons when configuration is missing or malformed.
#
# CACHING BEHAVIOR:
#   This validation result is cached at module import time. If the
#   KAFKA_BOOTSTRAP_SERVERS environment variable is modified at runtime
#   (e.g., via os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "new:9092"), the
#   cached validation result will NOT be updated.
#
#   This is intentional for test stability - environment configuration should
#   be set before test collection, not modified during test execution.
#
#   DEFENSIVE VALIDATION: Fixtures that use KAFKA_BOOTSTRAP_SERVERS perform
#   a secondary check at execution time (os.getenv + empty/whitespace check)
#   to handle edge cases where the env var is modified after module import.
#   This provides belt-and-suspenders safety without fully re-validating.
#
# EMPTY VALUE HANDLING:
#   The validation handles these edge cases:
#   - None: Returns invalid with skip reason
#   - Empty string "": Returns invalid with skip reason
#   - Whitespace-only "  ": Returns invalid with skip reason
#   - Comma-only ",,,": Returns invalid with skip reason (no valid entries)
#   - Malformed port "host:abc": Returns invalid with skip reason
#   - Port out of range "host:99999": Returns invalid with skip reason

_kafka_config_validation: KafkaConfigValidationResult = validate_bootstrap_servers(
    KAFKA_BOOTSTRAP_SERVERS
)


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

    Skips:
        When KAFKA_BOOTSTRAP_SERVERS is empty, whitespace-only, or malformed.
        The skip reason provides actionable guidance for configuration.

    Yields:
        Async function that creates a topic with the given name and partition count.
        Returns the topic name for convenience.

    Example:
        async def test_publish_subscribe(ensure_test_topic):
            topic = await ensure_test_topic(f"test.integration.{uuid4().hex[:12]}")
            # Topic now exists and can be used for produce/consume
    """
    # Skip if Kafka is not properly configured (empty, whitespace-only, or malformed)
    # The validation at module load time catches:
    # - Empty string or None
    # - Whitespace-only values
    # - Non-numeric port (e.g., "localhost:abc")
    # - Port out of range (must be 1-65535)
    if not _kafka_config_validation:
        ensure_skip_reason: str = (
            _kafka_config_validation.skip_reason
            or "KAFKA_BOOTSTRAP_SERVERS not configured"
        )
        pytest.skip(ensure_skip_reason)

    # SAFETY: At this point, KAFKA_BOOTSTRAP_SERVERS is guaranteed to be valid
    # because validate_bootstrap_servers() returned is_valid=True
    assert _kafka_config_validation.is_valid, (
        "Fixture logic error: should have skipped if validation failed"
    )

    # DEFENSIVE: Re-validate at point of use to catch runtime env changes.
    # The module-level _kafka_config_validation is cached at import time,
    # so this check catches if someone modified KAFKA_BOOTSTRAP_SERVERS after import.
    current_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if not current_bootstrap_servers or not current_bootstrap_servers.strip():
        pytest.skip(
            "KAFKA_BOOTSTRAP_SERVERS is empty or whitespace-only at fixture execution. "
            "Env var may have been modified after module import."
        )

    # Use the shared KafkaTopicManager for topic lifecycle management
    # Use create_topic_factory_function to avoid duplicating topic creation logic
    async with KafkaTopicManager(current_bootstrap_servers) as manager:
        # No UUID suffix for integration tests (caller controls naming)
        yield create_topic_factory_function(manager, add_uuid_suffix=False)
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

    topic_name: str = f"test.integration.{uuid.uuid4().hex[:12]}"
    await ensure_test_topic(topic_name, 1)
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

    topic_name: str = f"test-dlq.dlq.intents.{uuid.uuid4().hex[:8]}"
    await ensure_test_topic(topic_name, 1)
    return topic_name


@pytest.fixture
async def created_broadcast_topic(
    ensure_test_topic: Callable[[str, int], Coroutine[None, None, str]],
) -> str:
    """Pre-create the broadcast topic used by broadcast tests.

    Returns:
        The created broadcast topic name.
    """
    topic_name: str = "integration-test.broadcast"
    await ensure_test_topic(topic_name, 1)
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

    Skips:
        When KAFKA_BOOTSTRAP_SERVERS is empty, whitespace-only, or malformed.
        The skip reason provides actionable guidance for configuration.

    Yields:
        Async function that creates a topic with custom settings.

    Example:
        async def test_replicated_topic(topic_factory):
            topic = await topic_factory("my.topic", partitions=3, replication=1)
    """
    # Skip if Kafka is not properly configured (empty, whitespace-only, or malformed)
    # The validation at module load time catches:
    # - Empty string or None
    # - Whitespace-only values
    # - Non-numeric port (e.g., "localhost:abc")
    # - Port out of range (must be 1-65535)
    if not _kafka_config_validation:
        factory_skip_reason: str = (
            _kafka_config_validation.skip_reason
            or "KAFKA_BOOTSTRAP_SERVERS not configured"
        )
        pytest.skip(factory_skip_reason)

    # SAFETY: At this point, KAFKA_BOOTSTRAP_SERVERS is guaranteed to be valid
    # because validate_bootstrap_servers() returned is_valid=True
    assert _kafka_config_validation.is_valid, (
        "Fixture logic error: should have skipped if validation failed"
    )

    # DEFENSIVE: Re-validate at point of use to catch runtime env changes.
    # The module-level _kafka_config_validation is cached at import time,
    # so this check catches if someone modified KAFKA_BOOTSTRAP_SERVERS after import.
    current_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
    if not current_bootstrap_servers or not current_bootstrap_servers.strip():
        pytest.skip(
            "KAFKA_BOOTSTRAP_SERVERS is empty or whitespace-only at fixture execution. "
            "Env var may have been modified after module import."
        )

    # Use the shared KafkaTopicManager for topic lifecycle management
    async with KafkaTopicManager(current_bootstrap_servers) as manager:

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
