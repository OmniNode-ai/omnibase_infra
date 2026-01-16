# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka testing utilities for integration tests.

This module provides shared utilities for Kafka-based integration tests,
including consumer readiness polling and topic management helpers.

Available Utilities:
    - wait_for_consumer_ready: Poll for Kafka consumer readiness with exponential backoff
    - wait_for_topic_metadata: Wait for topic metadata propagation after creation
    - KafkaTopicManager: Async context manager for topic lifecycle management
    - parse_bootstrap_servers: Parse bootstrap servers string into (host, port) tuple
    - validate_bootstrap_servers: Validate configuration with skip reasons for tests

Configuration Validation:
    Use validate_bootstrap_servers() to check configuration before running tests:

    >>> result = validate_bootstrap_servers(os.getenv("KAFKA_BOOTSTRAP_SERVERS", ""))
    >>> if not result:
    ...     pytest.skip(result.skip_reason)

    This handles:
    - Empty/whitespace-only values
    - Malformed port numbers (non-numeric, out of range)
    - Clear skip reasons for test output

Topic Management Pattern:
    Use KafkaTopicManager for consistent topic creation and cleanup in tests:

    >>> async with KafkaTopicManager(bootstrap_servers) as manager:
    ...     topic = await manager.create_topic("test.topic")
    ...     # Test logic using the topic
    ...     # Topics are automatically cleaned up when context exits

Error Remediation:
    The module includes remediation hints for common Kafka error codes.
    See KAFKA_ERROR_REMEDIATION_HINTS for actionable hints on error resolution.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiokafka.admin import AIOKafkaAdminClient

    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

# Module-level logger for diagnostics
logger = logging.getLogger(__name__)


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

# Named constants for Kafka error codes (for readable conditionals)
KAFKA_ERROR_TOPIC_ALREADY_EXISTS = 36
KAFKA_ERROR_INVALID_PARTITIONS = 37  # Also used for memory limit in Redpanda


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


class KafkaConfigValidationResult:
    """Result of KAFKA_BOOTSTRAP_SERVERS validation.

    Attributes:
        is_valid: True if the configuration is valid and usable.
        host: Parsed host (or "<not set>" if invalid).
        port: Parsed port (or "29092" default if not specified).
        error_message: Human-readable error message if invalid, None if valid.
        skip_reason: Pytest skip reason if tests should be skipped, None if valid.
    """

    __slots__ = ("error_message", "host", "is_valid", "port", "skip_reason")

    def __init__(
        self,
        *,
        is_valid: bool,
        host: str,
        port: str,
        error_message: str | None = None,
        skip_reason: str | None = None,
    ) -> None:
        self.is_valid = is_valid
        self.host = host
        self.port = port
        self.error_message = error_message
        self.skip_reason = skip_reason

    def __bool__(self) -> bool:
        """Return True if configuration is valid."""
        return self.is_valid


def validate_bootstrap_servers(
    bootstrap_servers: str | None,
) -> KafkaConfigValidationResult:
    """Validate KAFKA_BOOTSTRAP_SERVERS and return detailed result.

    Performs comprehensive validation of the bootstrap servers string:
    - Checks for empty/whitespace-only values
    - Validates host:port format
    - Validates port is numeric and in valid range (1-65535)
    - Returns structured result with skip reason for tests

    Args:
        bootstrap_servers: The KAFKA_BOOTSTRAP_SERVERS value from environment.

    Returns:
        KafkaConfigValidationResult with validation status and details.

    Example:
        >>> result = validate_bootstrap_servers("")
        >>> if not result:
        ...     pytest.skip(result.skip_reason)

        >>> result = validate_bootstrap_servers("localhost:9092")
        >>> assert result.is_valid
        >>> assert result.host == "localhost"
        >>> assert result.port == "9092"
    """
    # Handle None (defensive)
    if bootstrap_servers is None:
        return KafkaConfigValidationResult(
            is_valid=False,
            host="<not set>",
            port="29092",
            error_message="KAFKA_BOOTSTRAP_SERVERS is not set (None)",
            skip_reason=(
                "KAFKA_BOOTSTRAP_SERVERS not configured. "
                "Set environment variable to enable Kafka integration tests. "
                "Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:29092"
            ),
        )

    # Handle empty/whitespace-only
    if not bootstrap_servers or not bootstrap_servers.strip():
        return KafkaConfigValidationResult(
            is_valid=False,
            host="<not set>",
            port="29092",
            error_message="KAFKA_BOOTSTRAP_SERVERS is empty or whitespace-only",
            skip_reason=(
                "KAFKA_BOOTSTRAP_SERVERS is empty or not set. "
                "Set environment variable to enable Kafka integration tests. "
                "Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:29092"
            ),
        )

    stripped = bootstrap_servers.strip()

    # Parse host and port
    host, port = parse_bootstrap_servers(stripped)

    # Validate port is numeric (when explicitly provided)
    if ":" in stripped:
        # Extract port part for validation
        if stripped.startswith("["):
            # IPv6 format: [::1]:9092
            bracket_close = stripped.rfind("]")
            if bracket_close != -1 and bracket_close < len(stripped) - 1:
                port_str = stripped[bracket_close + 2 :]
            else:
                port_str = ""
        else:
            # Standard format: host:port
            port_str = stripped.rsplit(":", 1)[-1] if ":" in stripped else ""

        if port_str and not port_str.isdigit():
            return KafkaConfigValidationResult(
                is_valid=False,
                host=host,
                port=port_str,
                error_message=(
                    f"KAFKA_BOOTSTRAP_SERVERS has invalid port: '{port_str}' "
                    f"(must be numeric)"
                ),
                skip_reason=(
                    f"KAFKA_BOOTSTRAP_SERVERS has invalid port: '{port_str}'. "
                    f"Port must be a number. "
                    f"Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:29092"
                ),
            )

        if port_str:
            port_num = int(port_str)
            if port_num < 1 or port_num > 65535:
                return KafkaConfigValidationResult(
                    is_valid=False,
                    host=host,
                    port=port_str,
                    error_message=(
                        f"KAFKA_BOOTSTRAP_SERVERS has invalid port: {port_num} "
                        f"(must be 1-65535)"
                    ),
                    skip_reason=(
                        f"KAFKA_BOOTSTRAP_SERVERS has invalid port: {port_num}. "
                        f"Port must be between 1 and 65535. "
                        f"Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:29092"
                    ),
                )

    # Valid configuration
    return KafkaConfigValidationResult(
        is_valid=True,
        host=host,
        port=port,
        error_message=None,
        skip_reason=None,
    )


def parse_bootstrap_servers(bootstrap_servers: str) -> tuple[str, str]:
    """Parse KAFKA_BOOTSTRAP_SERVERS into (host, port) tuple for error messages.

    Handles various formats safely:
    - Empty/whitespace-only: Returns ("<not set>", "29092")
    - "hostname:port": Returns ("hostname", "port")
    - "hostname" (no port): Returns ("hostname", "29092")
    - "[::1]:9092" (IPv6): Returns ("[::1]", "9092")
    - IPv6 without brackets: Returns (address, "29092") - malformed but tolerated

    Note:
        This function is primarily for error message generation. For validation
        with skip reasons, use validate_bootstrap_servers() instead.

    Args:
        bootstrap_servers: The KAFKA_BOOTSTRAP_SERVERS value.

    Returns:
        Tuple of (host, port) for use in error messages.
    """
    # Handle empty/whitespace-only input
    if not bootstrap_servers or not bootstrap_servers.strip():
        return ("<not set>", "29092")

    stripped = bootstrap_servers.strip()

    # Handle IPv6 with brackets: [::1]:9092
    if stripped.startswith("["):
        bracket_close = stripped.rfind("]")
        if bracket_close != -1 and bracket_close < len(stripped) - 1:
            # Has closing bracket and something after it
            if stripped[bracket_close + 1] == ":":
                host = stripped[: bracket_close + 1]
                port = stripped[bracket_close + 2 :] or "29092"
                return (host, port)
        # Malformed IPv6 - return as-is with default port
        return (stripped, "29092")

    # Standard host:port format - use rsplit to handle single colon
    if ":" in stripped:
        parts = stripped.rsplit(":", 1)
        host = parts[0] or "<not set>"
        port = parts[1] if len(parts) > 1 and parts[1] else "29092"
        return (host, port)

    # No colon - just hostname
    return (stripped, "29092")


async def wait_for_consumer_ready(
    event_bus: EventBusKafka,
    topic: str,
    max_wait: float = 10.0,
    initial_backoff: float = 0.1,
    max_backoff: float = 1.0,
    backoff_multiplier: float = 1.5,
) -> bool:
    """Wait for Kafka consumer to be ready to receive messages using polling.

    This is a **best-effort** readiness check that always returns True. It attempts
    to detect when the consumer is ready by polling health checks, but falls back
    gracefully on timeout to avoid blocking tests indefinitely.

    Kafka consumers require time to join the consumer group and start receiving
    messages after subscription. This helper polls the event bus health check
    until the consumer count increases, indicating the consumer task is running.

    Behavior Summary:
        1. Polls event_bus.health_check() with exponential backoff
        2. If consumer_count increases within max_wait: returns True (early exit)
        3. If max_wait exceeded: returns True anyway (graceful fallback)

    Why Always Return True?
        The purpose is to REDUCE flakiness by waiting for actual readiness when
        possible, not to DETECT failures. Test assertions should verify expected
        outcomes, not this helper's return value.

    Implementation:
        Uses exponential backoff polling (initial_backoff * backoff_multiplier^n)
        to check consumer registration, capped at max_backoff per iteration.
        This is more reliable than a fixed sleep as it:
        - Returns early when consumer is ready (reduces test time)
        - Adapts to variable Kafka/Redpanda startup times
        - Reduces flakiness compared to fixed-duration sleeps

    Args:
        event_bus: The EventBusKafka instance to check for readiness.
        topic: The topic to wait for (used for logging only, not filtering).
        max_wait: Maximum time in seconds to poll before giving up. The function
            will return True regardless of whether consumer became ready.
            Default: 10.0s. Actual wait may exceed max_wait by up to max_backoff
            (on timeout) or +0.1s stabilization delay (on success).
        initial_backoff: Initial polling delay in seconds (default 0.1s).
        max_backoff: Maximum polling delay cap in seconds (default 1.0s).
        backoff_multiplier: Multiplier for exponential backoff (default 1.5).

    Returns:
        Always True. Do not use return value for failure detection.
        Use test assertions to verify expected outcomes.

    Example:
        # Best-effort wait for consumer readiness (default max_wait=10.0s)
        await wait_for_consumer_ready(bus, topic)

        # Shorter wait for fast tests
        await wait_for_consumer_ready(bus, topic, max_wait=2.0)

        # Consumer MAY be ready here, but test should not rely on this
        # Use assertions on actual test outcomes instead
    """
    start_time = asyncio.get_running_loop().time()
    current_backoff = initial_backoff

    # Get initial consumer count for comparison
    initial_health = await event_bus.health_check()
    initial_consumer_count = initial_health.get("consumer_count", 0)

    # Poll until consumer count increases or timeout
    while (asyncio.get_running_loop().time() - start_time) < max_wait:
        health = await event_bus.health_check()
        consumer_count = health.get("consumer_count", 0)

        # If consumer count has increased, the subscription is active
        if consumer_count > initial_consumer_count:
            # Add a small additional delay for the consumer loop to start
            # processing messages after registration
            await asyncio.sleep(0.1)
            return True

        # Check if we've timed out after health check (prevents unnecessary sleep)
        elapsed = asyncio.get_running_loop().time() - start_time
        if elapsed >= max_wait:
            break

        # Exponential backoff with cap
        await asyncio.sleep(current_backoff)
        current_backoff = min(current_backoff * backoff_multiplier, max_backoff)

    # Return True even on timeout (graceful fallback)
    # Log at debug level for diagnostics
    logger.debug(
        "wait_for_consumer_ready timed out after %.2fs for topic %s",
        max_wait,
        topic,
    )
    return True


async def wait_for_topic_metadata(
    admin_client: AIOKafkaAdminClient,
    topic_name: str,
    timeout: float = 10.0,
    expected_partitions: int = 1,
) -> bool:
    """Wait for topic metadata with partitions to be available in the broker.

    After topic creation, there's a delay before the broker metadata is updated.
    This function polls until the topic appears with the expected number of
    partitions available.

    This function handles both response formats from aiokafka's describe_topics():
    - **List format** (older versions): `[{'error_code': 0, 'topic': 'name', 'partitions': [...]}]`
    - **Dict format** (aiokafka 0.11.0+): `{'topic_name': TopicDescription(error_code=0, ...)}`

    Args:
        admin_client: The AIOKafkaAdminClient instance for broker communication.
        topic_name: The topic to wait for.
        timeout: Maximum time to wait in seconds.
        expected_partitions: Minimum number of partitions to wait for.

    Returns:
        True if topic was found with expected partitions, False if timed out.

    Example:
        >>> admin = AIOKafkaAdminClient(bootstrap_servers="localhost:9092")
        >>> await admin.start()
        >>> await admin.create_topics([NewTopic(name="my-topic", ...)])
        >>> await wait_for_topic_metadata(admin, "my-topic", expected_partitions=3)
        True
    """
    start_time = asyncio.get_running_loop().time()

    while (asyncio.get_running_loop().time() - start_time) < timeout:
        try:
            # describe_topics is a coroutine method on AIOKafkaAdminClient
            description = await admin_client.describe_topics([topic_name])

            if not description:
                logger.debug("Topic %s: empty describe_topics response", topic_name)
                await asyncio.sleep(0.5)
                continue

            # Handle dict format (aiokafka 0.11.0+): {'topic_name': TopicDescription}
            if isinstance(description, dict):
                if topic_name in description:
                    topic_info = description[topic_name]
                    # TopicDescription may be an object with attributes or dict-like
                    error_code = (
                        getattr(topic_info, "error_code", None)
                        if hasattr(topic_info, "error_code")
                        else topic_info.get("error_code", -1)
                        if isinstance(topic_info, dict)
                        else -1
                    )
                    partitions = (
                        getattr(topic_info, "partitions", [])
                        if hasattr(topic_info, "partitions")
                        else topic_info.get("partitions", [])
                        if isinstance(topic_info, dict)
                        else []
                    )

                    if (error_code is None or error_code == 0) and len(
                        partitions
                    ) >= expected_partitions:
                        logger.debug(
                            "Topic %s ready with %d partitions (dict format)",
                            topic_name,
                            len(partitions),
                        )
                        return True
                    logger.debug(
                        "Topic %s not ready: error_code=%s, partitions=%d (dict format)",
                        topic_name,
                        error_code,
                        len(partitions),
                    )
                else:
                    # Dict response but topic not found - may still be propagating
                    logger.debug(
                        "Topic %s not in dict response keys: %s",
                        topic_name,
                        list(description.keys()),
                    )

            # Handle list format (older aiokafka): [{'error_code': 0, 'topic': ..., 'partitions': [...]}]
            elif isinstance(description, list) and len(description) > 0:
                topic_info = description[0]
                # List items are typically dicts
                if isinstance(topic_info, dict):
                    error_code = topic_info.get("error_code", -1)
                    partitions = topic_info.get("partitions", [])
                else:
                    # Object with attributes
                    error_code = getattr(topic_info, "error_code", -1)
                    partitions = getattr(topic_info, "partitions", [])

                if error_code == 0 and len(partitions) >= expected_partitions:
                    logger.debug(
                        "Topic %s ready with %d partitions (list format)",
                        topic_name,
                        len(partitions),
                    )
                    return True
                logger.debug(
                    "Topic %s not ready: error_code=%s, partitions=%d (list format)",
                    topic_name,
                    error_code,
                    len(partitions),
                )
            else:
                # Unknown format but truthy - accept as success for compatibility
                logger.debug(
                    "Topic %s: unknown response format, accepting as ready: %s",
                    topic_name,
                    type(description),
                )
                return True

        except Exception as e:
            logger.debug("Topic %s metadata check failed: %s", topic_name, e)

        await asyncio.sleep(0.5)  # Poll every 500ms

    logger.warning(
        "Timeout waiting for topic %s metadata after %.1fs",
        topic_name,
        timeout,
    )
    return False


class KafkaTopicManager:
    """Async context manager for Kafka topic lifecycle management.

    This class encapsulates the common pattern of creating topics for tests
    and cleaning them up afterwards. It handles:

    - Lazy admin client initialization with comprehensive error messages
    - Topic creation with error response handling
    - Waiting for topic metadata propagation
    - Automatic cleanup of created topics on context exit

    Usage:
        >>> async with KafkaTopicManager("localhost:9092") as manager:
        ...     topic1 = await manager.create_topic("test.topic.1")
        ...     topic2 = await manager.create_topic("test.topic.2", partitions=3)
        ...     # Topics are automatically deleted when context exits

    Attributes:
        bootstrap_servers: Kafka bootstrap servers string.
        created_topics: List of topic names created by this manager.

    Note:
        This class is designed for test fixtures. Production code should use
        proper topic management through infrastructure tooling.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        """Initialize the topic manager.

        Args:
            bootstrap_servers: Kafka bootstrap servers (e.g., "localhost:9092").
        """
        self.bootstrap_servers = bootstrap_servers
        self.created_topics: list[str] = []
        self._admin: AIOKafkaAdminClient | None = None

    async def __aenter__(self) -> KafkaTopicManager:
        """Enter the async context manager.

        Returns:
            Self for use in the context.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the async context manager, cleaning up topics and admin client."""
        await self.cleanup()

    async def _ensure_admin(self) -> AIOKafkaAdminClient:
        """Ensure admin client is initialized and started.

        Returns:
            The started AIOKafkaAdminClient.

        Raises:
            RuntimeError: If connection to the broker fails.
        """
        if self._admin is not None:
            return self._admin

        from aiokafka.admin import AIOKafkaAdminClient

        self._admin = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        try:
            await self._admin.start()
        except Exception as conn_err:
            self._admin = None  # Reset to allow retry
            host, port = parse_bootstrap_servers(self.bootstrap_servers)
            raise RuntimeError(
                f"Failed to connect to Kafka broker at {self.bootstrap_servers}. "
                f"Hint: Verify the broker is running and accessible:\n"
                f"  1. Check container status: 'docker ps | grep redpanda'\n"
                f"  2. Test connectivity: 'nc -zv {host} {port}'\n"
                f"  3. For Redpanda, check health: 'curl -s http://<host>:9644/v1/status/ready'\n"
                f"  4. Verify KAFKA_BOOTSTRAP_SERVERS env var is correct\n"
                f"  5. If using Docker, ensure network connectivity to {self.bootstrap_servers}\n"
                f"Original error: {conn_err}"
            ) from conn_err

        return self._admin

    async def create_topic(
        self,
        topic_name: str,
        partitions: int = 1,
        replication_factor: int = 1,
    ) -> str:
        """Create a topic with the given configuration.

        Args:
            topic_name: Name of the topic to create.
            partitions: Number of partitions (default: 1).
            replication_factor: Replication factor (default: 1 for testing).

        Returns:
            The topic name (for chaining convenience).

        Raises:
            RuntimeError: If topic creation fails with a non-recoverable error.
        """
        from aiokafka.admin import NewTopic
        from aiokafka.errors import TopicAlreadyExistsError

        admin = await self._ensure_admin()

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

            # Check for errors in the response
            # Handle both aiokafka response formats for version compatibility:
            # - topic_errors: List of (topic, error_code, error_message) tuples (newer versions)
            # - topic_error_codes: Dict mapping topic to error_code (older versions)
            if hasattr(response, "topic_errors") and response.topic_errors:
                for topic_error in response.topic_errors:
                    _, error_code, error_message = topic_error
                    if error_code != 0:
                        if error_code == KAFKA_ERROR_TOPIC_ALREADY_EXISTS:
                            raise TopicAlreadyExistsError
                        raise RuntimeError(
                            f"Failed to create topic '{topic_name}': "
                            f"{get_kafka_error_hint(error_code, error_message)}"
                        )
            elif hasattr(response, "topic_error_codes") and response.topic_error_codes:
                # Older aiokafka format: dict mapping topic name to error code
                for topic, error_code in response.topic_error_codes.items():
                    if error_code != 0:
                        if error_code == KAFKA_ERROR_TOPIC_ALREADY_EXISTS:
                            raise TopicAlreadyExistsError
                        raise RuntimeError(
                            f"Failed to create topic '{topic}': "
                            f"{get_kafka_error_hint(error_code)}"
                        )

            self.created_topics.append(topic_name)

            # Wait for topic metadata to propagate
            await wait_for_topic_metadata(
                admin, topic_name, expected_partitions=partitions
            )

        except TopicAlreadyExistsError:
            # Topic already exists - still wait for metadata
            await wait_for_topic_metadata(
                admin, topic_name, timeout=5.0, expected_partitions=partitions
            )

        return topic_name

    async def cleanup(self) -> None:
        """Clean up created topics and close admin client.

        This method is safe to call multiple times. It logs warnings for
        cleanup failures but does not raise exceptions.
        """
        if self._admin is not None:
            if self.created_topics:
                try:
                    await self._admin.delete_topics(self.created_topics)
                except Exception as e:
                    logger.warning(
                        "Cleanup failed for Kafka topics %s: %s",
                        self.created_topics,
                        e,
                        exc_info=True,
                    )
                self.created_topics.clear()

            try:
                await self._admin.close()
            except Exception as e:
                logger.warning(
                    "Failed to close Kafka admin client: %s",
                    e,
                    exc_info=True,
                )
            finally:
                self._admin = None

    @property
    def admin_client(self) -> AIOKafkaAdminClient | None:
        """Get the underlying admin client (if initialized).

        Returns:
            The admin client or None if not yet initialized.

        Note:
            This property is primarily for advanced use cases where direct
            admin client access is needed. Most use cases should use the
            create_topic method instead.
        """
        return self._admin


__all__ = [
    # Consumer readiness
    "wait_for_consumer_ready",
    # Topic metadata
    "wait_for_topic_metadata",
    # Topic management
    "KafkaTopicManager",
    # Error handling
    "KAFKA_ERROR_REMEDIATION_HINTS",
    "KAFKA_ERROR_TOPIC_ALREADY_EXISTS",
    "KAFKA_ERROR_INVALID_PARTITIONS",
    "get_kafka_error_hint",
    # Utilities
    "parse_bootstrap_servers",
    # Validation
    "validate_bootstrap_servers",
    "KafkaConfigValidationResult",
]
