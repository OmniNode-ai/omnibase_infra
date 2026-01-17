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
    - IPv6 addresses (bracketed [::1]:9092 and bare ::1 formats)
    - Clear skip reasons for test output

IPv6 Address Support:
    Both bracketed and bare IPv6 addresses are supported:
    - Bracketed with port: "[::1]:9092" or "[2001:db8::1]:9092"
    - Bare without port: "::1" or "2001:db8::1" (uses default port 29092)

    Bare IPv6 addresses with apparent port suffixes (e.g., "::1:9092") are treated
    as the full IPv6 address with default port, since the format is ambiguous.
    For unambiguous IPv6 with custom port, always use the bracketed format.

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
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraUnavailableError,
    ModelInfraErrorContext,
)

if TYPE_CHECKING:
    from aiokafka.admin import AIOKafkaAdminClient

    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

# Module-level logger for diagnostics
logger = logging.getLogger(__name__)
# =============================================================================
# Kafka Error Code Constants
# =============================================================================
# Named constants for Kafka error codes (for readable conditionals and dict keys).
# Reference: https://kafka.apache.org/protocol.html#protocol_error_codes
#
# Using named constants instead of magic numbers improves:
# - Code readability: "if error_code == KAFKA_ERROR_TOPIC_ALREADY_EXISTS" is clearer
# - Maintainability: Single source of truth for error code values
# - Searchability: Easy to find all usages of a specific error code
#
# These constants are defined BEFORE KAFKA_ERROR_REMEDIATION_HINTS so they can
# be used as dictionary keys.
# =============================================================================

KAFKA_ERROR_TOPIC_ALREADY_EXISTS = 36
"""Topic with this name already exists (error_code=36)."""

KAFKA_ERROR_INVALID_PARTITIONS = 37
"""Invalid number of partitions (error_code=37). Also indicates memory limit issues in Redpanda."""

KAFKA_ERROR_INVALID_REPLICATION_FACTOR = 38
"""Invalid replication factor (error_code=38). Cannot exceed number of brokers."""

KAFKA_ERROR_INVALID_REPLICA_ASSIGNMENT = 39
"""Invalid replica assignment (error_code=39). Specified broker IDs may not exist."""

KAFKA_ERROR_INVALID_CONFIG = 40
"""Invalid topic configuration (error_code=40). Check topic config parameters."""

KAFKA_ERROR_NOT_CONTROLLER = 41
"""Not the cluster controller (error_code=41). Retriable - cluster may be electing new controller."""

KAFKA_ERROR_CLUSTER_AUTHORIZATION_FAILED = 29
"""Cluster authorization failed (error_code=29). Client lacks ClusterAction permission."""

KAFKA_ERROR_GROUP_AUTHORIZATION_FAILED = 30
"""Group authorization failed (error_code=30). Client lacks access to consumer group."""

KAFKA_ERROR_BROKER_RESOURCE_EXHAUSTED = 89
"""Broker resource exhausted (error_code=89). Out of memory or resource limit reached."""

# =============================================================================
# Kafka Error Code Remediation Hints
# =============================================================================
# Common Kafka/Redpanda error codes with actionable remediation hints.
# Reference: https://kafka.apache.org/protocol.html#protocol_error_codes
#
# These hints help developers quickly diagnose and fix common issues when
# running integration tests against Kafka/Redpanda brokers.
#
# Dictionary keys use the named constants defined above for consistency.
# =============================================================================

KAFKA_ERROR_REMEDIATION_HINTS: dict[int, str] = {
    # Topic management errors
    KAFKA_ERROR_TOPIC_ALREADY_EXISTS: (
        "Topic already exists. This is usually harmless in test environments. "
        "If you need a fresh topic, use a unique name with UUID suffix."
    ),
    KAFKA_ERROR_INVALID_PARTITIONS: (
        "Invalid number of partitions. "
        "Hint: Ensure partitions >= 1. For Redpanda, check that the broker has "
        "sufficient memory allocated (see Docker memory limits or "
        "'redpanda.developer_mode' setting)."
    ),
    KAFKA_ERROR_INVALID_REPLICATION_FACTOR: (
        "Invalid replication factor. "
        "Hint: Replication factor cannot exceed the number of brokers. "
        "For single-node test setups, use replication_factor=1."
    ),
    KAFKA_ERROR_INVALID_REPLICA_ASSIGNMENT: (
        "Invalid replica assignment. "
        "Hint: Check that all specified broker IDs exist in the cluster."
    ),
    KAFKA_ERROR_INVALID_CONFIG: (
        "Invalid topic configuration. "
        "Hint: Check topic config parameters (retention.ms, segment.bytes, etc.). "
        "Some Redpanda/Kafka versions have different config key names."
    ),
    # Cluster state errors
    KAFKA_ERROR_NOT_CONTROLLER: (
        "Not the cluster controller. This is retriable. "
        "Hint: The cluster may be electing a new controller. Retry after a brief delay."
    ),
    # Authorization errors
    KAFKA_ERROR_CLUSTER_AUTHORIZATION_FAILED: (
        "Cluster authorization failed. "
        "Hint: Check that your client has ClusterAction permission. "
        "For Redpanda, verify ACL configuration or disable authorization for tests."
    ),
    KAFKA_ERROR_GROUP_AUTHORIZATION_FAILED: (
        "Group authorization failed. "
        "Hint: Check that your client has access to the consumer group. "
        "Verify KAFKA_SASL_* environment variables if using SASL authentication."
    ),
    # Resource errors
    KAFKA_ERROR_BROKER_RESOURCE_EXHAUSTED: (
        "Out of memory or resource exhausted on broker. "
        "Hint: For Redpanda in Docker, increase container memory limit "
        "(e.g., 'docker update --memory 2g <container>'). "
        "Check 'docker stats' for current memory usage."
    ),
}

# =============================================================================
# Default Configuration Constants
# =============================================================================
# Configurable defaults for Kafka connection settings.
# These can be overridden via environment variables in test fixtures.
# =============================================================================

KAFKA_DEFAULT_PORT = 29092
"""Default Kafka/Redpanda port for external connections (outside Docker network).

The default port 29092 is the external advertised port for Redpanda/Kafka
when running in Docker. Internal Docker connections typically use port 9092.

This value is used when:
- Bootstrap servers string has no explicit port
- Bare IPv6 addresses are provided without port specification
- Error messages suggest default configuration

Override in tests via environment variable or explicit port in bootstrap_servers.
"""

# =============================================================================
# aiokafka Version Compatibility
# =============================================================================
# The aiokafka library has different response formats across versions.
# This module handles multiple formats to ensure compatibility.
#
# Response Format Detection:
#   - topic_errors (list of tuples): aiokafka 0.8.0+ (newer format)
#   - topic_error_codes (dict): aiokafka <0.8.0 (older format)
#
# Tuple Format Variations (for topic_errors):
#   - Protocol v0: (topic_name, error_code) - 2-tuple
#   - Protocol v1+: (topic_name, error_code, error_message) - 3-tuple
#
# describe_topics Response Formats:
#   - Dict format (aiokafka 0.11.0+): {'topic_name': TopicDescription(...)}
#   - List format (older): [{'error_code': 0, 'topic': 'name', 'partitions': [...]}]
#
# The code uses runtime detection (hasattr, isinstance, len checks) to handle
# all formats gracefully without requiring version-specific imports.
# =============================================================================

AIOKAFKA_TOPIC_ERRORS_MIN_TUPLE_LEN = 2
"""Minimum tuple length for topic_errors entries (topic, error_code)."""

AIOKAFKA_TOPIC_ERRORS_FULL_TUPLE_LEN = 3
"""Full tuple length for topic_errors entries (topic, error_code, error_message)."""


def get_kafka_error_hint(error_code: int, error_message: str = "") -> str:
    """Get a remediation hint for a Kafka error code.

    Args:
        error_code: The Kafka protocol error code.
        error_message: Optional error message from the broker (for context).

    Returns:
        A formatted error message with remediation hints if available.
    """
    base_msg: str = f"Kafka error_code={error_code}"
    if error_message:
        base_msg += f", message='{error_message}'"

    hint: str | None = KAFKA_ERROR_REMEDIATION_HINTS.get(error_code)
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
# IPv6 Detection and Parsing
# =============================================================================
# Bare IPv6 addresses (without brackets) are ambiguous when combined with ports.
# For example, "::1:9092" could be interpreted as either:
#   - IPv6 address "::1" with port "9092"
#   - IPv6 address "::1:9092" without a port
#
# Per RFC 3986, IPv6 addresses in URIs should be enclosed in brackets.
# This module treats bare IPv6 addresses as the full host with default port.
# =============================================================================

# Pattern for detecting bare IPv6 addresses (without brackets)
# Matches strings that:
# - Contain only hex digits, colons, and optionally dots (for IPv4-mapped addresses)
# - Have at least 2 colons (minimum for valid IPv6)
# - Do not start with '[' (which would be bracketed)
_BARE_IPV6_PATTERN = re.compile(r"^[0-9a-fA-F:.]+$")


def _is_likely_bare_ipv6(address: str) -> bool:
    """Detect if address appears to be a bare IPv6 address without brackets.

    This is a heuristic check based on:
    - Contains more than one colon (IPv6 has multiple colons)
    - Does not start with '[' (bracketed IPv6 uses [::1]:port format)
    - Contains only valid IPv6 characters (hex digits, colons, dots for v4 suffix)

    Note:
        Bare IPv6 addresses with ports are ambiguous. For example, "::1:9092"
        could be interpreted as IPv6 "::1" with port 9092, or as the full
        IPv6 address "::1:9092". This function treats such addresses as bare
        IPv6 (the entire string is the host, use default port).

        For unambiguous IPv6 with port, use bracketed format: [::1]:9092

    Args:
        address: The address string to check.

    Returns:
        True if the address appears to be a bare IPv6 address.

    Examples:
        >>> _is_likely_bare_ipv6("::1")
        True
        >>> _is_likely_bare_ipv6("2001:db8::1")
        True
        >>> _is_likely_bare_ipv6("::ffff:192.168.1.1")  # IPv4-mapped
        True
        >>> _is_likely_bare_ipv6("[::1]:9092")  # Bracketed - not bare
        False
        >>> _is_likely_bare_ipv6("localhost:9092")  # Only one colon
        False
        >>> _is_likely_bare_ipv6("192.168.1.1:9092")  # IPv4 with port
        False
    """
    if not address or address.startswith("["):
        return False

    # Count colons - IPv6 has at least 2 colons
    colon_count: int = address.count(":")
    if colon_count < 2:
        return False

    # Check if it contains only valid IPv6 characters
    # Allows: hex digits (0-9, a-f, A-F), colons, and dots (for IPv4-mapped)
    return bool(_BARE_IPV6_PATTERN.fullmatch(address))


def normalize_ipv6_bootstrap_server(bootstrap_server: str) -> str:
    """Normalize a bootstrap server string, wrapping bare IPv6 addresses in brackets.

    Kafka bootstrap servers require IPv6 addresses to be enclosed in brackets
    when a port is specified (RFC 3986 URI format). This function ensures bare
    IPv6 addresses are properly formatted for Kafka client connections.

    Args:
        bootstrap_server: A single bootstrap server string (host:port or bare IPv6).

    Returns:
        The normalized bootstrap server string with IPv6 addresses bracketed.

    Examples:
        >>> normalize_ipv6_bootstrap_server("localhost:9092")
        'localhost:9092'
        >>> normalize_ipv6_bootstrap_server("[::1]:9092")
        '[::1]:9092'
        >>> normalize_ipv6_bootstrap_server("::1")
        '[::1]:29092'
        >>> normalize_ipv6_bootstrap_server("2001:db8::1")
        '[2001:db8::1]:29092'
        >>> normalize_ipv6_bootstrap_server("192.168.1.1:9092")
        '192.168.1.1:9092'
    """
    if not bootstrap_server or not bootstrap_server.strip():
        return bootstrap_server

    stripped = bootstrap_server.strip()

    # Already bracketed IPv6 - return as-is
    if stripped.startswith("["):
        return stripped

    # Bare IPv6 - wrap in brackets and add default port
    if _is_likely_bare_ipv6(stripped):
        return f"[{stripped}]:{KAFKA_DEFAULT_PORT}"

    # Standard format (hostname:port or IPv4:port) - return as-is
    return stripped


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


def _validate_single_server(server: str) -> tuple[bool, str, str, str | None]:
    """Validate a single bootstrap server entry.

    Args:
        server: A single server string (already stripped of whitespace).

    Returns:
        Tuple of (is_valid, host, port, error_message).
        error_message is None if validation passes.
    """
    default_port: str = str(KAFKA_DEFAULT_PORT)

    # Parse host and port
    host: str
    port: str
    host, port = parse_bootstrap_servers(server)

    # Validate port is numeric (when explicitly provided)
    # Skip port validation for bare IPv6 addresses (they contain multiple colons
    # which are part of the address, not host:port separators)
    if ":" in server and not _is_likely_bare_ipv6(server):
        # Extract port part for validation
        port_str: str
        if server.startswith("["):
            # IPv6 format: [::1]:9092
            bracket_close: int = server.rfind("]")
            if bracket_close != -1 and bracket_close < len(server) - 1:
                port_str = server[bracket_close + 2 :]
            else:
                port_str = ""
        else:
            # Standard format: host:port
            port_str = server.rsplit(":", 1)[-1] if ":" in server else ""

        if port_str and not port_str.isdigit():
            return (
                False,
                host,
                port_str,
                f"invalid port '{port_str}' (must be numeric)",
            )

        if port_str:
            port_num: int = int(port_str)
            if port_num < 1 or port_num > 65535:
                return (
                    False,
                    host,
                    port_str,
                    f"invalid port {port_num} (must be 1-65535)",
                )
            port = port_str

    return (True, host, port, None)


def validate_bootstrap_servers(
    bootstrap_servers: str | None,
) -> KafkaConfigValidationResult:
    """Validate KAFKA_BOOTSTRAP_SERVERS and return detailed result.

    Performs comprehensive validation of the bootstrap servers string:
    - Checks for empty/whitespace-only values
    - Handles comma-separated lists of servers (e.g., "server1:9092,server2:9092")
    - Validates host:port format (including IPv4 and bracketed IPv6)
    - Validates port is numeric and in valid range (1-65535)
    - Handles bare IPv6 addresses (treats as host with default port)
    - Handles edge cases: trailing commas, whitespace between entries
    - Returns structured result with skip reason for tests

    Comma-Separated Server Lists:
        Supports multiple bootstrap servers in the standard Kafka format:
        - "server1:9092,server2:9092,server3:9092"
        - Whitespace around commas is trimmed: "server1:9092 , server2:9092"
        - Empty entries are filtered: "server1:9092,,server2:9092" -> valid
        - Trailing commas are handled: "server1:9092," -> valid (ignores empty)

        The returned host/port are from the FIRST valid server in the list.

    IPv6 Address Support:
        - Bracketed IPv6 with port: "[::1]:9092" - fully validated
        - Bare IPv6 without port: "::1", "2001:db8::1" - valid, uses default port
        - Bare IPv6 with ambiguous port: "::1:9092" - treated as bare IPv6, uses
          default port (the "9092" is considered part of the address)

        For unambiguous IPv6 with custom port, use bracketed format: [::1]:9092

    Args:
        bootstrap_servers: The KAFKA_BOOTSTRAP_SERVERS value from environment.

    Returns:
        KafkaConfigValidationResult with validation status and details.

    Examples:
        >>> result = validate_bootstrap_servers("")
        >>> if not result:
        ...     pytest.skip(result.skip_reason)

        >>> result = validate_bootstrap_servers("localhost:9092")
        >>> assert result.is_valid
        >>> assert result.host == "localhost"
        >>> assert result.port == "9092"

        >>> result = validate_bootstrap_servers("server1:9092,server2:9092")
        >>> assert result.is_valid
        >>> assert result.host == "server1"  # First server in list
        >>> assert result.port == "9092"

        >>> result = validate_bootstrap_servers("[::1]:9092")
        >>> assert result.is_valid
        >>> assert result.host == "[::1]"
        >>> assert result.port == "9092"

        >>> result = validate_bootstrap_servers("::1")
        >>> assert result.is_valid
        >>> assert result.host == "::1"
        >>> assert result.port == "29092"  # Default port for bare IPv6
    """
    # Use string conversion of default port for consistent return type
    default_port: str = str(KAFKA_DEFAULT_PORT)

    # Handle None (defensive)
    if bootstrap_servers is None:
        return KafkaConfigValidationResult(
            is_valid=False,
            host="<not set>",
            port=default_port,
            error_message="KAFKA_BOOTSTRAP_SERVERS is not set (None)",
            skip_reason=(
                "KAFKA_BOOTSTRAP_SERVERS not configured. "
                "Set environment variable to enable Kafka integration tests. "
                f"Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:{KAFKA_DEFAULT_PORT}"
            ),
        )

    # Handle empty/whitespace-only
    if not bootstrap_servers or not bootstrap_servers.strip():
        return KafkaConfigValidationResult(
            is_valid=False,
            host="<not set>",
            port=default_port,
            error_message="KAFKA_BOOTSTRAP_SERVERS is empty or whitespace-only",
            skip_reason=(
                "KAFKA_BOOTSTRAP_SERVERS is empty or not set. "
                "Set environment variable to enable Kafka integration tests. "
                f"Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:{KAFKA_DEFAULT_PORT}"
            ),
        )

    # Split on commas and filter out empty/whitespace-only entries
    # This handles: trailing commas, whitespace around commas, multiple commas
    raw_servers: list[str] = [
        s.strip() for s in bootstrap_servers.split(",") if s.strip()
    ]

    # If all entries were empty/whitespace after splitting
    if not raw_servers:
        return KafkaConfigValidationResult(
            is_valid=False,
            host="<not set>",
            port=default_port,
            error_message=(
                "KAFKA_BOOTSTRAP_SERVERS contains only commas or whitespace"
            ),
            skip_reason=(
                "KAFKA_BOOTSTRAP_SERVERS contains no valid server entries. "
                "Set environment variable to enable Kafka integration tests. "
                f"Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:{KAFKA_DEFAULT_PORT}"
            ),
        )

    # Validate each server in the list
    # Track first valid and first entry (for error messages when all invalid)
    first_valid_host: str | None = None
    first_valid_port: str | None = None
    first_entry_host: str | None = None
    first_entry_port: str | None = None
    validation_errors: list[str] = []

    for server in raw_servers:
        is_valid, host, port, error_msg = _validate_single_server(server)

        # Always track first entry for error messages
        if first_entry_host is None:
            first_entry_host = host
            first_entry_port = port

        if is_valid:
            if first_valid_host is None:
                first_valid_host = host
                first_valid_port = port
        else:
            validation_errors.append(f"'{server}': {error_msg}")

    # If there were any validation errors
    if validation_errors:
        # Format error message
        if len(validation_errors) == 1:
            error_detail = validation_errors[0]
        else:
            error_detail = "; ".join(validation_errors)

        # Use first valid host if available, otherwise first entry host
        # (for better error messages showing what was provided)
        display_host = first_valid_host or first_entry_host or "<invalid>"
        display_port = first_valid_port or first_entry_port or default_port

        return KafkaConfigValidationResult(
            is_valid=False,
            host=display_host,
            port=display_port,
            error_message=f"KAFKA_BOOTSTRAP_SERVERS has invalid entries: {error_detail}",
            skip_reason=(
                f"KAFKA_BOOTSTRAP_SERVERS has invalid entries: {error_detail}. "
                f"Example: export KAFKA_BOOTSTRAP_SERVERS=localhost:{KAFKA_DEFAULT_PORT}"
            ),
        )

    # Valid configuration - return first server's host/port for display
    return KafkaConfigValidationResult(
        is_valid=True,
        host=first_valid_host or "<not set>",
        port=first_valid_port or default_port,
        error_message=None,
        skip_reason=None,
    )


def parse_bootstrap_servers(bootstrap_servers: str) -> tuple[str, str]:
    """Parse KAFKA_BOOTSTRAP_SERVERS into (host, port) tuple for error messages.

    Handles various formats safely:
    - Empty/whitespace-only: Returns ("<not set>", "29092")
    - "hostname:port": Returns ("hostname", "port")
    - "hostname" (no port): Returns ("hostname", "29092")
    - "[::1]:9092" (bracketed IPv6 with port): Returns ("[::1]", "9092")
    - "::1" (bare IPv6 without port): Returns ("::1", "29092")
    - "2001:db8::1" (bare IPv6 without port): Returns ("2001:db8::1", "29092")
    - "::ffff:192.168.1.1" (IPv4-mapped IPv6): Returns ("::ffff:192.168.1.1", "29092")

    IPv6 Address Handling:
        Bare IPv6 addresses (without brackets) are treated as the full host with
        the default port. This is because bare IPv6 with port is ambiguous - for
        example, "::1:9092" could mean either:
        - IPv6 address "::1" with port 9092, OR
        - IPv6 address "::1:9092" without a port

        For unambiguous IPv6 with port specification, use the bracketed format:
        "[::1]:9092" or "[2001:db8::1]:9092"

    Note:
        This function is primarily for error message generation. For validation
        with skip reasons, use validate_bootstrap_servers() instead.

    Args:
        bootstrap_servers: The KAFKA_BOOTSTRAP_SERVERS value.

    Returns:
        Tuple of (host, port) for use in error messages.

    Examples:
        >>> parse_bootstrap_servers("localhost:9092")
        ('localhost', '9092')
        >>> parse_bootstrap_servers("[::1]:9092")
        ('[::1]', '9092')
        >>> parse_bootstrap_servers("::1")
        ('::1', '29092')
        >>> parse_bootstrap_servers("2001:db8::1")
        ('2001:db8::1', '29092')
    """
    # Use string conversion of default port for consistent return type
    default_port: str = str(KAFKA_DEFAULT_PORT)

    # Handle empty/whitespace-only input
    if not bootstrap_servers or not bootstrap_servers.strip():
        return ("<not set>", default_port)

    stripped: str = bootstrap_servers.strip()

    # Handle IPv6 with brackets: [::1]:9092
    if stripped.startswith("["):
        bracket_close: int = stripped.rfind("]")
        if bracket_close != -1 and bracket_close < len(stripped) - 1:
            # Has closing bracket and something after it
            if stripped[bracket_close + 1] == ":":
                host: str = stripped[: bracket_close + 1]
                port: str = stripped[bracket_close + 2 :] or default_port
                return (host, port)
        # Malformed bracketed IPv6 - return as-is with default port
        return (stripped, default_port)

    # Handle bare IPv6 addresses (without brackets)
    # These contain multiple colons which are part of the address, not separators
    if _is_likely_bare_ipv6(stripped):
        return (stripped, default_port)

    # Standard host:port format - use rsplit to handle single colon
    if ":" in stripped:
        parts: list[str] = stripped.rsplit(":", 1)
        host = parts[0] or "<not set>"
        port = parts[1] if len(parts) > 1 and parts[1] else default_port
        return (host, port)

    # No colon - just hostname
    return (stripped, default_port)


async def wait_for_consumer_ready(
    event_bus: EventBusKafka,
    topic: str,
    max_wait: float = 10.0,
    initial_backoff: float = 0.1,
    max_backoff: float = 1.0,
    backoff_multiplier: float = 1.5,
    strict: bool = False,
) -> bool:
    """Wait for Kafka consumer to be ready to receive messages using polling.

    This is a **best-effort** readiness check that by default always returns True.
    It attempts to detect when the consumer is ready by polling health checks, but
    falls back gracefully on timeout to avoid blocking tests indefinitely.

    Kafka consumers require time to join the consumer group and start receiving
    messages after subscription. This helper polls the event bus health check
    until the consumer count increases, indicating the consumer task is running.

    Behavior Summary:
        1. Polls event_bus.health_check() with exponential backoff
        2. If consumer_count increases within max_wait: returns True (early exit)
        3. If max_wait exceeded:
           - strict=False (default): returns True anyway (graceful fallback)
           - strict=True: raises TimeoutError (fail-fast mode)

    Why Always Return True (default)?
        The purpose is to REDUCE flakiness by waiting for actual readiness when
        possible, not to DETECT failures. Test assertions should verify expected
        outcomes, not this helper's return value.

    Strict Mode:
        When strict=True, the function raises TimeoutError if the consumer does
        not become ready within max_wait. This is useful for tests that require
        the consumer to be ready before proceeding, and prefer a clear failure
        over a silent fallback.

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
        max_wait: Maximum time in seconds to poll before giving up. With
            strict=False (default), the function will return True regardless of
            whether consumer became ready. With strict=True, raises TimeoutError.
            Default: 10.0s. Actual wait may exceed max_wait by up to max_backoff
            (on timeout) or +0.1s stabilization delay (on success).
        initial_backoff: Initial polling delay in seconds (default 0.1s).
        max_backoff: Maximum polling delay cap in seconds (default 1.0s).
        backoff_multiplier: Multiplier for exponential backoff (default 1.5).
        strict: If True, raise TimeoutError when consumer doesn't become ready
            within max_wait. If False (default), return True on timeout for
            graceful fallback. Default: False.

    Returns:
        True when consumer is ready or when timeout occurs with strict=False.
        With strict=False, always returns True - do not use return value for
        failure detection. Use test assertions to verify expected outcomes.

    Raises:
        TimeoutError: If strict=True and consumer does not become ready within
            max_wait seconds.

    Example:
        # Best-effort wait for consumer readiness (default max_wait=10.0s)
        await wait_for_consumer_ready(bus, topic)

        # Shorter wait for fast tests
        await wait_for_consumer_ready(bus, topic, max_wait=2.0)

        # Fail-fast mode: raise TimeoutError if consumer not ready
        await wait_for_consumer_ready(bus, topic, strict=True)

        # Consumer MAY be ready here (with strict=False), but test should not
        # rely on this. Use assertions on actual test outcomes instead.
    """
    start_time: float = asyncio.get_running_loop().time()
    current_backoff: float = initial_backoff

    # Get initial consumer count for comparison
    initial_health: dict[str, object] = await event_bus.health_check()
    initial_consumer_count: int = initial_health.get("consumer_count", 0)  # type: ignore[assignment]

    # Poll until consumer count increases or timeout
    while (asyncio.get_running_loop().time() - start_time) < max_wait:
        health: dict[str, object] = await event_bus.health_check()
        consumer_count: int = health.get("consumer_count", 0)  # type: ignore[assignment]

        # If consumer count has increased, the subscription is active
        if consumer_count > initial_consumer_count:
            # Add a small additional delay for the consumer loop to start
            # processing messages after registration
            await asyncio.sleep(0.1)
            return True

        # Check if we've timed out after health check (prevents unnecessary sleep)
        elapsed: float = asyncio.get_running_loop().time() - start_time
        if elapsed >= max_wait:
            break

        # Exponential backoff with cap
        await asyncio.sleep(current_backoff)
        current_backoff = min(current_backoff * backoff_multiplier, max_backoff)

    # Log at debug level for diagnostics
    logger.debug(
        "wait_for_consumer_ready timed out after %.2fs for topic %s",
        max_wait,
        topic,
    )

    # Strict mode: raise TimeoutError for fail-fast behavior
    if strict:
        raise TimeoutError(
            f"Consumer did not become ready for topic '{topic}' within {max_wait}s"
        )

    # Default: return True even on timeout (graceful fallback)
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
    start_time: float = asyncio.get_running_loop().time()

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
                    topic_info: object = description[topic_name]
                    # TopicDescription may be an object with attributes or dict-like
                    error_code: int | None = (
                        getattr(topic_info, "error_code", None)
                        if hasattr(topic_info, "error_code")
                        else topic_info.get("error_code", -1)
                        if isinstance(topic_info, dict)
                        else -1
                    )
                    partitions: list[object] = (
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
                topic_info_item: object = description[0]
                # List items are typically dicts
                list_error_code: int
                list_partitions: list[object]
                if isinstance(topic_info_item, dict):
                    list_error_code = topic_info_item.get("error_code", -1)
                    list_partitions = topic_info_item.get("partitions", [])
                else:
                    # Object with attributes
                    list_error_code = getattr(topic_info_item, "error_code", -1)
                    list_partitions = getattr(topic_info_item, "partitions", [])

                if list_error_code == 0 and len(list_partitions) >= expected_partitions:
                    logger.debug(
                        "Topic %s ready with %d partitions (list format)",
                        topic_name,
                        len(list_partitions),
                    )
                    return True
                logger.debug(
                    "Topic %s not ready: error_code=%s, partitions=%d (list format)",
                    topic_name,
                    list_error_code,
                    len(list_partitions),
                )
            else:
                # Unknown format but truthy - log warning and accept for compatibility
                # This may indicate a new aiokafka version with different response format
                logger.warning(
                    "Topic %s: unknown describe_topics response format (type=%s). "
                    "Accepting as ready for compatibility, but this may indicate a new "
                    "aiokafka version. Consider updating util_kafka.py to handle this format. "
                    "Response: %r",
                    topic_name,
                    type(description).__name__,
                    description,
                )
                return True

        except Exception as e:
            # Log exception type for diagnostics - helps identify if specific
            # Kafka exceptions should be added to the handler
            exc_type_name = type(e).__name__
            logger.debug(
                "Topic %s metadata check failed (%s): %s", topic_name, exc_type_name, e
            )

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
            InfraConnectionError: If connection to the broker fails. Includes
                correlation ID, transport type, and remediation hints for
                common connection issues.
        """
        if self._admin is not None:
            return self._admin

        from aiokafka.admin import AIOKafkaAdminClient

        self._admin = AIOKafkaAdminClient(bootstrap_servers=self.bootstrap_servers)
        try:
            await self._admin.start()
        except Exception as conn_err:
            self._admin = None  # Reset to allow retry
            host: str
            port: str
            host, port = parse_bootstrap_servers(self.bootstrap_servers)
            # Include exception type in error message for better diagnostics
            exc_type_name = type(conn_err).__name__

            # Create error context with correlation ID for tracing
            correlation_id = uuid4()
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.KAFKA,
                operation="connect_admin_client",
                target_name=self.bootstrap_servers,
            )

            raise InfraConnectionError(
                f"Failed to connect to Kafka broker at {self.bootstrap_servers} "
                f"({exc_type_name}). "
                f"Hint: Verify the broker is running and accessible:\n"
                f"  1. Check container status: 'docker ps | grep redpanda'\n"
                f"  2. Test connectivity: 'nc -zv {host} {port}'\n"
                f"  3. For Redpanda, check health: 'curl -s http://<host>:9644/v1/status/ready'\n"
                f"  4. Verify KAFKA_BOOTSTRAP_SERVERS env var is correct\n"
                f"  5. If using Docker, ensure network connectivity to {self.bootstrap_servers}\n"
                f"Original error: {conn_err}",
                context=context,
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
            InfraUnavailableError: If topic creation fails with a non-recoverable
                Kafka error. Includes correlation ID, transport type, error code,
                and remediation hints.
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
            has_topic_errors: bool = hasattr(response, "topic_errors")
            has_topic_error_codes: bool = hasattr(response, "topic_error_codes")
            logger.debug(
                "create_topics response for '%s': has_topic_errors=%s, has_topic_error_codes=%s",
                topic_name,
                has_topic_errors,
                has_topic_error_codes,
            )

            if has_topic_errors and response.topic_errors:
                logger.debug(
                    "Using topic_errors format (newer aiokafka): %s",
                    response.topic_errors,
                )
                for topic_error in response.topic_errors:
                    # Handle both protocol v0 (2-tuple) and v1+ (3-tuple) formats:
                    # - Protocol v0: (topic_name, error_code)
                    # - Protocol v1+: (topic_name, error_code, error_message)
                    # Use length guard to safely extract elements
                    topic_error_tuple: tuple[object, ...] = (
                        tuple(topic_error)
                        if not isinstance(topic_error, tuple)
                        else topic_error
                    )
                    if len(topic_error_tuple) < AIOKAFKA_TOPIC_ERRORS_MIN_TUPLE_LEN:
                        logger.warning(
                            "Unexpected topic_error format (len=%d, min=%d): %r",
                            len(topic_error_tuple),
                            AIOKAFKA_TOPIC_ERRORS_MIN_TUPLE_LEN,
                            topic_error,
                        )
                        continue

                    _topic_name: str = str(topic_error_tuple[0])
                    topic_error_code: int = int(topic_error_tuple[1])
                    # Protocol v1+ includes error_message as third element
                    topic_error_message: str = (
                        str(topic_error_tuple[2])
                        if len(topic_error_tuple)
                        >= AIOKAFKA_TOPIC_ERRORS_FULL_TUPLE_LEN
                        else ""
                    )

                    if topic_error_code != 0:
                        if topic_error_code == KAFKA_ERROR_TOPIC_ALREADY_EXISTS:
                            raise TopicAlreadyExistsError

                        # Create error context with correlation ID for tracing
                        correlation_id = uuid4()
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.KAFKA,
                            operation="create_topic",
                            target_name=topic_name,
                        )
                        raise InfraUnavailableError(
                            f"Failed to create topic '{topic_name}': "
                            f"{get_kafka_error_hint(topic_error_code, topic_error_message)}",
                            context=context,
                            kafka_error_code=topic_error_code,
                        )
            elif has_topic_error_codes and response.topic_error_codes:
                # Older aiokafka format: dict mapping topic name to error code
                logger.debug(
                    "Using topic_error_codes format (older aiokafka): %s",
                    response.topic_error_codes,
                )
                topic_key: str
                error_code_value: int
                for topic_key, error_code_value in response.topic_error_codes.items():
                    if error_code_value != 0:
                        if error_code_value == KAFKA_ERROR_TOPIC_ALREADY_EXISTS:
                            raise TopicAlreadyExistsError

                        # Create error context with correlation ID for tracing
                        correlation_id = uuid4()
                        context = ModelInfraErrorContext.with_correlation(
                            correlation_id=correlation_id,
                            transport_type=EnumInfraTransportType.KAFKA,
                            operation="create_topic",
                            target_name=topic_key,
                        )
                        raise InfraUnavailableError(
                            f"Failed to create topic '{topic_key}': "
                            f"{get_kafka_error_hint(error_code_value)}",
                            context=context,
                            kafka_error_code=error_code_value,
                        )
            else:
                # Neither format has errors - topic created successfully
                # or response format is unknown (may indicate new aiokafka version)
                logger.debug(
                    "Topic '%s' created successfully (no errors in response)",
                    topic_name,
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
                    # Log exception type for diagnostics - helps identify if specific
                    # Kafka exceptions should be handled differently
                    exc_type_name = type(e).__name__
                    logger.warning(
                        "Cleanup failed for Kafka topics %s (%s): %s",
                        self.created_topics,
                        exc_type_name,
                        e,
                        exc_info=True,
                    )
                self.created_topics.clear()

            try:
                await self._admin.close()
            except Exception as e:
                # Log exception type for diagnostics
                exc_type_name = type(e).__name__
                logger.warning(
                    "Failed to close Kafka admin client (%s): %s",
                    exc_type_name,
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
    # Error code constants (Kafka protocol error codes)
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
    "get_kafka_error_hint",
    # Configuration constants
    "KAFKA_DEFAULT_PORT",
    # aiokafka version compatibility constants
    "AIOKAFKA_TOPIC_ERRORS_MIN_TUPLE_LEN",
    "AIOKAFKA_TOPIC_ERRORS_FULL_TUPLE_LEN",
    # Utilities
    "parse_bootstrap_servers",
    "normalize_ipv6_bootstrap_server",
    # Validation
    "validate_bootstrap_servers",
    "KafkaConfigValidationResult",
]
