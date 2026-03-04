# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Error resolution catalog mapping transport types to retry guidance (OMN-518)."""

from __future__ import annotations

from dataclasses import dataclass

from omnibase_infra.enums import EnumInfraTransportType


@dataclass(frozen=True)
class ErrorResolution:
    """A suggested resolution for an infrastructure error.

    Attributes:
        suggestion: Human-readable resolution guidance.
        retry_after_seconds: Recommended retry delay (None if not retryable).
        is_retryable: Whether the error is typically transient and retryable.
    """

    suggestion: str
    retry_after_seconds: float | None = None
    is_retryable: bool = False


# ---------------------------------------------------------------------------
# Resolution catalog keyed by (error_class_name, transport_type | None).
# A None transport_type key acts as a fallback for that error class.
# ---------------------------------------------------------------------------

_CATALOG: dict[tuple[str, EnumInfraTransportType | None], ErrorResolution] = {
    # --- InfraConnectionError ---
    ("InfraConnectionError", EnumInfraTransportType.DATABASE): ErrorResolution(
        suggestion=(
            "Check that PostgreSQL is running and accepting connections. "
            "Verify host, port, and authentication config in ~/.omnibase/.env."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", EnumInfraTransportType.KAFKA): ErrorResolution(
        suggestion=(
            "Check that Redpanda/Kafka is running. "
            "Verify KAFKA_BOOTSTRAP_SERVERS in ~/.omnibase/.env. "
            "For local Docker: localhost:19092."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", EnumInfraTransportType.HTTP): ErrorResolution(
        suggestion=(
            "Verify the target service URL is reachable. "
            "Check network connectivity and firewall rules."
        ),
        retry_after_seconds=2.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", EnumInfraTransportType.QDRANT): ErrorResolution(
        suggestion=(
            "Check that Qdrant is running on localhost:6333. "
            "Verify QDRANT_URL in ~/.omnibase/.env."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", EnumInfraTransportType.VALKEY): ErrorResolution(
        suggestion=(
            "Check that Valkey is running on localhost:16379. "
            "Verify the service is healthy with docker ps."
        ),
        retry_after_seconds=3.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", EnumInfraTransportType.INFISICAL): ErrorResolution(
        suggestion=(
            "Check that Infisical is running on localhost:8880. "
            "Verify INFISICAL_ADDR is set in ~/.omnibase/.env."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", EnumInfraTransportType.LLM): ErrorResolution(
        suggestion=(
            "Check that the LLM endpoint is running. "
            "Verify LLM_CODER_URL or LLM_EMBEDDING_URL in ~/.omnibase/.env."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    ("InfraConnectionError", None): ErrorResolution(
        suggestion="Verify the target service is running and network is reachable.",
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    # --- InfraTimeoutError ---
    ("InfraTimeoutError", EnumInfraTransportType.DATABASE): ErrorResolution(
        suggestion=(
            "Database query exceeded timeout. Consider optimizing the query, "
            "adding indexes, or increasing the timeout threshold."
        ),
        retry_after_seconds=10.0,
        is_retryable=True,
    ),
    ("InfraTimeoutError", EnumInfraTransportType.HTTP): ErrorResolution(
        suggestion=(
            "HTTP request timed out. The target service may be overloaded. "
            "Consider increasing timeout or implementing pagination."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
    ("InfraTimeoutError", EnumInfraTransportType.KAFKA): ErrorResolution(
        suggestion=(
            "Kafka operation timed out. The broker may be under heavy load. "
            "Check broker health and consider increasing request.timeout.ms."
        ),
        retry_after_seconds=10.0,
        is_retryable=True,
    ),
    ("InfraTimeoutError", None): ErrorResolution(
        suggestion="Operation timed out. The target service may be overloaded or unreachable.",
        retry_after_seconds=10.0,
        is_retryable=True,
    ),
    # --- InfraAuthenticationError ---
    ("InfraAuthenticationError", EnumInfraTransportType.DATABASE): ErrorResolution(
        suggestion=(
            "Database authentication failed. "
            "Verify POSTGRES_PASSWORD in ~/.omnibase/.env."
        ),
        is_retryable=False,
    ),
    ("InfraAuthenticationError", EnumInfraTransportType.INFISICAL): ErrorResolution(
        suggestion=(
            "Infisical authentication failed. "
            "Verify INFISICAL_CLIENT_ID and INFISICAL_CLIENT_SECRET "
            "in ~/.omnibase/.env. Re-run provision-infisical.py if needed."
        ),
        is_retryable=False,
    ),
    ("InfraAuthenticationError", None): ErrorResolution(
        suggestion="Authentication failed. Verify credentials in ~/.omnibase/.env.",
        is_retryable=False,
    ),
    # --- InfraUnavailableError ---
    ("InfraUnavailableError", None): ErrorResolution(
        suggestion=(
            "Service is unavailable. Check that the Docker infrastructure is running: "
            "docker compose -f omnibase_infra/docker/docker-compose.infra.yml up -d"
        ),
        retry_after_seconds=15.0,
        is_retryable=True,
    ),
    # --- InfraRateLimitedError ---
    ("InfraRateLimitedError", None): ErrorResolution(
        suggestion=(
            "Rate limit exceeded. Respect the Retry-After header value. "
            "Consider implementing request batching or reducing call frequency."
        ),
        retry_after_seconds=30.0,
        is_retryable=True,
    ),
    # --- ProtocolConfigurationError ---
    ("ProtocolConfigurationError", None): ErrorResolution(
        suggestion=(
            "Configuration validation failed. "
            "Check contract.yaml and ~/.omnibase/.env for required fields."
        ),
        is_retryable=False,
    ),
    # --- SecretResolutionError ---
    ("SecretResolutionError", None): ErrorResolution(
        suggestion=(
            "Secret not found. Verify Infisical is seeded: "
            "uv run python scripts/seed-infisical.py --contracts-dir src/omnibase_infra/nodes --execute"
        ),
        is_retryable=False,
    ),
    # --- InfraRequestRejectedError ---
    ("InfraRequestRejectedError", None): ErrorResolution(
        suggestion=(
            "Request was rejected by the target service. "
            "Check the request payload for validation errors."
        ),
        is_retryable=False,
    ),
    # --- InfraProtocolError ---
    ("InfraProtocolError", None): ErrorResolution(
        suggestion=(
            "Received unexpected response format from service. "
            "Check that the target endpoint is correct and the service is healthy."
        ),
        retry_after_seconds=5.0,
        is_retryable=True,
    ),
}


def get_resolution(
    error_class: str,
    transport_type: EnumInfraTransportType | None = None,
) -> ErrorResolution | None:
    """Look up a resolution suggestion for an error class and transport type.

    First checks for an exact (error_class, transport_type) match, then falls
    back to (error_class, None) as a generic suggestion for that error class.

    Args:
        error_class: The error class name (e.g., "InfraConnectionError").
        transport_type: The transport type from the error context.

    Returns:
        An ErrorResolution if a match is found, otherwise None.

    Example:
        >>> resolution = get_resolution(
        ...     "InfraConnectionError",
        ...     EnumInfraTransportType.DATABASE,
        ... )
        >>> if resolution:
        ...     print(resolution.suggestion)
        Check that PostgreSQL is running ...
    """
    # Try exact match first
    result = _CATALOG.get((error_class, transport_type))
    if result is not None:
        return result
    # Fall back to generic match for the error class
    return _CATALOG.get((error_class, None))


__all__ = ["ErrorResolution", "get_resolution"]
