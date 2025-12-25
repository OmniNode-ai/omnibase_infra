# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for _sanitize_backend_error function.

Tests validate that the error sanitization function correctly:
1. Extracts safe error patterns from raw error messages
2. Prevents exposure of sensitive information (connection strings, credentials)
3. Provides useful debugging context without security risks

Security Rule (from CLAUDE.md):
    NEVER include in error messages:
    - Passwords, API keys, tokens, secrets
    - Full connection strings with credentials
    - PII (names, emails, SSNs)

    SAFE to include:
    - Error codes
    - Operation names
    - Exception type names

Related:
    - PR #78: Error handling security review
    - OMN-954: Effect idempotency and retry behavior
"""

import pytest

from omnibase_infra.nodes.effects.registry_effect import _sanitize_backend_error


class TestSanitizeBackendError:
    """Test suite for _sanitize_backend_error function."""

    # =====================================================================
    # Safe Pattern Detection Tests
    # =====================================================================

    @pytest.mark.parametrize(
        ("raw_error", "expected_suffix"),
        [
            # Connection-related patterns (more specific first in tuple)
            ("Connection refused", "connection refused"),
            ("connection refused by server", "connection refused"),
            ("Connection reset by peer", "connection reset"),
            ("Connection timeout after 30s", "connection timeout"),
            ("Connection closed unexpectedly", "connection closed"),
            # Network patterns
            ("Network unreachable", "network unreachable"),
            ("Host not found: db.internal.example.com", "host not found"),
            ("DNS lookup failed for hostname", "dns lookup failed"),
            # Availability patterns
            ("Service unavailable", "service unavailable"),
            ("service unavailable - please retry", "service unavailable"),
            # Resource patterns
            ("Too many connections", "too many connections"),
            ("too many connections to database pool", "too many connections"),
            ("Resource exhausted", "resource exhausted"),
            # Authentication patterns (safe to expose type, not details)
            ("Authentication failed", "authentication failed"),
            ("authentication failed for user admin", "authentication failed"),
            ("Permission denied", "permission denied"),
            ("Access denied to resource", "access denied"),
            # Timeout patterns (generic timeout last)
            ("Timeout", "timeout"),
            ("Read timeout", "timeout"),
            ("Operation timeout", "timeout"),
            # Not found patterns
            ("Not found", "not found"),
            ("Resource not found", "not found"),
            # Conflict patterns
            ("Already exists", "already exists"),
            ("Resource already exists", "already exists"),
            ("Conflict detected", "conflict"),
            # Unavailable pattern
            ("Backend unavailable", "unavailable"),
        ],
    )
    def test_safe_patterns_extracted(
        self, raw_error: str, expected_suffix: str
    ) -> None:
        """Verify safe error patterns are correctly extracted."""
        result = _sanitize_backend_error("TestBackend", raw_error)
        assert result == f"TestBackend operation failed: {expected_suffix}"

    # =====================================================================
    # Secret Scrubbing Tests - CRITICAL SECURITY
    # =====================================================================

    @pytest.mark.parametrize(
        "raw_error",
        [
            # Connection strings with credentials
            "postgresql://admin:secret123@db.internal.example.com:5432/mydb",
            "Failed to connect: postgres://user:p@ssw0rd@localhost/db",
            "Connection string: mongodb://root:hunter2@mongo.internal:27017",
            # API keys and tokens
            "API key invalid: sk-1234567890abcdef",
            "Bearer token expired: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "Authorization failed with key: AKIAIOSFODNN7EXAMPLE",
            # Embedded credentials
            "Error with password=mysecret in config",
            "Failed: api_key=abcd1234 not accepted",
            "Token abc123xyz expired",
            # Internal hostnames without safe patterns
            "Cannot reach internal-service.prod.cluster.local",
            # Arbitrary technical errors
            "Unexpected error in module XYZ: data corruption detected",
            "Internal server error at /api/v1/sensitive/endpoint",
        ],
    )
    def test_secrets_not_exposed(self, raw_error: str) -> None:
        """Verify sensitive information is NOT exposed in sanitized errors.

        CRITICAL: These tests ensure that raw error messages containing
        potentially sensitive information are sanitized to generic messages.
        """
        result = _sanitize_backend_error("Backend", raw_error)

        # Should NOT contain the raw error
        assert raw_error.lower() not in result.lower()

        # Should be a generic message
        assert result == "Backend operation failed"

    def test_connection_string_with_credentials_sanitized(self) -> None:
        """Verify connection strings with embedded credentials are sanitized."""
        raw_error = (
            "Failed to connect to postgresql://admin:MyS3cr3tP@ss@"
            "db-primary.prod.internal:5432/production_db"
        )

        result = _sanitize_backend_error("PostgreSQL", raw_error)

        # Must not contain any part of the credentials or hostname
        assert "admin" not in result
        assert "MyS3cr3tP@ss" not in result
        assert "db-primary.prod.internal" not in result
        assert "production_db" not in result

        # Should be sanitized to generic message
        assert result == "PostgreSQL operation failed"

    # =====================================================================
    # Edge Case Tests
    # =====================================================================

    def test_none_error(self) -> None:
        """Verify None error produces generic message."""
        result = _sanitize_backend_error("Consul", None)
        assert result == "Consul operation failed"

    def test_empty_string_error(self) -> None:
        """Verify empty string error produces generic message."""
        result = _sanitize_backend_error("Consul", "")
        assert result == "Consul operation failed"

    def test_whitespace_only_error(self) -> None:
        """Verify whitespace-only error produces generic message."""
        result = _sanitize_backend_error("Consul", "   ")
        assert result == "Consul operation failed"

    def test_dict_error_converted(self) -> None:
        """Verify dict errors are converted to string for analysis."""
        raw_error = {"code": "TIMEOUT", "message": "Operation timeout"}

        result = _sanitize_backend_error("Kafka", raw_error)

        # Should detect "timeout" in the string representation
        assert result == "Kafka operation failed: timeout"

    def test_exception_object(self) -> None:
        """Verify exception objects are handled correctly."""
        raw_error = ConnectionError("Connection refused by remote host")

        result = _sanitize_backend_error("Redis", raw_error)

        # Should detect "connection refused" in the exception message
        assert result == "Redis operation failed: connection refused"

    def test_case_insensitive_matching(self) -> None:
        """Verify pattern matching is case-insensitive."""
        assert (
            _sanitize_backend_error("Backend", "CONNECTION REFUSED")
            == "Backend operation failed: connection refused"
        )
        assert (
            _sanitize_backend_error("Backend", "Timeout")
            == "Backend operation failed: timeout"
        )
        assert (
            _sanitize_backend_error("Backend", "SERVICE UNAVAILABLE")
            == "Backend operation failed: service unavailable"
        )

    # =====================================================================
    # Backend Name Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "backend_name",
        ["Consul", "PostgreSQL", "Kafka", "Redis", "Vault", "HTTP API"],
    )
    def test_backend_name_preserved(self, backend_name: str) -> None:
        """Verify backend name is correctly included in sanitized message."""
        result = _sanitize_backend_error(backend_name, "timeout error")
        assert result.startswith(f"{backend_name} operation failed")

    # =====================================================================
    # Integration with Real Error Scenarios
    # =====================================================================

    def test_consul_registration_failure(self) -> None:
        """Test realistic Consul registration failure scenario."""
        # Consul might return this type of error
        raw_error = (
            "Service registration failed: service unavailable at consul.internal:8500"
        )

        result = _sanitize_backend_error("Consul", raw_error)

        # Should extract "service unavailable" but not expose hostname/port
        assert result == "Consul operation failed: service unavailable"
        assert "consul.internal" not in result
        assert "8500" not in result

    def test_postgres_connection_failure(self) -> None:
        """Test realistic PostgreSQL connection failure scenario."""
        # psycopg2 might return this type of error
        raw_error = (
            "could not connect to server: Connection refused\n"
            '\tIs the server running on host "db.internal" and accepting '
            "TCP/IP connections on port 5432?"
        )

        result = _sanitize_backend_error("PostgreSQL", raw_error)

        # Should extract "connection refused" but not expose details
        assert result == "PostgreSQL operation failed: connection refused"
        assert "db.internal" not in result

    def test_kafka_broker_unavailable(self) -> None:
        """Test realistic Kafka broker failure scenario."""
        raw_error = "NoBrokersAvailable: unable to connect to kafka-0.internal:9092"

        result = _sanitize_backend_error("Kafka", raw_error)

        # No safe pattern matches, should be generic
        # (NoBrokersAvailable is not in safe list)
        assert result == "Kafka operation failed"
        assert "kafka-0.internal" not in result
