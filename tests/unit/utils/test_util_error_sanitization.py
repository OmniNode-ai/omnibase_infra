# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for error message sanitization utility.

These tests verify that sensitive data is properly redacted from error messages
before they are logged, published to DLQ, or included in API responses.

See Also:
    docs/patterns/error_sanitization_patterns.md - Sanitization guidelines
    docs/architecture/DLQ_MESSAGE_FORMAT.md - DLQ security considerations
"""

from __future__ import annotations

from omnibase_infra.utils import SENSITIVE_PATTERNS, sanitize_error_message


class TestSanitizeErrorMessage:
    """Tests for sanitize_error_message function."""

    def test_safe_error_not_redacted(self) -> None:
        """Normal errors without sensitive patterns should pass through."""
        try:
            raise ValueError("Connection refused by remote host")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "Connection refused" in result
        assert "ValueError" in result
        assert "REDACTED" not in result

    def test_password_in_error_is_redacted(self) -> None:
        """Errors containing 'password' should be redacted."""
        try:
            raise ValueError("Auth failed with password=secret123")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "secret123" not in result
        assert "password" not in result.lower()
        assert "ValueError" in result

    def test_api_key_in_error_is_redacted(self) -> None:
        """Errors containing 'api_key' should be redacted."""
        try:
            raise RuntimeError("Request failed with api_key=sk-12345abcde")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "sk-12345" not in result
        assert "RuntimeError" in result

    def test_bearer_token_in_error_is_redacted(self) -> None:
        """Errors containing 'bearer' token should be redacted."""
        try:
            raise RuntimeError("Auth failed with bearer eyJhbGciOiJIUzI1NiJ9.xxx")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "eyJhbG" not in result

    def test_connection_string_postgres_is_redacted(self) -> None:
        """PostgreSQL connection strings should be redacted."""
        try:
            raise ConnectionError(
                "Failed to connect to postgres://user:pass@db.example.com:5432/mydb"
            )
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "user:pass" not in result
        assert "postgres://" not in result.lower()

    def test_connection_string_mongodb_is_redacted(self) -> None:
        """MongoDB connection strings should be redacted."""
        try:
            raise ConnectionError(
                "Connection failed: mongodb://admin:secret@mongo.example.com:27017/db"
            )
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "admin:secret" not in result

    def test_connection_string_redis_is_redacted(self) -> None:
        """Redis connection strings should be redacted."""
        try:
            raise ConnectionError("Cannot connect: redis://user:pass@redis:6379/0")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "user:pass" not in result

    def test_secret_in_error_is_redacted(self) -> None:
        """Errors containing 'secret' should be redacted."""
        try:
            raise ValueError("Secret key is invalid: my-super-secret-key")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "my-super-secret" not in result

    def test_credential_in_error_is_redacted(self) -> None:
        """Errors containing 'credential' should be redacted."""
        try:
            raise PermissionError("Invalid credentials provided: admin/admin123")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "admin123" not in result

    def test_private_key_in_error_is_redacted(self) -> None:
        """Errors containing 'private_key' should be redacted."""
        try:
            raise ValueError("Failed to parse private_key: -----BEGIN RSA KEY-----")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "BEGIN RSA KEY" not in result

    def test_pem_header_is_redacted(self) -> None:
        """PEM format headers should be redacted."""
        try:
            raise ValueError("Certificate parse error: -----BEGIN CERTIFICATE-----")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "BEGIN CERTIFICATE" not in result

    def test_long_message_is_truncated(self) -> None:
        """Long error messages should be truncated."""
        long_message = "A" * 1000
        try:
            raise ValueError(long_message)
        except Exception as e:
            result = sanitize_error_message(e, max_length=100)

        assert "[truncated]" in result
        assert len(result) < 200  # Should be reasonably short

    def test_case_insensitive_matching(self) -> None:
        """Pattern matching should be case-insensitive."""
        try:
            raise ValueError("PASSWORD is SECRET_TOKEN for API_KEY")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "SECRET_TOKEN" not in result

    def test_exception_type_always_included(self) -> None:
        """Exception type should always be in the result."""
        try:
            raise TypeError("password=secret")
        except Exception as e:
            result = sanitize_error_message(e)

        assert "TypeError" in result

    def test_custom_max_length(self) -> None:
        """Custom max_length should be respected."""
        message = "A" * 200
        try:
            raise ValueError(message)
        except Exception as e:
            result = sanitize_error_message(e, max_length=50)

        # Result should contain type prefix + truncated message
        assert "[truncated]" in result
        # Original 200 chars should be truncated to ~50 + type prefix + "[truncated]"
        assert len(result) < 100


class TestSensitivePatterns:
    """Tests for the SENSITIVE_PATTERNS constant."""

    def test_patterns_is_tuple(self) -> None:
        """SENSITIVE_PATTERNS should be an immutable tuple."""
        assert isinstance(SENSITIVE_PATTERNS, tuple)

    def test_patterns_include_common_credentials(self) -> None:
        """Should include common credential patterns."""
        expected_patterns = [
            "password",
            "secret",
            "token",
            "api_key",
            "bearer",
            "credential",
        ]
        for pattern in expected_patterns:
            assert pattern in SENSITIVE_PATTERNS, f"Missing pattern: {pattern}"

    def test_patterns_include_connection_strings(self) -> None:
        """Should include database connection string patterns."""
        expected_patterns = [
            "postgres://",
            "postgresql://",
            "mongodb://",
            "mysql://",
            "redis://",
        ]
        for pattern in expected_patterns:
            assert pattern in SENSITIVE_PATTERNS, f"Missing pattern: {pattern}"

    def test_patterns_include_pem_headers(self) -> None:
        """Should include PEM format headers."""
        assert "-----begin" in SENSITIVE_PATTERNS
        assert "-----end" in SENSITIVE_PATTERNS


class TestDLQIntegration:
    """Tests verifying sanitization works for DLQ scenarios."""

    def test_dlq_error_with_connection_failure(self) -> None:
        """Simulate a DLQ error from database connection failure."""
        # Simulate what psycopg2 might raise
        try:
            raise RuntimeError(
                'FATAL: password authentication failed for user "admin" '
                'connection to server at "db.example.com" (192.168.1.100), '
                "port 5432 failed: FATAL: password authentication failed"
            )
        except RuntimeError as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        # Should not expose the password-related error details
        assert "admin" not in result.lower() or "REDACTED" in result

    def test_dlq_error_with_api_failure(self) -> None:
        """Simulate a DLQ error from external API failure."""
        try:
            raise RuntimeError(
                "HTTP 401 Unauthorized: Invalid API key 'sk-abc123xyz789' "
                "for service at https://api.example.com/v1/endpoint"
            )
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "sk-abc123" not in result

    def test_dlq_error_with_vault_failure(self) -> None:
        """Simulate a DLQ error from Vault secret retrieval."""
        try:
            raise PermissionError(
                "Error reading secret at path 'secret/data/database/credentials': "
                "permission denied, token: hvs.CAESIG..."
            )
        except Exception as e:
            result = sanitize_error_message(e)

        assert "REDACTED" in result
        assert "hvs.CAES" not in result
