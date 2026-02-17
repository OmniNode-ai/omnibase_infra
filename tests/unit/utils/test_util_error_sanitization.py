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

from omnibase_infra.utils import (
    SENSITIVE_PATTERNS,
    sanitize_consul_key,
    sanitize_error_message,
    sanitize_secret_path,
    sanitize_url,
)


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


class TestSanitizeSecretPath:
    """Tests for sanitize_secret_path function."""

    def test_none_returns_none(self) -> None:
        """None input should return None."""
        assert sanitize_secret_path(None) is None

    def test_empty_string_returns_empty(self) -> None:
        """Empty string should return empty string."""
        assert sanitize_secret_path("") == ""

    def test_single_segment_unchanged(self) -> None:
        """Single segment paths should be unchanged."""
        assert sanitize_secret_path("secret") == "secret"
        assert sanitize_secret_path("kv") == "kv"

    def test_multi_segment_path_sanitized(self) -> None:
        """Multi-segment paths should be sanitized."""
        result = sanitize_secret_path("secret/data/myapp/database/credentials")
        assert result == "secret/***/***"
        assert "myapp" not in result
        assert "database" not in result
        assert "credentials" not in result

    def test_two_segment_path_sanitized(self) -> None:
        """Two-segment paths should be sanitized."""
        result = sanitize_secret_path("secret/data")
        assert result == "secret/***/***"
        assert "data" not in result

    def test_different_mount_points(self) -> None:
        """Different mount points should be preserved."""
        assert sanitize_secret_path("kv/production/api-keys") == "kv/***/***"
        assert sanitize_secret_path("pki/issue/my-role") == "pki/***/***"
        assert sanitize_secret_path("transit/encrypt/my-key") == "transit/***/***"

    def test_preserves_mount_point_only(self) -> None:
        """Only mount point should be visible in sanitized path."""
        result = sanitize_secret_path(
            "secret/data/production/database/postgres/password"
        )
        assert result.startswith("secret/")
        assert "production" not in result
        assert "postgres" not in result
        assert "password" not in result


class TestSanitizeConsulKey:
    """Tests for sanitize_consul_key function."""

    def test_none_returns_none(self) -> None:
        """None input should return None."""
        assert sanitize_consul_key(None) is None

    def test_empty_string_returns_empty(self) -> None:
        """Empty string should return empty string."""
        assert sanitize_consul_key("") == ""

    def test_single_segment_unchanged(self) -> None:
        """Single segment keys should be unchanged."""
        assert sanitize_consul_key("config") == "config"
        assert sanitize_consul_key("services") == "services"

    def test_multi_segment_key_sanitized(self) -> None:
        """Multi-segment keys should be sanitized."""
        result = sanitize_consul_key("config/database/connection")
        assert result == "config/***/***"
        assert "database" not in result
        assert "connection" not in result

    def test_two_segment_key_sanitized(self) -> None:
        """Two-segment keys should be sanitized."""
        result = sanitize_consul_key("config/myapp")
        assert result == "config/***/***"
        assert "myapp" not in result

    def test_service_keys_sanitized(self) -> None:
        """Service registry keys should be sanitized."""
        result = sanitize_consul_key("services/api-gateway/endpoints")
        assert result == "services/***/***"
        assert "api-gateway" not in result
        assert "endpoints" not in result

    def test_preserves_prefix_only(self) -> None:
        """Only the first segment should be visible in sanitized key."""
        result = sanitize_consul_key("config/production/api/internal/database/primary")
        assert result.startswith("config/")
        assert "production" not in result
        assert "internal" not in result
        assert "primary" not in result


class TestSanitizeUrl:
    """Tests for sanitize_url function."""

    def test_strips_query_params(self) -> None:
        """Query parameters should be removed."""
        result = sanitize_url("http://host:8000/v1?token=secret&key=abc")
        assert result == "http://host:8000/v1"
        assert "secret" not in result
        assert "abc" not in result

    def test_strips_fragment(self) -> None:
        """Fragments should be removed."""
        result = sanitize_url("https://example.com/health#secret-anchor")
        assert result == "https://example.com/health"
        assert "secret-anchor" not in result

    def test_strips_query_and_fragment(self) -> None:
        """Both query params and fragments should be removed."""
        result = sanitize_url("http://host:8000/v1?token=x#frag")
        assert result == "http://host:8000/v1"
        assert "token" not in result
        assert "frag" not in result

    def test_preserves_scheme_host_port_path(self) -> None:
        """Scheme, host, port, and path should be preserved."""
        result = sanitize_url("https://192.168.86.201:8000/v1/models")
        assert result == "https://192.168.86.201:8000/v1/models"

    def test_plain_url_unchanged(self) -> None:
        """URLs without query or fragment should be unchanged."""
        url = "http://example.com:9999/health"
        assert sanitize_url(url) == url

    def test_non_url_passthrough(self) -> None:
        """Non-URL strings should pass through without error."""
        result = sanitize_url("not-a-url")
        assert result == "not-a-url"

    def test_strips_userinfo_credentials(self) -> None:
        """Userinfo (username:password) embedded in the URL must be stripped."""
        result = sanitize_url("http://admin:s3cret@host:8000/v1?token=x")
        assert result == "http://host:8000/v1"
        assert "admin" not in result
        assert "s3cret" not in result
        assert "token" not in result

    def test_strips_userinfo_without_port(self) -> None:
        """Userinfo should be stripped even when port is absent."""
        result = sanitize_url("https://user:pass@example.com/path")
        assert result == "https://example.com/path"
        assert "user" not in result
        assert "pass" not in result

    def test_strips_username_only_userinfo(self) -> None:
        """Username-only userinfo (no password) should also be stripped."""
        result = sanitize_url("http://admin@host:9000/health")
        assert result == "http://host:9000/health"
        assert "admin" not in result

    def test_ipv6_loopback_with_port_and_query(self) -> None:
        """IPv6 loopback URL should preserve brackets and strip query params."""
        result = sanitize_url("http://[::1]:8000/v1?token=x")
        assert result == "http://[::1]:8000/v1"
        assert "token" not in result

    def test_ipv6_full_address_with_port(self) -> None:
        """Full IPv6 address should be reconstructed with brackets."""
        result = sanitize_url("http://[2001:db8::1]:9090/health")
        assert result == "http://[2001:db8::1]:9090/health"

    def test_ipv6_loopback_without_port(self) -> None:
        """IPv6 URL without port should still use brackets."""
        result = sanitize_url("http://[::1]/path")
        assert result == "http://[::1]/path"

    def test_ipv6_with_userinfo_stripped(self) -> None:
        """Userinfo should be stripped from IPv6 URLs."""
        result = sanitize_url("http://admin:pass@[::1]:8000/v1?key=secret")
        assert result == "http://[::1]:8000/v1"
        assert "admin" not in result
        assert "pass" not in result
        assert "secret" not in result

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert sanitize_url("") == ""

    # -- hostname-is-None credential leak edge cases (PR #352) --

    def test_no_hostname_userinfo_stripped(self) -> None:
        """Credentials must be stripped when hostname is None (bare userinfo)."""
        result = sanitize_url("http://user:pass@")
        assert "user" not in result
        assert "pass" not in result
        assert result == "http://"

    def test_no_hostname_userinfo_with_path_stripped(self) -> None:
        """Credentials must be stripped when hostname is None but path exists."""
        result = sanitize_url("http://user:pass@/path")
        assert "user" not in result
        assert "pass" not in result
        assert "/path" in result

    def test_no_hostname_userinfo_with_port_stripped(self) -> None:
        """Credentials must be stripped when hostname is None but port exists."""
        result = sanitize_url("http://user:pass@:8080/path")
        assert "user" not in result
        assert "pass" not in result
        assert ":8080" in result
        assert "/path" in result

    def test_no_hostname_empty_userinfo(self) -> None:
        """Empty userinfo (bare @) must not leak anything."""
        result = sanitize_url("http://@/path")
        assert "@" not in result
        assert "/path" in result

    def test_no_hostname_empty_userinfo_with_port(self) -> None:
        """Empty userinfo with port must not leak the @ sign."""
        result = sanitize_url("http://@:8080/path")
        assert "@" not in result
        assert ":8080" in result
        assert "/path" in result

    def test_non_url_still_passthrough(self) -> None:
        """Non-URL strings should remain unchanged after the hostname-None fix."""
        assert sanitize_url("not-a-url") == "not-a-url"
        assert sanitize_url("plain-hostname:8080") == "plain-hostname:8080"


class TestErrorClassSanitization:
    """Tests verifying error classes sanitize paths correctly."""

    def test_infra_vault_error_sanitizes_secret_path(self) -> None:
        """InfraVaultError should sanitize secret_path in extra_context."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraVaultError, ModelInfraErrorContext

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.VAULT,
            operation="read_secret",
            target_name="vault-primary",
        )

        error = InfraVaultError(
            "Failed to read secret",
            context=context,
            secret_path="secret/data/myapp/database/credentials",  # noqa: S106
        )

        # The error should have sanitized secret_path
        error_str = str(error)
        # Full path should not be in the string representation
        assert "myapp" not in error_str or "***" in error_str
        assert "database/credentials" not in error_str

    def test_infra_consul_error_sanitizes_consul_key(self) -> None:
        """InfraConsulError should sanitize consul_key in extra_context."""
        from omnibase_infra.enums import EnumInfraTransportType
        from omnibase_infra.errors import InfraConsulError, ModelInfraErrorContext

        context = ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.CONSUL,
            operation="kv_get",
            target_name="consul-primary",
        )

        error = InfraConsulError(
            "Failed to read key",
            context=context,
            consul_key="config/database/connection",
        )

        # The error should have sanitized consul_key
        error_str = str(error)
        # Full path should not be in the string representation
        assert "database/connection" not in error_str
