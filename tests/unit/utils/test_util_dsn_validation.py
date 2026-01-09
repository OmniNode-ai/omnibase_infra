# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for PostgreSQL DSN validation utility.

This test suite validates DSN parsing and validation for:
- Standard formats (user:pass@host:port/db)
- IPv6 addresses ([::1]:5432)
- Special characters in passwords (URL-encoded)
- Missing components (no password, no port, no user)
- Query parameters (sslmode=require)
- Multiple hosts (host1:port1,host2:port2)
- Invalid formats
"""

from __future__ import annotations

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.utils.util_dsn_validation import parse_and_validate_dsn


class TestDsnValidation:
    """Test DSN validation utility with comprehensive edge cases."""

    def test_valid_standard_dsn(self) -> None:
        """Test standard DSN format with all components."""
        dsn = "postgresql://user:password@localhost:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.scheme == "postgresql"
        assert result.username == "user"
        assert result.password == "password"
        assert result.hostname == "localhost"
        assert result.port == 5432
        assert result.database == "mydb"
        assert result.query == {}

    def test_valid_postgres_prefix(self) -> None:
        """Test 'postgres://' prefix (alternative to 'postgresql://')."""
        dsn = "postgres://user:password@localhost:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.scheme == "postgres"
        assert result.hostname == "localhost"

    def test_valid_no_password(self) -> None:
        """Test DSN without password (trust auth or cert-based)."""
        dsn = "postgresql://user@localhost:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.username == "user"
        assert result.password is None
        assert result.hostname == "localhost"

    def test_valid_no_port(self) -> None:
        """Test DSN without port (defaults to 5432)."""
        dsn = "postgresql://user:password@localhost/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.hostname == "localhost"
        assert result.port is None  # Will default to 5432 at connection time
        assert result.database == "mydb"

    def test_valid_no_user_password(self) -> None:
        """Test DSN with only host/port/database (local trust)."""
        dsn = "postgresql://localhost:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.username is None
        assert result.password is None
        assert result.hostname == "localhost"
        assert result.port == 5432

    def test_valid_ipv6_address(self) -> None:
        """Test DSN with IPv6 address in brackets."""
        dsn = "postgresql://user:pass@[::1]:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.hostname == "::1"
        assert result.port == 5432

    def test_valid_ipv6_full_address(self) -> None:
        """Test DSN with full IPv6 address."""
        dsn = "postgresql://user:pass@[2001:db8::1]:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.hostname == "2001:db8::1"
        assert result.port == 5432

    def test_valid_ipv4_address(self) -> None:
        """Test DSN with IPv4 address."""
        dsn = "postgresql://user:pass@192.168.1.100:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.hostname == "192.168.1.100"
        assert result.port == 5432

    def test_valid_url_encoded_password(self) -> None:
        """Test DSN with URL-encoded special characters in password."""
        # Password: p@ss:w/rd%special!
        # Encoded: p%40ss%3Aw%2Frd%25special%21
        dsn = "postgresql://user:p%40ss%3Aw%2Frd%25special%21@localhost:5432/mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.username == "user"
        # urllib.parse.unquote decodes the password
        assert result.password == "p@ss:w/rd%special!"

    def test_valid_query_parameters(self) -> None:
        """Test DSN with query parameters (sslmode, connect_timeout, etc.)."""
        dsn = "postgresql://user:pass@localhost:5432/mydb?sslmode=require&connect_timeout=10"
        result = parse_and_validate_dsn(dsn)

        assert result.database == "mydb"
        assert result.query["sslmode"] == "require"
        assert result.query["connect_timeout"] == "10"

    def test_valid_minimal_dsn(self) -> None:
        """Test minimal valid DSN (scheme + database name)."""
        dsn = "postgresql:///mydb"
        result = parse_and_validate_dsn(dsn)

        assert result.scheme == "postgresql"
        assert result.hostname is None  # Defaults to Unix socket
        assert result.database == "mydb"

    def test_valid_unix_socket_path(self) -> None:
        """Test DSN with Unix socket path."""
        dsn = "postgresql:///mydb?host=/var/run/postgresql"
        result = parse_and_validate_dsn(dsn)

        assert result.database == "mydb"
        assert result.query["host"] == "/var/run/postgresql"

    def test_invalid_missing_scheme(self) -> None:
        """Test DSN without scheme."""
        dsn = "user:pass@localhost:5432/mydb"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        assert "dsn must start with" in str(exc_info.value)
        assert "postgresql://" in str(exc_info.value)

    def test_invalid_wrong_scheme(self) -> None:
        """Test DSN with incorrect scheme."""
        dsn = "mysql://user:pass@localhost:5432/mydb"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        assert "dsn must start with" in str(exc_info.value)

    def test_invalid_empty_string(self) -> None:
        """Test empty DSN string."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn("")

        assert "dsn cannot be empty" in str(exc_info.value)

    def test_invalid_whitespace_only(self) -> None:
        """Test DSN with only whitespace."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn("   ")

        assert "dsn cannot be empty" in str(exc_info.value)

    def test_invalid_none_value(self) -> None:
        """Test None value (type checking)."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(None)  # type: ignore[arg-type]

        assert "dsn cannot be None" in str(exc_info.value)

    def test_invalid_non_string_type(self) -> None:
        """Test non-string type."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(12345)  # type: ignore[arg-type]

        assert "dsn must be a string" in str(exc_info.value)

    def test_invalid_missing_database_name(self) -> None:
        """Test DSN without database name."""
        dsn = "postgresql://user:pass@localhost:5432"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        assert "database name" in str(exc_info.value).lower()

    def test_invalid_port_not_numeric(self) -> None:
        """Test DSN with non-numeric port."""
        dsn = "postgresql://user:pass@localhost:abc/mydb"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        # urllib.parse will raise ValueError for invalid port
        assert (
            "port" in str(exc_info.value).lower()
            or "invalid" in str(exc_info.value).lower()
        )

    def test_invalid_port_out_of_range(self) -> None:
        """Test DSN with port out of valid range (1-65535)."""
        dsn = "postgresql://user:pass@localhost:99999/mydb"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        assert "port" in str(exc_info.value).lower()

    def test_multiple_hosts_not_supported(self) -> None:
        """Test DSN with multiple hosts (not supported by urllib.parse).

        Note: PostgreSQL supports multiple hosts like:
        postgresql://host1:5432,host2:5433/mydb

        However, urllib.parse doesn't handle this format and will raise
        an error when trying to parse the port (it sees "5432,host2:5433"
        as the port value, which is invalid).

        Multi-host DSNs are NOT supported. If multi-host support is needed,
        use a PostgreSQL-specific parser like psycopg2.conninfo_to_dict.
        """
        dsn = "postgresql://user:pass@host1:5432,host2:5433/mydb"

        # urllib.parse will raise ValueError when accessing the port
        # because it sees "5432,host2:5433" as the port value
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        assert "port" in str(exc_info.value).lower()

    def test_sanitization_no_credential_leakage(self) -> None:
        """Test that validation errors don't leak credentials.

        This is a security test to ensure error messages never contain
        the actual DSN with credentials.
        """
        dsn = "postgresql://admin:super_secret_password@localhost:5432"

        # Missing database name will trigger error
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        error_msg = str(exc_info.value)

        # Error message should NOT contain password
        assert "super_secret_password" not in error_msg
        # Error message should NOT contain username
        assert "admin" not in error_msg
        # Should see [REDACTED] instead
        assert "[REDACTED]" in error_msg or "database name" in error_msg.lower()

    def test_edge_case_database_with_slash(self) -> None:
        """Test database name parsing with complex path."""
        # Some DSNs might have database names that look like paths
        dsn = "postgresql://user:pass@localhost:5432/my/db"

        result = parse_and_validate_dsn(dsn)

        # urllib.parse will parse full path as database
        assert result.database == "my/db"

    def test_edge_case_empty_password(self) -> None:
        """Test DSN with empty password (user: format)."""
        dsn = "postgresql://user:@localhost:5432/mydb"

        result = parse_and_validate_dsn(dsn)

        assert result.username == "user"
        # Empty password is valid (different from no password)
        assert result.password == ""

    def test_edge_case_at_sign_in_password(self) -> None:
        """Test @ sign in password (must be URL-encoded)."""
        # Password: p@ssword
        # Encoded: p%40ssword
        dsn = "postgresql://user:p%40ssword@localhost:5432/mydb"

        result = parse_and_validate_dsn(dsn)

        assert result.password == "p@ssword"

    def test_edge_case_colon_in_password(self) -> None:
        """Test colon in password (must be URL-encoded)."""
        # Password: pass:word
        # Encoded: pass%3Aword
        dsn = "postgresql://user:pass%3Aword@localhost:5432/mydb"

        result = parse_and_validate_dsn(dsn)

        assert result.password == "pass:word"


class TestDsnEdgeCasesIntegration:
    """Integration tests for DSN validation with config models."""

    def test_config_model_accepts_valid_dsn(self) -> None:
        """Test that config models accept valid DSNs after update."""
        from omnibase_infra.idempotency.models.model_postgres_idempotency_store_config import (
            ModelPostgresIdempotencyStoreConfig,
        )

        # Standard format
        config = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@localhost:5432/mydb"
        )
        assert config.dsn == "postgresql://user:pass@localhost:5432/mydb"

        # IPv6
        config_ipv6 = ModelPostgresIdempotencyStoreConfig(
            dsn="postgresql://user:pass@[::1]:5432/mydb"
        )
        assert "[::1]" in config_ipv6.dsn

    def test_config_model_rejects_invalid_dsn(self) -> None:
        """Test that config models reject invalid DSNs."""
        from pydantic import ValidationError

        from omnibase_infra.idempotency.models.model_postgres_idempotency_store_config import (
            ModelPostgresIdempotencyStoreConfig,
        )

        # Missing database name
        with pytest.raises((ProtocolConfigurationError, ValidationError)):
            ModelPostgresIdempotencyStoreConfig(
                dsn="postgresql://user:pass@localhost:5432"
            )

        # Wrong scheme
        with pytest.raises((ProtocolConfigurationError, ValidationError)):
            ModelPostgresIdempotencyStoreConfig(
                dsn="mysql://user:pass@localhost:5432/mydb"
            )


class TestModelParsedDSNValidation:
    """Tests for ModelParsedDSN Pydantic model validation.

    This test class validates the Pydantic model constraints including:
    - Immutability (frozen=True behavior)
    - Port range validation (1-65535)
    - Scheme validation (Literal["postgresql", "postgres"])
    """

    def test_frozen_immutability(self) -> None:
        """Test that ModelParsedDSN is immutable (frozen=True).

        The model uses ConfigDict(frozen=True), which should prevent
        modification of any field after instantiation.
        """
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            username="user",
            password="pass",  # noqa: S106 - test value
            hostname="localhost",
            port=5432,
            database="mydb",
        )

        # Attempting to modify a frozen model should raise ValidationError
        with pytest.raises(ValidationError):
            dsn.hostname = "newhost"  # type: ignore[misc]

    def test_frozen_immutability_all_fields(self) -> None:
        """Test immutability applies to all fields."""
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            hostname="localhost",
            port=5432,
            database="mydb",
        )

        # All fields should be immutable
        with pytest.raises(ValidationError):
            dsn.scheme = "postgres"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            dsn.port = 5433  # type: ignore[misc]

        with pytest.raises(ValidationError):
            dsn.database = "otherdb"  # type: ignore[misc]

    def test_port_validation_too_low(self) -> None:
        """Test that port 0 is rejected.

        Port must be >= 1 per the Field(ge=1) constraint.
        """
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedDSN(
                scheme="postgresql",
                hostname="localhost",
                port=0,
                database="db",
            )

        # Verify the error is about the port constraint
        assert "port" in str(exc_info.value).lower()

    def test_port_validation_negative(self) -> None:
        """Test that negative port values are rejected."""
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedDSN(
                scheme="postgresql",
                hostname="localhost",
                port=-1,
                database="db",
            )

        assert "port" in str(exc_info.value).lower()

    def test_port_validation_too_high(self) -> None:
        """Test that port > 65535 is rejected.

        Port must be <= 65535 per the Field(le=65535) constraint.
        """
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedDSN(
                scheme="postgresql",
                hostname="localhost",
                port=65536,
                database="db",
            )

        assert "port" in str(exc_info.value).lower()

    def test_port_validation_way_too_high(self) -> None:
        """Test that extremely high port values are rejected."""
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError):
            ModelParsedDSN(
                scheme="postgresql",
                hostname="localhost",
                port=99999,
                database="db",
            )

    def test_port_validation_valid_range(self) -> None:
        """Test valid port numbers across the acceptable range."""
        from omnibase_infra.types import ModelParsedDSN

        # Test boundary values and common ports
        valid_ports = [1, 80, 443, 5432, 5433, 8080, 65535]

        for port in valid_ports:
            dsn = ModelParsedDSN(
                scheme="postgresql",
                hostname="localhost",
                port=port,
                database="db",
            )
            assert dsn.port == port

    def test_port_validation_none_allowed(self) -> None:
        """Test that None port is valid (optional field)."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            hostname="localhost",
            database="db",
            # port not specified, defaults to None
        )
        assert dsn.port is None

    def test_scheme_validation_postgresql(self) -> None:
        """Test that 'postgresql' scheme is accepted."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            hostname="localhost",
            database="db",
        )
        assert dsn.scheme == "postgresql"

    def test_scheme_validation_postgres(self) -> None:
        """Test that 'postgres' scheme is accepted (alternative form)."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgres",
            hostname="localhost",
            database="db",
        )
        assert dsn.scheme == "postgres"

    def test_scheme_validation_invalid_mysql(self) -> None:
        """Test that 'mysql' scheme is rejected.

        The Literal type constrains scheme to only 'postgresql' or 'postgres'.
        """
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedDSN(
                scheme="mysql",  # type: ignore[arg-type]
                hostname="localhost",
                database="db",
            )

        # Verify the error mentions scheme or the invalid value
        error_str = str(exc_info.value).lower()
        assert "scheme" in error_str or "mysql" in error_str

    def test_scheme_validation_invalid_mongodb(self) -> None:
        """Test that 'mongodb' scheme is rejected."""
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError):
            ModelParsedDSN(
                scheme="mongodb",  # type: ignore[arg-type]
                hostname="localhost",
                database="db",
            )

    def test_scheme_validation_invalid_empty(self) -> None:
        """Test that empty string scheme is rejected."""
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError):
            ModelParsedDSN(
                scheme="",  # type: ignore[arg-type]
                hostname="localhost",
                database="db",
            )

    def test_scheme_validation_case_sensitive(self) -> None:
        """Test that scheme validation is case-sensitive.

        'PostgreSQL' (capitalized) should be rejected since only
        lowercase 'postgresql' and 'postgres' are valid.
        """
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError):
            ModelParsedDSN(
                scheme="PostgreSQL",  # type: ignore[arg-type]
                hostname="localhost",
                database="db",
            )

    def test_database_required(self) -> None:
        """Test that database field is required (no default)."""
        from pydantic import ValidationError

        from omnibase_infra.types import ModelParsedDSN

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedDSN(
                scheme="postgresql",
                hostname="localhost",
            )  # type: ignore[call-arg]

        assert "database" in str(exc_info.value).lower()

    def test_repr_masks_password(self) -> None:
        """Test that __repr__ masks the password for security."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            username="admin",
            password="super_secret",  # noqa: S106 - test value
            hostname="localhost",
            port=5432,
            database="mydb",
        )

        repr_str = repr(dsn)

        # Password should be masked
        assert "super_secret" not in repr_str
        assert "[REDACTED]" in repr_str

        # Other fields should be visible
        assert "admin" in repr_str
        assert "localhost" in repr_str
        assert "mydb" in repr_str

    def test_str_masks_password(self) -> None:
        """Test that __str__ also masks the password."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            password="another_secret",  # noqa: S106 - test value
            hostname="localhost",
            database="mydb",
        )

        str_output = str(dsn)

        assert "another_secret" not in str_output
        assert "[REDACTED]" in str_output

    def test_password_still_accessible(self) -> None:
        """Test that password is still accessible via attribute despite masking."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            password="real_password",  # noqa: S106 - test value
            hostname="localhost",
            database="mydb",
        )

        # The actual password should be accessible
        assert dsn.password == "real_password"

        # But not visible in string representations
        assert "real_password" not in repr(dsn)

    def test_to_sanitized_dict_with_password(self) -> None:
        """Test that to_sanitized_dict() masks password when set."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            username="admin",
            password="super_secret",  # noqa: S106 - test value
            hostname="localhost",
            port=5432,
            database="mydb",
            query={"sslmode": "require"},
        )

        result = dsn.to_sanitized_dict()

        # Password should be masked
        assert result["password"] == "[REDACTED]"
        assert "super_secret" not in str(result)

        # Other fields should be present and correct
        assert result["scheme"] == "postgresql"
        assert result["username"] == "admin"
        assert result["hostname"] == "localhost"
        assert result["port"] == 5432
        assert result["database"] == "mydb"
        assert result["query"] == {"sslmode": "require"}

    def test_to_sanitized_dict_without_password(self) -> None:
        """Test that to_sanitized_dict() leaves None password as None."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            username="admin",
            hostname="localhost",
            port=5432,
            database="mydb",
        )

        result = dsn.to_sanitized_dict()

        # None password should remain None (not masked)
        assert result["password"] is None

        # Other fields should be present
        assert result["scheme"] == "postgresql"
        assert result["username"] == "admin"

    def test_to_sanitized_dict_with_empty_password(self) -> None:
        """Test that to_sanitized_dict() leaves empty string password as-is.

        An empty password is technically falsy, so it should not be masked.
        This is correct behavior since empty string is different from a real password.
        """
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            username="admin",
            password="",  # Empty password
            hostname="localhost",
            database="mydb",
        )

        result = dsn.to_sanitized_dict()

        # Empty password should remain empty (falsy, so not masked)
        assert result["password"] == ""

    def test_to_sanitized_dict_returns_new_dict(self) -> None:
        """Test that to_sanitized_dict() returns a new dict each time."""
        from omnibase_infra.types import ModelParsedDSN

        dsn = ModelParsedDSN(
            scheme="postgresql",
            password="secret",  # noqa: S106 - test value
            hostname="localhost",
            database="mydb",
        )

        result1 = dsn.to_sanitized_dict()
        result2 = dsn.to_sanitized_dict()

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2


__all__: list[str] = [
    "TestDsnValidation",
    "TestDsnEdgeCasesIntegration",
    "TestModelParsedDSNValidation",
]
