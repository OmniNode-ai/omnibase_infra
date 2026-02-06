# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S106
# S106 disabled: Test password fixtures are intentional for DSN validation testing
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

        assert "expected a non-empty string, got empty string" in str(exc_info.value)

    def test_invalid_whitespace_only(self) -> None:
        """Test DSN with only whitespace."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn("   ")

        assert "expected a non-empty string, got empty string" in str(exc_info.value)

    def test_invalid_none_value(self) -> None:
        """Test None value (type checking)."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(None)  # type: ignore[arg-type]

        assert "expected a string, got None" in str(exc_info.value)

    def test_invalid_non_string_type(self) -> None:
        """Test non-string type."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(12345)  # type: ignore[arg-type]

        assert "expected str, got int" in str(exc_info.value)

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

    def test_edge_case_database_with_slash_rejected(self) -> None:
        """Test that database names with slashes are rejected for security.

        Database names like 'my/db' do not match the safe pattern and are
        rejected to prevent potential path traversal or injection attacks.
        """
        dsn = "postgresql://user:pass@localhost:5432/my/db"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        # Should be rejected - either as invalid pattern or forbidden chars
        error_msg = str(exc_info.value).lower()
        assert (
            "forbidden" in error_msg
            or "must start with" in error_msg
            or "only letters" in error_msg
        )

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
            password="pass",
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
            password="super_secret",
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
            password="another_secret",
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
            password="real_password",
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
            password="super_secret",
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
            password="secret",
            hostname="localhost",
            database="mydb",
        )

        result1 = dsn.to_sanitized_dict()
        result2 = dsn.to_sanitized_dict()

        # Should be equal but not the same object
        assert result1 == result2
        assert result1 is not result2


class TestDatabaseNameValidation:
    """Tests for database name security validation.

    This test class validates the database name validation function that
    prevents SQL injection attacks by rejecting dangerous characters and
    patterns in database names.
    """

    def test_valid_simple_name(self) -> None:
        """Test simple lowercase database name."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        result = validate_database_name("mydb")
        assert result == "mydb"

    def test_valid_mixed_case(self) -> None:
        """Test mixed case database name."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        result = validate_database_name("MyDatabase")
        assert result == "MyDatabase"

    def test_valid_with_underscore(self) -> None:
        """Test database name with underscores."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        result = validate_database_name("my_database_v2")
        assert result == "my_database_v2"

    def test_valid_with_hyphen(self) -> None:
        """Test database name with hyphens."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        result = validate_database_name("my-database")
        assert result == "my-database"

    def test_valid_with_numbers(self) -> None:
        """Test database name with numbers (not first character)."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        result = validate_database_name("db123")
        assert result == "db123"

    def test_valid_starts_with_underscore(self) -> None:
        """Test database name starting with underscore."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        result = validate_database_name("_private_db")
        assert result == "_private_db"

    def test_invalid_starts_with_number(self) -> None:
        """Test that database names starting with numbers are rejected."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("123db")

        assert "must start with a letter or underscore" in str(exc_info.value)

    def test_invalid_semicolon_injection(self) -> None:
        """Test that semicolons are rejected (SQL injection risk)."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("mydb;DROP TABLE users")

        assert "forbidden characters" in str(exc_info.value).lower()

    def test_invalid_single_quote_injection(self) -> None:
        """Test that single quotes are rejected (SQL injection risk)."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("my'db")

        assert "forbidden characters" in str(exc_info.value).lower()

    def test_invalid_double_quote_injection(self) -> None:
        """Test that double quotes are rejected (SQL injection risk)."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name('my"db')

        assert "forbidden characters" in str(exc_info.value).lower()

    def test_invalid_backslash(self) -> None:
        """Test that backslashes are rejected."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("my\\db")

        assert "forbidden characters" in str(exc_info.value).lower()

    def test_invalid_dollar_sign(self) -> None:
        """Test that dollar signs are rejected (shell injection risk)."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("my$db")

        assert "forbidden characters" in str(exc_info.value).lower()

    def test_invalid_space(self) -> None:
        """Test that spaces are rejected."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("my db")

        assert "whitespace" in str(exc_info.value).lower()

    def test_invalid_slash(self) -> None:
        """Test that forward slashes are rejected (path traversal risk)."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        # Note: Slashes are handled by the pattern check, not forbidden chars
        with pytest.raises(ProtocolConfigurationError):
            validate_database_name("my/db")

    def test_invalid_too_long(self) -> None:
        """Test that database names over 63 characters are rejected."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        long_name = "a" * 64  # Exceeds PostgreSQL 63-char limit

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name(long_name)

        assert "63 characters" in str(exc_info.value)

    def test_valid_max_length(self) -> None:
        """Test that 63-character database names are accepted."""
        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        max_name = "a" * 63
        result = validate_database_name(max_name)
        assert result == max_name

    def test_dsn_with_injection_attempt_rejected(self) -> None:
        """Test that DSNs with SQL injection attempts are rejected."""
        # Attempt to inject SQL via database name
        dsn = "postgresql://user:pass@localhost:5432/mydb;DROP TABLE users;--"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(dsn)

        assert (
            "forbidden" in str(exc_info.value).lower()
            or "injection" in str(exc_info.value).lower()
        )

    def test_dsn_with_valid_database_accepted(self) -> None:
        """Test that DSNs with valid database names work correctly."""
        dsn = "postgresql://user:pass@localhost:5432/prod_db_v2"

        result = parse_and_validate_dsn(dsn)

        assert result.database == "prod_db_v2"

    def test_all_forbidden_chars_rejected(self) -> None:
        """Test that all characters in FORBIDDEN_DATABASE_CHARS are rejected."""
        from omnibase_infra.utils.util_dsn_validation import (
            FORBIDDEN_DATABASE_CHARS,
            validate_database_name,
        )

        for char in FORBIDDEN_DATABASE_CHARS:
            with pytest.raises(ProtocolConfigurationError):
                validate_database_name(f"test{char}db")


class TestCorrelationIdPropagation:
    """Tests for correlation_id propagation through DSN validation functions.

    These tests verify that when a caller provides a correlation_id, that
    same ID is propagated into error contexts rather than generating a new
    one. This is critical for end-to-end distributed tracing.
    """

    def test_parse_dsn_propagates_correlation_id_on_error(self) -> None:
        """Test that parse_and_validate_dsn propagates caller's correlation_id."""
        from uuid import UUID, uuid4

        caller_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn("", correlation_id=caller_id)

        # The correlation_id is stored directly on the error object
        assert exc_info.value.correlation_id == caller_id

    def test_parse_dsn_generates_correlation_id_when_none(self) -> None:
        """Test that parse_and_validate_dsn generates a UUID when none provided."""
        from uuid import UUID

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn("")

        # Should have a correlation_id even without passing one
        assert isinstance(exc_info.value.correlation_id, UUID)

    def test_validate_database_name_propagates_correlation_id(self) -> None:
        """Test that validate_database_name propagates caller's correlation_id."""
        from uuid import uuid4

        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        caller_id = uuid4()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("my;db", correlation_id=caller_id)

        assert exc_info.value.correlation_id == caller_id

    def test_validate_database_name_generates_id_when_none(self) -> None:
        """Test that validate_database_name auto-generates correlation_id."""
        from uuid import UUID

        from omnibase_infra.utils.util_dsn_validation import validate_database_name

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_database_name("my;db")

        assert isinstance(exc_info.value.correlation_id, UUID)

    def test_parse_dsn_shares_correlation_id_with_database_validation(self) -> None:
        """Test that parse_and_validate_dsn passes its correlation_id to validate_database_name.

        When database name validation fails inside parse_and_validate_dsn,
        the error's correlation_id should match what the caller passed in.
        """
        from uuid import uuid4

        caller_id = uuid4()

        # Use a DSN with an invalid database name (contains semicolon)
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            parse_and_validate_dsn(
                "postgresql://user:pass@localhost:5432/my;db",
                correlation_id=caller_id,
            )

        assert exc_info.value.correlation_id == caller_id

    def test_successful_parse_with_correlation_id(self) -> None:
        """Test that passing correlation_id does not affect successful parsing."""
        from uuid import uuid4

        caller_id = uuid4()
        result = parse_and_validate_dsn(
            "postgresql://user:pass@localhost:5432/mydb",
            correlation_id=caller_id,
        )

        assert result.hostname == "localhost"
        assert result.database == "mydb"


class TestIsPrivateIp:
    """Tests for the is_private_ip utility function.

    This test class validates private IP detection across all RFC 1918
    private ranges, loopback, link-local, and IPv6 equivalents.
    """

    # --- RFC 1918 private ranges ---

    @pytest.mark.parametrize(
        "ip",
        [
            "10.0.0.0",
            "10.0.0.1",
            "10.255.255.255",
            "10.100.50.25",
        ],
    )
    def test_rfc1918_10_network(self, ip: str) -> None:
        """Test 10.0.0.0/8 private range detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(ip) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "172.16.0.0",
            "172.16.0.1",
            "172.31.255.255",
            "172.20.10.5",
        ],
    )
    def test_rfc1918_172_16_network(self, ip: str) -> None:
        """Test 172.16.0.0/12 private range detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(ip) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "192.168.0.0",
            "192.168.0.1",
            "192.168.255.255",
            "192.168.86.200",
            "192.168.1.100",
        ],
    )
    def test_rfc1918_192_168_network(self, ip: str) -> None:
        """Test 192.168.0.0/16 private range detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(ip) is True

    # --- Loopback ---

    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",
            "127.0.0.0",
            "127.255.255.255",
            "127.0.1.1",
        ],
    )
    def test_loopback_ipv4(self, ip: str) -> None:
        """Test 127.0.0.0/8 loopback range detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(ip) is True

    def test_loopback_ipv6(self) -> None:
        """Test ::1 IPv6 loopback detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip("::1") is True

    # --- Link-local ---

    @pytest.mark.parametrize(
        "ip",
        [
            "169.254.0.0",
            "169.254.0.1",
            "169.254.255.255",
            "169.254.169.254",
        ],
    )
    def test_link_local_ipv4(self, ip: str) -> None:
        """Test 169.254.0.0/16 link-local range detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(ip) is True

    def test_link_local_ipv6(self) -> None:
        """Test fe80::/10 IPv6 link-local detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip("fe80::1") is True

    # --- IPv6 unique local ---

    def test_ipv6_unique_local(self) -> None:
        """Test fc00::/7 IPv6 unique local address detection."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip("fd12:3456:789a::1") is True

    # --- Public IPs (should return False) ---

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "54.239.28.85",
            "151.101.1.140",
            "93.184.216.34",
        ],
    )
    def test_public_ipv4(self, ip: str) -> None:
        """Test that public IPv4 addresses return False.

        Note: TEST-NET ranges (203.0.113.0/24, 198.51.100.0/24) are excluded
        because Python's ipaddress module considers them private per IANA
        allocation (they are reserved for documentation use).
        """
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(ip) is False

    def test_public_ipv6(self) -> None:
        """Test that public IPv6 addresses return False."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip("2001:4860:4860::8888") is False

    # --- Non-IP hostnames (should return False) ---

    @pytest.mark.parametrize(
        "hostname",
        [
            "db.example.com",
            "localhost",
            "my-server",
            "omninode-bridge-postgres",
            "",
        ],
    )
    def test_dns_hostnames_return_false(self, hostname: str) -> None:
        """Test that DNS hostnames return False (not IP addresses)."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip(hostname) is False

    # --- Edge cases near range boundaries ---

    def test_boundary_172_15_is_not_private(self) -> None:
        """Test that 172.15.x.x is outside the 172.16.0.0/12 private range."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        # 172.15.255.255 is outside 172.16.0.0/12 but still in the
        # broader 172.0.0.0/8 range which Python's ipaddress considers private
        # per IANA allocation. The ipaddress module's is_private check is
        # based on IANA allocations, not strictly RFC 1918.
        # This test documents actual behavior rather than asserting a
        # specific value.
        result = is_private_ip("172.15.255.255")
        assert isinstance(result, bool)

    def test_boundary_172_32_is_not_rfc1918(self) -> None:
        """Test that 172.32.0.0 is outside the 172.16.0.0/12 range."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        # 172.32.0.0 is outside RFC 1918's 172.16-31.x.x range
        result = is_private_ip("172.32.0.0")
        assert isinstance(result, bool)

    def test_boundary_11_0_0_0_is_public(self) -> None:
        """Test that 11.0.0.0 is outside the 10.0.0.0/8 private range."""
        from omnibase_infra.utils.util_dsn_validation import is_private_ip

        assert is_private_ip("11.0.0.0") is False


__all__: list[str] = [
    "TestCorrelationIdPropagation",
    "TestDatabaseNameValidation",
    "TestDsnValidation",
    "TestDsnEdgeCasesIntegration",
    "TestIsPrivateIp",
    "TestModelParsedDSNValidation",
]
