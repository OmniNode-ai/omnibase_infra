# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL testing utilities for integration tests.

This module provides shared utilities for PostgreSQL-based integration tests,
including DSN building, reachability checking, and migration skip logic.

Available Utilities:
    - build_postgres_dsn: Build PostgreSQL DSN from components
    - check_postgres_reachable: Check if PostgreSQL server is reachable via TCP
    - PostgresConfig: Configuration dataclass for PostgreSQL connections
    - CONCURRENT_DDL_PATTERN: Regex pattern for detecting CONCURRENTLY DDL statements
    - should_skip_migration: Check if a migration should be skipped (CONCURRENTLY DDL)

Usage:
    >>> from tests.helpers.util_postgres import PostgresConfig, check_postgres_reachable
    >>> config = PostgresConfig.from_env()
    >>> if config.is_configured and check_postgres_reachable(config):
    ...     dsn = config.build_dsn()
    ...     # Use DSN for database connection

Migration Skip Pattern:
    Some migrations contain CONCURRENTLY DDL statements that cannot run inside
    a transaction block. Use should_skip_migration() to detect these:

    >>> if should_skip_migration(migration_sql):
    ...     logger.debug("Skipping production-only migration")
    ...     continue
"""
# NOTE: This module is the canonical location for PostgreSQL test utilities.
# Use PostgresConfig.from_env() and build_dsn() instead of inlining DSN
# construction in individual test conftest files.

from __future__ import annotations

import logging
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import ParseResult, quote_plus, unquote, urlparse
from uuid import uuid4

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from tests.infrastructure_config import DEFAULT_POSTGRES_PORT

logger = logging.getLogger(__name__)


# =============================================================================
# Migration Skip Patterns
# =============================================================================

# Regex pattern to match CONCURRENTLY DDL statements that cannot run in transactions.
# These migrations are production-only and should be skipped in test environments.
#
# Pattern matches:
#   - CREATE INDEX CONCURRENTLY
#   - CREATE UNIQUE INDEX CONCURRENTLY
#   - REINDEX [options] {INDEX|TABLE|SCHEMA|DATABASE|SYSTEM} CONCURRENTLY
#
# PostgreSQL REINDEX CONCURRENTLY syntax (from docs):
#   REINDEX [ ( option [, ...] ) ] { INDEX | TABLE | SCHEMA | DATABASE | SYSTEM }
#       [ CONCURRENTLY ] name
#
# The pattern uses word boundaries (\b) to avoid partial matches.
# The REINDEX portion explicitly matches the object type keywords to avoid
# false positives from `.*` greedy matching (e.g., matching across statements).
# Note: Comment/string stripping is handled by _strip_sql_comments() before matching.
CONCURRENT_DDL_PATTERN = re.compile(
    r"\b("
    r"CREATE\s+(UNIQUE\s+)?INDEX\s+CONCURRENTLY"
    r"|"
    # REINDEX with optional parenthesized options, then object type, then CONCURRENTLY
    # Options are like: (VERBOSE), (VERBOSE, TABLESPACE new_tablespace)
    r"REINDEX\s+(\([^)]*\)\s+)?(INDEX|TABLE|SCHEMA|DATABASE|SYSTEM)\s+CONCURRENTLY"
    r")\b",
    re.IGNORECASE,
)

# Pattern to match SQL single-line comments (-- to end of line)
_SQL_LINE_COMMENT_PATTERN = re.compile(r"--[^\n]*")

# Pattern to match SQL block comments (/* ... */)
# Uses non-greedy matching to handle multiple block comments correctly
_SQL_BLOCK_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)


def _strip_sql_comments(sql: str) -> str:
    """Strip SQL comments from a string before pattern matching.

    Removes both single-line (--) and block (/* */) comments to avoid
    false positives when detecting CONCURRENTLY DDL statements.

    Args:
        sql: Raw SQL content potentially containing comments.

    Returns:
        SQL with comments replaced by spaces (preserves position info for debugging).

    Note:
        This function does NOT strip string literals. While theoretically a string
        literal could contain DDL syntax, this is extremely rare in migration files
        and the complexity of handling escape sequences is not justified.

    Examples:
        >>> _strip_sql_comments("SELECT 1; -- comment")
        'SELECT 1;  '
        >>> _strip_sql_comments("SELECT /* inline */ 1;")
        'SELECT   1;'
    """
    # Replace block comments first (they can span multiple lines)
    result: str = _SQL_BLOCK_COMMENT_PATTERN.sub(" ", sql)
    # Then replace line comments
    result = _SQL_LINE_COMMENT_PATTERN.sub(" ", result)
    return result


def should_skip_migration(sql: str) -> bool:
    """Check if a migration SQL should be skipped in test environments.

    Migrations containing CONCURRENTLY DDL statements cannot run inside a
    transaction block, which is required for test isolation. These migrations
    are typically production-only for online schema changes.

    This function strips SQL comments before matching to avoid false positives
    from CONCURRENTLY appearing in comments.

    The pattern specifically matches:
      - CREATE INDEX CONCURRENTLY
      - CREATE UNIQUE INDEX CONCURRENTLY
      - REINDEX {INDEX|TABLE|SCHEMA|DATABASE|SYSTEM} CONCURRENTLY
      - REINDEX (options) {INDEX|TABLE|SCHEMA|DATABASE|SYSTEM} CONCURRENTLY

    The REINDEX pattern requires a valid object type keyword (INDEX, TABLE, etc.)
    to avoid false positives from overly broad matching.

    Args:
        sql: The SQL content of the migration file.

    Returns:
        True if the migration should be skipped, False otherwise.

    Examples - Statements that WILL be skipped (returns True):
        >>> should_skip_migration("CREATE INDEX CONCURRENTLY idx ON tbl(col);")
        True
        >>> should_skip_migration("CREATE UNIQUE INDEX CONCURRENTLY idx ON tbl(col);")
        True
        >>> should_skip_migration("REINDEX TABLE CONCURRENTLY my_table;")
        True
        >>> should_skip_migration("REINDEX INDEX CONCURRENTLY my_index;")
        True
        >>> should_skip_migration("REINDEX SCHEMA CONCURRENTLY my_schema;")
        True
        >>> should_skip_migration("REINDEX DATABASE CONCURRENTLY my_db;")
        True
        >>> should_skip_migration("REINDEX (VERBOSE) TABLE CONCURRENTLY my_table;")
        True

    Examples - Statements that will NOT be skipped (returns False):
        >>> should_skip_migration("CREATE INDEX idx ON tbl(col);")
        False
        >>> should_skip_migration("-- CREATE INDEX CONCURRENTLY idx ON tbl(col);")
        False
        >>> should_skip_migration("/* CREATE INDEX CONCURRENTLY */ CREATE INDEX idx ON t(c);")
        False
        >>> should_skip_migration("-- Note: use CREATE INDEX CONCURRENTLY in production")
        False

    Edge case - String literals (NOT handled, but rare in migrations):
        The pattern may still match CONCURRENTLY in string literals like:
        "SELECT 'CREATE INDEX CONCURRENTLY' AS example;"
        This is acceptable because:
        1. Migrations rarely contain DDL syntax in string literals
        2. Handling escape sequences adds significant complexity
        3. A false positive (skipping a safe migration) is harmless in tests
    """
    # Strip comments before matching to avoid false positives
    sql_without_comments: str = _strip_sql_comments(sql)
    return bool(CONCURRENT_DDL_PATTERN.search(sql_without_comments))


# =============================================================================
# PostgreSQL Configuration
# =============================================================================


def _extract_password(parsed: ParseResult, db_url_var: str) -> str | None:
    """Extract password from a parsed URL, warning on empty-password DSNs.

    Peer/trust-auth DSNs (e.g., ``postgresql://user:@host/db``) have an
    explicitly empty password which is falsy. This helper logs a warning
    so the operator knows why ``is_configured`` returns ``False``.
    """
    # urlparse ParseResult — password is str | None
    if parsed.password:
        return unquote(parsed.password)
    # Distinguish "no password at all" from "explicitly empty password"
    netloc: str = parsed.netloc
    if ":@" in netloc:
        logger.warning(
            "%s contains an explicitly empty password (peer/trust auth DSN). "
            "is_configured will return False; tests will be skipped. "
            "Set POSTGRES_PASSWORD or include a non-empty password in the DSN.",
            db_url_var,
        )
    return None


@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL connections.

    This dataclass encapsulates all PostgreSQL connection parameters and provides
    helper methods for building DSNs and checking configuration validity.

    Attributes:
        host: PostgreSQL server hostname.
        port: PostgreSQL server port.
        database: Database name.
        user: Database username.
        password: Database password (None if not configured).

    Example:
        >>> config = PostgresConfig.from_env()
        >>> if config.is_configured:
        ...     dsn = config.build_dsn()
        ...     pool = await asyncpg.create_pool(dsn)
    """

    host: str | None
    port: int
    database: str
    user: str
    password: str | None

    @classmethod
    def from_env(
        cls,
        *,
        db_url_var: str = "OMNIBASE_INFRA_DB_URL",
    ) -> PostgresConfig:
        """Create PostgresConfig from OMNIBASE_INFRA_DB_URL.

        Requires a full DSN in ``OMNIBASE_INFRA_DB_URL``. No fallback
        to individual ``POSTGRES_*`` env vars — if the DSN is missing
        or malformed, ``is_configured`` returns False and tests skip.

        Args:
            db_url_var: Environment variable holding a full PostgreSQL DSN.

        Returns:
            PostgresConfig instance. Check ``is_configured`` before use.
        """
        db_url: str | None = os.getenv(db_url_var)
        if db_url is not None:
            db_url = db_url.strip() or None

        if not db_url:
            logger.warning(
                "%s is not set. All integration tests requiring PostgreSQL "
                "will be skipped. Set it to a full DSN, e.g.: "
                "postgresql://user:pass@host:port/dbname",
                db_url_var,
            )
            return cls(
                host=None,
                port=DEFAULT_POSTGRES_PORT,
                database="",
                user="postgres",
                password=None,
            )

        parsed = urlparse(db_url)

        if parsed.scheme not in ("postgresql", "postgres"):
            logger.warning(
                "%s has invalid scheme '%s' (expected 'postgresql' or 'postgres').",
                db_url_var,
                parsed.scheme,
            )
            return cls(
                host=None,
                port=DEFAULT_POSTGRES_PORT,
                database="",
                user="postgres",
                password=None,
            )

        try:
            port = parsed.port or DEFAULT_POSTGRES_PORT
        except ValueError:
            logger.warning("%s contains a non-numeric port.", db_url_var)
            return cls(
                host=None,
                port=DEFAULT_POSTGRES_PORT,
                database="",
                user="postgres",
                password=None,
            )

        database = unquote((parsed.path or "").lstrip("/"))
        if not database or "/" in database:
            logger.warning(
                "%s has invalid database name (parsed: %r). DSN must end with /DBNAME.",
                db_url_var,
                parsed.path,
            )
            return cls(
                host=parsed.hostname,
                port=port,
                database="",
                user="postgres",
                password=None,
            )

        return cls(
            host=parsed.hostname or None,
            port=port,
            database=database,
            user=unquote(parsed.username) if parsed.username else "postgres",
            password=_extract_password(parsed, db_url_var),
        )

    @property
    def is_configured(self) -> bool:
        """Check if the configuration is complete for database connections.

        Returns:
            True if host, password, and database are all set, False otherwise.
        """
        return (
            self.host is not None and self.password is not None and bool(self.database)
        )

    def build_dsn(self) -> str:
        """Build PostgreSQL DSN from configuration.

        Credentials (user and password) are URL-encoded using quote_plus() to
        handle special characters like @, :, /, %, etc. that would otherwise
        break the DSN format.

        Returns:
            PostgreSQL connection string in standard format.

        Raises:
            ProtocolConfigurationError: If host or password is not configured.
                Includes correlation ID, transport type, and remediation hints.

        Example:
            >>> config = PostgresConfig(host="localhost", port=5432,
            ...     database="test", user="postgres", password="secret")
            >>> config.build_dsn()
            'postgresql://postgres:secret@localhost:5432/test'
            >>> config = PostgresConfig(host="localhost", port=5432,
            ...     database="test", user="user@domain", password="p@ss:word#123")
            >>> config.build_dsn()
            'postgresql://user%40domain:p%40ss%3Aword%23123@localhost:5432/test'
            >>> config = PostgresConfig(host="localhost", port=5432,
            ...     database="test", user="postgres", password="p@ss/word")
            >>> config.build_dsn()
            'postgresql://postgres:p%40ss%2Fword@localhost:5432/test'
        """
        if not self.is_configured:
            missing: list[str] = []
            if self.host is None:
                missing.append("host")
            if self.password is None:
                missing.append("password")
            if not self.database:
                missing.append("database")

            # Create error context with correlation ID for tracing
            correlation_id = uuid4()
            context = ModelInfraErrorContext.with_correlation(
                correlation_id=correlation_id,
                transport_type=EnumInfraTransportType.DATABASE,
                operation="build_dsn",
                target_name=self.database or "(unset)",
            )
            raise ProtocolConfigurationError(
                f"PostgreSQL configuration incomplete. Missing: {', '.join(missing)}. "
                "Hint: Set OMNIBASE_INFRA_DB_URL to a full PostgreSQL DSN "
                "(e.g., postgresql://user:pass@host:5432/omnibase_infra).",
                context=context,
            )

        # URL-encode credentials to handle special characters (@, :, /, %, etc.)
        # Using quote_plus for robust encoding of all special characters in credentials
        # Assert to help mypy understand is_configured ensures password is not None
        assert self.password is not None  # Verified by is_configured check above
        encoded_user: str = quote_plus(self.user, safe="")
        encoded_password: str = quote_plus(self.password, safe="")

        return (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


def build_postgres_dsn(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str | None,
) -> str:
    """Build PostgreSQL DSN from individual components.

    This is a standalone function for cases where a full PostgresConfig
    is not needed.

    Credentials (user and password) are URL-encoded using quote_plus() to handle
    special characters like @, :, /, %, etc. that would otherwise break the DSN
    format.

    Args:
        host: PostgreSQL server hostname (must not be empty).
        port: PostgreSQL server port.
        database: Database name.
        user: Database username.
        password: Database password. Empty strings are normalized to None.

    Returns:
        PostgreSQL connection string in standard format.

    Raises:
        ProtocolConfigurationError: If host is empty, or if password is None or empty
            (after normalization). Includes correlation ID, transport type, and
            remediation hints.

    Example:
        >>> build_postgres_dsn("localhost", 5432, "test", "postgres", "secret")
        'postgresql://postgres:secret@localhost:5432/test'
        >>> build_postgres_dsn("localhost", 5432, "test", "user@domain", "p@ss:word#123")
        'postgresql://user%40domain:p%40ss%3Aword%23123@localhost:5432/test'
        >>> build_postgres_dsn("localhost", 5432, "test", "postgres", "p@ss/word")
        'postgresql://postgres:p%40ss%2Fword@localhost:5432/test'
    """
    # Validate host is not empty or whitespace-only
    # This prevents malformed DSN like "postgresql://user:pass@:5432/db"
    if not host or not host.strip():
        # Create error context with correlation ID for tracing
        correlation_id = uuid4()
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="build_dsn",
            target_name=database,
        )
        raise ProtocolConfigurationError(
            "PostgreSQL host is required. Empty or whitespace-only host "
            "is not supported to prevent malformed DSN construction. "
            "Hint: Provide a valid hostname or IP address.",
            context=context,
        )

    # Normalize empty or whitespace-only password to None
    # This prevents malformed DSN like "postgresql://user:@host:5432/db"
    normalized_password: str | None = password
    if normalized_password is not None and not normalized_password.strip():
        normalized_password = None

    if normalized_password is None:
        # Create error context with correlation ID for tracing
        correlation_id = uuid4()
        context = ModelInfraErrorContext.with_correlation(
            correlation_id=correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation="build_dsn",
            target_name=database,
        )
        raise ProtocolConfigurationError(
            "PostgreSQL password is required. Empty or whitespace-only passwords "
            "are not supported to prevent malformed DSN construction. "
            "Hint: Ensure POSTGRES_PASSWORD environment variable is set.",
            context=context,
        )

    # URL-encode credentials to handle special characters (@, :, /, %, etc.)
    # Using quote_plus for robust encoding of all special characters in credentials
    encoded_user: str = quote_plus(user, safe="")
    encoded_password: str = quote_plus(normalized_password, safe="")

    return f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/{database}"


def check_postgres_reachable(
    config: PostgresConfig,
    timeout: float = 5.0,
) -> bool:
    """Check if PostgreSQL server is reachable via TCP connection.

    This function verifies actual network connectivity to the PostgreSQL server,
    not just whether environment variables are set. This prevents tests from
    failing with connection errors when the database is unreachable (e.g., when
    running outside the Docker network where hostname resolution may fail).

    Args:
        config: PostgreSQL configuration with host and port.
        timeout: Connection timeout in seconds.

    Returns:
        True if PostgreSQL is reachable, False otherwise.

    Example:
        >>> config = PostgresConfig.from_env()
        >>> if check_postgres_reachable(config):
        ...     # Safe to attempt connection
        ...     pass
    """
    if not config.is_configured:
        return False

    # Host should never be None here due to is_configured check
    host: str = config.host or ""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result: int = sock.connect_ex((host, config.port))
            return result == 0
    except (OSError, TimeoutError, socket.gaierror):
        return False


def check_postgres_reachable_simple(
    host: str,
    port: int,
    timeout: float = 5.0,
) -> bool:
    """Check if PostgreSQL server is reachable via TCP connection (simple version).

    Standalone function for cases where a full PostgresConfig is not needed.

    Args:
        host: PostgreSQL server hostname.
        port: PostgreSQL server port.
        timeout: Connection timeout in seconds.

    Returns:
        True if PostgreSQL is reachable, False otherwise.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result: int = sock.connect_ex((host, port))
            return result == 0
    except (OSError, TimeoutError, socket.gaierror):
        return False


__all__ = [
    # Configuration
    "PostgresConfig",
    # DSN building
    "build_postgres_dsn",
    # Reachability checks
    "check_postgres_reachable",
    "check_postgres_reachable_simple",
    # Migration skip patterns
    "CONCURRENT_DDL_PATTERN",
    "should_skip_migration",
]
