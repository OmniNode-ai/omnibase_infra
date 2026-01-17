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

from __future__ import annotations

import logging
import os
import re
import socket
from dataclasses import dataclass
from urllib.parse import quote

from tests.infrastructure_config import DEFAULT_POSTGRES_PORT, REMOTE_INFRA_HOST

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
#   - REINDEX ... CONCURRENTLY (any REINDEX variant with CONCURRENTLY)
#
# The pattern uses word boundaries (\b) to avoid partial matches.
# Note: Comment/string stripping is handled by _strip_sql_comments() before matching.
CONCURRENT_DDL_PATTERN = re.compile(
    r"\b(CREATE\s+(UNIQUE\s+)?INDEX\s+CONCURRENTLY|REINDEX\s+.*CONCURRENTLY)\b",
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
        host_var: str = "POSTGRES_HOST",
        port_var: str = "POSTGRES_PORT",
        database_var: str = "POSTGRES_DATABASE",
        user_var: str = "POSTGRES_USER",
        password_var: str = "POSTGRES_PASSWORD",  # noqa: S107 (env var name, not a password)
        default_database: str = "omninode_bridge",
        default_user: str = "postgres",
        fallback_host: str | None = None,
    ) -> PostgresConfig:
        """Create PostgresConfig from environment variables.

        Args:
            host_var: Environment variable name for host.
            port_var: Environment variable name for port.
            database_var: Environment variable name for database.
            user_var: Environment variable name for user.
            password_var: Environment variable name for password.
            default_database: Default database name if not set.
            default_user: Default username if not set.
            fallback_host: Fallback host if POSTGRES_HOST not set. If None,
                uses REMOTE_INFRA_HOST from infrastructure_config.

        Returns:
            PostgresConfig instance with values from environment.
        """
        host: str | None = os.getenv(host_var)
        if host is None and fallback_host is not None:
            host = fallback_host
        elif host is None:
            host = os.getenv("REMOTE_INFRA_HOST", REMOTE_INFRA_HOST)

        password: str | None = os.getenv(password_var)
        # Normalize empty password to None
        if password and not password.strip():
            password = None

        return cls(
            host=host,
            port=int(os.getenv(port_var, str(DEFAULT_POSTGRES_PORT))),
            database=os.getenv(database_var, default_database),
            user=os.getenv(user_var, default_user),
            password=password,
        )

    @property
    def is_configured(self) -> bool:
        """Check if the configuration is complete for database connections.

        Returns:
            True if host and password are set, False otherwise.
        """
        return self.host is not None and self.password is not None

    def build_dsn(self) -> str:
        """Build PostgreSQL DSN from configuration.

        Credentials (user and password) are URL-encoded to handle special
        characters like @, :, #, %, etc. that would otherwise break the DSN
        format.

        Returns:
            PostgreSQL connection string in standard format.

        Raises:
            ValueError: If host or password is not configured.

        Example:
            >>> config = PostgresConfig(host="localhost", port=5432,
            ...     database="test", user="postgres", password="secret")
            >>> config.build_dsn()
            'postgresql://postgres:secret@localhost:5432/test'
            >>> config = PostgresConfig(host="localhost", port=5432,
            ...     database="test", user="user@domain", password="p@ss:word#123")
            >>> config.build_dsn()
            'postgresql://user%40domain:p%40ss%3Aword%23123@localhost:5432/test'
        """
        if not self.is_configured:
            missing: list[str] = []
            if self.host is None:
                missing.append("host")
            if self.password is None:
                missing.append("password")
            raise ValueError(
                f"PostgreSQL configuration incomplete. Missing: {', '.join(missing)}"
            )

        # URL-encode credentials to handle special characters (@, :, #, %, etc.)
        # Assert to help mypy understand is_configured ensures password is not None
        assert self.password is not None  # Verified by is_configured check above
        encoded_user: str = quote(self.user, safe="")
        encoded_password: str = quote(self.password, safe="")

        return (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


def build_postgres_dsn(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
) -> str:
    """Build PostgreSQL DSN from individual components.

    This is a standalone function for cases where a full PostgresConfig
    is not needed.

    Credentials (user and password) are URL-encoded to handle special
    characters like @, :, #, %, etc. that would otherwise break the DSN
    format.

    Args:
        host: PostgreSQL server hostname.
        port: PostgreSQL server port.
        database: Database name.
        user: Database username.
        password: Database password.

    Returns:
        PostgreSQL connection string in standard format.

    Example:
        >>> build_postgres_dsn("localhost", 5432, "test", "postgres", "secret")
        'postgresql://postgres:secret@localhost:5432/test'
        >>> build_postgres_dsn("localhost", 5432, "test", "user@domain", "p@ss:word#123")
        'postgresql://user%40domain:p%40ss%3Aword%23123@localhost:5432/test'
    """
    # URL-encode credentials to handle special characters (@, :, #, %, etc.)
    encoded_user: str = quote(user, safe="")
    encoded_password: str = quote(password, safe="")

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
