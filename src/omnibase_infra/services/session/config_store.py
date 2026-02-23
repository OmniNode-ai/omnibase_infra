"""Configuration for session snapshot storage.

Reads standard POSTGRES_* environment variables sourced from ~/.omnibase/.env
at shell startup. No env_prefix and no env_file — values come entirely from
the shell environment, consistent with the zero-repo-env policy (OMN-2287).

Note: This module intentionally uses individual POSTGRES_* env vars rather
than a single DSN. The session storage may target a different database than
the main OMNIBASE_INFRA_DB_URL. Migration to DSN-based configuration is
tracked separately from the OMN-2065 DB split.

Moved from omniclaude as part of OMN-1526 architectural cleanup.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigSessionStorage(BaseSettings):
    """Configuration for session snapshot PostgreSQL storage.

    Reads standard POSTGRES_* environment variables directly from the shell
    environment (no prefix). The env_file is explicitly disabled so that no
    repository-local .env file is silently discovered, in compliance with the
    zero-repo-env policy. Source ~/.omnibase/.env in your shell profile to
    supply the required values.

    Note: Using an empty prefix means any POSTGRES_* variables already set in the
    environment (e.g. by a test runner or CI matrix) will be used here. This is an
    intentional trade-off of the zero-repo-env policy; ensure POSTGRES_* variables
    in the shell match the intended session storage target.

    Warning: Ambient environment risk — callers (including tests) that construct
    ``ConfigSessionStorage()`` without explicitly controlling the process environment
    may silently pick up POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_DATABASE,
    or POSTGRES_PASSWORD from an unrelated source (e.g. a CI matrix job, a sourced
    ~/.omnibase/.env, or a parent test fixture). Always isolate the environment before
    constructing this class in tests — use ``monkeypatch.delenv`` for every POSTGRES_*
    field listed in the environment variable mapping above, and ``monkeypatch.setenv``
    for the values under test. See ``tests/unit/services/session/test_config_store.py``
    for examples of correct isolation.

    Environment variable mapping:

    - ``postgres_host``      ← ``POSTGRES_HOST``
    - ``postgres_port``      ← ``POSTGRES_PORT``
    - ``postgres_database``  ← ``POSTGRES_DATABASE``
    - ``postgres_user``      ← ``POSTGRES_USER``
    - ``postgres_password``  ← ``POSTGRES_PASSWORD``
    - ``pool_min_size``      ← ``POSTGRES_POOL_MIN_SIZE`` (primary) or ``pool_min_size`` (fallback)
    - ``pool_max_size``      ← ``POSTGRES_POOL_MAX_SIZE`` (primary) or ``pool_max_size`` (fallback)
    - ``query_timeout_seconds`` ← ``QUERY_TIMEOUT_SECONDS`` (seconds; intentionally
      distinct from ``POSTGRES_TIMEOUT_MS`` in ``transport_config_map.py``, which is
      the shared platform key expressed in milliseconds — different unit and scope)

    The pool fields use ``AliasChoices`` so that both the canonical shared key
    (e.g. ``POSTGRES_POOL_MIN_SIZE``, as declared in
    ``config/shared_key_registry.yaml``) and the bare field name work.

    Example: export POSTGRES_HOST=db.example.com
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=None,  # Disable .env file discovery — reads from process env only (sourced via ~/.omnibase/.env).
        # A .env file in the CWD would NOT be read even if present.
        # case_sensitive=False: POSTGRES_HOST, postgres_host, and Postgres_Host
        # all resolve the same field. This is intentional — shell env conventions vary.
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,  # Required for AliasChoices on pool fields to pass mypy [pydantic-alias]
    )

    # PostgreSQL connection fields
    #
    # These fields do NOT need AliasChoices because pydantic-settings resolves
    # them automatically when env_prefix="" and case_sensitive=False: the field
    # name "postgres_host" matches the env var "POSTGRES_HOST" (uppercased by
    # pydantic-settings), and similarly for postgres_port, postgres_database,
    # postgres_user, and postgres_password.
    #
    # The pool fields below (pool_min_size / pool_max_size) are the exception:
    # their field names do NOT start with "postgres_", so pydantic-settings
    # would look for "POOL_MIN_SIZE" / "POOL_MAX_SIZE" — which are not the
    # canonical shared keys.  AliasChoices maps the canonical names
    # (POSTGRES_POOL_MIN_SIZE, POSTGRES_POOL_MAX_SIZE, per
    # config/shared_key_registry.yaml) first, then falls back to the bare
    # field names so that direct construction (e.g. in tests) still works.
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL host",
    )
    postgres_port: int = Field(
        default=5436,
        ge=1,
        le=65535,
        description="PostgreSQL port",
    )
    postgres_database: str = Field(
        default="omnibase_infra",
        description="PostgreSQL database name",
    )
    postgres_user: str = Field(
        default="postgres",
        description="PostgreSQL user",
    )
    postgres_password: SecretStr = Field(
        ...,  # Required
        description="PostgreSQL password - set via POSTGRES_PASSWORD env var",
    )

    # Connection pool
    # AliasChoices maps POSTGRES_POOL_MIN_SIZE (canonical shared key per
    # config/shared_key_registry.yaml) first, then the bare field name as
    # a fallback so that direct construction (e.g. in tests) still works.
    pool_min_size: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Minimum connection pool size (env: POSTGRES_POOL_MIN_SIZE)",
        validation_alias=AliasChoices("POSTGRES_POOL_MIN_SIZE", "pool_min_size"),
    )
    pool_max_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum connection pool size (env: POSTGRES_POOL_MAX_SIZE)",
        validation_alias=AliasChoices("POSTGRES_POOL_MAX_SIZE", "pool_max_size"),
    )

    # Query timeouts
    # AliasChoices maps QUERY_TIMEOUT_SECONDS (existing primary env var) first,
    # then POSTGRES_QUERY_TIMEOUT_SECONDS as a forward-compat alias in case the key
    # gains a POSTGRES_ prefix in a future shared_key_registry.yaml rename.
    # The primary alias QUERY_TIMEOUT_SECONDS must remain first so existing env
    # configs continue to work without any changes.
    query_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Query timeout in seconds (env: QUERY_TIMEOUT_SECONDS or POSTGRES_QUERY_TIMEOUT_SECONDS)",
        validation_alias=AliasChoices(
            "QUERY_TIMEOUT_SECONDS",
            # POSTGRES_QUERY_TIMEOUT_SECONDS is a provisional forward-compat alias only.
            # It is NOT seeded in Infisical (/shared/db/ holds QUERY_TIMEOUT_SECONDS, not
            # this prefixed variant) and does NOT appear in config/shared_key_registry.yaml
            # or runtime/config_discovery/transport_config_map.py.  If QUERY_TIMEOUT_SECONDS
            # is ever renamed in the registry, this alias should be promoted to primary and
            # the old name retired.
            "POSTGRES_QUERY_TIMEOUT_SECONDS",  # forward-compat alias — not in registry/Infisical  # TODO(OMN-2065): promote POSTGRES_QUERY_TIMEOUT_SECONDS to registry or remove alias
        ),
    )

    @model_validator(mode="after")
    def validate_pool_sizes(self) -> ConfigSessionStorage:
        """Validate that pool_min_size <= pool_max_size.

        Returns:
            Self if validation passes.

        Raises:
            ValueError: If pool_min_size > pool_max_size.
        """
        if self.pool_min_size > self.pool_max_size:
            raise ValueError(
                f"pool_min_size ({self.pool_min_size}) must be <= "
                f"pool_max_size ({self.pool_max_size})"
            )
        return self

    @staticmethod
    def _format_host(host: str) -> str:
        """Format host for DSN, wrapping IPv6 addresses in brackets.

        Uses ``ipaddress.IPv6Address`` for definitive detection rather than
        a ``":" in host`` heuristic, which would false-positive on strings
        like ``host:port`` accidentally passed as a bare hostname.

        Args:
            host: Hostname or IP address.

        Returns:
            Host string suitable for embedding in a DSN.
        """
        try:
            ipaddress.IPv6Address(host)
        except ValueError:
            return host
        return f"[{host}]"

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN from components.

        Credentials, database name, and host are URL-encoded or formatted
        to handle special characters that would otherwise break the DSN.

        Returns:
            PostgreSQL connection string.
        """
        encoded_user = quote_plus(self.postgres_user, safe="")
        encoded_password = quote_plus(
            self.postgres_password.get_secret_value(), safe=""
        )
        encoded_database = quote_plus(self.postgres_database, safe="")
        host = self._format_host(self.postgres_host)
        return (
            f"postgresql://{encoded_user}:{encoded_password}"
            f"@{host}:{self.postgres_port}"
            f"/{encoded_database}"
        )

    @property
    def dsn_async(self) -> str:
        """Build async PostgreSQL DSN for asyncpg.

        Credentials, database name, and host are URL-encoded or formatted
        to handle special characters that would otherwise break the DSN.

        Returns:
            PostgreSQL connection string with postgresql+asyncpg scheme.
        """
        encoded_user = quote_plus(self.postgres_user, safe="")
        encoded_password = quote_plus(
            self.postgres_password.get_secret_value(), safe=""
        )
        encoded_database = quote_plus(self.postgres_database, safe="")
        host = self._format_host(self.postgres_host)
        return (
            f"postgresql+asyncpg://{encoded_user}:{encoded_password}"
            f"@{host}:{self.postgres_port}"
            f"/{encoded_database}"
        )

    @property
    def dsn_safe(self) -> str:
        """Build PostgreSQL DSN with password masked (safe for logging).

        Returns:
            PostgreSQL connection string with password replaced by ***.
        """
        encoded_user = quote_plus(self.postgres_user, safe="")
        encoded_database = quote_plus(self.postgres_database, safe="")
        host = self._format_host(self.postgres_host)
        return (
            f"postgresql://{encoded_user}:***"
            f"@{host}:{self.postgres_port}"
            f"/{encoded_database}"
        )

    def __repr__(self) -> str:
        """Safe string representation that doesn't expose credentials.

        Returns:
            String representation with masked password.
        """
        return f"ConfigSessionStorage(dsn={self.dsn_safe!r})"
