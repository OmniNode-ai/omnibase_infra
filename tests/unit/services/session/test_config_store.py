# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ConfigSessionStorage.

Covers env var resolution via AliasChoices for the pool size fields, which use
non-standard names (POSTGRES_POOL_MIN_SIZE / POSTGRES_POOL_MAX_SIZE) that do
not match pydantic-settings' automatic bare-name mapping.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.services.session.config_store import ConfigSessionStorage


@pytest.mark.unit
class TestConfigSessionStorageAliasChoices:
    """Tests verifying AliasChoices env var resolution for pool size fields."""

    def test_pool_min_size_resolved_from_canonical_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSTGRES_POOL_MIN_SIZE (canonical shared key) resolves to pool_min_size."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("POSTGRES_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("POSTGRES_POOL_MAX_SIZE", "20")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

        config = ConfigSessionStorage()

        assert config.pool_min_size == 3

    def test_pool_max_size_resolved_from_canonical_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSTGRES_POOL_MAX_SIZE (canonical shared key) resolves to pool_max_size."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("POSTGRES_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("POSTGRES_POOL_MAX_SIZE", "8")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

        config = ConfigSessionStorage()

        assert config.pool_max_size == 8

    def test_pool_sizes_resolved_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both POSTGRES_POOL_MIN_SIZE and POSTGRES_POOL_MAX_SIZE resolve correctly."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("POSTGRES_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("POSTGRES_POOL_MAX_SIZE", "8")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

        config = ConfigSessionStorage()

        assert config.pool_min_size == 3
        assert config.pool_max_size == 8

    def test_pool_sizes_fallback_to_defaults_when_env_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pool sizes fall back to defaults when env vars are not set."""
        monkeypatch.delenv("POSTGRES_POOL_MIN_SIZE", raising=False)
        monkeypatch.delenv("POSTGRES_POOL_MAX_SIZE", raising=False)
        monkeypatch.delenv("pool_min_size", raising=False)
        monkeypatch.delenv("pool_max_size", raising=False)
        monkeypatch.delenv("QUERY_TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
        # Clear ambient connection vars so CI environments don't silently pollute the
        # constructed config and cause misleading failures if the test is extended.
        monkeypatch.delenv("POSTGRES_HOST", raising=False)
        monkeypatch.delenv("POSTGRES_PORT", raising=False)
        monkeypatch.delenv("POSTGRES_USER", raising=False)
        monkeypatch.delenv("POSTGRES_DATABASE", raising=False)

        config = ConfigSessionStorage()

        assert config.pool_min_size == 2
        assert config.pool_max_size == 10

    def test_direct_kwarg_construction_with_populate_by_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Direct Python kwargs use the field name when populate_by_name=True.

        This test verifies that ConfigSessionStorage accepts ``pool_min_size`` and
        ``pool_max_size`` as direct constructor kwargs.  It does NOT test env var
        resolution — AliasChoices aliases (POSTGRES_POOL_MIN_SIZE, etc.) are
        irrelevant here because direct kwargs bypass env var lookup entirely.
        The ``populate_by_name=True`` setting is what allows the Python field name
        to be used instead of the first AliasChoices alias.
        """
        # Clear ambient POSTGRES_* env vars so CI environments with these set
        # do not silently influence AliasChoices resolution and make the assertion
        # pass for the wrong reason (e.g. ambient POSTGRES_POOL_MIN_SIZE=5).
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("QUERY_TIMEOUT_SECONDS", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")  # required field
        config = ConfigSessionStorage(
            pool_min_size=5,
            pool_max_size=15,
            postgres_password="testpass",  # type: ignore[arg-type]
        )

        assert config.pool_min_size == 5
        assert config.pool_max_size == 15

    def test_construction_fails_without_postgres_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that missing POSTGRES_PASSWORD raises ValidationError.

        Note: ConfigSessionStorage uses env_prefix="" and case_sensitive=False,
        which means ANY ambient POSTGRES_* variable already present in the process
        environment (e.g. from a CI matrix job or a sourced ~/.omnibase/.env) will
        be silently consumed. All POSTGRES_* fields are cleared here to ensure the
        test only exercises the missing-password code path and is not accidentally
        satisfied by a password injected from an unrelated env var.
        """
        # Clear all POSTGRES_* vars that ConfigSessionStorage reads so that no
        # ambient value from the CI environment or sourced .env silently satisfies
        # the required postgres_password field (or changes host/port/user/database
        # in ways that mask the failure being tested here).
        for key in (
            "POSTGRES_PASSWORD",
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(ValidationError):
            ConfigSessionStorage()

    def test_pool_min_size_greater_than_max_raises_validation_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that pool_min_size > pool_max_size raises ValidationError."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
        with pytest.raises(ValidationError):
            ConfigSessionStorage(pool_min_size=10, pool_max_size=5)

    def test_query_timeout_resolved_from_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify QUERY_TIMEOUT_SECONDS env var resolves to query_timeout_seconds."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
        monkeypatch.setenv("QUERY_TIMEOUT_SECONDS", "60")
        config = ConfigSessionStorage()
        assert config.query_timeout_seconds == 60

    def test_dsn_safe_masks_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify dsn_safe masks the password but retains host, port, and database."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
        monkeypatch.setenv("POSTGRES_PORT", "5436")
        monkeypatch.setenv("POSTGRES_DATABASE", "omnibase_infra")
        monkeypatch.setenv("POSTGRES_PASSWORD", "supersecretpassword")

        config = ConfigSessionStorage()
        safe = config.dsn_safe

        assert "supersecretpassword" not in safe
        assert "***" in safe
        assert "db.example.com" in safe
        assert "5436" in safe
        assert "omnibase_infra" in safe

    def test_dsn_returns_postgresql_url_with_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify dsn returns a postgresql:// URL with password included."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
        monkeypatch.setenv("POSTGRES_PORT", "5436")
        monkeypatch.setenv("POSTGRES_DATABASE", "omnibase_infra")
        monkeypatch.setenv("POSTGRES_PASSWORD", "mypassword")

        config = ConfigSessionStorage()
        dsn = config.dsn

        assert dsn.startswith("postgresql://")
        assert "mypassword" in dsn
        assert "db.example.com" in dsn
        assert "5436" in dsn
        assert "omnibase_infra" in dsn
        assert "***" not in dsn

    def test_dsn_async_returns_asyncpg_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify dsn_async returns a postgresql+asyncpg:// URL with password included."""
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_HOST", "db.example.com")
        monkeypatch.setenv("POSTGRES_PORT", "5436")
        monkeypatch.setenv("POSTGRES_DATABASE", "omnibase_infra")
        monkeypatch.setenv("POSTGRES_PASSWORD", "mypassword")

        config = ConfigSessionStorage()
        dsn_async = config.dsn_async

        assert dsn_async.startswith("postgresql+asyncpg://")
        assert "mypassword" in dsn_async
        assert "db.example.com" in dsn_async
        assert "5436" in dsn_async
        assert "omnibase_infra" in dsn_async
        assert "***" not in dsn_async

    def test_dsn_url_encodes_special_characters_in_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify dsn and dsn_async URL-encode special characters in the password.

        A password containing characters like @, /, ?, #, or spaces would
        break DSN parsing if not encoded. The implementation uses quote_plus()
        so these are percent-encoded and the resulting DSN is safe to pass to
        asyncpg or SQLAlchemy without manual escaping.
        """
        for key in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_DATABASE",
            "POSTGRES_POOL_MIN_SIZE",
            "POSTGRES_POOL_MAX_SIZE",
            "QUERY_TIMEOUT_SECONDS",
        ):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        monkeypatch.setenv("POSTGRES_HOST", "localhost")
        monkeypatch.setenv("POSTGRES_DATABASE", "mydb")
        monkeypatch.setenv("POSTGRES_PASSWORD", "p@ss/word?foo#bar")

        config = ConfigSessionStorage()

        # The raw password must NOT appear unencoded in either DSN — an
        # unencoded '@' or '/' would cause the driver to misparse the URL.
        assert "p@ss/word?foo#bar" not in config.dsn
        assert "p@ss/word?foo#bar" not in config.dsn_async

        # Encoded form: '@' → '%40', '/' → '%2F', '?' → '%3F', '#' → '%23'
        assert "%40" in config.dsn
        assert "%40" in config.dsn_async
