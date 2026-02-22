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
        monkeypatch.setenv("POSTGRES_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("POSTGRES_POOL_MAX_SIZE", "20")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

        config = ConfigSessionStorage()

        assert config.pool_min_size == 3

    def test_pool_max_size_resolved_from_canonical_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """POSTGRES_POOL_MAX_SIZE (canonical shared key) resolves to pool_max_size."""
        monkeypatch.setenv("POSTGRES_POOL_MIN_SIZE", "3")
        monkeypatch.setenv("POSTGRES_POOL_MAX_SIZE", "8")
        monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")

        config = ConfigSessionStorage()

        assert config.pool_max_size == 8

    def test_pool_sizes_resolved_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both POSTGRES_POOL_MIN_SIZE and POSTGRES_POOL_MAX_SIZE resolve correctly."""
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

    def test_direct_construction_uses_field_name_fallback(self) -> None:
        """Direct kwarg construction bypasses env var resolution and uses the value directly."""
        config = ConfigSessionStorage(
            pool_min_size=5,
            pool_max_size=15,
            postgres_password="testpass",  # type: ignore[arg-type]  # noqa: S106
        )

        assert config.pool_min_size == 5
        assert config.pool_max_size == 15

    def test_construction_fails_without_postgres_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that missing POSTGRES_PASSWORD raises ValidationError."""
        monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
        with pytest.raises(ValidationError):
            ConfigSessionStorage()
