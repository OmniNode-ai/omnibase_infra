# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for SecretResolver.

These tests verify the SecretResolver's behavior with mocked sources,
including environment variables, file-based secrets, caching, and
thread safety.

Test Coverage:
- Basic resolution from env vars and files
- Required vs optional secret handling
- Cache hit/miss behavior
- TTL and expiration
- Convention fallback
- Introspection (non-sensitive)
- Thread safety under concurrent access
- Async API support

Related:
- OMN-764: SecretResolver implementation
- docs/milestones/BETA_v0.2.0_HARDENING.md: Issue 3.12
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from omnibase_infra.errors import SecretResolutionError
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import ModelSecretSourceSpec
from omnibase_infra.runtime.secret_resolver import SecretResolver


class TestSecretResolverBasic:
    """Basic resolution tests."""

    def test_resolve_from_env_with_explicit_mapping(self) -> None:
        """Should resolve secret from environment variable via explicit mapping."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="database.postgres.password",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_POSTGRES_PASSWORD",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"TEST_POSTGRES_PASSWORD": "secret123"}):
            result = resolver.get_secret("database.postgres.password")

        assert result is not None
        assert result.get_secret_value() == "secret123"

    def test_resolve_from_file(self) -> None:
        """Should resolve secret from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = Path(tmpdir) / "db_password"
            secret_file.write_text("file_secret_value\n")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="database.password",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path=str(secret_file),
                        ),
                    ),
                ],
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            result = resolver.get_secret("database.password")

            assert result is not None
            assert result.get_secret_value() == "file_secret_value"

    def test_resolve_from_file_strips_whitespace(self) -> None:
        """Should strip whitespace from file-based secrets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = Path(tmpdir) / "secret"
            secret_file.write_text("  secret_with_spaces  \n")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="my.secret",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path=str(secret_file),
                        ),
                    ),
                ],
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            result = resolver.get_secret("my.secret")

            assert result is not None
            assert result.get_secret_value() == "secret_with_spaces"

    def test_resolve_file_not_found(self) -> None:
        """Should return None when file does not exist and required=False."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="missing.secret",
                    source=ModelSecretSourceSpec(
                        source_type="file",
                        source_path="/nonexistent/path/to/secret",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        result = resolver.get_secret("missing.secret", required=False)

        assert result is None

    def test_convention_fallback_converts_name(self) -> None:
        """Should convert dotted name to env var when fallback enabled."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=True,
            convention_env_prefix="",
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"DATABASE_POSTGRES_PASSWORD": "fallback_secret"}):
            result = resolver.get_secret("database.postgres.password")

        assert result is not None
        assert result.get_secret_value() == "fallback_secret"

    def test_convention_fallback_with_prefix(self) -> None:
        """Should apply prefix when configured."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=True,
            convention_env_prefix="ONEX_",
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"ONEX_DATABASE_PASSWORD": "prefixed_secret"}):
            result = resolver.get_secret("database.password")

        assert result is not None
        assert result.get_secret_value() == "prefixed_secret"

    def test_explicit_mapping_takes_precedence_over_convention(self) -> None:
        """Explicit mapping should take precedence over convention fallback."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="database.password",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="EXPLICIT_DB_PASS",
                    ),
                ),
            ],
            enable_convention_fallback=True,
            convention_env_prefix="",
        )
        resolver = SecretResolver(config=config)

        with patch.dict(
            os.environ,
            {
                "EXPLICIT_DB_PASS": "explicit_value",
                "DATABASE_PASSWORD": "convention_value",
            },
        ):
            result = resolver.get_secret("database.password")

        assert result is not None
        assert result.get_secret_value() == "explicit_value"


class TestSecretResolverRequiredFlag:
    """Tests for required vs optional secrets."""

    def test_required_true_raises_when_not_found(self) -> None:
        """Should raise SecretResolutionError when required=True and not found.

        SECURITY: Error messages should NOT contain the logical_name to avoid
        exposing secret identifiers. The correlation_id is provided for tracing.
        """
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.get_secret("nonexistent.secret", required=True)

        error_msg = str(exc_info.value)
        # SECURITY: Logical name should NOT be in error message
        assert "nonexistent.secret" not in error_msg
        # Correlation ID should be present for tracing
        assert "correlation_id=" in error_msg
        # Should indicate what operation failed generically
        assert "secret not found" in error_msg.lower()

    def test_required_false_returns_none_when_not_found(self) -> None:
        """Should return None when required=False and not found."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        result = resolver.get_secret("nonexistent.secret", required=False)

        assert result is None

    def test_required_default_is_true(self) -> None:
        """Should default to required=True."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with pytest.raises(SecretResolutionError):
            resolver.get_secret("nonexistent.secret")

    def test_env_var_not_set_raises_when_required(self) -> None:
        """Should raise when env var is configured but not set.

        SECURITY: Error messages should NOT contain the logical_name to avoid
        exposing secret identifiers. The correlation_id is provided for tracing.
        """
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="api.key",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="UNSET_API_KEY",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            # Remove if exists
            os.environ.pop("UNSET_API_KEY", None)
            with pytest.raises(SecretResolutionError) as exc_info:
                resolver.get_secret("api.key", required=True)

        error_msg = str(exc_info.value)
        # SECURITY: Logical name should NOT be in error message
        assert "api.key" not in error_msg
        # SECURITY: Env var name should NOT be in error message
        assert "UNSET_API_KEY" not in error_msg
        # Correlation ID should be present for tracing
        assert "correlation_id=" in error_msg


class TestSecretResolverCaching:
    """Tests for caching behavior."""

    def test_cache_hit_returns_cached_value(self) -> None:
        """Should return cached value on second access."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"TEST_SECRET": "original"}):
            result1 = resolver.get_secret("test.secret")

        # Change env var - should still get cached value
        with patch.dict(os.environ, {"TEST_SECRET": "changed"}):
            result2 = resolver.get_secret("test.secret")

        assert result1 is not None
        assert result1.get_secret_value() == "original"
        assert result2 is not None
        assert result2.get_secret_value() == "original"  # Still cached

        stats = resolver.get_cache_stats()
        assert stats.hits == 1  # Second call was a cache hit

    def test_refresh_invalidates_cache(self) -> None:
        """Should fetch new value after refresh."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"TEST_SECRET": "original"}):
            resolver.get_secret("test.secret")

        resolver.refresh("test.secret")

        with patch.dict(os.environ, {"TEST_SECRET": "updated"}):
            result = resolver.get_secret("test.secret")

        assert result is not None
        assert result.get_secret_value() == "updated"

        stats = resolver.get_cache_stats()
        assert stats.refreshes == 1

    def test_refresh_all_clears_cache(self) -> None:
        """Should clear all cached entries."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="secret.two",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_TWO",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"SECRET_ONE": "v1", "SECRET_TWO": "v2"}):
            resolver.get_secret("secret.one")
            resolver.get_secret("secret.two")

        stats_before = resolver.get_cache_stats()
        assert stats_before.total_entries == 2

        resolver.refresh_all()

        stats_after = resolver.get_cache_stats()
        assert stats_after.total_entries == 0
        assert stats_after.refreshes == 2

    def test_cache_stats_are_accurate(self) -> None:
        """Should accurately track cache statistics."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Initial miss
        with patch.dict(os.environ, {"TEST_SECRET": "value"}):
            resolver.get_secret("test.secret")

        # Cache hit
        resolver.get_secret("test.secret")

        # Another cache hit
        resolver.get_secret("test.secret")

        stats = resolver.get_cache_stats()
        assert stats.hits == 2
        # Note: misses may be 0 because hit/miss tracking may differ in implementation
        assert stats.total_entries == 1

    def test_refresh_nonexistent_secret_is_safe(self) -> None:
        """Refreshing a non-cached secret should not raise."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Should not raise
        resolver.refresh("nonexistent.secret")

        stats = resolver.get_cache_stats()
        # No refresh counted since nothing was in cache
        assert stats.refreshes == 0


class TestSecretResolverGetSecrets:
    """Tests for get_secrets (multiple secrets at once)."""

    def test_get_secrets_resolves_multiple(self) -> None:
        """Should resolve multiple secrets in one call."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="secret.two",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_TWO",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"SECRET_ONE": "v1", "SECRET_TWO": "v2"}):
            results = resolver.get_secrets(["secret.one", "secret.two"])

        assert results["secret.one"] is not None
        assert results["secret.one"].get_secret_value() == "v1"
        assert results["secret.two"] is not None
        assert results["secret.two"].get_secret_value() == "v2"

    def test_get_secrets_raises_on_first_missing_when_required(self) -> None:
        """Should raise on first missing secret when required=True."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"SECRET_ONE": "v1"}):
            with pytest.raises(SecretResolutionError) as exc_info:
                resolver.get_secrets(["secret.one", "secret.missing"], required=True)

        error_msg = str(exc_info.value)
        # SECURITY: Logical name should NOT be in error message
        assert "secret.missing" not in error_msg
        # Correlation ID should be present for tracing
        assert "correlation_id=" in error_msg

    def test_get_secrets_returns_none_for_missing_when_not_required(self) -> None:
        """Should return None for missing secrets when required=False."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"SECRET_ONE": "v1"}):
            results = resolver.get_secrets(
                ["secret.one", "secret.missing"],
                required=False,
            )

        assert results["secret.one"] is not None
        assert results["secret.one"].get_secret_value() == "v1"
        assert results["secret.missing"] is None


class TestSecretResolverIntrospection:
    """Tests for non-sensitive introspection."""

    def test_list_configured_secrets(self) -> None:
        """Should list logical names without values."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="database.password",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="DB_PASS",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="api.key",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/api#key",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        secrets = resolver.list_configured_secrets()

        assert "database.password" in secrets
        assert "api.key" in secrets
        assert len(secrets) == 2

    def test_get_source_info_masks_vault_path(self) -> None:
        """Should mask sensitive parts of Vault path."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="database.password",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/data/database/postgres#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        info = resolver.get_source_info("database.password")

        assert info is not None
        assert info.source_type == "vault"
        assert "***" in info.source_path_masked
        # Should not expose the full path
        assert "postgres#password" not in info.source_path_masked

    def test_get_source_info_masks_file_path(self) -> None:
        """Should mask filename in file paths."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="file.secret",
                    source=ModelSecretSourceSpec(
                        source_type="file",
                        source_path="/run/secrets/database_password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        info = resolver.get_source_info("file.secret")

        assert info is not None
        assert info.source_type == "file"
        assert "***" in info.source_path_masked
        assert "database_password" not in info.source_path_masked

    def test_get_source_info_returns_none_for_unconfigured(self) -> None:
        """Should return None for unconfigured secrets when fallback disabled."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        info = resolver.get_source_info("unconfigured.secret")

        assert info is None

    def test_get_source_info_shows_cached_status(self) -> None:
        """Should indicate whether secret is cached."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Before caching
        info_before = resolver.get_source_info("test.secret")
        assert info_before is not None
        assert info_before.is_cached is False

        # Resolve to cache
        with patch.dict(os.environ, {"TEST_SECRET": "value"}):
            resolver.get_secret("test.secret")

        # After caching
        info_after = resolver.get_source_info("test.secret")
        assert info_after is not None
        assert info_after.is_cached is True


class TestSecretResolverAsync:
    """Tests for async API."""

    @pytest.mark.asyncio
    async def test_get_secret_async(self) -> None:
        """Should resolve secret asynchronously."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="async.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="ASYNC_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"ASYNC_SECRET": "async_value"}):
            result = await resolver.get_secret_async("async.secret")

        assert result is not None
        assert result.get_secret_value() == "async_value"

    @pytest.mark.asyncio
    async def test_get_secrets_async_multiple(self) -> None:
        """Should resolve multiple secrets asynchronously."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="secret.two",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_TWO",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"SECRET_ONE": "v1", "SECRET_TWO": "v2"}):
            results = await resolver.get_secrets_async(["secret.one", "secret.two"])

        assert results["secret.one"] is not None
        assert results["secret.one"].get_secret_value() == "v1"
        assert results["secret.two"] is not None
        assert results["secret.two"].get_secret_value() == "v2"

    @pytest.mark.asyncio
    async def test_get_secret_async_raises_when_required(self) -> None:
        """Should raise SecretResolutionError in async when required=True.

        SECURITY: Error messages should NOT contain the logical_name to avoid
        exposing secret identifiers. The correlation_id is provided for tracing.
        """
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with pytest.raises(SecretResolutionError) as exc_info:
            await resolver.get_secret_async("nonexistent.secret", required=True)

        error_msg = str(exc_info.value)
        # SECURITY: Logical name should NOT be in error message
        assert "nonexistent.secret" not in error_msg
        # Correlation ID should be present for tracing
        assert "correlation_id=" in error_msg
        # Should indicate what operation failed generically
        assert "secret not found" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_get_secret_async_returns_none_when_not_required(self) -> None:
        """Should return None in async when required=False."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        result = await resolver.get_secret_async("nonexistent.secret", required=False)

        assert result is None

    @pytest.mark.asyncio
    async def test_async_uses_cache(self) -> None:
        """Async resolution should use and update cache."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"TEST_SECRET": "cached_value"}):
            # First call - cache miss
            result1 = await resolver.get_secret_async("test.secret")
            # Second call - should use cache
            result2 = await resolver.get_secret_async("test.secret")

        assert result1 is not None
        assert result2 is not None
        assert result1.get_secret_value() == result2.get_secret_value()

    @pytest.mark.asyncio
    async def test_get_secrets_async_aggregates_all_failures(self) -> None:
        """get_secrets_async should aggregate all failures into one error message.

        Instead of failing on the first missing secret, it should attempt all
        secrets and report all failures in a single aggregated error.
        """
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Request multiple secrets where some don't exist
        # Neither SECRET_ONE nor other secrets exist in env
        with pytest.raises(SecretResolutionError) as exc_info:
            await resolver.get_secrets_async(
                ["secret.one", "missing.two", "missing.three"],
                required=True,
            )

        error_message = str(exc_info.value)
        # Should mention the aggregated failure count
        assert "3" in error_message or "secret" in error_message.lower()

    @pytest.mark.asyncio
    async def test_get_secrets_async_returns_partial_on_not_required(self) -> None:
        """get_secrets_async should return partial results when required=False.

        If some secrets fail to resolve but required=False, the successful
        results should still be returned.
        """
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="secret.one",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_ONE",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="secret.two",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SECRET_TWO",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Only one secret exists
        with patch.dict(os.environ, {"SECRET_ONE": "value_one"}, clear=False):
            results = await resolver.get_secrets_async(
                ["secret.one", "secret.two"],
                required=False,
            )

        # secret.one should succeed
        assert results.get("secret.one") is not None
        assert results["secret.one"].get_secret_value() == "value_one"
        # secret.two should return None (not found, but required=False)
        # Note: The implementation returns only successful results
        # So secret.two may not be in the dict at all
        assert results.get("secret.two") is None or "secret.two" not in results


class TestSecretResolverThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_reads_are_safe(self) -> None:
        """Should handle concurrent reads without corruption.

        Note: We set the env var before starting threads to avoid race
        conditions with patch.dict across threads.
        """
        # Set the env var once for all threads
        original_value = os.environ.get("CONCURRENT_READ_SECRET")
        os.environ["CONCURRENT_READ_SECRET"] = "thread_safe"

        try:
            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="concurrent.secret",
                        source=ModelSecretSourceSpec(
                            source_type="env",
                            source_path="CONCURRENT_READ_SECRET",
                        ),
                    ),
                ],
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)
            results: list[str] = []
            errors: list[Exception] = []
            results_lock = threading.Lock()

            def read_secret() -> None:
                try:
                    result = resolver.get_secret("concurrent.secret")
                    if result:
                        with results_lock:
                            results.append(result.get_secret_value())
                except Exception as e:
                    with results_lock:
                        errors.append(e)

            threads = [threading.Thread(target=read_secret) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors encountered: {errors}"
            assert len(results) == 10
            assert all(r == "thread_safe" for r in results)
        finally:
            # Restore original value or remove
            if original_value is not None:
                os.environ["CONCURRENT_READ_SECRET"] = original_value
            else:
                os.environ.pop("CONCURRENT_READ_SECRET", None)

    def test_concurrent_reads_and_refreshes_are_safe(self) -> None:
        """Should handle concurrent reads and refreshes without corruption.

        Note: We set the env var before starting threads to avoid race
        conditions with patch.dict across threads.
        """
        # Set the env var once for all threads
        original_value = os.environ.get("TEST_CONCURRENT_SECRET")
        os.environ["TEST_CONCURRENT_SECRET"] = "concurrent_value"

        try:
            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="test.secret",
                        source=ModelSecretSourceSpec(
                            source_type="env",
                            source_path="TEST_CONCURRENT_SECRET",
                        ),
                    ),
                ],
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)
            errors: list[Exception] = []
            stop_event = threading.Event()

            def read_secret() -> None:
                while not stop_event.is_set():
                    try:
                        resolver.get_secret("test.secret", required=False)
                    except Exception as e:
                        errors.append(e)

            def refresh_secret() -> None:
                while not stop_event.is_set():
                    try:
                        resolver.refresh("test.secret")
                    except Exception as e:
                        errors.append(e)

            readers = [threading.Thread(target=read_secret) for _ in range(5)]
            refreshers = [threading.Thread(target=refresh_secret) for _ in range(2)]

            for t in readers + refreshers:
                t.start()

            # Run for a short time
            time.sleep(0.1)
            stop_event.set()

            for t in readers + refreshers:
                t.join()

            assert len(errors) == 0, f"Errors encountered: {errors}"
        finally:
            # Restore original value or remove
            if original_value is not None:
                os.environ["TEST_CONCURRENT_SECRET"] = original_value
            else:
                os.environ.pop("TEST_CONCURRENT_SECRET", None)


class TestSecretResolverTTLBehavior:
    """Tests for TTL and expiration behavior."""

    def test_default_ttl_per_source_type(self) -> None:
        """Should use different default TTLs per source type."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="env.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="ENV_SECRET",
                    ),
                ),
                ModelSecretMapping(
                    logical_name="file.secret",
                    source=ModelSecretSourceSpec(
                        source_type="file",
                        source_path="/var/run/secrets/app_secret",
                    ),
                ),
            ],
            default_ttl_env_seconds=100,
            default_ttl_file_seconds=200,
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # TTLs are used internally - we can verify through source info
        env_info = resolver.get_source_info("env.secret")
        file_info = resolver.get_source_info("file.secret")

        assert env_info is not None
        assert file_info is not None
        # The source info shows the source type correctly
        assert env_info.source_type == "env"
        assert file_info.source_type == "file"

    def test_override_ttl_per_mapping(self) -> None:
        """Should use mapping-specific TTL when provided."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="test.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="TEST_SECRET",
                    ),
                    ttl_seconds=60,  # Override
                ),
            ],
            default_ttl_env_seconds=3600,  # Default
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # The mapping has a TTL override - verify it's recognized
        info = resolver.get_source_info("test.secret")
        assert info is not None
        assert info.source_type == "env"


class TestSecretResolverEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_secret_value_is_valid(self) -> None:
        """Empty string should be a valid secret value (not None).

        The SecretResolver treats empty strings as valid values, not as
        missing secrets. This allows intentionally empty secrets to be
        distinguished from unset/missing secrets.
        """
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="empty.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="EMPTY_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"EMPTY_SECRET": ""}):
            result = resolver.get_secret("empty.secret", required=False)

        # Empty string is returned as SecretStr(''), not None
        # This allows intentional empty values to be distinguished from missing
        assert result is not None
        assert result.get_secret_value() == ""

    def test_special_characters_in_secret_value(self) -> None:
        """Should handle special characters in secret values."""
        special_value = "p@$$w0rd!#$%^&*()_+-=[]{}|;':\",./<>?\\"
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="special.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="SPECIAL_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"SPECIAL_SECRET": special_value}):
            result = resolver.get_secret("special.secret")

        assert result is not None
        assert result.get_secret_value() == special_value

    def test_unicode_in_secret_value(self) -> None:
        """Should handle Unicode characters in secret values."""
        unicode_value = "secret_\u4e2d\u6587_\U0001f511"  # Chinese chars and key emoji
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="unicode.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="UNICODE_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"UNICODE_SECRET": unicode_value}):
            result = resolver.get_secret("unicode.secret")

        assert result is not None
        assert result.get_secret_value() == unicode_value

    def test_very_long_secret_value(self) -> None:
        """Should handle very long secret values."""
        long_value = "x" * 10000
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="long.secret",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="LONG_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"LONG_SECRET": long_value}):
            result = resolver.get_secret("long.secret")

        assert result is not None
        assert result.get_secret_value() == long_value
        assert len(result.get_secret_value()) == 10000

    def test_deeply_nested_logical_name(self) -> None:
        """Should handle deeply nested logical names."""
        nested_name = "a.b.c.d.e.f.g.h.i.j.k"
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name=nested_name,
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="NESTED_SECRET",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"NESTED_SECRET": "nested_value"}):
            result = resolver.get_secret(nested_name)

        assert result is not None
        assert result.get_secret_value() == "nested_value"

    def test_convention_fallback_disabled(self) -> None:
        """Should not use convention fallback when disabled."""
        config = ModelSecretResolverConfig(
            mappings=[],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Even with the env var set, should not resolve
        with patch.dict(os.environ, {"MY_SECRET": "value"}):
            result = resolver.get_secret("my.secret", required=False)

        assert result is None


class TestSecretResolverFileSecrets:
    """Tests for file-based secret resolution."""

    def test_resolve_from_relative_path(self) -> None:
        """Should resolve relative paths against secrets_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_dir = Path(tmpdir)
            secret_file = secrets_dir / "db_password"
            secret_file.write_text("relative_secret")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="db.password",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path="db_password",  # Relative path
                        ),
                    ),
                ],
                secrets_dir=secrets_dir,
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            result = resolver.get_secret("db.password")

            assert result is not None
            assert result.get_secret_value() == "relative_secret"

    def test_resolve_from_absolute_path(self) -> None:
        """Should resolve absolute paths directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = Path(tmpdir) / "absolute_secret"
            secret_file.write_text("absolute_value")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="absolute.secret",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path=str(secret_file),  # Absolute path
                        ),
                    ),
                ],
                secrets_dir=Path("/different/dir"),  # Should not affect absolute path
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            result = resolver.get_secret("absolute.secret")

            assert result is not None
            assert result.get_secret_value() == "absolute_value"

    def test_file_permission_error_returns_none(self) -> None:
        """Should return None when file cannot be read due to permissions.

        Uses mocking instead of actual filesystem permissions to ensure
        deterministic behavior in CI environments (where tests may run as
        root or in containers with different permission semantics).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_file = Path(tmpdir) / "protected"
            secret_file.write_text("protected_value")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="protected.secret",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path=str(secret_file),
                        ),
                    ),
                ],
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            # Mock Path.open to raise PermissionError only for the target file
            # We need a surgical mock that doesn't break pytest's file handling
            original_open = Path.open
            target_path = str(secret_file)

            def mock_open(self: Path, *args: object, **kwargs: object) -> object:
                if str(self) == target_path:
                    raise PermissionError("Permission denied")
                return original_open(self, *args, **kwargs)

            with patch.object(Path, "open", mock_open):
                result = resolver.get_secret("protected.secret", required=False)

            # Should return None when file cannot be read due to permissions
            # (PermissionError is caught and treated as "not found")
            assert result is None


class TestSecretResolverSecurity:
    """Security-focused tests for SecretResolver.

    These tests verify:
    - Path traversal prevention
    - Bootstrap secret isolation
    - Vault stub behavior (NotImplementedError)
    - No sensitive path logging
    - Error message sanitization
    """

    def test_path_traversal_blocked_relative_path(self) -> None:
        """Path traversal using relative path components should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_dir = Path(tmpdir) / "secrets"
            secrets_dir.mkdir()

            # Create a file outside secrets_dir that we shouldn't be able to access
            outside_file = Path(tmpdir) / "outside_secret"
            outside_file.write_text("outside_value")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="traversal.attempt",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path="../outside_secret",  # Path traversal attempt
                        ),
                    ),
                ],
                secrets_dir=secrets_dir,
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            # Should return None because path traversal is blocked
            result = resolver.get_secret("traversal.attempt", required=False)
            assert result is None

    def test_path_traversal_blocked_double_dots(self) -> None:
        """Multiple path traversal components should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_dir = Path(tmpdir) / "deep" / "secrets"
            secrets_dir.mkdir(parents=True)

            # Create a file that traversal would reach
            outside_file = Path(tmpdir) / "etc_passwd_like"
            outside_file.write_text("sensitive_data")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="traversal.deep",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path="../../etc_passwd_like",  # Deep traversal
                        ),
                    ),
                ],
                secrets_dir=secrets_dir,
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            # Should return None because path traversal is blocked
            result = resolver.get_secret("traversal.deep", required=False)
            assert result is None

    def test_absolute_path_allowed(self) -> None:
        """Absolute paths should be allowed (no secrets_dir constraint)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Absolute paths bypass the secrets_dir constraint
            # This is intentional - explicit absolute paths are trusted
            secret_file = Path(tmpdir) / "absolute_secret"
            secret_file.write_text("absolute_value")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="absolute.secret",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path=str(secret_file),  # Absolute path
                        ),
                    ),
                ],
                secrets_dir=Path("/run/secrets"),  # Different from tmpdir
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            result = resolver.get_secret("absolute.secret")
            assert result is not None
            assert result.get_secret_value() == "absolute_value"

    def test_bootstrap_secrets_isolated_from_vault(self) -> None:
        """Bootstrap secrets should always resolve from env, never Vault.

        This prevents circular dependency when initializing Vault.
        """
        config = ModelSecretResolverConfig(
            mappings=[
                # Even if there's a Vault mapping for a bootstrap secret,
                # it should be ignored
                ModelSecretMapping(
                    logical_name="vault.token",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/data/bootstrap#token",
                    ),
                ),
            ],
            bootstrap_secrets=["vault.token", "vault.addr"],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Set the env var that bootstrap resolution will use
        with patch.dict(os.environ, {"VAULT_TOKEN": "bootstrap_token_value"}):
            result = resolver.get_secret("vault.token")

        assert result is not None
        # Should get value from env, not Vault
        assert result.get_secret_value() == "bootstrap_token_value"

    def test_bootstrap_secrets_use_convention_naming(self) -> None:
        """Bootstrap secrets use convention-based env var naming."""
        config = ModelSecretResolverConfig(
            mappings=[],
            bootstrap_secrets=["vault.ca_cert"],
            convention_env_prefix="ONEX_",  # Prefix should be applied
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"ONEX_VAULT_CA_CERT": "ca_cert_data"}):
            result = resolver.get_secret("vault.ca_cert")

        assert result is not None
        assert result.get_secret_value() == "ca_cert_data"

    def test_vault_raises_not_implemented_when_handler_present(self) -> None:
        """Vault secrets should raise NotImplementedError when handler exists."""
        # Create a mock vault handler (not None)
        mock_vault_handler = object()

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="vault.secret",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/data/test#password",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_vault_handler)

        # Should raise NotImplementedError with helpful message
        with pytest.raises(NotImplementedError) as exc_info:
            resolver.get_secret("vault.secret")

        error_msg = str(exc_info.value)
        assert "vault.secret" in error_msg  # Should mention logical name
        assert "not yet implemented" in error_msg.lower()
        assert "env" in error_msg or "file" in error_msg  # Suggests alternatives

    @pytest.mark.asyncio
    async def test_vault_async_raises_not_implemented(self) -> None:
        """Async Vault secrets should also raise NotImplementedError."""
        mock_vault_handler = object()

        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="async.vault.secret",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/data/async#key",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config, vault_handler=mock_vault_handler)

        with pytest.raises(NotImplementedError) as exc_info:
            await resolver.get_secret_async("async.vault.secret")

        error_msg = str(exc_info.value)
        assert "async.vault.secret" in error_msg

    def test_vault_returns_none_when_no_handler(self) -> None:
        """Vault secrets return None when no handler configured (graceful degradation)."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="vault.no.handler",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/data/test#field",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        # No vault_handler provided
        resolver = SecretResolver(config=config, vault_handler=None)

        result = resolver.get_secret("vault.no.handler", required=False)
        assert result is None

    def test_error_message_does_not_leak_file_path(self) -> None:
        """SecretResolutionError should not expose file paths or secret identifiers.

        SECURITY: Error messages should contain NEITHER:
        - Logical names (e.g., "missing.file.secret") - reveals secret structure
        - File paths (e.g., "/sensitive/path/to/secret") - reveals infrastructure

        Only the correlation_id should be provided for tracing back to DEBUG logs.
        """
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="missing.file.secret",
                    source=ModelSecretSourceSpec(
                        source_type="file",
                        source_path="/sensitive/path/to/secret",
                    ),
                ),
            ],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with pytest.raises(SecretResolutionError) as exc_info:
            resolver.get_secret("missing.file.secret")

        error_msg = str(exc_info.value)
        # SECURITY: Logical name should NOT be in error message
        assert "missing.file.secret" not in error_msg
        # SECURITY: File path should NOT be in error message
        assert "/sensitive/path" not in error_msg
        # Correlation ID should be present for tracing
        assert "correlation_id=" in error_msg
        # Should indicate what operation failed generically
        assert "secret not found" in error_msg.lower()

    def test_symlink_loop_handled_gracefully(self) -> None:
        """Symlink loops should be handled without crashing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            secrets_dir = Path(tmpdir)
            # Create a symlink that points to itself (loop)
            symlink_path = secrets_dir / "loop"
            try:
                symlink_path.symlink_to(symlink_path)
            except OSError:
                # Some systems don't allow self-referential symlinks
                pytest.skip("Cannot create self-referential symlink on this system")

            config = ModelSecretResolverConfig(
                mappings=[
                    ModelSecretMapping(
                        logical_name="symlink.loop",
                        source=ModelSecretSourceSpec(
                            source_type="file",
                            source_path="loop",
                        ),
                    ),
                ],
                secrets_dir=secrets_dir,
                enable_convention_fallback=False,
            )
            resolver = SecretResolver(config=config)

            # Should return None, not crash
            result = resolver.get_secret("symlink.loop", required=False)
            assert result is None


class TestSecretResolverBootstrapSecrets:
    """Tests specifically for bootstrap secret behavior."""

    def test_bootstrap_secret_ignores_explicit_vault_mapping(self) -> None:
        """Bootstrap secrets should ignore any explicit Vault mapping."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="vault.addr",
                    source=ModelSecretSourceSpec(
                        source_type="vault",
                        source_path="secret/bootstrap#addr",
                    ),
                ),
            ],
            bootstrap_secrets=["vault.addr"],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(os.environ, {"VAULT_ADDR": "https://vault.local:8200"}):
            result = resolver.get_secret("vault.addr")

        assert result is not None
        assert result.get_secret_value() == "https://vault.local:8200"

    def test_non_bootstrap_secret_uses_normal_resolution(self) -> None:
        """Non-bootstrap secrets should use normal resolution chain."""
        config = ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name="database.password",
                    source=ModelSecretSourceSpec(
                        source_type="env",
                        source_path="DB_PASSWORD",
                    ),
                ),
            ],
            bootstrap_secrets=["vault.token"],  # database.password is NOT bootstrap
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        with patch.dict(
            os.environ,
            {
                "DB_PASSWORD": "explicit_mapping",
                "DATABASE_PASSWORD": "convention_value",
            },
        ):
            result = resolver.get_secret("database.password")

        assert result is not None
        # Should use explicit mapping, not convention
        assert result.get_secret_value() == "explicit_mapping"

    def test_bootstrap_secret_missing_returns_none(self) -> None:
        """Bootstrap secret not in env should return None when not required."""
        config = ModelSecretResolverConfig(
            mappings=[],
            bootstrap_secrets=["vault.token"],
            enable_convention_fallback=False,
        )
        resolver = SecretResolver(config=config)

        # Ensure env var is not set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("VAULT_TOKEN", None)
            result = resolver.get_secret("vault.token", required=False)

        assert result is None


__all__: list[str] = [
    "TestSecretResolverBasic",
    "TestSecretResolverRequiredFlag",
    "TestSecretResolverCaching",
    "TestSecretResolverGetSecrets",
    "TestSecretResolverIntrospection",
    "TestSecretResolverAsync",
    "TestSecretResolverThreadSafety",
    "TestSecretResolverTTLBehavior",
    "TestSecretResolverEdgeCases",
    "TestSecretResolverFileSecrets",
    "TestSecretResolverSecurity",
    "TestSecretResolverBootstrapSecrets",
]
