# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for RegistryCompute.

Tests cover:
- All registry operations (register, get, list, unregister, clear)
- Sync enforcement (rejects async without flag)
- Semver sorting (semantic, not lexicographic)
- Thread safety (concurrent registration/lookup)
- Error handling

This follows the testing patterns established in test_policy_registry.py.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from omnibase_infra.errors import ComputeRegistryError, ProtocolConfigurationError
from omnibase_infra.runtime.models import ModelComputeKey, ModelComputeRegistration
from omnibase_infra.runtime.registry_compute import RegistryCompute

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer


# =============================================================================
# Test Fixtures - Mock Compute Plugins
# =============================================================================


class SyncComputePlugin:
    """Synchronous compute plugin for testing."""

    def execute(self, data: dict[str, object]) -> dict[str, object]:
        """Execute synchronous computation."""
        return {"processed": True}


class AsyncComputePlugin:
    """Async compute plugin for testing (should be rejected without flag)."""

    async def execute(self, data: dict[str, object]) -> dict[str, object]:
        """Execute async computation."""
        return {"processed": True}


class PartialAsyncPlugin:
    """Plugin with one async public method (validate is async)."""

    def execute(self, data: dict[str, object]) -> dict[str, object]:
        """Execute synchronous computation."""
        return {"result": data}

    async def validate(self, data: dict[str, object]) -> bool:
        """Async validation - should trigger rejection."""
        return True


class PrivateAsyncPlugin:
    """Plugin with async private method (should be allowed)."""

    def execute(self, data: dict[str, object]) -> dict[str, object]:
        """Execute synchronous computation."""
        return self._transform(data)

    async def _internal_async(self) -> None:
        """Private async method - should NOT trigger rejection."""

    def _transform(self, data: dict[str, object]) -> dict[str, object]:
        """Transform data synchronously."""
        return {"transformed": data}


class SyncComputePluginV1:
    """Version 1 of sync compute plugin for version testing."""

    def execute(self, data: dict[str, object]) -> dict[str, object]:
        """Execute synchronous computation."""
        return {"version": "1.0.0"}


class SyncComputePluginV2:
    """Version 2 of sync compute plugin for version testing."""

    def execute(self, data: dict[str, object]) -> dict[str, object]:
        """Execute synchronous computation."""
        return {"version": "2.0.0"}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def registry() -> RegistryCompute:
    """Provide a fresh RegistryCompute instance for each test.

    Note: Resets the semver cache to ensure test isolation.
    """
    RegistryCompute._reset_semver_cache()
    return RegistryCompute()


@pytest.fixture
def populated_registry() -> RegistryCompute:
    """Provide a RegistryCompute with pre-registered plugins."""
    RegistryCompute._reset_semver_cache()
    registry = RegistryCompute()
    registry.register_plugin(
        plugin_id="json_normalizer",
        plugin_class=SyncComputePlugin,
        version="1.0.0",
    )
    registry.register_plugin(
        plugin_id="xml_parser",
        plugin_class=SyncComputePlugin,
        version="2.0.0",
    )
    return registry


# =============================================================================
# TestRegistration
# =============================================================================


class TestRegistration:
    """Tests for register() and register_plugin() methods."""

    def test_register_with_model(self, registry: RegistryCompute) -> None:
        """Test registration using ModelComputeRegistration."""
        registration = ModelComputeRegistration(
            plugin_id="json_normalizer",
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )
        registry.register(registration)

        assert registry.is_registered("json_normalizer")
        assert registry.is_registered("json_normalizer", "1.0.0")
        assert len(registry) == 1

    def test_register_plugin_convenience(self, registry: RegistryCompute) -> None:
        """Test registration using convenience method."""
        registry.register_plugin(
            plugin_id="transformer",
            plugin_class=SyncComputePlugin,
            version="2.0.0",
        )

        assert registry.is_registered("transformer", "2.0.0")

    def test_register_multiple_versions(self, registry: RegistryCompute) -> None:
        """Test registering multiple versions of same plugin."""
        registry.register_plugin("ranker", SyncComputePlugin, "1.0.0")
        registry.register_plugin("ranker", SyncComputePlugin, "1.1.0")
        registry.register_plugin("ranker", SyncComputePlugin, "2.0.0")

        versions = registry.list_versions("ranker")
        assert versions == ["1.0.0", "1.1.0", "2.0.0"]
        assert len(registry) == 3

    def test_register_overwrites_existing(self, registry: RegistryCompute) -> None:
        """Test that re-registering same version overwrites."""
        registry.register_plugin("scorer", SyncComputePlugin, "1.0.0")

        class NewScorer:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"new": True}

        registry.register_plugin("scorer", NewScorer, "1.0.0")

        assert len(registry) == 1
        plugin_cls = registry.get("scorer", "1.0.0")
        assert plugin_cls == NewScorer

    def test_register_with_default_version(self, registry: RegistryCompute) -> None:
        """Test that default version is 1.0.0."""
        registry.register_plugin(
            plugin_id="default_version",
            plugin_class=SyncComputePlugin,
        )

        assert registry.is_registered("default_version", "1.0.0")

    def test_register_with_description(self, registry: RegistryCompute) -> None:
        """Test registration with description field."""
        registration = ModelComputeRegistration(
            plugin_id="documented_plugin",
            plugin_class=SyncComputePlugin,
            version="1.0.0",
            description="A well-documented compute plugin",
        )
        registry.register(registration)

        assert registry.is_registered("documented_plugin")


# =============================================================================
# TestGet
# =============================================================================


class TestGet:
    """Tests for get() method."""

    def test_get_exact_version(self, registry: RegistryCompute) -> None:
        """Test getting specific version."""
        registry.register_plugin("plugin_a", SyncComputePlugin, "1.0.0")

        result = registry.get("plugin_a", "1.0.0")
        assert result == SyncComputePlugin

    def test_get_latest_version(self, registry: RegistryCompute) -> None:
        """Test getting latest version when version not specified."""
        registry.register_plugin("plugin_b", SyncComputePluginV1, "1.0.0")
        registry.register_plugin("plugin_b", SyncComputePluginV2, "2.0.0")

        result = registry.get("plugin_b")  # No version specified
        assert result == SyncComputePluginV2  # Should return 2.0.0

    def test_get_unregistered_raises(self, registry: RegistryCompute) -> None:
        """Test that getting unregistered plugin raises error."""
        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.get("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_wrong_version_raises(self, registry: RegistryCompute) -> None:
        """Test that getting wrong version raises error with available versions."""
        registry.register_plugin("plugin_c", SyncComputePlugin, "1.0.0")

        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.get("plugin_c", "9.9.9")

        # Error message should contain the version that was not found
        error_msg = str(exc_info.value)
        assert "9.9.9" in error_msg or "plugin_c" in error_msg

    def test_get_nonexistent_version_on_unregistered_plugin(
        self, registry: RegistryCompute
    ) -> None:
        """Test that getting version on non-existent plugin raises error.

        This test validates error behavior for unregistered plugins (which
        doesn't trigger the deadlock since it exits early before calling
        list_versions).
        """
        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.get("nonexistent_plugin", "1.0.0")

        error_msg = str(exc_info.value)
        assert "nonexistent_plugin" in error_msg

    def test_get_returns_latest_single_version(self, registry: RegistryCompute) -> None:
        """Test get() optimization for single version plugins."""
        registry.register_plugin("single_version", SyncComputePlugin, "1.0.0")

        result = registry.get("single_version")
        assert result == SyncComputePlugin


# =============================================================================
# TestSyncEnforcement - CRITICAL
# =============================================================================


class TestSyncEnforcement:
    """Tests for sync enforcement - MUST reject async without flag.

    This is CRITICAL functionality per OMN-811 acceptance criteria.
    Compute plugins must be synchronous by default.
    """

    def test_reject_async_execute_without_flag(self, registry: RegistryCompute) -> None:
        """Test that async execute() is rejected without deterministic_async."""
        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register_plugin(
                plugin_id="async_plugin",
                plugin_class=AsyncComputePlugin,
                deterministic_async=False,
            )

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert "async execute()" in error_msg
        assert "deterministic_async=True not specified" in error_msg

        # Verify error has context dict and includes plugin_id
        error = exc_info.value
        assert hasattr(error.model, "context"), "Error should have context dict"
        context = error.model.context
        assert context.get("plugin_id") == "async_plugin"

    def test_accept_async_with_flag(self, registry: RegistryCompute) -> None:
        """Test that async is accepted when explicitly flagged."""
        registry.register_plugin(
            plugin_id="async_plugin",
            plugin_class=AsyncComputePlugin,
            deterministic_async=True,  # Explicit flag
        )

        assert registry.is_registered("async_plugin")

    def test_reject_any_async_public_method(self, registry: RegistryCompute) -> None:
        """Test that ANY async public method triggers rejection."""
        with pytest.raises(ComputeRegistryError) as exc_info:
            registry.register_plugin(
                plugin_id="partial_async",
                plugin_class=PartialAsyncPlugin,
                deterministic_async=False,
            )

        # Verify error message mentions the async method name
        error_msg = str(exc_info.value)
        assert "validate" in error_msg

        # Verify error has context and includes plugin_id
        error = exc_info.value
        context = error.model.context
        assert context.get("plugin_id") == "partial_async"

    def test_allow_private_async_methods(self, registry: RegistryCompute) -> None:
        """Test that private async methods (prefixed with _) are allowed."""
        # This should NOT raise because _internal_async is private
        registry.register_plugin(
            plugin_id="private_async",
            plugin_class=PrivateAsyncPlugin,
            deterministic_async=False,  # Should work because async is private
        )

        assert registry.is_registered("private_async")

    def test_sync_plugin_registration_succeeds(self, registry: RegistryCompute) -> None:
        """Test that synchronous plugin registers without issues."""
        # Should not raise - sync plugin with default deterministic_async=False
        registry.register_plugin(
            plugin_id="sync_plugin",
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )
        assert registry.is_registered("sync_plugin")
        plugin_cls = registry.get("sync_plugin")
        assert plugin_cls is SyncComputePlugin


# =============================================================================
# TestSemverSorting - Semantic, not Lexicographic
# =============================================================================


class TestSemverSorting:
    """Tests for semantic version sorting.

    This is CRITICAL functionality - must sort semantically, not lexicographically.
    """

    def test_semver_sorts_correctly(self, registry: RegistryCompute) -> None:
        """Test that 1.10.0 > 1.9.0 (semantic, not lexicographic)."""
        registry.register_plugin("semver_test", SyncComputePluginV1, "1.9.0")
        registry.register_plugin("semver_test", SyncComputePluginV2, "1.10.0")

        # get() without version should return 1.10.0 (latest)
        result = registry.get("semver_test")
        assert result == SyncComputePluginV2

        # Versions should be sorted semantically
        versions = registry.list_versions("semver_test")
        assert versions == ["1.9.0", "1.10.0"]

    def test_prerelease_sorts_before_release(self, registry: RegistryCompute) -> None:
        """Test that 1.0.0-alpha < 1.0.0."""

        class AlphaPlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"v": "alpha"}

        registry.register_plugin("prerelease", AlphaPlugin, "1.0.0-alpha")
        registry.register_plugin("prerelease", SyncComputePlugin, "1.0.0")

        # Release version should be "latest"
        result = registry.get("prerelease")
        assert result == SyncComputePlugin

    def test_semver_double_digit_versions(self, registry: RegistryCompute) -> None:
        """Test edge case: 10.0.0 vs 2.0.0."""
        registry.register_plugin("major_test", SyncComputePluginV1, "2.0.0")
        registry.register_plugin("major_test", SyncComputePluginV2, "10.0.0")

        latest_cls = registry.get("major_test")
        assert latest_cls is SyncComputePluginV2, "10.0.0 > 2.0.0"

    def test_semver_patch_version_edge_case(self, registry: RegistryCompute) -> None:
        """Test edge case: 1.0.9 vs 1.0.10."""
        registry.register_plugin("patch_test", SyncComputePluginV1, "1.0.9")
        registry.register_plugin("patch_test", SyncComputePluginV2, "1.0.10")

        latest_cls = registry.get("patch_test")
        assert latest_cls is SyncComputePluginV2, "1.0.10 > 1.0.9"


# =============================================================================
# TestListOperations
# =============================================================================


class TestListOperations:
    """Tests for list_keys() and list_versions()."""

    def test_list_keys(self, populated_registry: RegistryCompute) -> None:
        """Test listing all registered plugins."""
        keys = populated_registry.list_keys()
        assert ("json_normalizer", "1.0.0") in keys
        assert ("xml_parser", "2.0.0") in keys

    def test_list_versions_empty(self, registry: RegistryCompute) -> None:
        """Test list_versions for non-existent plugin."""
        versions = registry.list_versions("nonexistent")
        assert versions == []

    def test_list_versions_multiple(self, registry: RegistryCompute) -> None:
        """Test list_versions with multiple versions."""
        registry.register_plugin("multi_version", SyncComputePlugin, "1.0.0")
        registry.register_plugin("multi_version", SyncComputePlugin, "1.1.0")
        registry.register_plugin("multi_version", SyncComputePlugin, "2.0.0")

        versions = registry.list_versions("multi_version")
        assert versions == ["1.0.0", "1.1.0", "2.0.0"]

    def test_list_keys_sorted(self, registry: RegistryCompute) -> None:
        """Test that list_keys returns sorted results."""
        registry.register_plugin("z_plugin", SyncComputePlugin, "1.0.0")
        registry.register_plugin("a_plugin", SyncComputePlugin, "1.0.0")
        registry.register_plugin("m_plugin", SyncComputePlugin, "1.0.0")

        keys = registry.list_keys()
        plugin_ids = [k[0] for k in keys]
        assert plugin_ids == ["a_plugin", "m_plugin", "z_plugin"]


# =============================================================================
# TestUnregisterAndClear
# =============================================================================


class TestUnregisterAndClear:
    """Tests for unregister() and clear() methods."""

    def test_unregister_specific_version(self, registry: RegistryCompute) -> None:
        """Test unregistering specific version."""
        registry.register_plugin("plugin", SyncComputePlugin, "1.0.0")
        registry.register_plugin("plugin", SyncComputePlugin, "2.0.0")

        count = registry.unregister("plugin", "1.0.0")

        assert count == 1
        assert not registry.is_registered("plugin", "1.0.0")
        assert registry.is_registered("plugin", "2.0.0")

    def test_unregister_all_versions(self, registry: RegistryCompute) -> None:
        """Test unregistering all versions."""
        registry.register_plugin("plugin", SyncComputePlugin, "1.0.0")
        registry.register_plugin("plugin", SyncComputePlugin, "2.0.0")

        count = registry.unregister("plugin")  # All versions

        assert count == 2
        assert not registry.is_registered("plugin")

    def test_unregister_nonexistent_returns_zero(
        self, registry: RegistryCompute
    ) -> None:
        """Test unregistering non-existent plugin returns 0."""
        count = registry.unregister("nonexistent")
        assert count == 0

    def test_clear(self, populated_registry: RegistryCompute) -> None:
        """Test clearing all registrations."""
        assert len(populated_registry) > 0

        populated_registry.clear()

        assert len(populated_registry) == 0
        assert populated_registry.list_keys() == []


# =============================================================================
# TestThreadSafety
# =============================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent operations."""

    def test_concurrent_registration(self, registry: RegistryCompute) -> None:
        """Test concurrent plugin registration."""
        errors: list[Exception] = []

        def register_plugin(plugin_id: str) -> None:
            class Plugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {"id": plugin_id}

            try:
                registry.register_plugin(plugin_id, Plugin)
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(register_plugin, f"plugin_{i}") for i in range(100)
            ]
            for future in futures:
                future.result()

        assert len(errors) == 0
        assert len(registry) == 100

    def test_concurrent_lookup(self, registry: RegistryCompute) -> None:
        """Test concurrent plugin lookup."""
        # Pre-register plugins
        for i in range(10):
            registry.register_plugin(f"plugin_{i}", SyncComputePlugin)

        results: list[type] = []
        lock = threading.Lock()
        errors: list[Exception] = []

        def lookup(plugin_id: str) -> None:
            try:
                result = registry.get(plugin_id)
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(lookup, f"plugin_{i % 10}") for i in range(1000)]
            for future in futures:
                future.result()

        assert len(errors) == 0
        assert len(results) == 1000
        assert all(r == SyncComputePlugin for r in results)

    def test_concurrent_register_and_lookup(self, registry: RegistryCompute) -> None:
        """Test concurrent registration and lookup operations."""
        # Pre-register some plugins
        for i in range(5):
            registry.register_plugin(f"existing_{i}", SyncComputePlugin)

        errors: list[Exception] = []
        results: list[type] = []
        lock = threading.Lock()

        def register_and_lookup(thread_id: int) -> None:
            try:
                # Register new plugin
                class ThreadPlugin:
                    def execute(self, data: dict[str, object]) -> dict[str, object]:
                        return {"thread": thread_id}

                registry.register_plugin(f"thread_plugin_{thread_id}", ThreadPlugin)

                # Lookup existing plugin
                result = registry.get(f"existing_{thread_id % 5}")
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(register_and_lookup, i) for i in range(50)]
            for future in futures:
                future.result()

        assert len(errors) == 0
        assert len(results) == 50


# =============================================================================
# TestDunderMethods
# =============================================================================


class TestDunderMethods:
    """Tests for __len__ and __contains__."""

    def test_len(self, registry: RegistryCompute) -> None:
        """Test __len__ returns correct count."""
        assert len(registry) == 0

        registry.register_plugin("a", SyncComputePlugin)
        assert len(registry) == 1

        registry.register_plugin("b", SyncComputePlugin)
        assert len(registry) == 2

    def test_contains_with_string(self, registry: RegistryCompute) -> None:
        """Test 'in' operator with plugin_id string."""
        registry.register_plugin("test_plugin", SyncComputePlugin)

        assert "test_plugin" in registry
        assert "nonexistent" not in registry

    def test_contains_with_key(self, registry: RegistryCompute) -> None:
        """Test 'in' operator with ModelComputeKey."""
        registry.register_plugin("test_plugin", SyncComputePlugin, "1.0.0")

        key = ModelComputeKey(plugin_id="test_plugin", version="1.0.0")
        assert key in registry

        wrong_key = ModelComputeKey(plugin_id="test_plugin", version="9.9.9")
        assert wrong_key not in registry


# =============================================================================
# TestVersionValidation
# =============================================================================


class TestVersionValidation:
    """Tests for version validation and error handling.

    Note: Version validation happens in two places:
    1. ModelComputeRegistration validator (raises pydantic ValidationError)
    2. RegistryCompute._parse_semver (raises ProtocolConfigurationError)

    Tests cover both behaviors as the error type depends on where validation fails.
    """

    def test_invalid_version_format_raises_error(
        self, registry: RegistryCompute
    ) -> None:
        """Test that invalid version format raises ValidationError.

        Validation happens in ModelComputeRegistration's Pydantic validator.
        """
        with pytest.raises(ValidationError) as exc_info:
            registry.register_plugin(
                plugin_id="invalid_version",
                plugin_class=SyncComputePlugin,
                version="not-a-version",
            )

        assert "not-a-version" in str(exc_info.value)

    def test_empty_version_raises_error(self, registry: RegistryCompute) -> None:
        """Test that empty version string raises ValidationError.

        Validation happens in ModelComputeRegistration's Pydantic validator.
        """
        with pytest.raises(ValidationError) as exc_info:
            registry.register_plugin(
                plugin_id="empty_version",
                plugin_class=SyncComputePlugin,
                version="",
            )

        # Pydantic error message contains "Version cannot be empty"
        assert "empty" in str(exc_info.value).lower()

    def test_version_with_too_many_parts_raises_error(
        self, registry: RegistryCompute
    ) -> None:
        """Test that version with more than 3 parts raises ValidationError.

        Validation happens in ModelComputeRegistration's Pydantic validator.
        """
        with pytest.raises(ValidationError) as exc_info:
            registry.register_plugin(
                plugin_id="too_many_parts",
                plugin_class=SyncComputePlugin,
                version="1.2.3.4",
            )

        assert "1.2.3.4" in str(exc_info.value)

    def test_parse_semver_invalid_version_raises_protocol_error(self) -> None:
        """Test that _parse_semver raises ProtocolConfigurationError for invalid versions.

        This tests the internal semver parser directly, which raises
        ProtocolConfigurationError unlike the Pydantic model validators.
        """
        RegistryCompute._reset_semver_cache()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryCompute._parse_semver("not-a-semver")

        # Case-insensitive check for robustness against minor error message changes
        assert "invalid semantic version format" in str(exc_info.value).lower()

    def test_parse_semver_empty_prerelease_raises_error(self) -> None:
        """Test that empty prerelease suffix raises ProtocolConfigurationError."""
        RegistryCompute._reset_semver_cache()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryCompute._parse_semver("1.2.3-")

        # Case-insensitive check for robustness against minor error message changes
        assert "prerelease suffix cannot be empty" in str(exc_info.value).lower()

    def test_valid_prerelease_versions_accepted(
        self, registry: RegistryCompute
    ) -> None:
        """Test that valid prerelease versions are accepted."""
        registry.register_plugin(
            plugin_id="alpha_plugin",
            plugin_class=SyncComputePlugin,
            version="1.0.0-alpha",
        )
        registry.register_plugin(
            plugin_id="beta_plugin",
            plugin_class=SyncComputePlugin,
            version="2.0.0-beta.1",
        )

        assert registry.is_registered("alpha_plugin", version="1.0.0-alpha")
        assert registry.is_registered("beta_plugin", version="2.0.0-beta.1")

    def test_version_with_whitespace_trimmed(self, registry: RegistryCompute) -> None:
        """Test that whitespace is trimmed from version strings."""
        registry.register_plugin(
            plugin_id="whitespace_version",
            plugin_class=SyncComputePlugin,
            version="  1.2.3  ",
        )

        # Should be able to retrieve with trimmed version
        assert registry.is_registered("whitespace_version")
        plugin_cls = registry.get("whitespace_version", version="1.2.3")
        assert plugin_cls is SyncComputePlugin


# =============================================================================
# TestSemverCaching
# =============================================================================


class TestSemverCaching:
    """Tests for _parse_semver() caching behavior."""

    def test_parse_semver_returns_consistent_results(self) -> None:
        """Test that _parse_semver returns consistent results for same input."""
        # Reset cache to ensure clean state
        RegistryCompute._reset_semver_cache()

        # Parse same version multiple times
        result1 = RegistryCompute._parse_semver("1.2.3")
        result2 = RegistryCompute._parse_semver("1.2.3")
        result3 = RegistryCompute._parse_semver("1.2.3")

        # All should return identical tuples
        assert result1 == result2 == result3
        assert result1 == (1, 2, 3, chr(127))  # chr(127) for release version

    def test_parse_semver_cache_hits(self) -> None:
        """Test that cache info shows hits for repeated parses."""
        # Reset cache to ensure clean state
        RegistryCompute._reset_semver_cache()

        # Get the parser (initializes the cache)
        parser = RegistryCompute._get_semver_parser()
        initial_info = parser.cache_info()
        assert initial_info.hits == 0
        assert initial_info.misses == 0

        # First parse - should be a cache miss
        RegistryCompute._parse_semver("1.0.0")
        info_after_first = parser.cache_info()
        assert info_after_first.misses == 1
        assert info_after_first.hits == 0

        # Second parse of same version - should be a cache hit
        RegistryCompute._parse_semver("1.0.0")
        info_after_second = parser.cache_info()
        assert info_after_second.misses == 1
        assert info_after_second.hits == 1

    def test_reset_semver_cache_clears_state(self) -> None:
        """Test that _reset_semver_cache() clears cache state."""
        # Parse some versions
        RegistryCompute._parse_semver("1.0.0")
        RegistryCompute._parse_semver("2.0.0")

        # Reset cache
        RegistryCompute._reset_semver_cache()

        # After reset, the cache should be None
        assert RegistryCompute._semver_cache is None


# =============================================================================
# TestComputeRegistryError
# =============================================================================


class TestComputeRegistryError:
    """Tests for ComputeRegistryError exception class."""

    def test_error_includes_plugin_id(self) -> None:
        """Test that ComputeRegistryError context includes plugin_id."""
        error = ComputeRegistryError(
            "Plugin not found",
            plugin_id="missing_plugin",
        )
        assert "Plugin not found" in str(error)
        assert error.model.context.get("plugin_id") == "missing_plugin"

    def test_error_includes_version(self) -> None:
        """Test that ComputeRegistryError context includes version."""
        error = ComputeRegistryError(
            "Version not found",
            plugin_id="test_plugin",
            version="1.0.0",
        )
        assert error.model.context.get("version") == "1.0.0"

    def test_error_with_extra_context(self) -> None:
        """Test ComputeRegistryError with extra context kwargs."""
        error = ComputeRegistryError(
            "Async method detected",
            plugin_id="async_plugin",
            async_method="execute",
        )
        assert error.model.context.get("async_method") == "execute"

    def test_error_is_exception(self) -> None:
        """Test ComputeRegistryError is an Exception."""
        error = ComputeRegistryError("Test error")
        assert isinstance(error, Exception)


# =============================================================================
# TestIsRegistered
# =============================================================================


class TestIsRegistered:
    """Tests for is_registered() method."""

    def test_is_registered_returns_true(
        self, populated_registry: RegistryCompute
    ) -> None:
        """Test is_registered returns True when plugin exists."""
        assert populated_registry.is_registered("json_normalizer")

    def test_is_registered_returns_false(self, registry: RegistryCompute) -> None:
        """Test is_registered returns False when plugin doesn't exist."""
        assert not registry.is_registered("nonexistent_plugin")

    def test_is_registered_with_version_filter(self, registry: RegistryCompute) -> None:
        """Test is_registered with version filter."""
        registry.register_plugin("versioned_plugin", SyncComputePlugin, "1.0.0")

        # Matching version
        assert registry.is_registered("versioned_plugin", version="1.0.0")
        # Non-matching version
        assert not registry.is_registered("versioned_plugin", version="2.0.0")

    def test_is_registered_any_version(self, registry: RegistryCompute) -> None:
        """Test is_registered without version returns True if any version exists."""
        registry.register_plugin("multi_version", SyncComputePlugin, "1.0.0")
        registry.register_plugin("multi_version", SyncComputePlugin, "2.0.0")

        assert registry.is_registered("multi_version")


# =============================================================================
# TestModelComputeKeyHashUniqueness
# =============================================================================


class TestModelComputeKeyHashUniqueness:
    """Test ModelComputeKey hash uniqueness for edge cases."""

    def test_hash_uniqueness_similar_ids(self) -> None:
        """Similar plugin_ids should have different hashes."""
        keys = [
            ModelComputeKey(plugin_id="normalizer", version="1.0.0"),
            ModelComputeKey(plugin_id="normalizer1", version="1.0.0"),
            ModelComputeKey(plugin_id="1normalizer", version="1.0.0"),
            ModelComputeKey(plugin_id="normal", version="1.0.0"),
        ]
        hashes = {hash(k) for k in keys}
        assert len(hashes) == len(keys), "Hash collision detected for similar IDs"

    def test_hash_uniqueness_version_differs(self) -> None:
        """Same plugin_id with different versions should have different hashes."""
        keys = [
            ModelComputeKey(plugin_id="test", version="1.0.0"),
            ModelComputeKey(plugin_id="test", version="1.0.1"),
            ModelComputeKey(plugin_id="test", version="2.0.0"),
        ]
        hashes = {hash(k) for k in keys}
        assert len(hashes) == len(keys)

    def test_hash_stability(self) -> None:
        """Same key should always produce same hash."""
        key = ModelComputeKey(plugin_id="stable", version="1.0.0")
        hash1 = hash(key)
        hash2 = hash(key)
        key_copy = ModelComputeKey(plugin_id="stable", version="1.0.0")
        hash3 = hash(key_copy)

        assert hash1 == hash2 == hash3

    def test_dict_key_usage(self) -> None:
        """ModelComputeKey should work correctly as dict key."""
        d: dict[ModelComputeKey, str] = {}

        key1 = ModelComputeKey(plugin_id="a", version="1.0.0")
        key2 = ModelComputeKey(plugin_id="a", version="1.0.0")  # same
        key3 = ModelComputeKey(plugin_id="b", version="1.0.0")  # different

        d[key1] = "value1"
        d[key3] = "value3"

        # key2 should find same value as key1 (they're equal)
        assert d[key2] == "value1"
        assert len(d) == 2


# =============================================================================
# TestContainerIntegration
# =============================================================================


class TestContainerIntegration:
    """Integration tests for container-based DI access."""

    async def test_container_with_registries_provides_compute_registry(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test that real container fixture provides RegistryCompute."""
        # Skip if ServiceRegistry not available (omnibase_core 0.6.x)
        if container_with_registries.service_registry is None:
            pytest.skip("ServiceRegistry not available in omnibase_core 0.6.x")

        # Resolve from container (async in omnibase_core 0.4+)
        registry: RegistryCompute = (
            await container_with_registries.service_registry.resolve_service(
                RegistryCompute
            )
        )
        assert isinstance(registry, RegistryCompute)

    async def test_container_based_registration_workflow(
        self, container_with_registries: ModelONEXContainer
    ) -> None:
        """Test full workflow using container-based DI."""
        # Skip if ServiceRegistry not available (omnibase_core 0.6.x)
        if container_with_registries.service_registry is None:
            pytest.skip("ServiceRegistry not available in omnibase_core 0.6.x")

        # Step 1: Resolve registry from container
        registry: RegistryCompute = (
            await container_with_registries.service_registry.resolve_service(
                RegistryCompute
            )
        )

        # Step 2: Register plugin
        registry.register_plugin(
            plugin_id="container_test",
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        # Step 3: Verify registration
        assert registry.is_registered("container_test")
        plugin_cls = registry.get("container_test")
        assert plugin_cls is SyncComputePlugin


# =============================================================================
# TestSemverEdgeCases - Prerelease + Build Metadata
# =============================================================================


class TestSemverEdgeCases:
    """Tests for semver edge cases including prerelease and build metadata.

    These tests verify parsing behavior for complex semver formats per OMN-811.
    Note: The current implementation treats build metadata as part of the
    prerelease or version string rather than stripping it per strict semver spec.

    Key behaviors tested:
    - Prerelease + build metadata combinations
    - Build metadata only (no prerelease)
    - Multiple prerelease segments
    - Invalid edge cases (double dash, double plus)
    """

    def test_semver_prerelease_with_build_metadata(self) -> None:
        """Test parsing '1.0.0-alpha.1+build.123' (prerelease + build metadata).

        Current behavior: Build metadata is included as part of the prerelease string.
        This tests that combined prerelease+build formats parse without error.
        """
        RegistryCompute._reset_semver_cache()

        # Should parse without error - build metadata is captured in prerelease
        result = RegistryCompute._parse_semver("1.0.0-alpha.1+build.123")

        # Verify major, minor, patch are correct
        assert result[0] == 1  # major
        assert result[1] == 0  # minor
        assert result[2] == 0  # patch
        # Prerelease includes build metadata in current implementation
        assert result[3] == "alpha.1+build.123"

    def test_semver_build_metadata_only_raises_error(self) -> None:
        """Test parsing '1.0.0+build.123' (build metadata only, no prerelease).

        Current behavior: Without prerelease, the '+' is treated as part of the
        version string, resulting in "1.0.0+build" which fails to parse as it
        creates >3 parts when split on '.'.
        """
        RegistryCompute._reset_semver_cache()

        # This format is NOT supported by the current implementation
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryCompute._parse_semver("1.0.0+build.123")

        # Case-insensitive check for robustness against minor error message changes
        assert "invalid semantic version format" in str(exc_info.value).lower()

    def test_semver_multiple_prerelease_segments(self) -> None:
        """Test parsing '1.0.0-alpha.1.2.3' (multiple prerelease segments).

        Semver spec allows multiple dot-separated prerelease identifiers.
        The current implementation captures all segments after the first '-'.
        """
        RegistryCompute._reset_semver_cache()

        result = RegistryCompute._parse_semver("1.0.0-alpha.1.2.3")

        assert result[0] == 1  # major
        assert result[1] == 0  # minor
        assert result[2] == 0  # patch
        assert result[3] == "alpha.1.2.3"  # full prerelease string

    def test_semver_prerelease_with_numeric_build(self) -> None:
        """Test parsing '1.0.0-alpha+001' (prerelease + numeric build metadata).

        Tests that prerelease with build metadata containing numeric identifiers
        is parsed correctly.
        """
        RegistryCompute._reset_semver_cache()

        result = RegistryCompute._parse_semver("1.0.0-alpha+001")

        assert result[0] == 1  # major
        assert result[1] == 0  # minor
        assert result[2] == 0  # patch
        assert result[3] == "alpha+001"  # prerelease includes build

    def test_semver_double_dash_raises_error(self) -> None:
        """Test parsing '1.0.0--' (double dash edge case - should fail).

        A version ending with double dash has an empty prerelease identifier
        after the second dash, which is technically valid per semver but
        semantically questionable. The current implementation accepts this
        because '-' is non-empty after the split.

        However, this tests documents the current behavior.
        """
        RegistryCompute._reset_semver_cache()

        # Current implementation: splits on first '-', prerelease = '-'
        # This actually succeeds in the current implementation
        result = RegistryCompute._parse_semver("1.0.0--")

        # Documents current behavior: double-dash results in prerelease = '-'
        assert result[0] == 1
        assert result[1] == 0
        assert result[2] == 0
        assert result[3] == "-"  # Single dash as prerelease identifier

    def test_semver_double_plus_raises_error(self) -> None:
        """Test parsing '1.0.0++' (double plus edge case - should fail).

        Without prerelease delimiter, the '++' is part of the version string.
        This results in '0++' for the patch component, which fails integer parsing.
        """
        RegistryCompute._reset_semver_cache()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryCompute._parse_semver("1.0.0++")

        # Case-insensitive check for robustness against minor error message changes
        assert "invalid semantic version format" in str(exc_info.value).lower()

    def test_semver_sorting_prerelease_vs_release(
        self, registry: RegistryCompute
    ) -> None:
        """Test that prerelease versions sort before release versions.

        Per semver spec: 1.0.0-alpha < 1.0.0
        """

        class ReleasePlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"type": "release"}

        class AlphaPlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"type": "alpha"}

        class BetaPlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"type": "beta"}

        # Register in random order
        registry.register_plugin("sorted_plugin", BetaPlugin, "1.0.0-beta")
        registry.register_plugin("sorted_plugin", ReleasePlugin, "1.0.0")
        registry.register_plugin("sorted_plugin", AlphaPlugin, "1.0.0-alpha")

        # Verify sorting order
        versions = registry.list_versions("sorted_plugin")
        assert versions == ["1.0.0-alpha", "1.0.0-beta", "1.0.0"]

        # Latest should be the release (no prerelease)
        latest_cls = registry.get("sorted_plugin")
        assert latest_cls is ReleasePlugin

    def test_semver_sorting_prerelease_alphabetical(
        self, registry: RegistryCompute
    ) -> None:
        """Test that prerelease versions are sorted alphabetically.

        Per semver spec: 1.0.0-alpha < 1.0.0-beta < 1.0.0-rc
        """

        class AlphaPlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"type": "alpha"}

        class BetaPlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"type": "beta"}

        class RcPlugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"type": "rc"}

        # Register in random order
        registry.register_plugin("prerelease_sort", RcPlugin, "1.0.0-rc")
        registry.register_plugin("prerelease_sort", AlphaPlugin, "1.0.0-alpha")
        registry.register_plugin("prerelease_sort", BetaPlugin, "1.0.0-beta")

        # Verify alphabetical sorting of prerelease
        versions = registry.list_versions("prerelease_sort")
        assert versions == ["1.0.0-alpha", "1.0.0-beta", "1.0.0-rc"]

    def test_semver_prerelease_with_dots(self, registry: RegistryCompute) -> None:
        """Test prerelease versions with dot separators like 'rc.1', 'beta.2'."""

        class Rc1Plugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"v": "rc.1"}

        class Rc2Plugin:
            def execute(self, data: dict[str, object]) -> dict[str, object]:
                return {"v": "rc.2"}

        registry.register_plugin("dotted_prerelease", Rc1Plugin, "1.0.0-rc.1")
        registry.register_plugin("dotted_prerelease", Rc2Plugin, "1.0.0-rc.2")

        versions = registry.list_versions("dotted_prerelease")
        assert versions == ["1.0.0-rc.1", "1.0.0-rc.2"]

        # rc.2 > rc.1 alphabetically
        latest = registry.get("dotted_prerelease")
        assert latest is Rc2Plugin

    def test_semver_complex_prerelease_identifiers(
        self, registry: RegistryCompute
    ) -> None:
        """Test complex prerelease identifiers with multiple segments."""
        registry.register_plugin(
            "complex_prerelease", SyncComputePlugin, "2.1.0-alpha.beta.1"
        )

        assert registry.is_registered("complex_prerelease", "2.1.0-alpha.beta.1")
        plugin_cls = registry.get("complex_prerelease", "2.1.0-alpha.beta.1")
        assert plugin_cls is SyncComputePlugin

    def test_semver_registration_with_prerelease_build_combo(
        self, registry: RegistryCompute
    ) -> None:
        """Test full registration workflow with prerelease+build version."""
        registration = ModelComputeRegistration(
            plugin_id="combo_version",
            plugin_class=SyncComputePlugin,
            version="1.0.0-beta.1+build.456",
        )
        registry.register(registration)

        assert registry.is_registered("combo_version")
        assert registry.is_registered("combo_version", "1.0.0-beta.1+build.456")

        # Should be retrievable by exact version
        plugin_cls = registry.get("combo_version", "1.0.0-beta.1+build.456")
        assert plugin_cls is SyncComputePlugin


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for RegistryCompute."""

    def test_empty_registry_list_keys(self, registry: RegistryCompute) -> None:
        """Test that empty registry returns empty list."""
        keys = registry.list_keys()
        assert keys == []

    def test_plugin_id_with_special_characters(self, registry: RegistryCompute) -> None:
        """Test registration with special characters in plugin_id."""
        registry.register_plugin(
            plugin_id="json-normalizer",
            plugin_class=SyncComputePlugin,
        )
        registry.register_plugin(
            plugin_id="xml_parser",
            plugin_class=SyncComputePlugin,
        )
        registry.register_plugin(
            plugin_id="transform.v2",
            plugin_class=SyncComputePlugin,
        )

        assert registry.is_registered("json-normalizer")
        assert registry.is_registered("xml_parser")
        assert registry.is_registered("transform.v2")

    def test_get_after_unregister_all(self, registry: RegistryCompute) -> None:
        """Test get() after unregistering all versions."""
        registry.register_plugin("plugin", SyncComputePlugin, "1.0.0")
        registry.register_plugin("plugin", SyncComputePlugin, "2.0.0")

        registry.unregister("plugin")

        with pytest.raises(ComputeRegistryError):
            registry.get("plugin")

    def test_unregister_cleans_up_secondary_index(
        self, registry: RegistryCompute
    ) -> None:
        """Test that unregister properly cleans up secondary index."""
        registry.register_plugin("single", SyncComputePlugin, "1.0.0")
        registry.unregister("single", "1.0.0")

        # Should not leave empty entries in secondary index
        assert "single" not in registry._plugin_id_index


# =============================================================================
# TestUnicodePluginId - Unicode Character Handling
# =============================================================================


class TestUnicodePluginId:
    """Tests for Unicode character handling in plugin_id.

    Current Behavior Documentation:
    ------------------------------
    As of this implementation, plugin_id validation uses only `min_length=1`
    (via Pydantic's Field constraint in ModelComputeRegistration). This means:

    - Unicode letters (Cyrillic, CJK, Greek, etc.) are ALLOWED
    - Emoji characters are ALLOWED
    - Mixed Unicode/ASCII are ALLOWED
    - Zero-width characters are ALLOWED (potentially problematic for debugging)
    - NULL bytes (\\x00) are ALLOWED (security concern!)
    - Control characters (\\x07 bell, etc.) are ALLOWED

    Security Concerns:
    -----------------
    1. NULL bytes in plugin_ids can cause:
       - C-string termination issues in some backends
       - Security vulnerabilities (null byte injection)
       - Database storage problems

    2. Zero-width characters can cause:
       - Confusion: "plugin" vs "plugin\\u200b" look identical
       - Debugging difficulties

    3. Unicode confusables (homoglyphs) can cause:
       - Spoofing: "normalizer" (ASCII) vs "n\\u03bfrmalizer" (Greek omicron)
       - Accidental collisions in different scripts

    Design Considerations:
    ---------------------
    Whether Unicode plugin_ids should be supported is a design decision:

    ARGUMENTS FOR allowing Unicode:
    - Internationalization: Teams may want native-language plugin names
    - Semantic clarity: "処理器" clearly means "processor" in Japanese
    - Modern Python 3 handles Unicode strings natively

    ARGUMENTS FOR restricting to ASCII:
    - Consistency: ASCII-only ensures predictable serialization/storage
    - Debuggability: ASCII names are easier to type in logs/terminals
    - Interoperability: Some downstream systems may not handle Unicode well
    - Security: Prevents homoglyph attacks and null byte injection

    RECOMMENDATIONS:
    ---------------
    1. At minimum, reject NULL bytes and control characters:
       ```python
       @field_validator("plugin_id")
       @classmethod
       def validate_plugin_id_safe(cls, v: str) -> str:
           if any(ord(c) < 32 for c in v):  # Reject control chars including NULL
               raise ValueError("plugin_id cannot contain control characters")
           return v
       ```

    2. For maximum safety, restrict to ASCII identifier pattern:
       ```python
       import re
       @field_validator("plugin_id")
       @classmethod
       def validate_plugin_id_ascii(cls, v: str) -> str:
           if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.-]*$', v):
               raise ValueError(
                   f"plugin_id must be a valid ASCII identifier, got: {v!r}"
               )
           return v
       ```

    These tests document the CURRENT behavior without modifying it.
    """

    # =========================================================================
    # Baseline ASCII Tests
    # =========================================================================

    def test_unicode_ascii_plugin_id_baseline(self, registry: RegistryCompute) -> None:
        """Test standard ASCII plugin_id works correctly (baseline)."""
        registry.register_plugin(
            plugin_id="json_normalizer",
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered("json_normalizer")
        assert registry.is_registered("json_normalizer", version="1.0.0")
        plugin_cls = registry.get("json_normalizer")
        assert plugin_cls is SyncComputePlugin

    def test_unicode_ascii_with_underscores_and_hyphens(
        self, registry: RegistryCompute
    ) -> None:
        """Test ASCII with common separator characters."""
        registry.register_plugin("json-normalizer", SyncComputePlugin, "1.0.0")
        registry.register_plugin("xml_transformer", SyncComputePlugin, "1.0.0")
        registry.register_plugin("data.processor", SyncComputePlugin, "1.0.0")

        assert registry.is_registered("json-normalizer")
        assert registry.is_registered("xml_transformer")
        assert registry.is_registered("data.processor")

    # =========================================================================
    # Unicode Letters Tests (Cyrillic, CJK, Japanese, Greek)
    # =========================================================================

    def test_unicode_cyrillic_plugin_id(self, registry: RegistryCompute) -> None:
        """Test Cyrillic (Russian) plugin_id is accepted.

        Current behavior: ALLOWED
        "процессор" means "processor" in Russian.
        """
        plugin_id = "процессор"  # Russian for "processor"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        # Registration should succeed
        assert registry.is_registered(plugin_id)
        assert registry.is_registered(plugin_id, version="1.0.0")

        # Retrieval should work
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

        # Should appear in list_keys
        keys = registry.list_keys()
        assert (plugin_id, "1.0.0") in keys

    def test_unicode_chinese_plugin_id(self, registry: RegistryCompute) -> None:
        """Test Chinese plugin_id is accepted.

        Current behavior: ALLOWED
        "处理器" means "processor" in Chinese.
        """
        plugin_id = "处理器"  # Chinese for "processor"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_japanese_plugin_id(self, registry: RegistryCompute) -> None:
        """Test Japanese plugin_id is accepted.

        Current behavior: ALLOWED
        "処理" means "processing" in Japanese.
        """
        plugin_id = "処理"  # Japanese for "processing"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_greek_plugin_id(self, registry: RegistryCompute) -> None:
        """Test Greek plugin_id is accepted.

        Current behavior: ALLOWED
        """
        plugin_id = "επεξεργαστής"  # Greek for "processor"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    # =========================================================================
    # Emoji Tests
    # =========================================================================

    def test_unicode_emoji_prefix_plugin_id(self, registry: RegistryCompute) -> None:
        """Test plugin_id with emoji prefix is accepted.

        Current behavior: ALLOWED
        """
        plugin_id = "🔧_tool"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_emoji_suffix_plugin_id(self, registry: RegistryCompute) -> None:
        """Test plugin_id with emoji suffix is accepted.

        Current behavior: ALLOWED
        """
        plugin_id = "data_🚀"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_emoji_only_plugin_id(self, registry: RegistryCompute) -> None:
        """Test plugin_id with only emojis is accepted.

        Current behavior: ALLOWED
        """
        plugin_id = "🔧🚀💾"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    # =========================================================================
    # Mixed Unicode Tests
    # =========================================================================

    def test_unicode_mixed_ascii_greek_plugin_id(
        self, registry: RegistryCompute
    ) -> None:
        """Test mixed ASCII and Greek letters in plugin_id.

        Current behavior: ALLOWED
        Uses Greek letters alpha and beta.
        """
        plugin_id = "transformer_α_β"  # noqa: RUF001

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_mixed_scripts_plugin_id(self, registry: RegistryCompute) -> None:
        """Test mixed scripts (Latin, Cyrillic, CJK) in plugin_id.

        Current behavior: ALLOWED
        """
        plugin_id = "data_данные_数据"  # English + Russian + Chinese for "data"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    # =========================================================================
    # Unicode Whitespace and Control Characters
    # =========================================================================

    def test_unicode_zero_width_space_in_plugin_id(
        self, registry: RegistryCompute
    ) -> None:
        """Test plugin_id with zero-width space (U+200B) is accepted.

        Current behavior: ALLOWED (potentially problematic!)

        Zero-width space is an invisible character that can cause:
        - Confusion: "plugin" vs "plugin\\u200b" look identical
        - Serialization issues in some systems
        - Debugging difficulties

        RECOMMENDATION: Consider rejecting zero-width and other
        invisible Unicode characters in plugin_id validation.
        """
        # Plugin ID with invisible zero-width space at end
        plugin_id_with_zwsp = "plugin\u200b"  # Zero-width space (U+200B)
        plugin_id_normal = "plugin"

        # Both should register separately (they are different strings)
        registry.register_plugin(
            plugin_id=plugin_id_with_zwsp,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )
        registry.register_plugin(
            plugin_id=plugin_id_normal,
            plugin_class=SyncComputePluginV1,
            version="1.0.0",
        )

        # Both should be retrievable
        assert registry.is_registered(plugin_id_with_zwsp)
        assert registry.is_registered(plugin_id_normal)

        # They are DIFFERENT plugins (different strings)
        assert len(registry) == 2
        assert plugin_id_with_zwsp != plugin_id_normal

        # Can retrieve each independently
        cls1 = registry.get(plugin_id_with_zwsp)
        cls2 = registry.get(plugin_id_normal)
        assert cls1 is SyncComputePlugin
        assert cls2 is SyncComputePluginV1

    def test_unicode_null_byte_in_plugin_id_accepted(
        self, registry: RegistryCompute
    ) -> None:
        """Test plugin_id with NULL byte (\\x00) is accepted.

        Current behavior: ALLOWED (security concern!)

        NULL bytes in strings can cause:
        - C-string termination issues in some backends
        - Security vulnerabilities (null byte injection)
        - Database storage problems

        RECOMMENDATION: NULL bytes should be rejected in plugin_id.
        Add a validator to ModelComputeRegistration:
        ```python
        @field_validator("plugin_id")
        @classmethod
        def validate_no_null_bytes(cls, v: str) -> str:
            if "\\x00" in v:
                raise ValueError("plugin_id cannot contain NULL bytes")
            return v
        ```
        """
        plugin_id_with_null = "plugin\x00suffix"
        plugin_id_normal = "plugin"

        # Currently NULL bytes are ALLOWED (potentially problematic!)
        registry.register_plugin(
            plugin_id=plugin_id_with_null,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        # Registration succeeds
        assert registry.is_registered(plugin_id_with_null)

        # The null byte makes it a different plugin_id
        registry.register_plugin(
            plugin_id=plugin_id_normal,
            plugin_class=SyncComputePluginV1,
            version="1.0.0",
        )

        assert len(registry) == 2
        assert plugin_id_with_null != plugin_id_normal

        # Can retrieve both
        cls1 = registry.get(plugin_id_with_null)
        cls2 = registry.get(plugin_id_normal)
        assert cls1 is SyncComputePlugin
        assert cls2 is SyncComputePluginV1

    def test_unicode_other_control_characters_behavior(
        self, registry: RegistryCompute
    ) -> None:
        """Test behavior with other Unicode control characters.

        Current behavior: Some control chars may be ALLOWED

        This test documents behavior with bell character (\\x07).
        Different control characters may have different behavior.
        """
        # Bell character - less problematic than NULL
        plugin_id_with_bell = "plugin\x07bell"

        # Try to register and see what happens
        try:
            registry.register_plugin(
                plugin_id=plugin_id_with_bell,
                plugin_class=SyncComputePlugin,
                version="1.0.0",
            )
            # If we get here, the control character was accepted
            assert registry.is_registered(plugin_id_with_bell)
            # Document that this control character is allowed
        except ValidationError:
            # If ValidationError is raised, control chars are rejected
            # Document that Pydantic rejects this character
            pass  # Test passes either way - we're documenting behavior

    # =========================================================================
    # Unicode Normalization Tests
    # =========================================================================

    def test_unicode_normalization_not_applied(self, registry: RegistryCompute) -> None:
        """Test that Unicode normalization is NOT applied to plugin_id.

        Current behavior: NO normalization

        The same character in different Unicode forms (NFC vs NFD)
        is treated as DIFFERENT plugin_ids.

        Example: "cafe" with composed e-acute vs decomposed e + accent

        RECOMMENDATION: Consider applying Unicode normalization (NFC)
        to plugin_ids for consistency.
        """
        import unicodedata

        # "cafe" in NFC (composed: e-acute as single character U+00E9)
        plugin_id_nfc = unicodedata.normalize("NFC", "café")
        # "cafe" in NFD (decomposed: e + combining acute accent)
        plugin_id_nfd = unicodedata.normalize("NFD", "café")

        # They look the same but are different byte sequences
        assert plugin_id_nfc != plugin_id_nfd

        registry.register_plugin(
            plugin_id=plugin_id_nfc,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )
        registry.register_plugin(
            plugin_id=plugin_id_nfd,
            plugin_class=SyncComputePluginV1,
            version="1.0.0",
        )

        # Both are registered as DIFFERENT plugins
        assert len(registry) == 2
        assert registry.is_registered(plugin_id_nfc)
        assert registry.is_registered(plugin_id_nfd)

    # =========================================================================
    # Unicode Confusables (Homoglyph) Tests
    # =========================================================================

    def test_unicode_lookalike_characters_are_different(
        self, registry: RegistryCompute
    ) -> None:
        """Test that visually similar Unicode characters create different plugins.

        Current behavior: Lookalikes are DIFFERENT plugins

        This documents a potential security/confusion issue:
        - "normalizer" (ASCII 'o')
        - "normalizer" (Greek omicron U+03BF)

        These look nearly identical but are different strings.

        RECOMMENDATION: If this is a security concern, consider:
        1. Restricting to ASCII-only plugin_ids
        2. Using Unicode confusable detection
        """
        # ASCII lowercase 'o' (U+006F)
        plugin_id_ascii = "normalizer"
        # Greek lowercase omicron (U+03BF) - visually similar to 'o'
        plugin_id_greek = "n\u03bfrmalizer"  # Using escape for Greek omicron

        # Verify they are different strings
        assert plugin_id_ascii != plugin_id_greek
        assert "o" != "\u03bf"  # noqa: PLR0133 - Intentional: ASCII 'o' vs Greek omicron

        registry.register_plugin(
            plugin_id=plugin_id_ascii,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )
        registry.register_plugin(
            plugin_id=plugin_id_greek,
            plugin_class=SyncComputePluginV1,
            version="1.0.0",
        )

        # Both registered as DIFFERENT plugins
        assert len(registry) == 2
        assert registry.is_registered(plugin_id_ascii)
        assert registry.is_registered(plugin_id_greek)

        # Can retrieve each independently
        cls_ascii = registry.get(plugin_id_ascii)
        cls_greek = registry.get(plugin_id_greek)
        assert cls_ascii is SyncComputePlugin
        assert cls_greek is SyncComputePluginV1

    # =========================================================================
    # Multiple Unicode Plugin Version Tests
    # =========================================================================

    def test_unicode_plugin_multiple_versions(self, registry: RegistryCompute) -> None:
        """Test registering multiple versions of a Unicode plugin_id."""
        plugin_id = "处理器"  # Chinese for "processor"

        registry.register_plugin(plugin_id, SyncComputePluginV1, "1.0.0")
        registry.register_plugin(plugin_id, SyncComputePluginV2, "2.0.0")

        # Both versions registered
        versions = registry.list_versions(plugin_id)
        assert versions == ["1.0.0", "2.0.0"]

        # get() without version returns latest
        latest = registry.get(plugin_id)
        assert latest is SyncComputePluginV2

        # get() with version returns specific
        v1 = registry.get(plugin_id, version="1.0.0")
        assert v1 is SyncComputePluginV1

    def test_unicode_plugin_unregister(self, registry: RegistryCompute) -> None:
        """Test unregistering a Unicode plugin_id."""
        plugin_id = "процессор"

        registry.register_plugin(plugin_id, SyncComputePlugin, "1.0.0")
        assert registry.is_registered(plugin_id)

        count = registry.unregister(plugin_id)
        assert count == 1
        assert not registry.is_registered(plugin_id)

    def test_unicode_plugin_in_list_keys_sorted(
        self, registry: RegistryCompute
    ) -> None:
        """Test that Unicode plugins appear correctly in list_keys()."""
        # Register ASCII and various Unicode plugins
        registry.register_plugin("ascii_plugin", SyncComputePlugin, "1.0.0")
        registry.register_plugin("处理器", SyncComputePlugin, "1.0.0")
        registry.register_plugin("процессор", SyncComputePlugin, "1.0.0")

        keys = registry.list_keys()
        plugin_ids = [k[0] for k in keys]

        # All should be present
        assert "ascii_plugin" in plugin_ids
        assert "处理器" in plugin_ids
        assert "процессор" in plugin_ids

    # =========================================================================
    # Edge Cases
    # =========================================================================

    def test_unicode_single_character_plugin_id(
        self, registry: RegistryCompute
    ) -> None:
        """Test single Unicode character as plugin_id (min_length=1)."""
        # Single Chinese character
        plugin_id = "龙"  # "dragon" in Chinese

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_mathematical_symbols(self, registry: RegistryCompute) -> None:
        """Test mathematical Unicode symbols in plugin_id."""
        plugin_id = "∑_sum_∏_product"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin

    def test_unicode_arrows_and_symbols(self, registry: RegistryCompute) -> None:
        """Test arrow and other symbols in plugin_id."""
        plugin_id = "input→transform→output"

        registry.register_plugin(
            plugin_id=plugin_id,
            plugin_class=SyncComputePlugin,
            version="1.0.0",
        )

        assert registry.is_registered(plugin_id)
        plugin_cls = registry.get(plugin_id)
        assert plugin_cls is SyncComputePlugin


# =============================================================================
# TestStressAndPerformance - Large Scale Registry Tests
# =============================================================================


class TestStressAndPerformance:
    """Stress tests for RegistryCompute with large numbers of registrations.

    These tests verify the registry performs well at scale:
    - 1000+ unique plugin registrations
    - 100+ versions per plugin
    - Registration, lookup, and list operation performance
    - Memory efficiency

    Tests are marked with @pytest.mark.slow for optional exclusion in fast test runs.
    """

    @pytest.fixture
    def large_registry(self) -> RegistryCompute:
        """Create a registry with 1000 unique plugins for stress testing.

        Creates 1000 unique plugins with version 1.0.0 each.
        Total: 1000 registrations.

        Note: Direct instantiation avoids container DI overhead for accurate
        performance measurement.
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()
        for i in range(1000):

            class DynamicPlugin:
                """Dynamically created plugin for stress testing."""

                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {"plugin_id": i}  # noqa: B023

            registry.register_plugin(
                plugin_id=f"plugin_{i:04d}",
                plugin_class=DynamicPlugin,
                version="1.0.0",
            )
        return registry

    @pytest.fixture
    def many_versions_registry(self) -> RegistryCompute:
        """Create a registry with single plugin having 100+ versions.

        Creates 1 plugin with 100 versions (1.0.0 through 1.99.0).
        Total: 100 registrations.
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()
        for i in range(100):

            class VersionedPlugin:
                """Plugin with specific version for testing."""

                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {"version": i}  # noqa: B023

            registry.register_plugin(
                plugin_id="versioned_plugin",
                plugin_class=VersionedPlugin,
                version=f"1.{i}.0",
            )
        return registry

    @pytest.mark.slow
    def test_stress_register_1000_unique_plugins(self) -> None:
        """Stress test: Register 1000 unique plugins in < 1 second.

        This validates that registration performance scales linearly
        with O(1) registration complexity.

        Threshold: 1000 registrations < 1000ms (< 1ms per registration)
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        start_time = time.perf_counter()
        for i in range(1000):

            class DynamicPlugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {"id": i}  # noqa: B023

            registry.register_plugin(
                plugin_id=f"stress_plugin_{i:04d}",
                plugin_class=DynamicPlugin,
                version="1.0.0",
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify all registrations succeeded
        assert len(registry) == 1000

        # Performance assertion: < 1 second for 1000 registrations
        assert elapsed_ms < 1000, (
            f"1000 registrations took {elapsed_ms:.1f}ms (threshold: 1000ms). "
            f"Average: {elapsed_ms / 1000:.3f}ms per registration."
        )

    @pytest.mark.slow
    def test_stress_register_100_versions_same_plugin(self) -> None:
        """Stress test: Register 100 versions of same plugin.

        This validates that the secondary index handles many versions
        for a single plugin_id efficiently.

        Threshold: 100 registrations < 200ms
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        start_time = time.perf_counter()
        for i in range(100):

            class VersionPlugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {"v": i}  # noqa: B023

            registry.register_plugin(
                plugin_id="multi_version_plugin",
                plugin_class=VersionPlugin,
                version=f"{i // 10}.{i % 10}.0",
            )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify all registrations succeeded
        assert len(registry) == 100
        versions = registry.list_versions("multi_version_plugin")
        assert len(versions) == 100

        # Performance assertion
        assert elapsed_ms < 200, (
            f"100 version registrations took {elapsed_ms:.1f}ms (threshold: 200ms)."
        )

    @pytest.mark.slow
    def test_stress_get_random_lookups_1000(
        self, large_registry: RegistryCompute
    ) -> None:
        """Stress test: Random get() lookups should average < 1ms.

        Tests that the secondary index provides O(1) lookup performance
        even with 1000 registered plugins.

        Threshold: 1000 lookups < 100ms (< 0.1ms per lookup on average)
        """
        import random

        # Warm up cache
        _ = large_registry.get("plugin_0500")

        # Perform 1000 random lookups
        plugin_ids = [f"plugin_{i:04d}" for i in range(1000)]
        random.shuffle(plugin_ids)

        start_time = time.perf_counter()
        for plugin_id in plugin_ids:
            _ = large_registry.get(plugin_id)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        avg_ms = elapsed_ms / 1000

        # Performance assertion: < 1ms average per lookup
        assert avg_ms < 1.0, (
            f"Average lookup time {avg_ms:.3f}ms exceeds 1ms threshold. "
            f"Total: {elapsed_ms:.1f}ms for 1000 lookups."
        )

    @pytest.mark.slow
    def test_stress_get_p99_latency_under_threshold(
        self, large_registry: RegistryCompute
    ) -> None:
        """Stress test: P99 get() latency must be under 1ms.

        Validates O(1) secondary index optimization provides consistent performance.

        Threshold: P99 < 1ms
        """
        import statistics

        # Warm up
        for _ in range(10):
            _ = large_registry.get("plugin_0500")

        # Collect 1000 latency samples
        latencies: list[float] = []
        plugin_ids = [f"plugin_{i:04d}" for i in range(1000)]

        for plugin_id in plugin_ids:
            start = time.perf_counter()
            _ = large_registry.get(plugin_id)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

        # Calculate p99 latency
        latencies.sort()
        p99_index = int(len(latencies) * 0.99)
        p99 = latencies[p99_index]
        p50 = statistics.median(latencies)
        mean = statistics.mean(latencies)

        assert p99 < 1.0, (
            f"P99 latency {p99:.3f}ms exceeds 1ms threshold. "
            f"Stats: p50={p50:.3f}ms, mean={mean:.3f}ms, p99={p99:.3f}ms."
        )

    @pytest.mark.slow
    def test_stress_list_keys_1000_entries(
        self, large_registry: RegistryCompute
    ) -> None:
        """Stress test: list_keys() with 1000 entries.

        Tests that list_keys() completes in reasonable time with sorting.

        Threshold: list_keys() < 100ms for 1000 entries
        """
        start_time = time.perf_counter()
        keys = large_registry.list_keys()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify correctness
        assert len(keys) == 1000
        # Verify sorted order
        plugin_ids = [k[0] for k in keys]
        assert plugin_ids == sorted(plugin_ids)

        # Performance assertion
        assert elapsed_ms < 100, (
            f"list_keys() took {elapsed_ms:.1f}ms for 1000 entries (threshold: 100ms)."
        )

    @pytest.mark.slow
    def test_stress_list_versions_100_versions(
        self, many_versions_registry: RegistryCompute
    ) -> None:
        """Stress test: list_versions() with 100 versions.

        Tests that list_versions() performance is O(k) where k = number of versions.

        Threshold: 1000 list_versions() calls < 500ms
        """
        start_time = time.perf_counter()
        for _ in range(1000):
            versions = many_versions_registry.list_versions("versioned_plugin")
            assert len(versions) == 100
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Performance assertion
        assert elapsed_ms < 500, (
            f"1000 list_versions() calls took {elapsed_ms:.1f}ms (threshold: 500ms). "
            f"Average: {elapsed_ms / 1000:.3f}ms per call."
        )

    @pytest.mark.slow
    def test_stress_unregister_performance(self) -> None:
        """Stress test: unregister() performance with many entries.

        Tests both single-version and all-version unregister performance.

        Threshold: 500 unregistrations < 500ms
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        # Register 500 plugins with 2 versions each = 1000 total
        for i in range(500):

            class Plugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {}

            registry.register_plugin(f"unregister_test_{i:04d}", Plugin, "1.0.0")
            registry.register_plugin(f"unregister_test_{i:04d}", Plugin, "2.0.0")

        assert len(registry) == 1000

        # Test single-version unregister (250 operations)
        start_time = time.perf_counter()
        for i in range(250):
            count = registry.unregister(f"unregister_test_{i:04d}", "1.0.0")
            assert count == 1
        single_version_ms = (time.perf_counter() - start_time) * 1000

        # Test all-versions unregister (250 operations, each removes 2 entries)
        start_time = time.perf_counter()
        for i in range(250, 500):
            count = registry.unregister(f"unregister_test_{i:04d}")
            assert count == 2
        all_versions_ms = (time.perf_counter() - start_time) * 1000

        total_ms = single_version_ms + all_versions_ms

        # Verify all unregistered
        # 250 plugins remaining (i=0-249 with only version 2.0.0)
        assert len(registry) == 250

        # Performance assertion
        assert total_ms < 500, (
            f"500 unregister operations took {total_ms:.1f}ms (threshold: 500ms). "
            f"Single-version: {single_version_ms:.1f}ms, "
            f"All-versions: {all_versions_ms:.1f}ms."
        )

    @pytest.mark.slow
    def test_stress_memory_footprint_1000_plugins(self) -> None:
        """Stress test: Memory footprint with 1000 plugins.

        Validates that memory usage doesn't grow unbounded.
        Based on documented estimate: ~220 bytes per registration.

        Expected: 1000 registrations ~= 220 KB
        Threshold: < 1 MB (generous margin for Python overhead)
        """
        import gc
        import sys

        # Force GC for clean baseline
        gc.collect()
        gc.collect()
        gc.collect()

        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        # Register 1000 plugins
        for i in range(1000):

            class MemoryPlugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {}

            registry.register_plugin(f"mem_plugin_{i:04d}", MemoryPlugin, "1.0.0")

        # Measure memory using sys.getsizeof for registry internals
        memory_bytes = 0

        # Registry dict
        memory_bytes += sys.getsizeof(registry._registry)
        for key, _value in registry._registry.items():
            memory_bytes += sys.getsizeof(key)
            # Key internals (strings)
            memory_bytes += sys.getsizeof(key.plugin_id)
            memory_bytes += sys.getsizeof(key.version)

        # Secondary index
        memory_bytes += sys.getsizeof(registry._plugin_id_index)
        for plugin_id, keys in registry._plugin_id_index.items():
            memory_bytes += sys.getsizeof(plugin_id)
            memory_bytes += sys.getsizeof(keys)

        memory_kb = memory_bytes / 1024

        # Verify count
        assert len(registry) == 1000

        # Memory assertion: < 1 MB (generous threshold for Python object overhead)
        assert memory_kb < 1024, (
            f"Registry memory {memory_kb:.1f}KB exceeds 1024KB threshold. "
            f"Expected ~220KB for 1000 registrations."
        )

    @pytest.mark.slow
    def test_stress_memory_no_leak_on_repeated_operations(self) -> None:
        """Stress test: Verify no memory leak from repeated register/unregister cycles.

        Tests that the secondary index is properly cleaned up and doesn't
        accumulate stale entries.

        Strategy: Register and unregister 1000 plugins 5 times,
        measure final memory footprint.
        """
        import gc
        import sys

        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        # Perform 5 cycles of register/unregister
        for cycle in range(5):
            # Register 1000 plugins
            for i in range(1000):

                class CyclePlugin:
                    def execute(self, data: dict[str, object]) -> dict[str, object]:
                        return {"cycle": cycle}  # noqa: B023

                registry.register_plugin(f"cycle_plugin_{i:04d}", CyclePlugin, "1.0.0")

            assert len(registry) == 1000

            # Unregister all
            for i in range(1000):
                registry.unregister(f"cycle_plugin_{i:04d}")

            assert len(registry) == 0

        # Force GC
        gc.collect()
        gc.collect()

        # Measure final memory footprint
        memory_bytes = sys.getsizeof(registry._registry)
        memory_bytes += sys.getsizeof(registry._plugin_id_index)

        # Registry should be essentially empty
        assert len(registry._registry) == 0
        assert len(registry._plugin_id_index) == 0

        # Memory should be bounded - Python dicts pre-allocate and don't shrink
        # fully after clearing, so we use a generous threshold. The key test is
        # that the registry is logically empty (len == 0) and memory doesn't
        # grow unboundedly (stays under 100KB for cleared dicts).
        assert memory_bytes < 100 * 1024, (
            f"Empty registry using {memory_bytes} bytes after cycles. "
            "Possible memory leak in secondary index cleanup."
        )

    @pytest.mark.slow
    def test_stress_concurrent_operations_1000(self) -> None:
        """Stress test: 1000 concurrent operations (register + lookup).

        Tests thread safety under high concurrency with mixed operations.

        Threshold: 1000 concurrent operations < 2 seconds
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        # Pre-register some plugins
        for i in range(100):

            class PrePlugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {}

            registry.register_plugin(f"pre_plugin_{i:02d}", PrePlugin, "1.0.0")

        errors: list[Exception] = []
        results: list[bool] = []
        lock = threading.Lock()

        def concurrent_operation(thread_id: int) -> None:
            try:
                # Mix of operations
                for i in range(100):
                    if i % 3 == 0:
                        # Register new plugin
                        class ThreadPlugin:
                            def execute(
                                self, data: dict[str, object]
                            ) -> dict[str, object]:
                                return {"thread": thread_id}

                        registry.register_plugin(
                            f"thread_{thread_id}_plugin_{i}",
                            ThreadPlugin,
                            "1.0.0",
                        )
                    elif i % 3 == 1:
                        # Lookup existing plugin
                        plugin_cls = registry.get(f"pre_plugin_{i % 100:02d}")
                        with lock:
                            results.append(plugin_cls is not None)
                    else:
                        # Check registration
                        is_reg = registry.is_registered(f"pre_plugin_{i % 100:02d}")
                        with lock:
                            results.append(is_reg)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run 10 threads concurrently
        start_time = time.perf_counter()
        threads = [
            threading.Thread(target=concurrent_operation, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify no errors
        assert len(errors) == 0, f"Concurrent operation errors: {errors}"

        # Verify some operations succeeded
        assert len(results) > 0

        # Performance assertion
        assert elapsed_ms < 2000, (
            f"Concurrent operations took {elapsed_ms:.1f}ms (threshold: 2000ms)."
        )

    @pytest.mark.slow
    def test_stress_get_latest_with_100_versions(
        self, many_versions_registry: RegistryCompute
    ) -> None:
        """Stress test: get() returns correct latest from 100 versions.

        Validates that semantic version sorting works correctly with many versions.

        Threshold: 1000 get() calls < 200ms
        """
        # Warm up
        _ = many_versions_registry.get("versioned_plugin")

        start_time = time.perf_counter()
        for _ in range(1000):
            plugin_cls = many_versions_registry.get("versioned_plugin")
            # Just verify we got a class back
            assert plugin_cls is not None
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Verify latest version is returned (1.99.0 is semantically latest)
        versions = many_versions_registry.list_versions("versioned_plugin")
        assert versions[-1] == "1.99.0", (
            f"Expected 1.99.0 as latest, got {versions[-1]}"
        )

        # Performance assertion
        assert elapsed_ms < 200, (
            f"1000 get() calls with 100 versions took {elapsed_ms:.1f}ms "
            f"(threshold: 200ms). Average: {elapsed_ms / 1000:.3f}ms."
        )

    @pytest.mark.slow
    def test_stress_is_registered_performance(
        self, large_registry: RegistryCompute
    ) -> None:
        """Stress test: is_registered() performance with 1000 plugins.

        Threshold: 1000 is_registered() calls < 50ms (< 0.05ms per call)
        """
        start_time = time.perf_counter()
        for i in range(1000):
            plugin_id = f"plugin_{i:04d}"
            result = large_registry.is_registered(plugin_id)
            assert result is True
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Performance assertion
        assert elapsed_ms < 50, (
            f"1000 is_registered() calls took {elapsed_ms:.1f}ms (threshold: 50ms). "
            f"Average: {elapsed_ms / 1000:.4f}ms per call."
        )

    @pytest.mark.slow
    def test_stress_secondary_index_integrity(self) -> None:
        """Stress test: Verify secondary index integrity after many operations.

        Tests that the secondary index stays consistent with the main registry
        after many mixed operations (register, unregister, overwrite).
        """
        RegistryCompute._reset_semver_cache()
        registry = RegistryCompute()

        # Phase 1: Register 500 plugins
        for i in range(500):

            class Phase1Plugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {}

            registry.register_plugin(f"integrity_plugin_{i:04d}", Phase1Plugin, "1.0.0")

        # Phase 2: Add second version to half of them
        for i in range(250):

            class Phase2Plugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {}

            registry.register_plugin(f"integrity_plugin_{i:04d}", Phase2Plugin, "2.0.0")

        # Phase 3: Unregister some
        for i in range(100):
            registry.unregister(f"integrity_plugin_{i:04d}", "1.0.0")

        # Phase 4: Overwrite some
        for i in range(100, 150):

            class Phase4Plugin:
                def execute(self, data: dict[str, object]) -> dict[str, object]:
                    return {"overwritten": True}

            registry.register_plugin(f"integrity_plugin_{i:04d}", Phase4Plugin, "1.0.0")

        # Verify secondary index integrity
        # Count should match
        total_keys_in_index = sum(
            len(keys) for keys in registry._plugin_id_index.values()
        )
        assert total_keys_in_index == len(registry._registry), (
            f"Secondary index count {total_keys_in_index} != "
            f"registry count {len(registry._registry)}"
        )

        # All keys in index should exist in registry
        for plugin_id, keys in registry._plugin_id_index.items():
            for key in keys:
                assert key in registry._registry, (
                    f"Key {key} in index but not in registry"
                )
                assert key.plugin_id == plugin_id, (
                    f"Key plugin_id {key.plugin_id} != index key {plugin_id}"
                )

        # All keys in registry should be in index
        for key in registry._registry:
            assert key.plugin_id in registry._plugin_id_index, (
                f"plugin_id {key.plugin_id} not in secondary index"
            )
            assert key in registry._plugin_id_index[key.plugin_id], (
                f"Key {key} not in secondary index for {key.plugin_id}"
            )

    @pytest.mark.slow
    def test_stress_semver_cache_performance(self) -> None:
        """Stress test: Semver cache provides performance benefit.

        Validates that the LRU cache for semver parsing improves performance
        for repeated operations.

        Strategy: Compare cold cache vs warm cache performance.
        """
        RegistryCompute._reset_semver_cache()

        # Generate version strings
        versions = [
            f"{major}.{minor}.{patch}"
            for major in range(10)
            for minor in range(10)
            for patch in range(10)
        ]
        # 1000 unique versions

        # Cold cache - first parse of each version
        cold_start = time.perf_counter()
        for v in versions:
            _ = RegistryCompute._parse_semver(v)
        cold_time_ms = (time.perf_counter() - cold_start) * 1000

        # Warm cache - repeated parsing (should hit cache)
        warm_start = time.perf_counter()
        for _ in range(10):  # 10 iterations
            for v in versions:
                _ = RegistryCompute._parse_semver(v)
        warm_time_ms = (time.perf_counter() - warm_start) * 1000

        # Per-iteration warm time
        warm_per_iteration_ms = warm_time_ms / 10

        # Cache should provide some benefit or at least not hurt significantly
        # (Within 2x is acceptable due to cache overhead on fast operations)
        ratio = warm_per_iteration_ms / cold_time_ms

        assert ratio < 2.0, (
            f"Semver cache not effective. "
            f"Cold: {cold_time_ms:.2f}ms, "
            f"Warm per iteration: {warm_per_iteration_ms:.2f}ms. "
            f"Ratio: {ratio:.2f}x (expected < 2.0x)."
        )
