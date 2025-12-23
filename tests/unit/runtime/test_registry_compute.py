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

        error_msg = str(exc_info.value)
        assert "async execute()" in error_msg
        assert "deterministic_async=True not specified" in error_msg

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

        # Should mention the async method name
        error_msg = str(exc_info.value)
        assert "validate" in error_msg

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

        assert "Invalid semantic version format" in str(exc_info.value)

    def test_parse_semver_empty_prerelease_raises_error(self) -> None:
        """Test that empty prerelease suffix raises ProtocolConfigurationError."""
        RegistryCompute._reset_semver_cache()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            RegistryCompute._parse_semver("1.2.3-")

        assert "Prerelease suffix cannot be empty" in str(exc_info.value)

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
