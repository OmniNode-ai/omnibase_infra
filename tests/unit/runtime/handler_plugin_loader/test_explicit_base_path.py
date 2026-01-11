# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader.discover_and_load explicit base_path usage.

This module tests the explicit base_path parameter in discover_and_load(),
verifying that it provides deterministic behavior independent of the current
working directory.

Part of OMN-1132: Handler Plugin Loader implementation.

Background:
    The discover_and_load method supports an optional base_path parameter that
    allows callers to specify an explicit base path for glob pattern resolution.
    This is important for:
    - Deterministic behavior in multi-threaded applications
    - Test isolation without modifying cwd
    - Consistent results regardless of where the code is invoked from
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


class TestHandlerPluginLoaderExplicitBasePath:
    """Tests for discover_and_load with explicit base_path parameter."""

    def test_discover_with_explicit_base_path(
        self, valid_contract_directory: Path
    ) -> None:
        """Test that explicit base_path enables discovery without changing cwd.

        This test verifies that passing base_path allows discovering contracts
        relative to that path, without needing to change the current working
        directory with monkeypatch.chdir().
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Use relative glob pattern with explicit base_path
        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=valid_contract_directory,
        )

        # Should find all 3 handlers
        assert len(handlers) == 3
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    def test_explicit_base_path_overrides_cwd(
        self,
        valid_contract_directory: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that explicit base_path takes precedence over cwd.

        This test verifies that when both cwd and base_path could be used,
        the explicit base_path is used for glob resolution, not cwd.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change cwd to an empty directory (no contracts)
        empty_dir = tmp_path / "empty_cwd"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        loader = HandlerPluginLoader()

        # Even though cwd is empty, base_path should find contracts
        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=valid_contract_directory,
        )

        # Should find all 3 handlers from base_path, not from cwd
        assert len(handlers) == 3
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    def test_explicit_base_path_with_subdirectory_pattern(
        self, valid_contract_directory: Path
    ) -> None:
        """Test explicit base_path with patterns targeting subdirectories.

        Verifies that patterns can target specific subdirectories when
        using an explicit base_path.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Pattern targeting only handler1 subdirectory
        pattern = "handler1/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=valid_contract_directory,
        )

        # Should find only handler1
        assert len(handlers) == 1
        assert handlers[0].handler_name == "handler.one"

    def test_explicit_base_path_with_nested_pattern(
        self, valid_contract_directory: Path
    ) -> None:
        """Test explicit base_path with patterns targeting nested directories.

        Verifies that recursive glob patterns work correctly with explicit
        base_path for deeply nested contract files.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Pattern targeting nested directory
        pattern = "nested/**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=valid_contract_directory,
        )

        # Should find only the nested handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "handler.nested.deep"

    def test_explicit_base_path_with_multiple_patterns(
        self, valid_contract_directory: Path
    ) -> None:
        """Test explicit base_path with multiple glob patterns.

        Verifies that multiple patterns are all resolved relative to
        the explicit base_path.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Multiple patterns targeting different directories
        patterns = [
            "handler1/handler_contract.yaml",
            "handler2/handler_contract.yaml",
        ]
        handlers = loader.discover_and_load(
            patterns,
            base_path=valid_contract_directory,
        )

        # Should find both handlers
        assert len(handlers) == 2
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two"}

    def test_explicit_base_path_with_correlation_id(
        self, valid_contract_directory: Path
    ) -> None:
        """Test explicit base_path works correctly with correlation_id.

        Verifies that both optional parameters can be used together.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            correlation_id=uuid4(),
            base_path=valid_contract_directory,
        )

        # Should find all 3 handlers
        assert len(handlers) == 3

    def test_explicit_base_path_no_matches_returns_empty(self, tmp_path: Path) -> None:
        """Test that explicit base_path with no matches returns empty list.

        Verifies that when base_path is valid but pattern doesn't match
        anything, an empty list is returned (not an error).
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create an empty directory
        empty_dir = tmp_path / "empty_base"
        empty_dir.mkdir()

        loader = HandlerPluginLoader()

        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=empty_dir,
        )

        # Should return empty list when no contracts found
        assert handlers == []

    def test_explicit_base_path_deduplication(
        self, valid_contract_directory: Path
    ) -> None:
        """Test that deduplication works with explicit base_path.

        Verifies that overlapping patterns with explicit base_path still
        result in deduplicated handler loading.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Multiple overlapping patterns
        patterns = [
            "**/handler_contract.yaml",
            "handler1/handler_contract.yaml",
            "*/handler_contract.yaml",
        ]
        handlers = loader.discover_and_load(
            patterns,
            base_path=valid_contract_directory,
        )

        # Should still find only 3 unique handlers (deduplicated)
        assert len(handlers) == 3
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    def test_explicit_base_path_with_absolute_base(
        self, valid_contract_directory: Path
    ) -> None:
        """Test explicit base_path with absolute path.

        Verifies that absolute base paths work correctly for glob resolution.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Ensure we're using an absolute path
        absolute_base = valid_contract_directory.resolve()
        assert absolute_base.is_absolute()

        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=absolute_base,
        )

        # Should find all 3 handlers
        assert len(handlers) == 3

    def test_explicit_base_path_graceful_failure(
        self, mixed_valid_invalid_directory: Path
    ) -> None:
        """Test graceful failure handling with explicit base_path.

        Verifies that invalid contracts are skipped gracefully when using
        explicit base_path, just like when using cwd-based resolution.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=mixed_valid_invalid_directory,
        )

        # Should load only the one valid handler, skipping invalid ones
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"


class TestHandlerPluginLoaderBasePathEdgeCases:
    """Edge case tests for base_path parameter."""

    def test_none_base_path_uses_cwd(
        self,
        valid_contract_directory: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that None base_path falls back to cwd (default behavior).

        Verifies that explicitly passing None for base_path is equivalent
        to not passing it at all - both use cwd for resolution.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the valid contract directory
        monkeypatch.chdir(valid_contract_directory)

        loader = HandlerPluginLoader()

        # Explicitly pass None for base_path
        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=None,
        )

        # Should find all 3 handlers using cwd
        assert len(handlers) == 3

    def test_base_path_with_single_file_pattern(
        self, valid_contract_directory: Path
    ) -> None:
        """Test base_path with a pattern matching a single specific file.

        Verifies that very specific patterns work with explicit base_path.
        """
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Very specific pattern for handler2
        pattern = "handler2/handler_contract.yaml"
        handlers = loader.discover_and_load(
            [pattern],
            base_path=valid_contract_directory,
        )

        # Should find exactly one handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "handler.two"
