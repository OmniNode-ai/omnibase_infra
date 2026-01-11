# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for HandlerPluginLoader.discover_and_load method.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest


class TestHandlerPluginLoaderDiscoverAndLoad:
    """Tests for discover_and_load method."""

    def test_discover_with_relative_glob_pattern(
        self, valid_contract_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test glob pattern matching with relative patterns."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(valid_contract_directory)

        loader = HandlerPluginLoader()

        # Use relative glob pattern to discover contracts
        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load([pattern])

        # Should find all 3 handlers
        assert len(handlers) == 3

    def test_discover_deduplicates_paths(
        self, valid_contract_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that duplicate paths are deduplicated."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(valid_contract_directory)

        loader = HandlerPluginLoader()

        # Use multiple patterns that overlap
        pattern1 = "**/handler_contract.yaml"
        pattern2 = "handler1/handler_contract.yaml"
        pattern3 = "*/handler_contract.yaml"

        handlers = loader.discover_and_load([pattern1, pattern2, pattern3])

        # Should still find only 3 unique handlers (deduplicated)
        assert len(handlers) == 3
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    def test_discover_empty_patterns_raises_error(self) -> None:
        """Test that empty patterns list raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            loader.discover_and_load([])

        # Verify error message indicates empty patterns
        assert "empty" in str(exc_info.value).lower()

    def test_discover_no_matches_returns_empty_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that pattern with no matches returns empty list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(tmp_path)

        loader = HandlerPluginLoader()

        # Pattern that won't match anything
        pattern = "**/nonexistent_file.yaml"
        handlers = loader.discover_and_load([pattern])

        assert handlers == []

    def test_discover_graceful_failure_on_invalid_contracts(
        self, mixed_valid_invalid_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test graceful handling of invalid contracts during discovery."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(mixed_valid_invalid_directory)

        loader = HandlerPluginLoader()

        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load([pattern])

        # Should still load the one valid handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"

    def test_discover_with_correlation_id(
        self, valid_contract_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test discovering with correlation_id parameter."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(valid_contract_directory)

        loader = HandlerPluginLoader()

        pattern = "**/handler_contract.yaml"
        handlers = loader.discover_and_load([pattern], correlation_id=uuid4())

        # Should find all 3 handlers
        assert len(handlers) == 3
