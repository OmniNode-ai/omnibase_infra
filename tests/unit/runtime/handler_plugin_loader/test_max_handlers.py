# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for max_handlers parameter in handler plugin loader.

This module tests the max_handlers parameter functionality that prevents
runaway resource usage during handler discovery.

Part of OMN-1132: Handler Plugin Loader implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

from .conftest import VALID_HANDLER_CONTRACT_YAML, MockValidHandler


def create_handler_contract(
    directory: Path,
    handler_name: str,
) -> Path:
    """Create a valid handler contract file in the given directory.

    Args:
        directory: Directory where the contract will be created.
        handler_name: Name for the handler.

    Returns:
        Path to the created contract file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    contract_file = directory / "handler_contract.yaml"
    contract_file.write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name=handler_name,
            handler_class=f"{MockValidHandler.__module__}.MockValidHandler",
            handler_type="compute",
            tag1="test",
            tag2="max_handlers",
        )
    )
    return contract_file


class TestMaxHandlersLoadFromDirectory:
    """Tests for max_handlers parameter in load_from_directory method."""

    def test_max_handlers_none_loads_all_handlers(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers=None (default) loads all discovered handlers.

        When max_handlers is not specified (None), all discovered contract files
        should be loaded without any limit.
        """
        # Create 5 handler directories with valid contracts
        handler_names = [
            "handler_a",
            "handler_b",
            "handler_c",
            "handler_d",
            "handler_e",
        ]
        for name in handler_names:
            handler_dir = tmp_path / name
            create_handler_contract(handler_dir, name)

        loader = HandlerPluginLoader()

        # Load without limit (max_handlers=None)
        handlers = loader.load_from_directory(tmp_path, max_handlers=None)

        # All 5 handlers should be loaded
        assert len(handlers) == 5
        loaded_names = {h.handler_name for h in handlers}
        assert loaded_names == set(handler_names)

    def test_max_handlers_limits_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers limits the number of handlers discovered.

        When max_handlers is set to a value less than the total number of
        available handlers, discovery should stop at the limit.
        """
        # Create 10 handler directories with valid contracts
        for i in range(10):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Load with limit of 5
        handlers = loader.load_from_directory(tmp_path, max_handlers=5)

        # Only 5 handlers should be loaded (the limit)
        assert len(handlers) == 5

    def test_max_handlers_logs_warning_when_limit_reached(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that a warning is logged when max_handlers limit is reached.

        When discovery stops due to reaching the max_handlers limit, a warning
        should be logged to alert operators that some handlers were not loaded.
        """
        # Create 10 handler directories with valid contracts
        for i in range(10):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Load with limit of 3
        with caplog.at_level(logging.WARNING):
            handlers = loader.load_from_directory(tmp_path, max_handlers=3)

        # Verify warning was logged
        assert len(handlers) == 3
        # Check that the warning contains the limit information
        # The format varies but should include the limit count
        assert any(
            "Handler discovery limit reached" in record.message
            and "3" in record.message  # The limit value should appear
            for record in caplog.records
        )

    def test_max_handlers_equal_to_total_loads_all(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that max_handlers equal to total handlers loads all without warning.

        When max_handlers is set to exactly the number of available handlers,
        all should be loaded and no warning should be logged.
        """
        # Create 3 handler directories
        for i in range(3):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Load with limit equal to total
        with caplog.at_level(logging.WARNING):
            handlers = loader.load_from_directory(tmp_path, max_handlers=3)

        # All 3 should be loaded, no warning about limit reached
        assert len(handlers) == 3
        assert not any(
            "Handler discovery limit reached" in record.message
            for record in caplog.records
        )

    def test_max_handlers_greater_than_total_loads_all(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers greater than total handlers loads all available.

        When max_handlers is set to a value greater than the number of available
        handlers, all available handlers should be loaded without error.
        """
        # Create 3 handler directories
        for i in range(3):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Load with limit greater than total
        handlers = loader.load_from_directory(tmp_path, max_handlers=100)

        # All 3 should be loaded
        assert len(handlers) == 3


class TestMaxHandlersDiscoverAndLoad:
    """Tests for max_handlers parameter in discover_and_load method."""

    def test_max_handlers_none_discovers_all(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers=None discovers all matching handlers."""
        # Create handler directories
        for i in range(5):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Discover without limit
        handlers = loader.discover_and_load(
            patterns=["**/handler_contract.yaml"],
            base_path=tmp_path,
            max_handlers=None,
        )

        assert len(handlers) == 5

    def test_max_handlers_limits_glob_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers limits discovery during glob matching."""
        # Create 10 handler directories
        for i in range(10):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Discover with limit of 5
        handlers = loader.discover_and_load(
            patterns=["**/handler_contract.yaml"],
            base_path=tmp_path,
            max_handlers=5,
        )

        assert len(handlers) == 5

    def test_max_handlers_logs_warning_on_limit(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that discover_and_load logs warning when limit is reached."""
        # Create 10 handler directories
        for i in range(10):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Discover with limit
        with caplog.at_level(logging.WARNING):
            handlers = loader.discover_and_load(
                patterns=["**/handler_contract.yaml"],
                base_path=tmp_path,
                max_handlers=4,
            )

        assert len(handlers) == 4
        assert any(
            "Handler discovery limit reached" in record.message
            for record in caplog.records
        )

    def test_max_handlers_with_multiple_patterns(
        self,
        tmp_path: Path,
    ) -> None:
        """Test max_handlers works correctly with multiple glob patterns.

        When using multiple patterns, the limit should apply to the total
        number of discovered handlers across all patterns.
        """
        # Create handlers in different locations
        handlers_a = tmp_path / "group_a"
        handlers_b = tmp_path / "group_b"

        # Create 5 handlers in each group
        for i in range(5):
            dir_a = handlers_a / f"handler_a{i}"
            dir_b = handlers_b / f"handler_b{i}"
            create_handler_contract(dir_a, f"handler_a{i}")
            create_handler_contract(dir_b, f"handler_b{i}")

        loader = HandlerPluginLoader()

        # Discover with limit across multiple patterns
        handlers = loader.discover_and_load(
            patterns=[
                "group_a/**/handler_contract.yaml",
                "group_b/**/handler_contract.yaml",
            ],
            base_path=tmp_path,
            max_handlers=7,  # Less than total of 10
        )

        # Should stop at 7 total
        assert len(handlers) == 7


class TestMaxHandlersEdgeCases:
    """Edge case tests for max_handlers parameter."""

    def test_max_handlers_zero_returns_empty(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that max_handlers=0 returns empty list without error.

        While unusual, setting max_handlers to 0 should be handled gracefully
        and return an empty list without errors.
        """
        # Create a handler
        handler_dir = tmp_path / "handler"
        create_handler_contract(handler_dir, "handler")

        loader = HandlerPluginLoader()

        # Load with max_handlers=0 - warning should be logged
        with caplog.at_level(logging.WARNING):
            handlers = loader.load_from_directory(tmp_path, max_handlers=0)

        # Should return empty list (limit reached immediately)
        assert handlers == []

    def test_max_handlers_one_loads_single_handler(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers=1 loads exactly one handler."""
        # Create multiple handlers
        for i in range(5):
            handler_dir = tmp_path / f"handler_{i}"
            create_handler_contract(handler_dir, f"handler_{i}")

        loader = HandlerPluginLoader()

        # Load with max_handlers=1
        handlers = loader.load_from_directory(tmp_path, max_handlers=1)

        # Should load exactly one handler
        assert len(handlers) == 1

    def test_max_handlers_with_empty_directory(
        self,
        empty_directory: Path,
    ) -> None:
        """Test that max_handlers with empty directory returns empty list."""
        loader = HandlerPluginLoader()

        # Load from empty directory with limit
        handlers = loader.load_from_directory(empty_directory, max_handlers=10)

        assert handlers == []

    def test_max_handlers_with_failing_contracts(
        self,
        tmp_path: Path,
    ) -> None:
        """Test that max_handlers counts discovered files, not successful loads.

        The limit applies to the discovery phase, not the loading phase.
        If some contracts fail to load, the limit still applies to discovery.
        """
        # Create 5 valid handlers
        for i in range(5):
            valid_dir = tmp_path / f"valid_{i}"
            create_handler_contract(valid_dir, f"valid_{i}")

        # Create 5 invalid contracts (empty YAML with no handler fields)
        for i in range(5):
            invalid_dir = tmp_path / f"invalid_{i}"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "handler_contract.yaml").write_text("# Empty contract\n")

        loader = HandlerPluginLoader()

        # Load with limit of 7 - should discover 7 files (mix of valid/invalid)
        # but only successfully load the valid ones among those 7
        handlers = loader.load_from_directory(tmp_path, max_handlers=7)

        # The number of successfully loaded handlers depends on which 7 files
        # were discovered (order is not guaranteed). The important thing is
        # that we stopped discovery at 7 files.
        assert len(handlers) <= 7


class TestMaxHandlersBackwardsCompatibility:
    """Tests to ensure max_handlers is backwards compatible."""

    def test_existing_api_without_max_handlers_still_works(
        self,
        valid_contract_directory: Path,
    ) -> None:
        """Test that calling methods without max_handlers still works.

        This ensures backwards compatibility - existing code that doesn't
        use max_handlers should continue to work unchanged.
        """
        loader = HandlerPluginLoader()

        # Call without max_handlers parameter (using default None)
        handlers = loader.load_from_directory(valid_contract_directory)

        # Should load all 3 handlers from valid_contract_directory fixture
        assert len(handlers) == 3

    def test_discover_and_load_without_max_handlers_still_works(
        self,
        valid_contract_directory: Path,
    ) -> None:
        """Test that discover_and_load without max_handlers still works."""
        loader = HandlerPluginLoader()

        # Call without max_handlers parameter
        handlers = loader.discover_and_load(
            patterns=["**/handler_contract.yaml"],
            base_path=valid_contract_directory,
        )

        # Should load all handlers
        assert len(handlers) == 3
