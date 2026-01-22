# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for HandlerPluginLoader with real handler implementations.

This module validates that HandlerPluginLoader can discover, load, and validate
real handler classes from the omnibase_infra.handlers package. Unlike unit tests
that use mock handlers, these tests verify the loader works correctly with
production handler implementations.

Test Coverage:
- Loading real handlers (HttpRestHandler, HandlerDb, HandlerVault, HandlerConsul)
- Protocol compliance validation for real handlers
- Directory-based discovery with real handler contracts
- Glob pattern-based discovery with real handler contracts

Related:
    - OMN-1132: Handler Plugin Loader implementation
    - src/omnibase_infra/runtime/handler_plugin_loader.py
    - docs/patterns/handler_plugin_loader.md

Note:
    These tests create temporary handler contract YAML files that point to
    real handler classes. They do NOT require external infrastructure (database,
    vault, consul) because they only test handler loading, not handler execution.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumHandlerTypeCategory
from omnibase_infra.runtime.handler_plugin_loader import (
    CONTRACT_YAML_FILENAME,
    HANDLER_CONTRACT_FILENAME,
    HandlerPluginLoader,
)

# =============================================================================
# Constants for Real Handler Class Paths
# =============================================================================

# Real handler class paths from omnibase_infra.handlers
REAL_HANDLER_HTTP_CLASS = "omnibase_infra.handlers.handler_http.HandlerHttpRest"
REAL_HANDLER_DB_CLASS = "omnibase_infra.handlers.handler_db.HandlerDb"
REAL_HANDLER_VAULT_CLASS = "omnibase_infra.handlers.handler_vault.HandlerVault"
REAL_HANDLER_CONSUL_CLASS = "omnibase_infra.handlers.handler_consul.HandlerConsul"

# All real handlers for parametrized tests
REAL_HANDLERS = [
    ("http.rest.handler", REAL_HANDLER_HTTP_CLASS, "effect", ["http", "rest"]),
    ("db.postgres.handler", REAL_HANDLER_DB_CLASS, "effect", ["database", "postgres"]),
    ("vault.secrets.handler", REAL_HANDLER_VAULT_CLASS, "effect", ["vault", "secrets"]),
    (
        "consul.service.handler",
        REAL_HANDLER_CONSUL_CLASS,
        "effect",
        ["consul", "discovery"],
    ),
]

# Handler contract template for real handlers
REAL_HANDLER_CONTRACT_YAML = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "{handler_type}"
capability_tags:
  - {tag1}
  - {tag2}
"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def loader() -> HandlerPluginLoader:
    """Create a fresh HandlerPluginLoader instance."""
    return HandlerPluginLoader()


@pytest.fixture
def mock_container() -> MagicMock:
    """Create a mock ModelONEXContainer for handler instantiation."""
    return MagicMock(spec=ModelONEXContainer)


@pytest.fixture
def http_handler_contract_path(tmp_path: Path) -> Path:
    """Create a valid handler contract pointing to HttpRestHandler.

    Returns:
        Path to the contract file.
    """
    contract_dir = tmp_path / "http_handler"
    contract_dir.mkdir(parents=True)
    contract_file = contract_dir / HANDLER_CONTRACT_FILENAME
    contract_file.write_text(
        REAL_HANDLER_CONTRACT_YAML.format(
            handler_name="http.rest.handler",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            handler_type="effect",
            tag1="http",
            tag2="rest",
        )
    )
    return contract_file


@pytest.fixture
def db_handler_contract_path(tmp_path: Path) -> Path:
    """Create a valid handler contract pointing to HandlerDb.

    Returns:
        Path to the contract file.
    """
    contract_dir = tmp_path / "db_handler"
    contract_dir.mkdir(parents=True)
    contract_file = contract_dir / HANDLER_CONTRACT_FILENAME
    contract_file.write_text(
        REAL_HANDLER_CONTRACT_YAML.format(
            handler_name="db.postgres.handler",
            handler_class=REAL_HANDLER_DB_CLASS,
            handler_type="effect",
            tag1="database",
            tag2="postgres",
        )
    )
    return contract_file


@pytest.fixture
def all_real_handlers_directory(tmp_path: Path) -> Path:
    """Create a directory with contracts for all real handlers.

    Structure:
        tmp_path/
        |-- http/
        |   |-- handler_contract.yaml  (HttpRestHandler)
        |-- db/
        |   |-- handler_contract.yaml  (HandlerDb)
        |-- vault/
        |   |-- handler_contract.yaml  (HandlerVault)
        |-- consul/
        |   |-- handler_contract.yaml  (HandlerConsul)

    Returns:
        Path to the root directory.
    """
    for handler_name, handler_class, handler_type, tags in REAL_HANDLERS:
        # Create directory named after first part of handler name
        dir_name = handler_name.split(".")[0]
        handler_dir = tmp_path / dir_name
        handler_dir.mkdir(parents=True, exist_ok=True)
        (handler_dir / HANDLER_CONTRACT_FILENAME).write_text(
            REAL_HANDLER_CONTRACT_YAML.format(
                handler_name=handler_name,
                handler_class=handler_class,
                handler_type=handler_type,
                tag1=tags[0],
                tag2=tags[1],
            )
        )

    return tmp_path


@pytest.fixture
def nested_real_handlers_directory(tmp_path: Path) -> Path:
    """Create a nested directory structure with real handler contracts.

    Structure:
        tmp_path/
        |-- infra/
        |   |-- http/
        |   |   |-- handler_contract.yaml  (HttpRestHandler)
        |   |-- database/
        |   |   |-- postgres/
        |   |   |   |-- handler_contract.yaml  (HandlerDb)

    Returns:
        Path to the root directory.
    """
    # Nested HTTP handler
    http_dir = tmp_path / "infra" / "http"
    http_dir.mkdir(parents=True)
    (http_dir / HANDLER_CONTRACT_FILENAME).write_text(
        REAL_HANDLER_CONTRACT_YAML.format(
            handler_name="infra.http.handler",
            handler_class=REAL_HANDLER_HTTP_CLASS,
            handler_type="effect",
            tag1="http",
            tag2="client",
        )
    )

    # Deeply nested DB handler
    db_dir = tmp_path / "infra" / "database" / "postgres"
    db_dir.mkdir(parents=True)
    (db_dir / HANDLER_CONTRACT_FILENAME).write_text(
        REAL_HANDLER_CONTRACT_YAML.format(
            handler_name="infra.database.postgres.handler",
            handler_class=REAL_HANDLER_DB_CLASS,
            handler_type="effect",
            tag1="postgres",
            tag2="asyncpg",
        )
    )

    return tmp_path


# =============================================================================
# Test Classes
# =============================================================================


class TestLoadFromContractWithRealHandlers:
    """Integration tests for load_from_contract() with real handler classes.

    These tests verify that HandlerPluginLoader can load actual handler
    implementations from the omnibase_infra.handlers package and correctly
    validate their protocol compliance.
    """

    def test_load_http_rest_handler(
        self, loader: HandlerPluginLoader, http_handler_contract_path: Path
    ) -> None:
        """Verify HttpRestHandler can be loaded from a contract.

        HttpRestHandler is a real infrastructure handler that implements
        the ProtocolHandler interface for HTTP/REST operations.
        """
        result = loader.load_from_contract(http_handler_contract_path)

        assert result.handler_name == "http.rest.handler"
        assert result.handler_class == REAL_HANDLER_HTTP_CLASS
        assert result.handler_type == EnumHandlerTypeCategory.EFFECT
        assert "http" in result.capability_tags
        assert "rest" in result.capability_tags

    def test_load_db_handler(
        self, loader: HandlerPluginLoader, db_handler_contract_path: Path
    ) -> None:
        """Verify HandlerDb can be loaded from a contract.

        HandlerDb is a real infrastructure handler that implements
        the ProtocolHandler interface for PostgreSQL database operations.
        """
        result = loader.load_from_contract(db_handler_contract_path)

        assert result.handler_name == "db.postgres.handler"
        assert result.handler_class == REAL_HANDLER_DB_CLASS
        assert result.handler_type == EnumHandlerTypeCategory.EFFECT
        assert "database" in result.capability_tags
        assert "postgres" in result.capability_tags

    @pytest.mark.parametrize(
        ("handler_name", "handler_class", "handler_type", "tags"),
        REAL_HANDLERS,
        ids=["http", "db", "vault", "consul"],
    )
    def test_load_all_real_handlers_parametrized(
        self,
        loader: HandlerPluginLoader,
        tmp_path: Path,
        handler_name: str,
        handler_class: str,
        handler_type: str,
        tags: list[str],
    ) -> None:
        """Verify all real handlers can be loaded from contracts.

        This parametrized test ensures every production handler in
        omnibase_infra.handlers can be successfully loaded and validated.
        """
        # Create contract for this handler
        contract_dir = tmp_path / handler_name.replace(".", "_")
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / HANDLER_CONTRACT_FILENAME
        contract_file.write_text(
            REAL_HANDLER_CONTRACT_YAML.format(
                handler_name=handler_name,
                handler_class=handler_class,
                handler_type=handler_type,
                tag1=tags[0],
                tag2=tags[1],
            )
        )

        # Load and verify
        result = loader.load_from_contract(contract_file)

        assert result.handler_name == handler_name
        assert result.handler_class == handler_class
        assert result.handler_type == EnumHandlerTypeCategory.EFFECT
        for tag in tags:
            assert tag in result.capability_tags

    def test_loaded_handler_has_valid_contract_path(
        self, loader: HandlerPluginLoader, http_handler_contract_path: Path
    ) -> None:
        """Verify loaded handler records the correct contract path."""
        result = loader.load_from_contract(http_handler_contract_path)

        # Contract path should be absolute and match the input
        assert result.contract_path.is_absolute()
        assert result.contract_path.exists()
        assert result.contract_path.name == HANDLER_CONTRACT_FILENAME


class TestLoadFromDirectoryWithRealHandlers:
    """Integration tests for load_from_directory() with real handler classes.

    These tests verify that HandlerPluginLoader can discover and load
    multiple real handlers from a directory structure.
    """

    def test_load_all_handlers_from_directory(
        self, loader: HandlerPluginLoader, all_real_handlers_directory: Path
    ) -> None:
        """Verify all real handlers can be loaded from a directory.

        Creates contracts for all 4 real handlers and verifies they are
        all discovered and loaded correctly.
        """
        results = loader.load_from_directory(all_real_handlers_directory)

        # Should find all 4 handlers
        assert len(results) == 4

        # Verify each handler was loaded
        handler_names = {r.handler_name for r in results}
        expected_names = {h[0] for h in REAL_HANDLERS}
        assert handler_names == expected_names

    def test_load_handlers_from_nested_directories(
        self, loader: HandlerPluginLoader, nested_real_handlers_directory: Path
    ) -> None:
        """Verify handlers are discovered in nested directory structures.

        The loader should recursively scan directories to find handler
        contracts at any depth level.
        """
        results = loader.load_from_directory(nested_real_handlers_directory)

        # Should find both nested handlers
        assert len(results) == 2

        handler_names = {r.handler_name for r in results}
        assert "infra.http.handler" in handler_names
        assert "infra.database.postgres.handler" in handler_names

    def test_empty_directory_returns_empty_list(
        self, loader: HandlerPluginLoader, tmp_path: Path
    ) -> None:
        """Verify empty directory returns empty handler list."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        results = loader.load_from_directory(empty_dir)

        assert results == []


class TestDiscoverAndLoadWithRealHandlers:
    """Integration tests for discover_and_load() with real handler classes.

    These tests verify glob pattern-based discovery works correctly with
    real handler contracts.
    """

    def test_discover_with_glob_pattern(
        self, loader: HandlerPluginLoader, all_real_handlers_directory: Path
    ) -> None:
        """Verify handlers are discovered using glob patterns."""
        results = loader.discover_and_load(
            patterns=[f"**/{HANDLER_CONTRACT_FILENAME}"],
            base_path=all_real_handlers_directory,
        )

        # Should discover all 4 handlers
        assert len(results) == 4

    def test_discover_with_multiple_patterns(
        self, loader: HandlerPluginLoader, nested_real_handlers_directory: Path
    ) -> None:
        """Verify multiple glob patterns work correctly."""
        results = loader.discover_and_load(
            patterns=[
                f"infra/http/{HANDLER_CONTRACT_FILENAME}",
                f"infra/database/**/{HANDLER_CONTRACT_FILENAME}",
            ],
            base_path=nested_real_handlers_directory,
        )

        # Should find handlers matching both patterns
        assert len(results) == 2

    def test_discover_with_contract_yaml_filename(
        self, loader: HandlerPluginLoader, tmp_path: Path
    ) -> None:
        """Verify loader also discovers contract.yaml files (alternative name).

        The loader supports both handler_contract.yaml and contract.yaml
        as valid contract file names.
        """
        contract_dir = tmp_path / "alt_naming"
        contract_dir.mkdir()
        contract_file = contract_dir / CONTRACT_YAML_FILENAME  # contract.yaml
        contract_file.write_text(
            REAL_HANDLER_CONTRACT_YAML.format(
                handler_name="alt.naming.handler",
                handler_class=REAL_HANDLER_HTTP_CLASS,
                handler_type="effect",
                tag1="alt",
                tag2="naming",
            )
        )

        results = loader.discover_and_load(
            patterns=[f"**/{CONTRACT_YAML_FILENAME}"],
            base_path=tmp_path,
        )

        assert len(results) == 1
        assert results[0].handler_name == "alt.naming.handler"


class TestProtocolComplianceOfRealHandlers:
    """Verify that real handlers implement the required ProtocolHandler interface.

    These tests import the actual handler classes and verify they have all
    the methods required by ProtocolHandler. This is the same validation
    the loader performs, but we test it directly for completeness.

    Required ProtocolHandler methods:
    - handler_type (property)
    - initialize() - async method
    - shutdown() - async method
    - execute() - async method
    - describe() - sync method
    """

    @pytest.mark.parametrize(
        "handler_class_path",
        [
            REAL_HANDLER_HTTP_CLASS,
            REAL_HANDLER_DB_CLASS,
            REAL_HANDLER_VAULT_CLASS,
            REAL_HANDLER_CONSUL_CLASS,
        ],
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handler_has_handler_type_property(
        self, handler_class_path: str
    ) -> None:
        """Verify real handlers have handler_type property."""
        handler_class = self._import_handler_class(handler_class_path)
        assert hasattr(handler_class, "handler_type")

    @pytest.mark.parametrize(
        "handler_class_path",
        [
            REAL_HANDLER_HTTP_CLASS,
            REAL_HANDLER_DB_CLASS,
            REAL_HANDLER_VAULT_CLASS,
            REAL_HANDLER_CONSUL_CLASS,
        ],
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handler_has_initialize_method(self, handler_class_path: str) -> None:
        """Verify real handlers have async initialize() method."""
        handler_class = self._import_handler_class(handler_class_path)
        assert hasattr(handler_class, "initialize")
        assert callable(handler_class.initialize)

    @pytest.mark.parametrize(
        "handler_class_path",
        [
            REAL_HANDLER_HTTP_CLASS,
            REAL_HANDLER_DB_CLASS,
            REAL_HANDLER_VAULT_CLASS,
            REAL_HANDLER_CONSUL_CLASS,
        ],
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handler_has_shutdown_method(self, handler_class_path: str) -> None:
        """Verify real handlers have async shutdown() method."""
        handler_class = self._import_handler_class(handler_class_path)
        assert hasattr(handler_class, "shutdown")
        assert callable(handler_class.shutdown)

    @pytest.mark.parametrize(
        "handler_class_path",
        [
            REAL_HANDLER_HTTP_CLASS,
            REAL_HANDLER_DB_CLASS,
            REAL_HANDLER_VAULT_CLASS,
            REAL_HANDLER_CONSUL_CLASS,
        ],
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handler_has_execute_method(self, handler_class_path: str) -> None:
        """Verify real handlers have async execute() method."""
        handler_class = self._import_handler_class(handler_class_path)
        assert hasattr(handler_class, "execute")
        assert callable(handler_class.execute)

    @pytest.mark.parametrize(
        "handler_class_path",
        [
            REAL_HANDLER_HTTP_CLASS,
            REAL_HANDLER_DB_CLASS,
            REAL_HANDLER_VAULT_CLASS,
            REAL_HANDLER_CONSUL_CLASS,
        ],
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handler_has_describe_method(self, handler_class_path: str) -> None:
        """Verify real handlers have describe() method."""
        handler_class = self._import_handler_class(handler_class_path)
        assert hasattr(handler_class, "describe")
        assert callable(handler_class.describe)

    @pytest.mark.parametrize(
        "handler_class_path",
        [
            REAL_HANDLER_HTTP_CLASS,
            REAL_HANDLER_DB_CLASS,
            REAL_HANDLER_VAULT_CLASS,
            REAL_HANDLER_CONSUL_CLASS,
        ],
        ids=["http", "db", "vault", "consul"],
    )
    def test_real_handler_passes_full_protocol_validation(
        self, loader: HandlerPluginLoader, handler_class_path: str
    ) -> None:
        """Verify real handlers pass the loader's protocol validation.

        This tests the same validation the loader performs internally,
        ensuring all required methods are present.
        """
        handler_class = self._import_handler_class(handler_class_path)

        is_valid, missing_methods = loader._validate_handler_protocol(handler_class)

        assert is_valid, (
            f"Handler {handler_class_path} should pass protocol validation "
            f"but is missing methods: {missing_methods}"
        )
        assert missing_methods == []

    def _import_handler_class(self, class_path: str) -> type:
        """Import a handler class from its fully qualified path.

        Args:
            class_path: Fully qualified class path (e.g., 'module.Class')

        Returns:
            The imported class type.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


class TestRealHandlerInstantiation:
    """Verify that real handlers can be instantiated after loading.

    These tests verify that after loading handler metadata, we can
    actually instantiate the handler classes without errors.

    Note: We do NOT call initialize() as that would require external
    infrastructure. We only test that __init__() works.
    """

    @pytest.mark.parametrize(
        ("handler_name", "handler_class", "handler_type", "tags"),
        REAL_HANDLERS,
        ids=["http", "db", "vault", "consul"],
    )
    def test_loaded_handler_can_be_instantiated(
        self,
        loader: HandlerPluginLoader,
        tmp_path: Path,
        mock_container: MagicMock,
        handler_name: str,
        handler_class: str,
        handler_type: str,
        tags: list[str],
    ) -> None:
        """Verify loaded handler class can be instantiated.

        After loading a handler contract, we should be able to import
        the handler class and create an instance.
        """
        # Create and load contract
        contract_dir = tmp_path / handler_name.replace(".", "_")
        contract_dir.mkdir(parents=True)
        contract_file = contract_dir / HANDLER_CONTRACT_FILENAME
        contract_file.write_text(
            REAL_HANDLER_CONTRACT_YAML.format(
                handler_name=handler_name,
                handler_class=handler_class,
                handler_type=handler_type,
                tag1=tags[0],
                tag2=tags[1],
            )
        )

        result = loader.load_from_contract(contract_file)

        # Import and instantiate the handler class
        module_path, class_name = result.handler_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        handler_class_type = getattr(module, class_name)

        # This should not raise - handlers require container parameter
        handler_instance = handler_class_type(mock_container)

        # Verify instance has expected attributes
        assert hasattr(handler_instance, "handler_type")
        assert hasattr(handler_instance, "describe")

    def test_instantiated_handler_describe_returns_dict(
        self,
        loader: HandlerPluginLoader,
        http_handler_contract_path: Path,
        mock_container: MagicMock,
    ) -> None:
        """Verify instantiated handler's describe() returns a dict.

        The describe() method should return handler metadata as a dict
        without requiring initialization.
        """
        result = loader.load_from_contract(http_handler_contract_path)

        # Import and instantiate
        module_path, class_name = result.handler_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        handler_class_type = getattr(module, class_name)
        handler_instance = handler_class_type(mock_container)

        # describe() should return a dict
        description = handler_instance.describe()
        assert isinstance(description, dict)
        assert "handler_type" in description or "version" in description


class TestCorrelationIdTracking:
    """Verify correlation ID is properly tracked through loading operations."""

    def test_load_from_contract_with_correlation_id(
        self, loader: HandlerPluginLoader, http_handler_contract_path: Path
    ) -> None:
        """Verify correlation_id parameter is accepted and doesn't break loading."""
        correlation_id = UUID("12345678-1234-5678-1234-567812345678")

        # Should not raise - correlation_id is used for logging/tracing
        result = loader.load_from_contract(
            http_handler_contract_path, correlation_id=correlation_id
        )

        assert result.handler_name == "http.rest.handler"

    def test_load_from_directory_with_correlation_id(
        self, loader: HandlerPluginLoader, all_real_handlers_directory: Path
    ) -> None:
        """Verify correlation_id propagates through directory loading."""
        correlation_id = UUID("23456789-2345-6789-2345-678923456789")

        results = loader.load_from_directory(
            all_real_handlers_directory, correlation_id=correlation_id
        )

        assert len(results) == 4

    def test_discover_and_load_with_correlation_id(
        self, loader: HandlerPluginLoader, all_real_handlers_directory: Path
    ) -> None:
        """Verify correlation_id propagates through discovery and loading."""
        correlation_id = UUID("34567890-3456-7890-3456-789034567890")

        results = loader.discover_and_load(
            patterns=[f"**/{HANDLER_CONTRACT_FILENAME}"],
            base_path=all_real_handlers_directory,
            correlation_id=correlation_id,
        )

        assert len(results) == 4
