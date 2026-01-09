# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerPluginLoader.

Tests for the HandlerPluginLoader implementation including:
- Single contract loading from a specific path
- Directory-based discovery with recursive scanning
- Glob pattern-based discovery for flexible matching
- Error handling for invalid contracts
- Handler protocol validation

Part of OMN-1132: Handler Plugin Loader implementation.

Related:
    - src/omnibase_infra/runtime/handler_plugin_loader.py
    - src/omnibase_infra/models/runtime/model_loaded_handler.py
    - tests/unit/runtime/test_handler_contract_source.py (reference for patterns)
"""

from __future__ import annotations

from pathlib import Path

import pytest

# =============================================================================
# Constants for Test Contracts
# =============================================================================

VALID_HANDLER_CONTRACT_YAML = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
handler_type: "{handler_type}"
capability_tags:
  - {tag1}
  - {tag2}
"""

MINIMAL_HANDLER_CONTRACT_YAML = """
handler_name: "{handler_name}"
handler_class: "{handler_class}"
"""

HANDLER_CONTRACT_WITHOUT_NAME = """
handler_class: "test.handlers.TestHandler"
handler_type: "compute"
"""

HANDLER_CONTRACT_WITHOUT_CLASS = """
handler_name: "test.handler"
handler_type: "compute"
"""

INVALID_YAML_SYNTAX = """
handler_name: "test.handler"
handler_class: this is not valid yaml: [
    unclosed bracket
"""

EMPTY_CONTRACT_YAML = ""


# =============================================================================
# Mock Handler Class for Testing
# =============================================================================


class MockValidHandler:
    """Mock handler class that implements ProtocolHandler (has describe method)."""

    @classmethod
    def describe(cls) -> dict[str, object]:
        """Describe this handler per ProtocolHandler contract."""
        return {
            "handler_id": "mock.valid.handler",
            "version": "1.0.0",
            "description": "Mock handler for testing",
        }


class MockInvalidHandler:
    """Mock handler class that does NOT implement ProtocolHandler (no describe)."""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_contract_path(tmp_path: Path) -> Path:
    """Create a valid handler contract file.

    Returns:
        Path to the directory containing the valid contract file.
    """
    contract_dir = tmp_path / "valid_handler"
    contract_dir.mkdir(parents=True)
    contract_file = contract_dir / "handler_contract.yaml"
    contract_file.write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="test.valid.handler",
            handler_class=f"{__name__}.MockValidHandler",
            handler_type="compute",
            tag1="auth",
            tag2="validation",
        )
    )
    return contract_file


@pytest.fixture
def valid_contract_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple valid handler contracts.

    Structure:
        tmp_path/
        |-- handler1/
        |   |-- handler_contract.yaml
        |-- handler2/
        |   |-- handler_contract.yaml
        |-- nested/
        |   |-- deep/
        |   |   |-- handler_contract.yaml

    Returns:
        Path to the root directory containing contracts.
    """
    # Handler 1
    handler1_dir = tmp_path / "handler1"
    handler1_dir.mkdir(parents=True)
    (handler1_dir / "handler_contract.yaml").write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="handler.one",
            handler_class=f"{__name__}.MockValidHandler",
            handler_type="compute",
            tag1="compute",
            tag2="sync",
        )
    )

    # Handler 2
    handler2_dir = tmp_path / "handler2"
    handler2_dir.mkdir(parents=True)
    (handler2_dir / "handler_contract.yaml").write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="handler.two",
            handler_class=f"{__name__}.MockValidHandler",
            handler_type="effect",
            tag1="effect",
            tag2="async",
        )
    )

    # Nested handler
    nested_dir = tmp_path / "nested" / "deep"
    nested_dir.mkdir(parents=True)
    (nested_dir / "handler_contract.yaml").write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="handler.nested.deep",
            handler_class=f"{__name__}.MockValidHandler",
            handler_type="compute",
            tag1="nested",
            tag2="deep",
        )
    )

    return tmp_path


@pytest.fixture
def mixed_valid_invalid_directory(tmp_path: Path) -> Path:
    """Create a directory with both valid and invalid contracts.

    Structure:
        tmp_path/
        |-- valid/
        |   |-- handler_contract.yaml  (valid)
        |-- invalid_yaml/
        |   |-- handler_contract.yaml  (malformed YAML)
        |-- missing_class/
        |   |-- handler_contract.yaml  (missing handler_class field)

    Returns:
        Path to the root directory.
    """
    # Valid handler
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir(parents=True)
    (valid_dir / "handler_contract.yaml").write_text(
        VALID_HANDLER_CONTRACT_YAML.format(
            handler_name="valid.handler",
            handler_class=f"{__name__}.MockValidHandler",
            handler_type="compute",
            tag1="valid",
            tag2="test",
        )
    )

    # Invalid YAML syntax
    invalid_yaml_dir = tmp_path / "invalid_yaml"
    invalid_yaml_dir.mkdir(parents=True)
    (invalid_yaml_dir / "handler_contract.yaml").write_text(INVALID_YAML_SYNTAX)

    # Missing handler_class field
    missing_class_dir = tmp_path / "missing_class"
    missing_class_dir.mkdir(parents=True)
    (missing_class_dir / "handler_contract.yaml").write_text(
        HANDLER_CONTRACT_WITHOUT_CLASS
    )

    return tmp_path


@pytest.fixture
def empty_directory(tmp_path: Path) -> Path:
    """Create an empty directory with no contracts.

    Returns:
        Path to the empty directory.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True)
    return empty_dir


# =============================================================================
# Import Tests
# =============================================================================


class TestHandlerPluginLoaderImport:
    """Tests for HandlerPluginLoader import and instantiation."""

    def test_handler_plugin_loader_can_be_imported(self) -> None:
        """HandlerPluginLoader should be importable from omnibase_infra.runtime."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        assert HandlerPluginLoader is not None

    def test_handler_plugin_loader_implements_protocol(self) -> None:
        """HandlerPluginLoader should implement ProtocolHandlerPluginLoader."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader
        from omnibase_infra.runtime.protocol_handler_plugin_loader import (
            ProtocolHandlerPluginLoader,
        )

        loader = HandlerPluginLoader()

        # Protocol compliance via duck typing (ONEX convention)
        assert hasattr(loader, "load_from_contract")
        assert hasattr(loader, "load_from_directory")
        assert hasattr(loader, "discover_and_load")
        assert callable(loader.load_from_contract)
        assert callable(loader.load_from_directory)
        assert callable(loader.discover_and_load)

        # Runtime checkable protocol verification
        assert isinstance(loader, ProtocolHandlerPluginLoader)

    def test_constants_are_exported(self) -> None:
        """Module constants should be exported."""
        from omnibase_infra.runtime.handler_plugin_loader import (
            CONTRACT_YAML_FILENAME,
            HANDLER_CONTRACT_FILENAME,
            MAX_CONTRACT_SIZE,
        )

        assert HANDLER_CONTRACT_FILENAME == "handler_contract.yaml"
        assert CONTRACT_YAML_FILENAME == "contract.yaml"
        assert MAX_CONTRACT_SIZE == 10 * 1024 * 1024  # 10MB


# =============================================================================
# load_from_contract Tests
# =============================================================================


class TestHandlerPluginLoaderLoadFromContract:
    """Tests for load_from_contract method."""

    @pytest.mark.asyncio
    async def test_load_valid_handler_contract(self, valid_contract_path: Path) -> None:
        """Test loading a valid handler contract."""
        from omnibase_infra.models.runtime import ModelLoadedHandler
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        result = await loader.load_from_contract(valid_contract_path)

        # Verify result type
        assert isinstance(result, ModelLoadedHandler)

        # Verify handler metadata
        assert result.handler_name == "test.valid.handler"
        assert result.handler_class == f"{__name__}.MockValidHandler"
        assert result.contract_path == valid_contract_path.resolve()
        assert "auth" in result.capability_tags
        assert "validation" in result.capability_tags

    @pytest.mark.asyncio
    async def test_load_minimal_contract_defaults_handler_type(
        self, tmp_path: Path
    ) -> None:
        """Test that minimal contract without handler_type defaults to EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="minimal.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        result = await loader.load_from_contract(contract_file)

        # Default to EFFECT when handler_type not specified
        assert result.handler_type == EnumHandlerTypeCategory.EFFECT
        assert result.capability_tags == []  # No tags specified

    @pytest.mark.asyncio
    async def test_reject_missing_contract_file(self, tmp_path: Path) -> None:
        """Test error when contract file doesn't exist."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_path = tmp_path / "does_not_exist" / "handler_contract.yaml"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(nonexistent_path)

        # Verify error details via message content
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reject_contract_path_is_directory(self, tmp_path: Path) -> None:
        """Test error when contract path points to a directory, not a file."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        directory = tmp_path / "some_dir"
        directory.mkdir()

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(directory)

        assert "not a file" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reject_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error when contract has invalid YAML syntax."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(INVALID_YAML_SYNTAX)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates YAML parsing failure
        assert "yaml" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reject_empty_contract_file(self, tmp_path: Path) -> None:
        """Test error when contract file is empty."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(EMPTY_CONTRACT_YAML)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates empty contract
        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reject_missing_handler_name(self, tmp_path: Path) -> None:
        """Test error when handler_name field is missing."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(HANDLER_CONTRACT_WITHOUT_NAME)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates missing handler name
        error_msg = str(exc_info.value).lower()
        assert "handler_name" in error_msg or "name" in error_msg

    @pytest.mark.asyncio
    async def test_reject_missing_handler_class(self, tmp_path: Path) -> None:
        """Test error when handler_class field is missing."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(HANDLER_CONTRACT_WITHOUT_CLASS)

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates missing handler class
        assert "handler_class" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reject_invalid_handler_class_path(self, tmp_path: Path) -> None:
        """Test error when handler class cannot be imported."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="invalid.module.handler",
                handler_class="nonexistent.module.path.Handler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates module import failure
        error_msg = str(exc_info.value).lower()
        assert "module" in error_msg or "import" in error_msg

    @pytest.mark.asyncio
    async def test_reject_class_not_found_in_module(self, tmp_path: Path) -> None:
        """Test error when class doesn't exist in module."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        # Use a valid module but nonexistent class
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="missing.class.handler",
                handler_class=f"{__name__}.NonexistentClass",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates class not found
        error_msg = str(exc_info.value).lower()
        assert "class" in error_msg or "not found" in error_msg

    @pytest.mark.asyncio
    async def test_reject_handler_without_describe_method(self, tmp_path: Path) -> None:
        """Test error when handler doesn't implement ProtocolHandler (no describe)."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="invalid.protocol.handler",
                handler_class=f"{__name__}.MockInvalidHandler",
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates protocol validation failure
        error_msg = str(exc_info.value).lower()
        assert "describe" in error_msg or "protocol" in error_msg

    @pytest.mark.asyncio
    async def test_reject_unqualified_class_path(self, tmp_path: Path) -> None:
        """Test error when handler_class is not fully qualified (no dots)."""
        from omnibase_infra.errors import InfraConnectionError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="unqualified.handler",
                handler_class="JustClassName",  # Not fully qualified
            )
        )

        loader = HandlerPluginLoader()

        with pytest.raises(InfraConnectionError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates invalid class path
        error_msg = str(exc_info.value).lower()
        assert "class" in error_msg or "import" in error_msg or "module" in error_msg


# =============================================================================
# load_from_directory Tests
# =============================================================================


class TestHandlerPluginLoaderLoadFromDirectory:
    """Tests for load_from_directory method."""

    @pytest.mark.asyncio
    async def test_load_multiple_handlers_from_directory(
        self, valid_contract_directory: Path
    ) -> None:
        """Test loading multiple handlers from a directory tree."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = await loader.load_from_directory(valid_contract_directory)

        # Should find all 3 handlers (handler1, handler2, nested/deep)
        assert len(handlers) == 3

        # Verify all handler names are present
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    @pytest.mark.asyncio
    async def test_empty_directory_returns_empty_list(
        self, empty_directory: Path
    ) -> None:
        """Test that empty directory returns empty list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = await loader.load_from_directory(empty_directory)

        assert handlers == []

    @pytest.mark.asyncio
    async def test_graceful_failure_continues_on_invalid_contract(
        self, mixed_valid_invalid_directory: Path
    ) -> None:
        """Test that invalid contracts don't stop the whole load."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()
        handlers = await loader.load_from_directory(mixed_valid_invalid_directory)

        # Should still load the one valid handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"

    @pytest.mark.asyncio
    async def test_directory_not_found_raises_error(self, tmp_path: Path) -> None:
        """Test that nonexistent directory raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        nonexistent_dir = tmp_path / "does_not_exist"

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_directory(nonexistent_dir)

        # Verify error message indicates directory not found
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_path_is_file_not_directory_raises_error(
        self, valid_contract_path: Path
    ) -> None:
        """Test that file path instead of directory raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # valid_contract_path is a file, not a directory
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_directory(valid_contract_path)

        # Verify error message indicates path is not a directory
        assert "not a directory" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_loads_both_handler_contract_and_contract_yaml(
        self, tmp_path: Path
    ) -> None:
        """Test that loader finds both handler_contract.yaml and contract.yaml."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create handler_contract.yaml
        handler1_dir = tmp_path / "handler1"
        handler1_dir.mkdir()
        (handler1_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="handler.contract.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        # Create contract.yaml
        handler2_dir = tmp_path / "handler2"
        handler2_dir.mkdir()
        (handler2_dir / "contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="contract.yaml.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handlers = await loader.load_from_directory(tmp_path)

        assert len(handlers) == 2
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.contract.handler", "contract.yaml.handler"}


# =============================================================================
# discover_and_load Tests
# =============================================================================


class TestHandlerPluginLoaderDiscoverAndLoad:
    """Tests for discover_and_load method."""

    @pytest.mark.asyncio
    async def test_discover_with_relative_glob_pattern(
        self, valid_contract_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test glob pattern matching with relative patterns."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(valid_contract_directory)

        loader = HandlerPluginLoader()

        # Use relative glob pattern to discover contracts
        pattern = "**/handler_contract.yaml"
        handlers = await loader.discover_and_load([pattern])

        # Should find all 3 handlers
        assert len(handlers) == 3

    @pytest.mark.asyncio
    async def test_discover_deduplicates_paths(
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

        handlers = await loader.discover_and_load([pattern1, pattern2, pattern3])

        # Should still find only 3 unique handlers (deduplicated)
        assert len(handlers) == 3
        handler_names = {h.handler_name for h in handlers}
        assert handler_names == {"handler.one", "handler.two", "handler.nested.deep"}

    @pytest.mark.asyncio
    async def test_discover_empty_patterns_raises_error(self) -> None:
        """Test that empty patterns list raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.discover_and_load([])

        # Verify error message indicates empty patterns
        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_discover_no_matches_returns_empty_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that pattern with no matches returns empty list."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(tmp_path)

        loader = HandlerPluginLoader()

        # Pattern that won't match anything
        pattern = "**/nonexistent_file.yaml"
        handlers = await loader.discover_and_load([pattern])

        assert handlers == []

    @pytest.mark.asyncio
    async def test_discover_graceful_failure_on_invalid_contracts(
        self, mixed_valid_invalid_directory: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test graceful handling of invalid contracts during discovery."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Change to the directory so relative patterns work
        monkeypatch.chdir(mixed_valid_invalid_directory)

        loader = HandlerPluginLoader()

        pattern = "**/handler_contract.yaml"
        handlers = await loader.discover_and_load([pattern])

        # Should still load the one valid handler
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"


# =============================================================================
# Validation Tests
# =============================================================================


class TestHandlerPluginLoaderValidation:
    """Tests for handler validation logic."""

    def test_validate_handler_implements_protocol(self) -> None:
        """Test protocol validation for handler with describe method."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockValidHandler has describe method
        assert loader._validate_handler_protocol(MockValidHandler) is True

    def test_validate_handler_without_describe_method(self) -> None:
        """Test protocol validation for handler without describe method."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # MockInvalidHandler doesn't have describe method
        assert loader._validate_handler_protocol(MockInvalidHandler) is False

    def test_validate_non_callable_describe_rejected(self) -> None:
        """Test that non-callable describe attribute is rejected."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Create a class with describe as an attribute, not a method
        class HandlerWithNonCallableDescribe:
            describe = "not a method"

        result = loader._validate_handler_protocol(HandlerWithNonCallableDescribe)
        assert result is False

    @pytest.mark.asyncio
    async def test_capability_tags_extracted_correctly(self, tmp_path: Path) -> None:
        """Test that capability tags are extracted from contract."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            VALID_HANDLER_CONTRACT_YAML.format(
                handler_name="tagged.handler",
                handler_class=f"{__name__}.MockValidHandler",
                handler_type="compute",
                tag1="database",
                tag2="caching",
            )
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        assert "database" in handler.capability_tags
        assert "caching" in handler.capability_tags
        assert len(handler.capability_tags) == 2

    @pytest.mark.asyncio
    async def test_handler_type_categories_parsed_correctly(
        self, tmp_path: Path
    ) -> None:
        """Test that handler_type values map to correct enum values."""
        from omnibase_infra.enums import EnumHandlerTypeCategory
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        test_cases = [
            ("compute", EnumHandlerTypeCategory.COMPUTE),
            ("effect", EnumHandlerTypeCategory.EFFECT),
            (
                "nondeterministic_compute",
                EnumHandlerTypeCategory.NONDETERMINISTIC_COMPUTE,
            ),
            # Case insensitive
            ("COMPUTE", EnumHandlerTypeCategory.COMPUTE),
            ("Effect", EnumHandlerTypeCategory.EFFECT),
        ]

        for handler_type_str, expected_enum in test_cases:
            contract_file = tmp_path / f"handler_{handler_type_str}.yaml"
            contract_file.write_text(
                f"""
handler_name: test.handler.{handler_type_str}
handler_class: {__name__}.MockValidHandler
handler_type: {handler_type_str}
"""
            )

            handler = await loader.load_from_contract(contract_file)
            assert handler.handler_type == expected_enum, (
                f"Expected {expected_enum} for '{handler_type_str}', "
                f"got {handler.handler_type}"
            )

    @pytest.mark.asyncio
    async def test_invalid_handler_type_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid handler_type value raises error."""
        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            """
handler_name: invalid.type.handler
handler_class: test.handlers.TestHandler
handler_type: invalid_type
"""
        )

        loader = HandlerPluginLoader()

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            await loader.load_from_contract(contract_file)

        # Verify error message indicates invalid handler type
        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "type" in error_msg


# =============================================================================
# File Size Limit Tests
# =============================================================================


class TestHandlerPluginLoaderFileSizeLimit:
    """Tests for 10MB file size limit enforcement."""

    @pytest.mark.asyncio
    async def test_rejects_file_exceeding_10mb_limit(self, tmp_path: Path) -> None:
        """Test that files exceeding MAX_CONTRACT_SIZE are rejected."""
        from unittest.mock import patch

        from omnibase_infra.errors import ProtocolConfigurationError
        from omnibase_infra.runtime.handler_plugin_loader import (
            MAX_CONTRACT_SIZE,
            HandlerPluginLoader,
        )

        # Create a valid contract file
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="oversized.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        # Mock stat to return oversized file
        oversized_bytes = MAX_CONTRACT_SIZE + 1
        original_stat = Path.stat

        class MockStatResult:
            """Mock stat result with configurable st_size."""

            st_size = oversized_bytes
            st_mode = 0o100644
            st_ino = 1
            st_dev = 1
            st_nlink = 1
            st_uid = 1000
            st_gid = 1000
            st_atime = 0.0
            st_mtime = 0.0
            st_ctime = 0.0

        def mock_stat(self: Path, **kwargs: object) -> object:
            """Mock stat that returns oversized value for contract files."""
            if self.name == "handler_contract.yaml":
                return MockStatResult()
            return original_stat(self, **kwargs)

        loader = HandlerPluginLoader()

        with patch.object(Path, "stat", mock_stat):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                await loader.load_from_contract(contract_file)

        # Verify error message indicates size limit exceeded
        error_msg = str(exc_info.value).lower()
        assert "size" in error_msg or "limit" in error_msg or "exceeds" in error_msg

    @pytest.mark.asyncio
    async def test_accepts_file_under_10mb_limit(self, tmp_path: Path) -> None:
        """Test that files under MAX_CONTRACT_SIZE are accepted."""
        from omnibase_infra.runtime.handler_plugin_loader import (
            MAX_CONTRACT_SIZE,
            HandlerPluginLoader,
        )

        # Create a valid contract file (small, under limit)
        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="normal.size.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        # Verify file is under limit
        actual_size = contract_file.stat().st_size
        assert actual_size < MAX_CONTRACT_SIZE

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        assert handler.handler_name == "normal.size.handler"


# =============================================================================
# Idempotency Tests
# =============================================================================


class TestHandlerPluginLoaderIdempotency:
    """Tests for idempotency of load operations."""

    @pytest.mark.asyncio
    async def test_load_from_contract_is_idempotent(
        self, valid_contract_path: Path
    ) -> None:
        """Test that loading the same contract multiple times works correctly."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        # Load the same contract multiple times
        result1 = await loader.load_from_contract(valid_contract_path)
        result2 = await loader.load_from_contract(valid_contract_path)
        result3 = await loader.load_from_contract(valid_contract_path)

        # All should return equivalent results
        assert result1.handler_name == result2.handler_name == result3.handler_name
        assert result1.handler_class == result2.handler_class == result3.handler_class
        assert result1.handler_type == result2.handler_type == result3.handler_type

    @pytest.mark.asyncio
    async def test_load_from_directory_is_idempotent(
        self, valid_contract_directory: Path
    ) -> None:
        """Test loading from directory multiple times returns consistent results."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        loader = HandlerPluginLoader()

        result1 = await loader.load_from_directory(valid_contract_directory)
        result2 = await loader.load_from_directory(valid_contract_directory)
        result3 = await loader.load_from_directory(valid_contract_directory)

        # All should return same number of handlers
        assert len(result1) == len(result2) == len(result3) == 3

        # Handler names should be consistent
        names1 = {h.handler_name for h in result1}
        names2 = {h.handler_name for h in result2}
        names3 = {h.handler_name for h in result3}

        assert names1 == names2 == names3


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestHandlerPluginLoaderEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handles_single_tag_as_string(self, tmp_path: Path) -> None:
        """Test that single tag specified as string (not list) is handled."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            f"""
handler_name: single.tag.handler
handler_class: {__name__}.MockValidHandler
capability_tags: single-tag
"""
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        assert handler.capability_tags == ["single-tag"]

    @pytest.mark.asyncio
    async def test_filters_non_string_tags(self, tmp_path: Path) -> None:
        """Test that non-string tags are filtered out."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            f"""
handler_name: mixed.tags.handler
handler_class: {__name__}.MockValidHandler
capability_tags:
  - valid-tag
  - 123
  - true
  - another-valid
"""
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        # Only string tags should be included
        assert "valid-tag" in handler.capability_tags
        assert "another-valid" in handler.capability_tags

    @pytest.mark.asyncio
    async def test_handler_name_whitespace_stripped(self, tmp_path: Path) -> None:
        """Test that whitespace in handler_name is stripped."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            f"""
handler_name: "  whitespace.handler  "
handler_class: {__name__}.MockValidHandler
"""
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        assert handler.handler_name == "whitespace.handler"

    @pytest.mark.asyncio
    async def test_accepts_name_field_as_alternative_to_handler_name(
        self, tmp_path: Path
    ) -> None:
        """Test that 'name' field can be used instead of 'handler_name'."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            f"""
name: alternative.name.handler
handler_class: {__name__}.MockValidHandler
"""
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        assert handler.handler_name == "alternative.name.handler"

    @pytest.mark.asyncio
    async def test_accepts_tags_field_as_alternative_to_capability_tags(
        self, tmp_path: Path
    ) -> None:
        """Test that 'tags' field can be used instead of 'capability_tags'."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            f"""
handler_name: alternative.tags.handler
handler_class: {__name__}.MockValidHandler
tags:
  - alt-tag-1
  - alt-tag-2
"""
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        assert "alt-tag-1" in handler.capability_tags
        assert "alt-tag-2" in handler.capability_tags

    @pytest.mark.asyncio
    async def test_loaded_at_timestamp_is_set(self, valid_contract_path: Path) -> None:
        """Test that loaded_at timestamp is set during load."""
        from datetime import UTC, datetime

        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        before_load = datetime.now(UTC)

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(valid_contract_path)

        after_load = datetime.now(UTC)

        assert handler.loaded_at >= before_load
        assert handler.loaded_at <= after_load

    @pytest.mark.asyncio
    async def test_contract_path_is_resolved_to_absolute(self, tmp_path: Path) -> None:
        """Test that contract_path in result is resolved to absolute path."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        contract_file = tmp_path / "handler_contract.yaml"
        contract_file.write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="absolute.path.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handler = await loader.load_from_contract(contract_file)

        # Path should be absolute
        assert handler.contract_path.is_absolute()
        assert handler.contract_path == contract_file.resolve()


# =============================================================================
# Case Sensitivity Tests
# =============================================================================


class TestHandlerPluginLoaderCaseSensitivity:
    """Tests for case-sensitive file discovery."""

    @pytest.mark.asyncio
    async def test_only_discovers_exact_filename_match(self, tmp_path: Path) -> None:
        """Test that only exact filename matches are discovered."""
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader

        # Create correctly named file (should be discovered)
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        (valid_dir / "handler_contract.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="valid.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        # Create incorrectly named files (should NOT be discovered)
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()
        (invalid_dir / "HANDLER_CONTRACT.yaml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="uppercase.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        wrong_ext_dir = tmp_path / "wrong_ext"
        wrong_ext_dir.mkdir()
        (wrong_ext_dir / "handler_contract.yml").write_text(
            MINIMAL_HANDLER_CONTRACT_YAML.format(
                handler_name="wrong.ext.handler",
                handler_class=f"{__name__}.MockValidHandler",
            )
        )

        loader = HandlerPluginLoader()
        handlers = await loader.load_from_directory(tmp_path)

        # Should only find the correctly named file
        assert len(handlers) == 1
        assert handlers[0].handler_name == "valid.handler"
