# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared fixtures and constants for HandlerPluginLoader tests.

This module provides reusable test infrastructure for all handler plugin loader tests.

Part of OMN-1132: Handler Plugin Loader implementation.
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
handler_type: "compute"
"""

MINIMAL_CONTRACT_WITHOUT_HANDLER_TYPE = """
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
# Mock Handler Classes for Testing
# =============================================================================


class MockValidHandler:
    """Mock handler class that implements ProtocolHandler.

    Implements all 5 required protocol methods:
    - handler_type (property): Returns handler type identifier
    - initialize(): Async method for connection setup
    - shutdown(): Async method for resource cleanup
    - execute(): Async method for operation execution
    - describe(): Sync method for introspection

    Note: health_check() is optional per ProtocolHandler protocol and is not
    included here to match the existing handlers (HandlerHttp, HandlerDb, etc.)
    which also don't implement it.
    """

    @property
    def handler_type(self) -> str:
        """Return handler type identifier."""
        return "mock"

    async def initialize(self, config: dict[str, object]) -> None:
        """Initialize handler (mock implementation)."""

    async def shutdown(self, timeout_seconds: float = 30.0) -> None:
        """Shutdown handler (mock implementation)."""

    async def execute(
        self,
        request: object,
        operation_config: object,
    ) -> object:
        """Execute operation (mock implementation)."""
        return {}

    @classmethod
    def describe(cls) -> dict[str, object]:
        """Describe this handler per ProtocolHandler contract."""
        return {
            "handler_id": "mock.valid.handler",
            "version": "1.0.0",
            "description": "Mock handler for testing",
        }


class MockInvalidHandler:
    """Mock handler class that does NOT implement ProtocolHandler.

    Missing all required protocol methods to test validation rejection.
    """


class MockPartialHandler:
    """Mock handler with only describe() method.

    This tests that validation rejects handlers that only implement
    describe() but are missing other required methods (handler_type,
    initialize, shutdown, execute).
    """

    def describe(self) -> dict[str, object]:
        """Describe this handler (only method implemented)."""
        return {"handler_id": "mock.partial.handler"}


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
