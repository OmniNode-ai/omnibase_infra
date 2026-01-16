# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared fixtures for ContractHandlerDiscovery tests.

Part of OMN-1133: Handler Discovery Service implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime import (
    ContractHandlerDiscovery,
    HandlerPluginLoader,
    RegistryProtocolBinding,
)

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

INVALID_YAML_SYNTAX = """
handler_name: "test.handler"
handler_class: this is not valid yaml: [
    unclosed bracket
"""

HANDLER_CONTRACT_WITHOUT_CLASS = """
handler_name: "test.handler"
handler_type: "compute"
"""


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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler_registry() -> RegistryProtocolBinding:
    """Create a fresh, isolated handler registry for each test.

    This fixture uses pytest's default function scope, meaning each test
    gets its own fresh RegistryProtocolBinding instance. This ensures:
    1. No test pollution - registrations from one test don't affect another
    2. Predictable state - each test starts with an empty registry
    3. Isolation - tests can run in any order without side effects

    Note: RegistryProtocolBinding is NOT a singleton in this test context.
    Each call creates a new instance with empty registration state.

    Returns:
        A new, empty RegistryProtocolBinding instance.
    """
    return RegistryProtocolBinding()


@pytest.fixture
def plugin_loader() -> HandlerPluginLoader:
    """Create a handler plugin loader for testing.

    Returns:
        A HandlerPluginLoader instance.
    """
    return HandlerPluginLoader()


@pytest.fixture
def discovery_service(
    plugin_loader: HandlerPluginLoader,
    handler_registry: RegistryProtocolBinding,
) -> ContractHandlerDiscovery:
    """Create a ContractHandlerDiscovery instance for testing.

    This fixture composes the plugin_loader and handler_registry fixtures,
    so each test gets a fully isolated discovery service with:
    - Fresh plugin loader (no cached contracts from other tests)
    - Fresh handler registry (no registered handlers from other tests)

    Returns:
        A ContractHandlerDiscovery configured with fresh, isolated dependencies.
    """
    return ContractHandlerDiscovery(
        plugin_loader=plugin_loader,
        handler_registry=handler_registry,
    )


@pytest.fixture
def valid_contract_path(tmp_path: Path) -> Path:
    """Create a valid handler contract file.

    Returns:
        Path to the contract file.
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

    return tmp_path


@pytest.fixture
def mixed_valid_invalid_directory(tmp_path: Path) -> Path:
    """Create a directory with both valid and invalid contracts.

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
