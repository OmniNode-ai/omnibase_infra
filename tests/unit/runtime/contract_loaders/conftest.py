# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared fixtures and constants for contract loader tests.

This module provides reusable test infrastructure for all contract loader tests.

Part of OMN-1316: Contract-driven handler loading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# =============================================================================
# Constants for Test Contracts
# =============================================================================

VALID_HANDLER_ROUTING_CONTRACT_YAML = """
name: "test_orchestrator"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "ModelNodeIntrospectionEvent"
        module: "omnibase_infra.models.registration.model_node_introspection_event"
      handler:
        name: "HandlerNodeIntrospected"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_introspected"
      output_events:
        - ModelNodeRegistrationInitiated
    - event_model:
        name: "ModelRuntimeTick"
        module: "omnibase_infra.runtime.models.model_runtime_tick"
      handler:
        name: "HandlerRuntimeTick"
        module: "omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_runtime_tick"
      output_events:
        - ModelNodeRegistrationAckTimedOut
"""

MINIMAL_HANDLER_ROUTING_CONTRACT_YAML = """
name: "minimal_orchestrator"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "TestEventModel"
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
"""

CONTRACT_WITHOUT_HANDLER_ROUTING_YAML = """
name: "orchestrator_without_routing"
version: "1.0.0"
description: "Contract without handler_routing section"
"""

CONTRACT_WITH_EMPTY_HANDLERS_YAML = """
name: "orchestrator_with_empty_handlers"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers: []
"""

CONTRACT_WITH_INCOMPLETE_HANDLER_YAML = """
name: "orchestrator_with_incomplete_handler"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "TestEventModel"
      handler:
        name: "TestHandler"
    - event_model:
        module: "test.models"
      handler:
        module: "test.handlers"
"""

CONTRACT_WITH_MISSING_HANDLER_NAME_YAML = """
name: "orchestrator_with_missing_handler_name"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "TestEventModel"
        module: "test.models"
      handler:
        module: "test.handlers"
"""

CONTRACT_WITH_MISSING_EVENT_MODEL_NAME_YAML = """
name: "orchestrator_with_missing_event_model_name"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
"""

CONTRACT_WITH_INVALID_ROUTING_STRATEGY_YAML = """
name: "orchestrator_with_invalid_strategy"
version: "1.0.0"
handler_routing:
  routing_strategy: "first_match"
  handlers:
    - event_model:
        name: "TestEventModel"
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
"""

CONTRACT_WITH_UNKNOWN_ROUTING_STRATEGY_YAML = """
name: "orchestrator_with_unknown_strategy"
version: "1.0.0"
handler_routing:
  routing_strategy: "some_unknown_strategy"
  handlers:
    - event_model:
        name: "TestEventModel"
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
"""

INVALID_YAML_SYNTAX = """
name: "broken_contract"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers: [
    unclosed bracket
"""

EMPTY_CONTRACT_YAML = ""

YAML_WITH_ONLY_WHITESPACE = """


"""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_contract_path(tmp_path: Path) -> Path:
    """Create a valid handler routing contract file.

    Returns:
        Path to the contract.yaml file.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(VALID_HANDLER_ROUTING_CONTRACT_YAML)
    return contract_file


@pytest.fixture
def minimal_contract_path(tmp_path: Path) -> Path:
    """Create a minimal valid contract file.

    Returns:
        Path to the contract.yaml file with minimal configuration.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(MINIMAL_HANDLER_ROUTING_CONTRACT_YAML)
    return contract_file


@pytest.fixture
def contract_without_routing_path(tmp_path: Path) -> Path:
    """Create a contract file missing handler_routing section.

    Returns:
        Path to the contract.yaml file without handler_routing.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(CONTRACT_WITHOUT_HANDLER_ROUTING_YAML)
    return contract_file


@pytest.fixture
def contract_with_empty_handlers_path(tmp_path: Path) -> Path:
    """Create a contract file with empty handlers list.

    Returns:
        Path to the contract.yaml file with empty handlers.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(CONTRACT_WITH_EMPTY_HANDLERS_YAML)
    return contract_file


@pytest.fixture
def contract_with_incomplete_handler_path(tmp_path: Path) -> Path:
    """Create a contract file with incomplete handler entries.

    Returns:
        Path to the contract.yaml file with incomplete handlers.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(CONTRACT_WITH_INCOMPLETE_HANDLER_YAML)
    return contract_file


@pytest.fixture
def invalid_yaml_path(tmp_path: Path) -> Path:
    """Create a file with invalid YAML syntax.

    Returns:
        Path to the malformed contract.yaml file.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(INVALID_YAML_SYNTAX)
    return contract_file


@pytest.fixture
def empty_contract_path(tmp_path: Path) -> Path:
    """Create an empty contract file.

    Returns:
        Path to the empty contract.yaml file.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(EMPTY_CONTRACT_YAML)
    return contract_file


@pytest.fixture
def whitespace_only_contract_path(tmp_path: Path) -> Path:
    """Create a contract file with only whitespace.

    Returns:
        Path to the contract.yaml file with only whitespace.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(YAML_WITH_ONLY_WHITESPACE)
    return contract_file


@pytest.fixture
def nonexistent_contract_path(tmp_path: Path) -> Path:
    """Return a path to a nonexistent contract file.

    Returns:
        Path to a contract.yaml file that does not exist.
    """
    return tmp_path / "nonexistent" / "contract.yaml"


@pytest.fixture
def contract_with_invalid_routing_strategy_path(tmp_path: Path) -> Path:
    """Create a contract file with an unimplemented routing strategy.

    Uses "first_match" which was listed in the old constant but never implemented.

    Returns:
        Path to the contract.yaml file with invalid routing strategy.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(CONTRACT_WITH_INVALID_ROUTING_STRATEGY_YAML)
    return contract_file


@pytest.fixture
def contract_with_unknown_routing_strategy_path(tmp_path: Path) -> Path:
    """Create a contract file with a completely unknown routing strategy.

    Returns:
        Path to the contract.yaml file with unknown routing strategy.
    """
    contract_file = tmp_path / "contract.yaml"
    contract_file.write_text(CONTRACT_WITH_UNKNOWN_ROUTING_STRATEGY_YAML)
    return contract_file


@pytest.fixture
def oversized_contract_path(tmp_path: Path) -> Path:
    """Create a contract file that exceeds the maximum allowed size.

    Creates a file that is slightly larger than MAX_CONTRACT_FILE_SIZE_BYTES (10MB + 1 byte).
    This is used to test the file size security control.

    Returns:
        Path to the oversized contract.yaml file.
    """
    from omnibase_infra.runtime.contract_loaders import MAX_CONTRACT_FILE_SIZE_BYTES

    contract_file = tmp_path / "contract.yaml"
    # Create a file that exceeds the limit by 1 byte
    oversized_content = "x" * (MAX_CONTRACT_FILE_SIZE_BYTES + 1)
    contract_file.write_text(oversized_content)
    return contract_file


@pytest.fixture
def contract_at_size_limit_path(tmp_path: Path) -> Path:
    """Create a contract file that is exactly at the maximum allowed size.

    Creates a file that is exactly MAX_CONTRACT_FILE_SIZE_BYTES (10MB).
    This tests the boundary condition where the file should still be accepted.

    Returns:
        Path to the contract.yaml file at the size limit.
    """
    from omnibase_infra.runtime.contract_loaders import MAX_CONTRACT_FILE_SIZE_BYTES

    contract_file = tmp_path / "contract.yaml"
    # Create valid YAML content that is exactly at the limit
    # We'll pad a valid contract with comments to reach the limit
    base_content = """name: "test"
version: "1.0.0"
handler_routing:
  routing_strategy: "payload_type_match"
  handlers:
    - event_model:
        name: "TestEvent"
        module: "test.models"
      handler:
        name: "TestHandler"
        module: "test.handlers"
"""
    # Pad with comment characters to reach exact size
    padding_needed = MAX_CONTRACT_FILE_SIZE_BYTES - len(base_content.encode("utf-8"))
    if padding_needed > 0:
        # Add a comment line with the right amount of padding
        padded_content = base_content + "\n# " + ("x" * (padding_needed - 3))
    else:
        padded_content = base_content
    contract_file.write_text(padded_content)
    return contract_file


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Contract YAML constants
    "CONTRACT_WITH_EMPTY_HANDLERS_YAML",
    "CONTRACT_WITH_INCOMPLETE_HANDLER_YAML",
    "CONTRACT_WITH_INVALID_ROUTING_STRATEGY_YAML",
    "CONTRACT_WITH_MISSING_EVENT_MODEL_NAME_YAML",
    "CONTRACT_WITH_MISSING_HANDLER_NAME_YAML",
    "CONTRACT_WITH_UNKNOWN_ROUTING_STRATEGY_YAML",
    "CONTRACT_WITHOUT_HANDLER_ROUTING_YAML",
    "EMPTY_CONTRACT_YAML",
    "INVALID_YAML_SYNTAX",
    "MINIMAL_HANDLER_ROUTING_CONTRACT_YAML",
    "VALID_HANDLER_ROUTING_CONTRACT_YAML",
    "YAML_WITH_ONLY_WHITESPACE",
    # Fixtures
    "contract_at_size_limit_path",
    "contract_with_empty_handlers_path",
    "contract_with_incomplete_handler_path",
    "contract_with_invalid_routing_strategy_path",
    "contract_with_unknown_routing_strategy_path",
    "contract_without_routing_path",
    "empty_contract_path",
    "invalid_yaml_path",
    "minimal_contract_path",
    "nonexistent_contract_path",
    "oversized_contract_path",
    "valid_contract_path",
    "whitespace_only_contract_path",
]
