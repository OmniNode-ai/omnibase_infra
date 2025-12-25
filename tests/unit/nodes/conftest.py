# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for orchestrator node tests.

This module provides common fixtures used across multiple orchestrator test files,
extracted to reduce duplication and improve performance through module-level scoping.

Performance Rationale:
    All fixtures use `scope="module"` because:
    1. The contract.yaml file is static during test runs - parsing it once per module
       avoids repeated I/O and YAML parsing overhead.
    2. The node.py source code is static - AST parsing is expensive and should only
       be done once per module.
    3. Module scope ensures fixtures are shared across all tests in a single file
       while still isolating between different test modules.

Fixtures Provided:
    - contract_path: Path to the orchestrator contract.yaml file
    - contract_data: Parsed YAML content from contract.yaml
    - node_module_path: Path to the orchestrator node.py file
    - node_source_code: Raw source code from node.py
    - node_ast: Parsed AST from node.py

Usage:
    These fixtures are automatically discovered by pytest. Import is not needed.

Example::

    def test_contract_has_published_events(contract_data: dict) -> None:
        assert "published_events" in contract_data

Related Tickets:
    - OMN-952: Comprehensive orchestrator tests
    - PR #85: Extract duplicated fixtures to shared conftest.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

# =============================================================================
# Path Constants
# =============================================================================

# Base path to the orchestrator node directory
ORCHESTRATOR_NODE_DIR = Path("src/omnibase_infra/nodes/node_registration_orchestrator")

# Contract file path
ORCHESTRATOR_CONTRACT_PATH = ORCHESTRATOR_NODE_DIR / "contract.yaml"

# Node implementation file path
ORCHESTRATOR_NODE_PATH = ORCHESTRATOR_NODE_DIR / "node.py"


# =============================================================================
# Contract Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def contract_path() -> Path:
    """Return path to the orchestrator contract.yaml file.

    Uses module scope because the path is static - no need to recreate
    the Path object for every test function.

    Returns:
        Path to contract.yaml file.

    Raises:
        pytest.skip: If contract file doesn't exist.

    """
    if not ORCHESTRATOR_CONTRACT_PATH.exists():
        pytest.skip(f"Contract file not found: {ORCHESTRATOR_CONTRACT_PATH}")
    return ORCHESTRATOR_CONTRACT_PATH


@pytest.fixture(scope="module")
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as a dictionary.

    Uses module scope for performance - YAML parsing is done once per module
    rather than for every test function. This significantly reduces test
    execution time when multiple tests need contract data.

    Args:
        contract_path: Path fixture to contract.yaml (auto-injected by pytest).

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        pytest.skip: If contract file doesn't exist.
        pytest.fail: If contract file contains invalid YAML syntax.

    """
    if not contract_path.exists():
        pytest.skip(f"Contract file not found: {contract_path}")

    with contract_path.open(encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in contract file: {e}")


# =============================================================================
# Node Source Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def node_module_path() -> Path:
    """Return path to the orchestrator node.py file.

    Uses module scope because the path is static - no need to recreate
    the Path object for every test function.

    Returns:
        Path to node.py file.

    Raises:
        pytest.skip: If node file doesn't exist, allowing tests to be
            skipped gracefully.

    """
    if not ORCHESTRATOR_NODE_PATH.exists():
        pytest.skip(f"Orchestrator node file not found: {ORCHESTRATOR_NODE_PATH}")
    return ORCHESTRATOR_NODE_PATH


@pytest.fixture(scope="module")
def node_source_code(node_module_path: Path) -> str:
    """Load and return the orchestrator node.py source code.

    Uses module scope for performance - file I/O is done once per module
    rather than for every test function. Multiple AST analysis tests can
    share the same source code string.

    Args:
        node_module_path: Path fixture to node.py (auto-injected by pytest).

    Returns:
        Raw source code as a string.

    """
    return node_module_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def node_ast(node_source_code: str, node_module_path: Path) -> ast.Module:
    """Parse and return the orchestrator node.py AST.

    Uses module scope for performance - AST parsing is computationally
    expensive and should only be done once per module. All tests that
    need to analyze the node's AST can share this parsed tree.

    Args:
        node_source_code: Source code fixture (auto-injected by pytest).
        node_module_path: Path fixture for filename in AST (auto-injected).

    Returns:
        Parsed AST Module object representing the node.py file.

    Note:
        The AST is immutable and safe to share across tests. Tests should
        not attempt to modify the AST tree.

    """
    return ast.parse(node_source_code, filename=str(node_module_path))


# =============================================================================
# Contract-Derived Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def published_events(contract_data: dict) -> list[dict]:
    """Extract published_events list from contract data.

    Uses module scope as it derives from the module-scoped contract_data.

    Args:
        contract_data: Parsed contract dictionary (auto-injected by pytest).

    Returns:
        List of published event dictionaries from the contract.

    """
    return contract_data.get("published_events", [])


@pytest.fixture(scope="module")
def event_types_map(published_events: list[dict]) -> dict[str, dict]:
    """Build a map of event_type -> event definition for quick lookup.

    Uses module scope as it derives from module-scoped published_events.
    This provides O(1) lookup for individual event types during testing.

    Args:
        published_events: List of published events (auto-injected by pytest).

    Returns:
        Dictionary mapping event_type names to their full definitions.
        Events missing the 'event_type' field are silently skipped.

    Note:
        Malformed events (those without an 'event_type' field) are filtered
        out rather than raising a KeyError. This ensures test robustness
        when contract files may have incomplete event definitions during
        development or migration.

    """
    return {
        event_type: event
        for event in published_events
        if (event_type := event.get("event_type")) is not None
    }


@pytest.fixture(scope="module")
def execution_graph_nodes(contract_data: dict) -> list[dict]:
    """Extract execution graph nodes from contract data.

    Uses module scope as it derives from module-scoped contract_data.

    Args:
        contract_data: Parsed contract dictionary (auto-injected by pytest).

    Returns:
        List of node dictionaries from the execution graph.

    Raises:
        pytest.fail: If execution_graph.nodes is not found in contract.

    """
    try:
        return contract_data["workflow_coordination"]["workflow_definition"][
            "execution_graph"
        ]["nodes"]
    except KeyError as e:
        pytest.fail(f"Missing key in contract structure: {e}")


@pytest.fixture(scope="module")
def dependencies(contract_data: dict) -> list[dict]:
    """Extract dependencies list from contract data.

    Uses module scope as it derives from module-scoped contract_data.

    Args:
        contract_data: Parsed contract dictionary (auto-injected by pytest).

    Returns:
        List of dependency dictionaries from the contract.

    Raises:
        pytest.fail: If dependencies section is not found in contract.

    """
    if "dependencies" not in contract_data:
        pytest.fail("Missing 'dependencies' section in contract")
    return contract_data["dependencies"]


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "contract_data",
    "contract_path",
    "dependencies",
    "event_types_map",
    "execution_graph_nodes",
    "node_ast",
    "node_module_path",
    "node_source_code",
    "published_events",
]
