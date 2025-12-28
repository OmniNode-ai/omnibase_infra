# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for orchestrator integration tests.

This module provides common fixtures used across multiple orchestrator
integration test files, extracted to reduce duplication and ensure
consistency.

Fixtures Provided:
    - contract_path: Path to orchestrator contract.yaml
    - contract_data: Parsed YAML content from contract.yaml

Note:
    simple_mock_container is now provided by tests/conftest.py.
    Use it for basic orchestrator tests that only need container.config.

Usage:
    These fixtures are automatically discovered by pytest. Import is not needed.

Example::

    def test_orchestrator_with_container(simple_mock_container: MagicMock) -> None:
        orchestrator = NodeRegistrationOrchestrator(simple_mock_container)
        assert orchestrator is not None

Related:
    - tests/unit/nodes/conftest.py: Similar fixtures for unit tests
    - tests/conftest.py: Higher-level container fixtures with real wiring
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# =============================================================================
# Path Constants
# =============================================================================

ORCHESTRATOR_NODE_DIR = Path("src/omnibase_infra/nodes/node_registration_orchestrator")
ORCHESTRATOR_CONTRACT_PATH = ORCHESTRATOR_NODE_DIR / "contract.yaml"


# =============================================================================
# Contract Fixtures
# =============================================================================


@pytest.fixture
def contract_path() -> Path:
    """Return path to contract.yaml.

    Returns:
        Path to the orchestrator contract.yaml file.

    Raises:
        pytest.skip: If contract file doesn't exist.
    """
    if not ORCHESTRATOR_CONTRACT_PATH.exists():
        pytest.skip(f"Contract file not found: {ORCHESTRATOR_CONTRACT_PATH}")
    return ORCHESTRATOR_CONTRACT_PATH


@pytest.fixture
def contract_data(contract_path: Path) -> dict:
    """Load and return contract.yaml as dict.

    Args:
        contract_path: Path fixture to contract.yaml (auto-injected by pytest).

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        pytest.skip: If contract file doesn't exist.
        pytest.fail: If contract file contains invalid YAML.
    """
    if not contract_path.exists():
        pytest.skip(f"Contract file not found: {contract_path}")

    with contract_path.open(encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in contract file: {e}")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "contract_data",
    "contract_path",
]
