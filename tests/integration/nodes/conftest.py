# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared pytest fixtures for orchestrator integration tests.

This module provides common fixtures used across multiple orchestrator
integration test files, extracted to reduce duplication and ensure
consistency.

Fixtures Provided:
    - simple_mock_container: Basic mock container for orchestrator tests
    - contract_path: Path to orchestrator contract.yaml
    - contract_data: Parsed YAML content from contract.yaml

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
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from unittest.mock import MagicMock


# =============================================================================
# Path Constants
# =============================================================================

ORCHESTRATOR_NODE_DIR = Path("src/omnibase_infra/nodes/node_registration_orchestrator")
ORCHESTRATOR_CONTRACT_PATH = ORCHESTRATOR_NODE_DIR / "contract.yaml"


# =============================================================================
# Container Fixtures
# =============================================================================


@pytest.fixture
def simple_mock_container() -> MagicMock:
    """Create a simple mock ONEX container for orchestrator integration tests.

    This provides a minimal mock container with just the basic
    container.config attribute needed for NodeOrchestrator initialization.

    For integration tests requiring real container behavior, use
    the container_with_registries fixture from tests/conftest.py.

    Returns:
        MagicMock configured with minimal container.config attribute.

    Example::

        def test_orchestrator_creates(simple_mock_container: MagicMock) -> None:
            orchestrator = NodeRegistrationOrchestrator(simple_mock_container)
            assert orchestrator is not None

    """
    from unittest.mock import MagicMock

    container = MagicMock()
    container.config = MagicMock()
    return container


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

    with open(contract_path, encoding="utf-8") as f:
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
    "simple_mock_container",
]
