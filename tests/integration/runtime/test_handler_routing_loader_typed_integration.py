# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for load_and_validate_contract_yaml() against real contracts (OMN-9746).

Validates that the typed contract loading spine works against real on-disk
contract.yaml files, not just synthetic YAML written in tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.enums.enum_node_type import EnumNodeType
from omnibase_infra.runtime.contract_loaders.handler_routing_loader import (
    ModelContractNodeType,
    load_and_validate_contract_yaml,
)
from tests.helpers.path_utils import find_project_root

try:
    _NODES_ROOT = (
        find_project_root(start=Path(__file__).resolve().parent)
        / "src"
        / "omnibase_infra"
        / "nodes"
    )
except RuntimeError:
    _NODES_ROOT = None


@pytest.mark.integration
class TestHandlerRoutingLoaderRealContracts:
    """load_and_validate_contract_yaml() works against real on-disk contract.yaml files."""

    @pytest.mark.skipif(_NODES_ROOT is None, reason="project root not found")
    def test_compute_contract_returns_typed_model(self) -> None:
        contract_path = (
            _NODES_ROOT / "node_savings_estimation_compute" / "contract.yaml"
        )
        assert contract_path.exists(), f"Contract not found: {contract_path}"
        result = load_and_validate_contract_yaml(contract_path)
        assert isinstance(result, ModelContractNodeType)
        assert result.node_type == EnumNodeType.COMPUTE_GENERIC

    @pytest.mark.skipif(_NODES_ROOT is None, reason="project root not found")
    def test_orchestrator_contract_returns_typed_model(self) -> None:
        contract_path = _NODES_ROOT / "node_routing_orchestrator" / "contract.yaml"
        assert contract_path.exists(), f"Contract not found: {contract_path}"
        result = load_and_validate_contract_yaml(contract_path)
        assert isinstance(result, ModelContractNodeType)
        assert result.node_type == EnumNodeType.ORCHESTRATOR_GENERIC

    @pytest.mark.skipif(_NODES_ROOT is None, reason="project root not found")
    def test_result_preserves_raw_dict(self) -> None:
        contract_path = (
            _NODES_ROOT / "node_savings_estimation_compute" / "contract.yaml"
        )
        result = load_and_validate_contract_yaml(contract_path)
        assert isinstance(result.raw, dict)
        assert "node_type" in result.raw
