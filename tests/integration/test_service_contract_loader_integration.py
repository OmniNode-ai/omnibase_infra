# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for load_service_contract against real contract YAMLs.

Verifies that the actual runtime.contract.yaml and event_bus.contract.yaml
files in contracts/services/ parse correctly through ModelFeatureFlagContract.
No external infrastructure required — purely filesystem-based.

Ticket: OMN-9751
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.models.contracts.model_service_contract import (
    ModelFeatureFlagContract,
)
from omnibase_infra.runtime.service_contract_loader import load_service_contract

_CONTRACTS_DIR = Path(__file__).parent.parent.parent / "contracts" / "services"


def test_load_real_runtime_contract_yaml() -> None:
    path = _CONTRACTS_DIR / "runtime.contract.yaml"
    assert path.exists(), f"Expected contract file at {path}"
    m = load_service_contract(path)
    assert isinstance(m, ModelFeatureFlagContract)
    assert m.name == "runtime_service"
    assert isinstance(m.feature_flags, list)


def test_load_real_event_bus_contract_yaml() -> None:
    path = _CONTRACTS_DIR / "event_bus.contract.yaml"
    assert path.exists(), f"Expected contract file at {path}"
    m = load_service_contract(path)
    assert isinstance(m, ModelFeatureFlagContract)
    assert m.name == "event_bus_service"
    assert isinstance(m.feature_flags, list)
