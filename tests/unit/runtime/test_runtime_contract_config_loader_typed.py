# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for RuntimeContractConfigLoader.load_from_directory typed output.

Verifies that load_from_directory() invokes model_validate() and returns
list[ModelContractBase] instead of raw dicts. Part of OMN-9747.
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_core.models.contracts.model_contract_base import ModelContractBase
from omnibase_infra.runtime.runtime_contract_config_loader import (
    RuntimeContractConfigLoader,
)

MINIMAL_EFFECT_CONTRACT: dict[str, object] = {
    "name": "test_node",
    "node_type": "EFFECT_GENERIC",
    "contract_version": {"major": 1, "minor": 0, "patch": 0},
    "description": "Test effect node",
    "input_model": "omnibase_core.models.model_input.ModelInput",
    "output_model": "omnibase_core.models.model_output.ModelOutput",
    "io_operations": [{"operation_type": "api_call"}],
}

MINIMAL_EFFECT_CONTRACT_2: dict[str, object] = {
    "name": "test_node_two",
    "node_type": "EFFECT_GENERIC",
    "contract_version": {"major": 1, "minor": 0, "patch": 0},
    "description": "Second test effect node",
    "input_model": "omnibase_core.models.model_input.ModelInput",
    "output_model": "omnibase_core.models.model_output.ModelOutput",
    "io_operations": [{"operation_type": "file_read"}],
}


@pytest.mark.unit
def test_load_from_directory_returns_typed_contracts(
    tmp_path: pytest.TempPathFactory,
) -> None:
    contract_dir = tmp_path / "nodes" / "test_node"
    contract_dir.mkdir(parents=True)
    (contract_dir / "contract.yaml").write_text(yaml.dump(MINIMAL_EFFECT_CONTRACT))

    loader = RuntimeContractConfigLoader()
    results = loader.load_from_directory(tmp_path)

    assert len(results) == 1
    assert isinstance(results[0], ModelContractBase)
    assert results[0].name == "test_node"


@pytest.mark.unit
def test_load_from_directory_multiple_contracts(
    tmp_path: pytest.TempPathFactory,
) -> None:
    for i, contract in enumerate([MINIMAL_EFFECT_CONTRACT, MINIMAL_EFFECT_CONTRACT_2]):
        d = tmp_path / f"node_{i}"
        d.mkdir()
        (d / "contract.yaml").write_text(yaml.dump(contract))

    loader = RuntimeContractConfigLoader()
    results = loader.load_from_directory(tmp_path)

    assert len(results) == 2
    assert all(isinstance(r, ModelContractBase) for r in results)


@pytest.mark.unit
def test_load_from_directory_empty_directory(tmp_path: pytest.TempPathFactory) -> None:
    loader = RuntimeContractConfigLoader()
    results = loader.load_from_directory(tmp_path)
    assert results == []


@pytest.mark.unit
def test_load_from_directory_skips_invalid_yaml(
    tmp_path: pytest.TempPathFactory,
) -> None:
    valid_dir = tmp_path / "valid_node"
    valid_dir.mkdir()
    (valid_dir / "contract.yaml").write_text(yaml.dump(MINIMAL_EFFECT_CONTRACT))

    invalid_dir = tmp_path / "invalid_node"
    invalid_dir.mkdir()
    (invalid_dir / "contract.yaml").write_text(":::invalid yaml:::")

    loader = RuntimeContractConfigLoader()
    results = loader.load_from_directory(tmp_path)

    assert len(results) == 1
    assert isinstance(results[0], ModelContractBase)


@pytest.mark.unit
def test_load_from_directory_skips_non_contract_yamls(
    tmp_path: pytest.TempPathFactory,
) -> None:
    contract_dir = tmp_path / "nodes" / "test_node"
    contract_dir.mkdir(parents=True)
    (contract_dir / "contract.yaml").write_text(yaml.dump(MINIMAL_EFFECT_CONTRACT))
    (contract_dir / "other.yaml").write_text("not: a contract")

    loader = RuntimeContractConfigLoader()
    results = loader.load_from_directory(tmp_path)

    assert len(results) == 1
