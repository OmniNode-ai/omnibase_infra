# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

import logging
from pathlib import Path

import pytest

from omnibase_infra.runtime.runtime_contract_config_loader import (
    RuntimeContractConfigLoader,
)


@pytest.mark.unit
def test_loader_collects_hook_activations_from_packages(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "pkg" / "contracts"
    contracts_dir.mkdir(parents=True)
    yaml_content = """
hook_activations:
  - hook_bit: WORKTREE_GUARD
    enabled_by_default: true
"""
    (contracts_dir / "hook_activations.yaml").write_text(yaml_content)

    loader = RuntimeContractConfigLoader()
    activations = loader.load_hook_activations_from_path(contracts_dir)
    assert len(activations) == 1
    assert activations[0].hook_bit.name == "WORKTREE_GUARD"


@pytest.mark.unit
def test_loader_returns_empty_when_file_absent(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "pkg" / "contracts"
    contracts_dir.mkdir(parents=True)

    loader = RuntimeContractConfigLoader()
    activations = loader.load_hook_activations_from_path(contracts_dir)
    assert activations == []


@pytest.mark.unit
def test_loader_skips_malformed_hook_activations_yaml(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    contracts_dir = tmp_path / "pkg" / "contracts"
    contracts_dir.mkdir(parents=True)
    (contracts_dir / "hook_activations.yaml").write_text("not: valid: yaml: [")

    loader = RuntimeContractConfigLoader()
    with caplog.at_level(logging.WARNING):
        activations = loader.load_hook_activations_from_path(contracts_dir)
    assert activations == []


@pytest.mark.unit
def test_loader_skips_unknown_hook_bit(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    contracts_dir = tmp_path / "pkg" / "contracts"
    contracts_dir.mkdir(parents=True)
    yaml_content = """
hook_activations:
  - hook_bit: DOES_NOT_EXIST
    enabled_by_default: true
"""
    (contracts_dir / "hook_activations.yaml").write_text(yaml_content)

    loader = RuntimeContractConfigLoader()
    with caplog.at_level(logging.WARNING):
        activations = loader.load_hook_activations_from_path(contracts_dir)
    assert activations == []


@pytest.mark.unit
def test_loader_parses_multiple_activations(tmp_path: Path) -> None:
    contracts_dir = tmp_path / "pkg" / "contracts"
    contracts_dir.mkdir(parents=True)
    yaml_content = """
hook_activations:
  - hook_bit: WORKTREE_GUARD
    enabled_by_default: true
    description: "Enforces worktree discipline"
  - hook_bit: RUFF_FIX
    enabled_by_default: false
    description: "Auto-runs ruff after file writes"
"""
    (contracts_dir / "hook_activations.yaml").write_text(yaml_content)

    loader = RuntimeContractConfigLoader()
    activations = loader.load_hook_activations_from_path(contracts_dir)
    assert len(activations) == 2
    assert activations[1].enabled_by_default is False
