# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for ARCH-004 baseline identity in descriptive worktrees."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    BASELINE_RELATIVE_PATH,
    load_baseline,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VALIDATE_PATH = _REPO_ROOT / "scripts/validate.py"
_SPEC = importlib.util.spec_from_file_location(
    "omnibase_infra_validate", _VALIDATE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load validator from {_VALIDATE_PATH}")
validate = importlib.util.module_from_spec(_SPEC)
sys.modules["omnibase_infra_validate"] = validate
_SPEC.loader.exec_module(validate)

_ACCEPTED_H2_NODES = {
    "node_chain_orchestrator",
    "node_merge_sweep_workflow_orchestrator",
    "node_registration_orchestrator",
    "node_routing_orchestrator",
    "node_rsd_orchestrator",
    "node_runner_fleet_maintain_orchestrator",
    "node_scope_workflow_orchestrator",
}


@pytest.mark.unit
def test_descriptive_worktree_identity_recognizes_accepted_arch004_entries() -> None:
    """A linked worktree basename must not replace its committed repo identity."""
    repo_name = validate._canonical_repository_name(_REPO_ROOT)

    assert repo_name == "omnibase_infra"
    assert _REPO_ROOT.name != repo_name

    baseline = load_baseline(_REPO_ROOT / BASELINE_RELATIVE_PATH)
    assert {f"{repo_name}::{node}" for node in _ACCEPTED_H2_NODES}.issubset(baseline)


@pytest.mark.unit
def test_arch004_identity_fails_closed_without_project_name(tmp_path: Path) -> None:
    """No worktree basename fallback is allowed when repository metadata is absent."""
    descriptive_worktree = tmp_path / "omnibase_infra-ticket-description"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        "[project]\nversion = '0.0.0'\n", encoding="utf-8"
    )

    assert validate._canonical_repository_name(descriptive_worktree) is None
