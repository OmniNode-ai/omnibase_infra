# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for ARCH-004 baseline identity in descriptive worktrees."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    BASELINE_RELATIVE_PATH,
    load_baseline,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VALIDATE_PATH = _REPO_ROOT / "scripts/validate.py"

_ACCEPTED_H2_NODES = {
    "node_chain_orchestrator",
    "node_merge_sweep_workflow_orchestrator",
    "node_registration_orchestrator",
    "node_routing_orchestrator",
    "node_rsd_orchestrator",
    "node_runner_fleet_maintain_orchestrator",
    "node_scope_workflow_orchestrator",
}


@pytest.fixture
def validate_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the script without leaking module or import-path state between tests."""
    spec = importlib.util.spec_from_file_location(
        "omnibase_infra_validate", _VALIDATE_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load validator from {_VALIDATE_PATH}")

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "omnibase_infra_validate", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
@pytest.mark.parametrize(
    "declared_name", ["omnibase_infra", "OmniBase-Infra", "omnibase.infra"]
)
def test_pep503_identity_recognizes_accepted_arch004_entries(
    validate_module: Any, tmp_path: Path, declared_name: str
) -> None:
    """PEP 503 package spellings must map to the underscore baseline keys."""
    descriptive_worktree = tmp_path / "linked-checkout-with-descriptive-name"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        f"[project]\nname = {declared_name!r}\n", encoding="utf-8"
    )

    repo_name = validate_module._canonical_repository_name(descriptive_worktree)

    assert repo_name == "omnibase_infra"

    baseline = load_baseline(_REPO_ROOT / BASELINE_RELATIVE_PATH)
    assert {f"{repo_name}::{node}" for node in _ACCEPTED_H2_NODES}.issubset(baseline)


@pytest.mark.unit
def test_arch004_identity_uses_git_root_from_nested_worktree_directory(
    validate_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ARCH-004 runs from a nested directory against the worktree baseline."""
    monkeypatch.chdir(_REPO_ROOT / "scripts")

    assert validate_module._repository_root(Path.cwd()) == _REPO_ROOT.resolve()

    assert validate_module.run_imperative_orchestrators(
        files=[
            "src/omnibase_infra/nodes/node_chain_orchestrator/handlers/"
            "handler_chain_replay_complete.py"
        ]
    )


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_a_fake_nested_git_marker(
    validate_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A nested filesystem marker cannot replace the validator's Git root."""
    fake_checkout = tmp_path / "fake-checkout"
    (fake_checkout / ".git").mkdir(parents=True)
    (fake_checkout / "pyproject.toml").write_text(
        "[project]\nname = 'omnibase_infra'\n", encoding="utf-8"
    )
    monkeypatch.chdir(fake_checkout)

    assert validate_module._repository_root(Path.cwd()) is None


@pytest.mark.unit
def test_arch004_identity_fails_closed_without_project_name(
    validate_module: Any, tmp_path: Path
) -> None:
    """No worktree basename fallback is allowed when repository metadata is absent."""
    descriptive_worktree = tmp_path / "omnibase_infra-ticket-description"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        "[project]\nversion = '0.0.0'\n", encoding="utf-8"
    )

    assert validate_module._canonical_repository_name(descriptive_worktree) is None


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_malformed_metadata(
    validate_module: Any, tmp_path: Path
) -> None:
    """Malformed TOML must not produce a guessed architecture baseline key."""
    descriptive_worktree = tmp_path / "malformed-metadata-worktree"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").write_text(
        "[project\nname = 'omnibase_infra'\n", encoding="utf-8"
    )

    assert validate_module._canonical_repository_name(descriptive_worktree) is None


@pytest.mark.unit
def test_arch004_identity_fails_closed_for_unreadable_metadata(
    validate_module: Any, tmp_path: Path
) -> None:
    """An unreadable metadata path must not fall back to a directory basename."""
    descriptive_worktree = tmp_path / "unreadable-metadata-worktree"
    descriptive_worktree.mkdir()
    (descriptive_worktree / "pyproject.toml").mkdir()

    assert validate_module._canonical_repository_name(descriptive_worktree) is None
