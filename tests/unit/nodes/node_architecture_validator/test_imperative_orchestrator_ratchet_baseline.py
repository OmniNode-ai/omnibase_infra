# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the ARCH-004 imperative-orchestrator ratchet baseline (OMN-13485).

Covers:
    * the ``accepted_rationale`` optional baseline field round-trips through the
      ``BaselineEntry`` model and is only serialized when present, and
    * the committed baseline file carries the per-node owner-ticket
      re-attribution from OMN-13485 (each hard-fail points at its own
      decomposition ticket, not the catch-all OMN-13471).
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    BASELINE_RELATIVE_PATH,
    BaselineEntry,
    _baseline_key,
    load_baseline,
)

# Per-node owner-ticket attribution introduced by OMN-13485 (B0 re-point).
EXPECTED_OWNER_TICKETS: dict[str, str] = {
    "node_delegation_orchestrator": "OMN-13471",
    "pr_lifecycle_orchestrator": "OMN-13487",
    "autopilot_orchestrator": "OMN-13488",
    "node_chain_orchestrator": "OMN-13489",
    "node_merge_sweep_workflow_orchestrator": "OMN-13490",
    "node_registration_orchestrator": "OMN-13491",
    "node_routing_orchestrator": "OMN-13492",
    "node_rsd_orchestrator": "OMN-13493",
    "node_scope_workflow_orchestrator": "OMN-13494",
    "node_runner_fleet_maintain_orchestrator": "OMN-13942",
}

# Repo root = five parents up from the scanner module; the test file is deeper,
# so resolve relative to the repo root we already know via the baseline path.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_validate_module() -> ModuleType:
    """Load the executable validator script for its worktree identity helper."""
    script_path = _REPO_ROOT / "scripts" / "validate.py"
    spec = importlib.util.spec_from_file_location("validate_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_git(directory: Path, *args: str) -> None:
    """Run a Git setup command for an isolated temporary worktree."""
    subprocess.run(
        ["git", *args],
        check=True,
        cwd=directory,
        capture_output=True,
        text=True,
    )


def _create_linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    """Create canonical and arbitrarily named linked Git worktrees."""
    canonical_checkout = tmp_path / "omnibase_infra"
    canonical_checkout.mkdir()
    _run_git(canonical_checkout, "init")
    _run_git(canonical_checkout, "config", "user.email", "validator@example.test")
    _run_git(canonical_checkout, "config", "user.name", "Validator Test")
    (canonical_checkout / "README.md").write_text("fixture\n", encoding="utf-8")
    _run_git(canonical_checkout, "add", "README.md")
    _run_git(canonical_checkout, "commit", "-m", "fixture")

    linked_worktree = tmp_path / "arbitrarily-named-linked-worktree"
    _run_git(
        canonical_checkout,
        "worktree",
        "add",
        "-b",
        "linked-worktree",
        str(linked_worktree),
        "HEAD",
    )
    return canonical_checkout, linked_worktree


def _load_live_baseline() -> dict[str, BaselineEntry]:
    return load_baseline(_REPO_ROOT / BASELINE_RELATIVE_PATH)


@pytest.mark.unit
def test_accepted_rationale_round_trips() -> None:
    entry = BaselineEntry(
        repo="omnibase_infra",
        node="node_x",
        max_handler_path="src/x.py",
        line_count=70,
        risk_score=2,
        finding_codes=("H2",),
        owner_ticket="OMN-13490",
        accepted_rationale="thin decision handler, no decorative FSM",
    )
    restored = BaselineEntry.from_dict(entry.to_dict())
    assert restored == entry
    assert restored.accepted_rationale == "thin decision handler, no decorative FSM"


@pytest.mark.unit
def test_accepted_rationale_omitted_when_empty() -> None:
    entry = BaselineEntry(
        repo="omnibase_infra",
        node="node_x",
        max_handler_path="src/x.py",
        line_count=70,
        risk_score=2,
        finding_codes=("H2",),
        owner_ticket="OMN-13490",
    )
    assert "accepted_rationale" not in entry.to_dict()
    # Default is empty string, not None.
    assert entry.accepted_rationale == ""


@pytest.mark.unit
def test_live_baseline_owner_tickets_repointed() -> None:
    baseline = _load_live_baseline()
    by_node = {entry.node: entry for entry in baseline.values()}
    assert set(by_node) == set(EXPECTED_OWNER_TICKETS)
    for node, expected_ticket in EXPECTED_OWNER_TICKETS.items():
        assert by_node[node].owner_ticket == expected_ticket, (
            f"{node} owner_ticket should be {expected_ticket}, "
            f"got {by_node[node].owner_ticket}"
        )


@pytest.mark.unit
def test_live_baseline_risk2_entries_carry_accepted_rationale() -> None:
    baseline = _load_live_baseline()
    risk2 = [e for e in baseline.values() if e.risk_score == 2]
    # There are exactly seven risk-2 thin handlers accepted as baseline
    # (six from OMN-13485 plus node_runner_fleet_maintain_orchestrator, OMN-13942).
    assert len(risk2) == 7
    for entry in risk2:
        assert entry.accepted_rationale, (
            f"{entry.node}: risk-2 baseline entry must carry an accepted_rationale"
        )


@pytest.mark.unit
def test_every_live_baseline_entry_has_owner_ticket() -> None:
    baseline = _load_live_baseline()
    assert baseline, "baseline must not be empty"
    for key, entry in baseline.items():
        assert key == _baseline_key(entry.repo, entry.node)
        assert entry.owner_ticket, f"{entry.node} missing owner_ticket"


@pytest.mark.unit
def test_canonical_repo_identity_matches_arbitrarily_named_linked_worktree(
    tmp_path: Path,
) -> None:
    """A linked worktree must use the canonical baseline key, not its path name."""
    canonical_checkout, linked_worktree = _create_linked_worktree(tmp_path)
    validate_module = _load_validate_module()

    canonical_identity = validate_module.resolve_canonical_repo_identity(
        canonical_checkout
    )
    linked_identity = validate_module.resolve_canonical_repo_identity(linked_worktree)

    assert canonical_identity == "omnibase_infra"
    assert linked_identity == canonical_identity
    assert _baseline_key(linked_identity, "node_chain_orchestrator") == _baseline_key(
        canonical_identity, "node_chain_orchestrator"
    )


@pytest.mark.unit
def test_canonical_repo_identity_rejects_non_git_directory(tmp_path: Path) -> None:
    """Missing Git topology must fail closed rather than use a path basename."""
    validate_module = _load_validate_module()

    with pytest.raises(RuntimeError, match="Cannot establish canonical Git identity"):
        validate_module.resolve_canonical_repo_identity(tmp_path)
