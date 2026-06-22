# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
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

from pathlib import Path

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
}

# Repo root = five parents up from the scanner module; the test file is deeper,
# so resolve relative to the repo root we already know via the baseline path.
_REPO_ROOT = Path(__file__).resolve().parents[4]


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
    # There are exactly six risk-2 thin handlers accepted as baseline.
    assert len(risk2) == 6
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
