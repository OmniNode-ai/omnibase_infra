# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime role may read and write the lane_work_workflow table on every profile (OMN-26708).

omnimarket's node_projection_lane_work upserts one row per workflow into
``omninode_internal.lane_work_workflow`` as ``omninode_runtime``. The schema
grant already exists; the table carries no grant, so the Event Chain Gate
refuses the node. Every shipped profile grants the table SELECT, INSERT and
UPDATE to the runtime role alone, never DELETE (a workflow's row is never
removed) and never to any other principal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

_RUNTIME = "omninode_runtime"
_TABLE = "lane_work_workflow"


def _grants(profile: str) -> list[tuple[str, str, str]]:
    return list(declared_acl(load_topology_profile(profile)).table_grants)


def _declared(grants: list[tuple[str, str, str]], principal: str) -> set[str]:
    return {p for who, rel, p in grants if who == principal and rel == _TABLE}


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_may_read_and_write_the_lane_work_workflow_table(
    profile: str,
) -> None:
    assert _declared(_grants(profile), _RUNTIME) == {"SELECT", "INSERT", "UPDATE"}


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_no_other_principal_holds_a_grant_on_the_table(profile: str) -> None:
    others = {who for who, rel, _ in _grants(profile) if rel == _TABLE} - {_RUNTIME}
    assert others == set(), profile


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_runtime_role_cannot_delete_lane_work_workflow_rows(profile: str) -> None:
    grants = _grants(profile)
    assert _declared(grants, _RUNTIME), profile  # positive control: not vacuous
    assert "DELETE" not in _declared(grants, _RUNTIME), profile


def test_control_removing_the_grant_is_reported_as_missing() -> None:
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        grants = _grants(profile)
        assert _declared(grants, _RUNTIME), profile
        broken = [entry for entry in grants if entry[1] != _TABLE]
        assert _declared(broken, _RUNTIME) == set(), profile


_VENDORED = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_lane_work"
)
_LEDGER = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/_ledger/application-migrations.tsv"
)
_EXPECTED_SHAS = {
    "0000_create_lane_work_workflow.sql": (
        "c9538b91136fb35c651795be2364adc44c8dc8415b103cf40e7cb21257610361"
    ),
    "0001_grant_omninode_runtime_lane_work_workflow.sql": (
        "9a6f991296a50521d7904263147cad585855206f4e773024e33c58c09bb983ef"
    ),
}


def test_both_lane_work_migrations_are_vendored_with_their_ledger_rows() -> None:
    assert sorted(p.name for p in _VENDORED.glob("*.sql")) == sorted(_EXPECTED_SHAS)
    rows = [
        line.split("\t")
        for line in _LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for name, sha in _EXPECTED_SHAS.items():
        assert hashlib.sha256((_VENDORED / name).read_bytes()).hexdigest() == sha
        matching = [
            r for r in rows if r[0] == f"nodes/node_projection_lane_work/{name}"
        ]
        assert len(matching) == 1, name
        assert matching[0][5] == sha, name
