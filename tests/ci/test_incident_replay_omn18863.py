# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Replay the tree that reddened dev, against the real derivation (OMN-18863).

INCIDENT. `omnibase_infra#3821` merged at 2026-09-20T02:25:15Z, squash
`31ad10c847b80c622ff2bc41c9d6d3a060a57d62`. It vendored
`node_projection_lab_lane_health` and added `omninode_internal.lab_lane_health`
to the shipped topology instances. Its own enforcement check was green, because
the pull request body declared `Node-Migration-Source-PR: 2674` and the job
derived from the omnimarket BRANCH those trailers named. The push to `dev`
carries no pull-request payload, derived from the committed pin
`612980ace2218999331cb1d79241e90a26a305e8`, which declared no such relation,
and the same check failed at 02:25:52Z. Every open pull request in the
repository was blocked until an unrelated lane diagnosed it. It was the second
occurrence in nine minutes; `#3795` did the same thing with
`runtime_error_fingerprints` at 23:55:39Z.

THE ARTIFACT IS THE REAL TREE, not a reconstruction. It is
`src/omnibase_infra/topology/instances/local.yaml` read out of the object store
at that exact squash, byte for byte, re-fetchable at the locator recorded in
`tests/incident_replays/registry.yaml`. The shape matters and is why a
hand-written fixture would have been worthless here: the relation appears as
one entry inside `omninode_runtime`'s `omninode_internal` TABLE block among
eleven siblings, and the failure is entirely about whether the derivation can
reproduce THAT list from a given declaration set. A fixture that carried only
the interesting line would have proven nothing about the list.

WHAT IS DRIVEN. The real `derive_table_grants`, the same function the generator
and the enforcement gate call, against the real shipped topology profile. The
only thing varied is the declaration set, which is exactly what varied in the
incident: with the relation declared, the derivation reproduces what the tree
ships; without it, it does not, and the instance file is then unreproducible,
which is the state the push-side check refuses.

THE SECOND DIRECTION IS LOAD-BEARING. A guard that refused every tree would
satisfy the first assertion trivially and red every future merge, so the same
derivation is required to REPRODUCE the captured grant once the declaration is
present. Without that, this file would be a test that the derivation can fail.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    ContractTableDeclaration,
    derive_table_grants,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "omn18863"
    / "local-instance-31ad10c8.yaml.captured"
)
_FIXTURE_SHA256 = "cf13e168ad4583e432a83654ab6ece41b3abbe7bdc0133ceb557e48c67dc27b4"

_RELATION = "lab_lane_health"
_SCHEMA = "omninode_internal"
_ROLE = "omninode_runtime"
_MIGRATION = (
    "docker/migrations/forward/nodes/node_projection_lab_lane_health/"
    "0000_create_lab_lane_health.sql"
)


def _bridge() -> ContractTableDeclaration:
    """The declaration the trailer tree had and the pin did not."""
    return ContractTableDeclaration(
        node=f"replay:{_RELATION}",
        contract_path=Path(_MIGRATION),
        table=ModelDbTableDeclaration(
            name=_RELATION,
            database_ref="application",
            schema=_SCHEMA,
            migration=_MIGRATION,
            access="read_write",
            role=_RELATION,
        ),
    )


def _relations_the_captured_tree_grants() -> set[str]:
    """What the incident's own instance file declares for the internal role."""
    document = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))
    found: set[str] = set()
    for database in document["databases"].values():
        principal = (database.get("principals") or {}).get(_ROLE)
        if not principal:
            continue
        for grant in principal.get("grants", []) or []:
            if grant.get("schema") != _SCHEMA:
                continue
            if str(grant.get("object_type", "")).upper() != "TABLE":
                continue
            found.update(grant.get("objects", []) or [])
    assert found, (
        "the captured instance declares no internal TABLE grants for "
        f"{_ROLE}; the reader is looking in the wrong place and every "
        "assertion built on it would be vacuous"
    )
    return found


def _derived_relations(declarations: list[ContractTableDeclaration]) -> set[str]:
    topology = load_topology_profile("local")
    derived = derive_table_grants(topology, declarations)
    return {
        relation
        for grants in derived.grants.values()
        for grant in grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        for relation in grant.objects
    }


class TestTheArtifactIsTheOneThatFailed:
    def test_the_capture_is_unmodified(self) -> None:
        """A reformatted artifact is no longer the artifact that failed."""
        digest = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
        assert digest == _FIXTURE_SHA256

    def test_it_carries_the_relation_the_push_could_not_derive(self) -> None:
        """The premise. Without this the replay is about some other tree."""
        assert _RELATION in _relations_the_captured_tree_grants()


class TestTheReplay:
    def test_without_the_declaration_the_captured_tree_is_unreproducible(
        self,
    ) -> None:
        """The incident: the pin declared it nowhere, so nothing derived it.

        This is the verdict the trailered pull request never had to face, and
        the one the push to dev faced immediately.
        """
        shipped = _relations_the_captured_tree_grants()
        derived = _derived_relations([])
        assert _RELATION in shipped
        assert _RELATION not in derived, (
            "the derivation produced the relation from an empty declaration "
            "set, so this replay cannot distinguish the incident from a "
            "healthy tree"
        )

    def test_with_the_declaration_the_derivation_reproduces_it(self) -> None:
        """The discriminator, and the reason this is a guard and not a veto.

        A check that refused every tree would pass the assertion above and red
        every future merge. The remedy has to actually work.
        """
        assert _RELATION in _derived_relations([_bridge()])

    def test_the_declaration_adds_only_that_relation(self) -> None:
        """A bridge that widened the grant set would be its own defect."""
        without = _derived_relations([])
        with_bridge = _derived_relations([_bridge()])
        assert with_bridge - without == {_RELATION}
