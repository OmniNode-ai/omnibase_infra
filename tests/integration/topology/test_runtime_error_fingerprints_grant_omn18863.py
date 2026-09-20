# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A vendored internal projection derives the grant it ships with (OMN-18863).

`omnibase_infra` `dev` went red on `Application Database Domain Enforcement
(OMN-15361)` twice in nine minutes, at 2026-09-19T23:55:39Z and again at
2026-09-20T02:25:52Z, blocking every open pull request both times. Each was a
relation DECLARED in the shipped topology instances that no pinned omnimarket
contract declared: `omninode_internal.runtime_error_fingerprints`, then
`omninode_internal.lab_lane_health`. Both arrived through a deliberate
infra-first ordering, because omnimarket's node-migration-vendor-parity gate
refuses the producing pull request until the vendored counterpart is on this
repo's `dev`.

**Everything here runs without the cross-repo checkout, on purpose.** An earlier
revision gated these assertions on `.proof-dependencies/omnimarket`, which the
split test job does not provide -- so they were collected and never executed,
and the OMN-18776 skip ratchet refused them by name. It was right to: a test
that cannot run is not coverage, and baselining fourteen of them would have
bought a green at the cost of the thing being tested. The assertions were
rewritten to depend only on committed repository state.

What that leaves is the property the fix actually delivers, which needs no
foreign tree:

1. the shipped instances carry the relation, on every profile that ships
   application-database grants;
2. the derivation maps a declaration of it onto a TABLE grant for the internal
   projection role with exactly the privileges the OMN-15418 validator demands;
   and
3. removing the declaration removes the grant -- the direction that arms the
   OMN-18768 boot crash.

Point 3 is the one worth having. Deleting the declaration from the instances
also makes the enforcement check green, which is why it is the tempting fix and
why it is wrong: `0001` in each lineage still GRANTS the relation, the runtime
resolves a projection binding against the DECLARATION rather than the grant,
and auto-wiring is fail-closed at the process level, so one refused binding
takes the whole runtime down.

Whether the PINNED CONTRACTS have caught up, and therefore whether the interim
supplemental entry must now be deleted, is a different question against a
different input. It lives in `tests/ci/test_supplemental_declaration_expiry_omn18863.py`,
which the enforcement job runs with the checkout present.
"""

from __future__ import annotations

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
    WRITE_PRIVILEGES,
    ContractTableDeclaration,
    derive_table_grants,
    physical_grant_schema_for_table,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = "omninode_internal"

# The topology ROLE the internal projection binding resolves to. Not the same
# string as the wiring binding name, and the grant is keyed by this one: the
# reverse ratchet printed exactly `omninode_runtime -> omninode_internal.<rel>`
# on each occasion the declaration was missing.
_INTERNAL_PROJECTION_ROLE = "omninode_runtime"

# The profiles whose instances carry derived application-database grants. The
# rendered catalogs are projections of these, so asserting here covers both.
_PROFILES = ("local", "onex-dev", "onex-prod")

# Each relation with the migration lineage that creates it. Both are vendored
# into this repository, so both paths are committed state.
_VENDORED_RELATIONS: dict[str, str] = {
    "runtime_error_fingerprints": (
        "docker/migrations/forward/nodes/node_projection_runtime_error_fingerprints/"
        "0000_create_runtime_error_fingerprints.sql"
    ),
    "lab_lane_health": (
        "docker/migrations/forward/nodes/node_projection_lab_lane_health/"
        "0000_create_lab_lane_health.sql"
    ),
}


def _declaration_for(relation: str) -> ContractTableDeclaration:
    """The declaration shape both sources produce for one of these relations.

    Built here rather than read out of a tuple, so the test is indifferent to
    WHICH source currently owns the relation. Ownership moves from the
    supplemental manifest to the omnimarket contract the moment the pin
    advances, and a test bound to one source would go red on the handover and
    take its coverage with it.
    """
    return ContractTableDeclaration(
        node=f"test:{relation}",
        contract_path=Path(_VENDORED_RELATIONS[relation]),
        table=ModelDbTableDeclaration(
            name=relation,
            database_ref="application",
            schema=_SCHEMA,
            migration=_VENDORED_RELATIONS[relation],
            access="read_write",
            role=relation,
        ),
    )


def _instance_text(profile: str) -> str:
    return (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("relation", sorted(_VENDORED_RELATIONS))
class TestTheVendoredLineageIsPresent:
    def test_the_create_migration_is_in_the_tree(self, relation: str) -> None:
        """A grant for a relation this repo does not create is a dangling grant."""
        assert (_REPO_ROOT / _VENDORED_RELATIONS[relation]).is_file()

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_the_shipped_instance_declares_it(
        self, relation: str, profile: str
    ) -> None:
        """The regression, stated as the thing a reader can check by eye.

        Both reds were this assertion failing in production: the relation
        granted by the migration corpus and absent from the declaration the
        runtime reads.
        """
        document = yaml.safe_load(_instance_text(profile))
        declared = yaml.safe_dump(document)
        assert relation in declared, (
            f"{profile} no longer declares {relation}, which the vendored "
            "migration still grants. The runtime resolves a projection binding "
            "against the declaration, and auto-wiring is fail-closed at the "
            "process level, so this is a boot failure for the whole runtime"
        )


@pytest.mark.parametrize("relation", sorted(_VENDORED_RELATIONS))
@pytest.mark.parametrize("profile", _PROFILES)
class TestTheDerivationProducesTheShippedGrant:
    def test_exactly_the_internal_role_receives_it(
        self, relation: str, profile: str
    ) -> None:
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, [_declaration_for(relation)])
        physical_schema = physical_grant_schema_for_table(_SCHEMA, relation)
        principals = {
            principal
            for principal, grants in derived.grants.items()
            for grant in grants
            if grant.object_type is EnumDatabaseGrantObjectType.TABLE
            and grant.schema == physical_schema
            and relation in grant.objects
        }
        assert principals == {_INTERNAL_PROJECTION_ROLE}, (
            f"profile {profile} derives {sorted(principals)} for {relation}; a "
            "miss is a boot failure and an extra principal is over-granting"
        )

    def test_privileges_are_exactly_what_the_validator_demands(
        self, relation: str, profile: str
    ) -> None:
        """No wildcard, no widening. Over-granting is its own defect."""
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, [_declaration_for(relation)])
        seen = 0
        for grants in derived.grants.values():
            for grant in grants:
                if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                    continue
                if relation not in grant.objects:
                    continue
                assert set(grant.privileges) == WRITE_PRIVILEGES
                seen += 1
        assert seen, "no TABLE grant on the relation at all; the check is vacuous"

    def test_nothing_is_left_unmappable(self, relation: str, profile: str) -> None:
        """A residual means the derivation could not place it, which ships nothing."""
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, [_declaration_for(relation)])
        assert derived.unmappable == ()


class TestRemovingTheDeclarationIsWhatBreaksIt:
    """The negative control, and the reason the one-line fix is wrong.

    Without this the suite above passes just as happily on a tree where the
    declaration was deleted and the instances regenerated to match -- exactly
    the state that arms the OMN-18768 boot crash. This asserts the causal
    direction: the grant is present BECAUSE something declares the relation.
    """

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_an_empty_declaration_set_derives_no_grant(self, profile: str) -> None:
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, [])
        granted = {
            relation
            for grants in derived.grants.values()
            for grant in grants
            if grant.object_type is EnumDatabaseGrantObjectType.TABLE
            for relation in grant.objects
            if relation in _VENDORED_RELATIONS
        }
        assert granted == set()
