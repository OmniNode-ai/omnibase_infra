# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The vendored fingerprints relation derives the grant it is shipped with (OMN-18863).

`omnibase_infra` `dev` went red on `Application Database Domain Enforcement
(OMN-15361)` at 2026-09-19T23:55:39Z, blocking every open pull request, because
`omninode_internal.runtime_error_fingerprints` was DECLARED in the shipped
topology instances while no contract in the pinned omnimarket set declared it.
OMN-18770 vendored the relation's create and grant migrations into this repo
first, deliberately: omnimarket's node-migration-vendor-parity gate refuses the
producing pull request until the vendored counterpart is on this repo's `dev`.

These tests bind the supplemental declaration end to end, against the real
shipped topology profiles rather than a fixture, so the three things that have
to stay true stay true together:

1. the manifest entry exists and names the migration that actually creates the
   relation, so it cannot drift into naming nothing;
2. the derivation turns it into the grant the shipped instances carry, for the
   internal projection principal, with exactly the privileges the OMN-15418
   validator demands and no more; and
3. removing it is what deletes the shipped grant -- the direction that arms the
   OMN-18768 boot crash, asserted here as a property of the derivation rather
   than left as a claim in a comment.

Point 3 is the one worth having. The tempting fix for the red was to delete the
declaration from the instances, and the reason that is wrong is not local to any
one file: `0001` in this lineage still GRANTS the relation, the runtime resolves
a projection binding against the DECLARATION rather than the grant, and
auto-wiring is fail-closed at the process level, so one refused binding takes
the whole runtime down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    WRITE_PRIVILEGES,
    ContractTableDeclaration,
    derive_table_grants,
    load_contract_declarations,
    physical_grant_schema_for_table,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]

_RELATION = "runtime_error_fingerprints"
_SCHEMA = "omninode_internal"
_CREATE_MIGRATION = (
    "docker/migrations/forward/nodes/node_projection_runtime_error_fingerprints/"
    "0000_create_runtime_error_fingerprints.sql"
)
# The profiles whose instances carry derived application-database grants. The
# rendered catalogs are projections of these; asserting here covers both.
_PROFILES = ("local", "onex-dev", "onex-prod")

# The topology ROLE the internal projection binding resolves to. Not the same
# string as the wiring binding name, and the grant is keyed by this one -- see
# the docstring on the principal assertion below.
_INTERNAL_PROJECTION_ROLE = "omninode_runtime"


_PINNED_CONTRACTS = (
    _REPO_ROOT / ".proof-dependencies" / "omnimarket" / "src" / "omnimarket" / "nodes"
)

_SKIP_REASON = (
    "requires the pinned omnimarket checkout at .proof-dependencies/omnimarket, "
    "which the OMN-15361 enforcement job provides and a bare local run does not"
)


def _fingerprints_declarations() -> list[ContractTableDeclaration]:
    """Every declaration of this relation, from whichever source now owns it.

    OMN-18863 SOURCE HANDOVER. This relation was declared by a supplemental
    ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry during the infra-first window,
    and is declared by the omnimarket contract itself now that the pin has
    advanced to ``ac35d56338b3``. The hand entry was deleted in that same
    change, on the expiry test's instruction.

    These assertions are about the RELATION, not about which tuple happens to
    carry it, so they read both sources and are indifferent to the handover.
    That is deliberate: a test bound to the interim source would have had to be
    deleted alongside the entry, taking the grant coverage with it at exactly
    the moment the ownership changed.
    """
    declarations = [
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == _RELATION
    ]
    if _PINNED_CONTRACTS.is_dir():
        declarations.extend(
            declaration
            for declaration in load_contract_declarations(_PINNED_CONTRACTS)
            if declaration.table.name == _RELATION
        )
    return declarations


@pytest.mark.skipif(not _PINNED_CONTRACTS.is_dir(), reason=_SKIP_REASON)
class TestTheManifestEntry:
    def test_the_relation_is_declared_somewhere(self) -> None:
        """Exactly one owner. Two would mean the hand entry outlived the pin."""
        assert len(_fingerprints_declarations()) == 1

    def test_it_names_the_migration_that_creates_the_relation(self) -> None:
        """A declaration pointing at nothing is a declaration that rots."""
        table = _fingerprints_declarations()[0].table
        assert table.schema == _SCHEMA
        assert (_REPO_ROOT / _CREATE_MIGRATION).is_file(), (
            "the declared create migration is not in the vendored tree; the "
            "file moved or was removed without this declaration following it"
        )

    def test_write_access_matches_what_the_grant_migration_delivers(self) -> None:
        """`0001` grants SELECT, INSERT, UPDATE, so `read` alone would understate it.

        The sequence arm derives its USAGE requirement from a declared INSERT,
        so a narrower access mode here silently drops that too.
        """
        table = _fingerprints_declarations()[0].table
        assert table.access == "read_write"


@pytest.mark.skipif(not _PINNED_CONTRACTS.is_dir(), reason=_SKIP_REASON)
@pytest.mark.parametrize("profile", _PROFILES)
class TestTheDerivationProducesTheShippedGrant:
    def test_the_internal_principal_receives_the_relation(self, profile: str) -> None:
        """Exactly one principal, and it is the one the boot path resolves.

        ``_INTERNAL_PROJECTION_BINDING`` is the wiring BINDING name; the grant
        is keyed by the topology ROLE that binding resolves to, which is a
        different string. Asserting the role is what makes this test
        falsifiable against the live gate: the reverse ratchet printed exactly
        ``omninode_runtime -> omninode_internal.runtime_error_fingerprints``
        when the declaration was missing.
        """
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, _fingerprints_declarations())
        physical_schema = physical_grant_schema_for_table(_SCHEMA, _RELATION)
        principals = {
            principal
            for principal, grants in derived.grants.items()
            for grant in grants
            if grant.object_type is EnumDatabaseGrantObjectType.TABLE
            and grant.schema == physical_schema
            and _RELATION in grant.objects
        }
        assert principals == {_INTERNAL_PROJECTION_ROLE}, (
            f"profile {profile} derives {sorted(principals)} for {_RELATION}; "
            "the runtime validates the declared privilege of the binding "
            "principal before it will wire the handler, so a miss here is a "
            "boot failure and an extra principal is over-granting"
        )

    def test_privileges_are_exactly_what_the_validator_demands(
        self, profile: str
    ) -> None:
        """No wildcard, no widening. Over-granting is its own defect."""
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, _fingerprints_declarations())
        seen = 0
        for grants in derived.grants.values():
            for grant in grants:
                if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                    continue
                if _RELATION not in grant.objects:
                    continue
                assert set(grant.privileges) == WRITE_PRIVILEGES
                seen += 1
        assert seen, "no TABLE grant on the relation at all; the check is vacuous"

    def test_nothing_is_left_unmappable(self, profile: str) -> None:
        """A residual means the derivation could not place it, which ships nothing."""
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, _fingerprints_declarations())
        assert derived.unmappable == ()


class TestRemovingItIsWhatBreaksIt:
    """The negative control, and the reason the one-line fix is wrong.

    Without this test the suite above passes just as happily on a tree where
    the declaration was deleted and the instances were regenerated to match --
    which is exactly the state that arms the OMN-18768 boot crash. This asserts
    the causal direction: the grant is present BECAUSE the declaration is.
    """

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_an_empty_declaration_set_derives_no_grant(self, profile: str) -> None:
        topology = load_topology_profile(profile)
        derived = derive_table_grants(topology, [])
        assert not any(
            _RELATION in grant.objects
            for grants in derived.grants.values()
            for grant in grants
            if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        )
