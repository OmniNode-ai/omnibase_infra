# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The prod_promotion_gate_decisions bridge must derive the grant it ships (OMN-18999).

This repository vendors the create and grant migrations for
``omninode_internal.prod_promotion_gate_decisions`` BEFORE omnimarket lands the
node package whose contract declares the relation, because omnimarket's
``node-migration-vendor-parity-gate`` refuses a node migration that has no
vendored counterpart here. For the length of that window the shipped topology
instances declare a relation that the PINNED omnimarket contracts cannot
derive, and a supplemental entry in ``LEGACY_MIGRATION_TABLE_DECLARATIONS``
closes it.

``tests/ci/test_supplemental_declaration_expiry_omn18863.py`` asserts the entry
is DELETED once the pin catches up. It cannot assert the entry WORKS, because
that question needs the shipped topology instances rather than the foreign
tree, and it is the more important of the two: an entry that expires on
schedule but derives the wrong grant, or no grant, still boots a runtime that
cannot write its own projection. That is the OMN-18768 shape, where the
migration issues a grant the topology does not declare and the service crashes
on first write rather than at deploy.

So this module derives against the real instances and asserts the three things
the bridge exists to make true, each with the control that stops it passing
vacuously:

* the relation reaches ``omninode_runtime`` with write privileges on every
  shipped instance, not merely on the one someone happened to check;
* removing the bridge REMOVES it, which is what distinguishes "the bridge
  supplies this" from "something else already did and the entry is inert";
* the shipped instance files already carry it, so the derivation reproduces a
  grant rather than inventing one. Regenerating against a pin that does not
  declare the relation would have deleted it from three instances while the
  vendored migration still granted it, which is the failure this bridge was
  written instead of.

Needs no database and no foreign checkout: it reads the committed topology
instances, which is exactly the artifact the deploy ships.
"""

from __future__ import annotations

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    STATE_IO_TABLE_DECLARATIONS,
    ContractTableDeclaration,
    derive_topology_table_grants,
)

pytestmark = pytest.mark.integration

_RELATION = "prod_promotion_gate_decisions"
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"
_DATABASE = "application"

# The three instances the generator renders and the deploy ships. Named rather
# than derived from the profile map so that dropping an instance from the
# deploy is a visible edit here too.
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

# A write projection needs all three. SELECT alone would read green while the
# writer still could not write, which is the OMN-18768 crash with extra steps.
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})


def _supplemental() -> tuple[ContractTableDeclaration, ...]:
    return STATE_IO_TABLE_DECLARATIONS + LEGACY_MIGRATION_TABLE_DECLARATIONS


def _without_the_bridge() -> tuple[ContractTableDeclaration, ...]:
    return STATE_IO_TABLE_DECLARATIONS + tuple(
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name != _RELATION
    )


def _privileges_granted(
    instance: str, declarations: tuple[ContractTableDeclaration, ...]
) -> frozenset[str]:
    """Privileges the derivation gives _PRINCIPAL on _RELATION, or empty."""
    derived = derive_topology_table_grants(
        load_topology_profile(instance), declarations
    )
    database = derived.per_database.get(_DATABASE)
    if database is None:
        return frozenset()
    granted: set[str] = set()
    for grant in database.grants.get(_PRINCIPAL, ()):
        if grant.schema != _SCHEMA:
            continue
        if _RELATION not in tuple(grant.objects or ()):
            continue
        for privilege in grant.privileges:
            granted.add(getattr(privilege, "value", privilege))
    return frozenset(granted)


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
class TestTheBridgeDerivesTheGrantTheMigrationIssues:
    def test_the_runtime_principal_can_write_the_projection(
        self, instance: str
    ) -> None:
        """Every shipped instance, not just the one that was checked by hand."""
        granted = _privileges_granted(instance, _supplemental())

        missing = sorted(_REQUIRED_PRIVILEGES - granted)
        assert not missing, (
            f"instance {instance!r} derives no {missing} on "
            f"{_SCHEMA}.{_RELATION} for {_PRINCIPAL}; the vendored migration "
            "issues that grant, so a runtime booted from this topology would "
            "fail on first write rather than at deploy (OMN-18768)"
        )

    def test_removing_the_bridge_removes_the_grant(self, instance: str) -> None:
        """Negative control, and the reason the test above is not vacuous.

        If the relation were already derivable from somewhere else, the
        supplemental entry would be inert and this module would be asserting
        nothing while reading green. It fires the moment that becomes true,
        which is also the moment the entry should be deleted.
        """
        granted = _privileges_granted(instance, _without_the_bridge())

        assert not granted, (
            f"instance {instance!r} still derives {sorted(granted)} on "
            f"{_SCHEMA}.{_RELATION} with the supplemental entry removed, so the "
            "entry is redundant rather than load-bearing. Delete it from "
            "LEGACY_MIGRATION_TABLE_DECLARATIONS and from _INTERIM_ENTRIES in "
            "tests/ci/test_supplemental_declaration_expiry_omn18863.py"
        )

    def test_the_relation_is_not_an_undeliverable_residual(self, instance: str) -> None:
        """A declaration routed to no principal is reported, never granted.

        Asserted separately because a residual is not an absence: the grant
        lookup above would read empty for both, and only this distinguishes
        "nothing declared it" from "it was declared and could not be placed".
        """
        derived = derive_topology_table_grants(
            load_topology_profile(instance), _supplemental()
        )

        stranded = [
            residual for residual in derived.unmappable if residual.name == _RELATION
        ]
        assert not stranded, (
            f"instance {instance!r} classified {_SCHEMA}.{_RELATION} as an "
            f"undeliverable residual: {stranded}"
        )
