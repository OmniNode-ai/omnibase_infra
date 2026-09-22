# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegate_skill_command_claims bridge was retired (OMN-18993 / OMN-18863).

``omninode_internal.delegate_skill_command_claims`` reached the shipped
topology instances through the infra-first supplemental-bridge mechanism: this
repository vendors the migration that creates and grants the relation before
the omnimarket node package that declares it in its contract merges, because
omnimarket's ``node-migration-vendor-parity`` gate refuses the producing pull
request until the vendored counterpart is on this repo's ``dev``. A
hand-authored ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry carried it for the
length of that window.

The pin advance to ``622664a35575`` (OMN-17292) closes the window. It carries
omnimarket#2744, the retiring pull request the entry named in its own comment,
so the pinned contracts now declare the relation themselves and the entry
contributes byte-identical output. ``test_no_interim_entry_is_redundant`` in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py`` went red naming
it, and the deletion rides the commit that caused it -- the sequence OMN-18900
followed for ``dod_verify_runs`` on the previous pin advance.

The relation is the durable correlation-keyed claim that stops a redelivered
delegate-skill command from re-running the inference and billing it twice, so
losing its grant is not a cosmetic regression.

This module asserts the deletion happened and stuck, on committed repository
state alone: the pin-vs-declared-set comparison itself needs the omnimarket
checkout the CI enforcement job provides and lives in the expiry module rather
than here. The privilege half is asserted against the COMMITTED instance files,
which is the artifact the deploy applies. The OMN-18768 failure it guards is a
regeneration done wrong dropping the declaration from three instances while the
vendored migration still issues the grant, so the runtime fails on first write
rather than at deploy.

Needs no database and no foreign checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATION = "delegate_skill_command_claims"
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"

# The three instances the generator renders and the deploy ships. Named rather
# than derived from the profile map so that dropping an instance from the
# deploy is a visible edit here too.
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

# A write projection needs all three. SELECT alone would read green while the
# writer still could not write, which is the OMN-18768 crash with extra steps.
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

_MIGRATION = Path(
    "docker/migrations/forward/nodes/node_delegate_skill_orchestrator/"
    "0001_delegate_skill_command_claims.sql"
)


def _instance_document(profile: str) -> dict[str, Any]:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document


def _committed_privileges(profile: str) -> frozenset[str]:
    """Privileges the SHIPPED instance file grants _PRINCIPAL on _RELATION."""
    granted: set[str] = set()
    for database in _instance_document(profile).get("databases", {}).values():
        principals = database.get("principals", {})
        principal = principals.get(_PRINCIPAL)
        if principal is None:
            continue
        for grant in principal.get("grants", ()):
            if grant.get("object_type") != "TABLE":
                continue
            if grant.get("schema") != _SCHEMA:
                continue
            if _RELATION not in tuple(grant.get("objects") or ()):
                continue
            granted.update(grant.get("privileges", ()))
    return frozenset(granted)


class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert _RELATION not in carried, (
            f"{_RELATION} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry in table_grant_derivation.py, but the pinned "
            "omnimarket contracts (622664a35575, OMN-17292, carrying "
            "omnimarket#2744) declare it. A redundant bridge contributes "
            "byte-identical output and nothing else will tell you it is there."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self,
    ) -> None:
        """Deleting the bridge must never delete what it bridged."""
        assert (_REPO_ROOT / _MIGRATION).is_file(), (
            f"{_MIGRATION} is gone. The vendored migration is what issues the "
            "grant the instances declare; retiring the supplemental entry "
            "retires the DECLARATION SOURCE, never the migration"
        )


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
class TestTheShippedGrantSurvivedTheRetirement:
    """The half that actually protects the runtime, on every shipped instance."""

    def test_the_runtime_principal_can_still_write_the_projection(
        self, instance: str
    ) -> None:
        granted = _committed_privileges(instance)

        missing = sorted(_REQUIRED_PRIVILEGES - granted)
        assert not missing, (
            f"instance {instance!r} no longer grants {missing} on "
            f"{_SCHEMA}.{_RELATION} to {_PRINCIPAL}. Retiring the supplemental "
            "bridge must leave the shipped grant byte-identical, because the "
            "pinned contract now derives what the entry used to. The vendored "
            "migration still issues this grant, so a runtime booted from this "
            "topology would fail on first write rather than at deploy "
            "(OMN-18768)"
        )
