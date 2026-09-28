# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The topic_activity bridge was retired (OMN-19716 / OMN-17292).

OMN-19716 (#4165) vendored the topic-activity projection migrations and
declared ``omninode_internal.topic_activity`` by hand in
``LEGACY_MIGRATION_TABLE_DECLARATIONS``, because its producing contract was not
yet in the pinned omnimarket checkout. The entry's own comment named the
retiring pull request: omnimarket#2953.

The pin advance to ``2e7cec7d45ed`` (OMN-17292, omnibase_infra#4199) is that
moment -- it is ancestor-forward of ``ed5dbf2ba24d`` (verified: git merge-base
--is-ancestor ed5dbf2ba24d2c5a199c77e773d5583a5ce357a8
2e7cec7d45ed05ecf9daa3d874ac46e968bb3389 => true). It carries omnimarket#2953,
so the pinned contracts declare the relation themselves and the hand-authored
entry contributes byte-identical output.
``test_no_interim_entry_is_redundant`` in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py`` went red naming
it, and the deletion rides the commit that caused it -- the sequence OMN-18900
and OMN-18999 followed, whose retirement modules this one mirrors.

The privilege assertion is made against the COMMITTED instance files, which is
what the deploy applies, so it no longer depends on which declaration source
supplied the grant. The OMN-18768 failure this guards is unchanged: a
regeneration done wrong drops the declaration from the shipped instances while
the vendored migration still issues the grant, and the runtime then fails on
first write rather than at deploy.

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
_RELATION = "topic_activity"
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
    "docker/migrations/forward/nodes/node_projection_topic_activity/"
    "0000_create_topic_activity.sql"
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
            "omnimarket contracts (2e7cec7d45ed, OMN-17292, carrying "
            "omnimarket#2953) declare it. A redundant bridge contributes "
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
    """The half that actually protects the runtime, on every shipped instance.

    This is the assertion the replaced module made by derivation. Making it
    against the committed file instead is strictly stronger: the deploy applies
    this artifact, whatever derived it.
    """

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
