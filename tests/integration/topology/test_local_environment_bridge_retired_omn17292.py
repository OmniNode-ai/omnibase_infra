# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The local_environment bridge was retired (OMN-19988 / OMN-17292).

OMN-19988 declared ``public.local_environment`` by hand in
``LEGACY_MIGRATION_TABLE_DECLARATIONS``, because its producing contract was not
yet in the pinned omnimarket checkout.

The pin advance to ``39c422c1a325`` (OMN-17292, full commit
39c422c1a3251926fe3334bd1e184fc2aa1d5064) carries the
``node_projection_local_environment`` contract declaring the relation, so the
hand-authored entry was deleted from ``LEGACY_MIGRATION_TABLE_DECLARATIONS``
and from ``_INTERIM_ENTRIES`` in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py``. The vendored
``0000_create_local_environment.sql`` migration and its 0001 and 0002
successors remain under
``docker/migrations/forward/nodes/node_projection_local_environment/``.

The privilege assertion is made against the COMMITTED instance files, which is
what the deploy applies, so it no longer depends on which declaration source
supplied the grant. Every shipped instance under ``topology/instances`` grants
``tenant_projection_writer`` INSERT, SELECT, and UPDATE on
``public.local_environment``. The instances are discovered from that directory,
and the module refuses an empty discovery so the grant assertion cannot pass
vacuously. A regeneration done wrong drops that grant while
the vendored migration remains, and the projection writer then fails on first
write rather than at deploy.

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
_RELATION = "local_environment"
_SCHEMA = "public"
_PRINCIPAL = "tenant_projection_writer"

_INSTANCES = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in _INSTANCES.glob("*.yaml")))

# A write projection needs all three. SELECT alone would read green while the
# writer still could not write, which is the OMN-18768 crash with extra steps.
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

_MIGRATION = Path(
    "docker/migrations/forward/nodes/node_projection_local_environment/"
    "0000_create_local_environment.sql"
)


def _instance_document(profile: str) -> dict[str, Any]:
    path = _INSTANCES / f"{profile}.yaml"
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


def test_the_shipped_instances_are_discovered() -> None:
    assert _SHIPPED_INSTANCES, f"no topology instances found under {_INSTANCES}"


class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert _RELATION not in carried, (
            f"{_RELATION} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry in table_grant_derivation.py, but the pinned "
            "omnimarket contracts (39c422c1a325, OMN-17292, carrying "
            "node_projection_local_environment) declare it. The OMN-19988 "
            "bridge contributes byte-identical output and nothing else will "
            "tell you it is there."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self,
    ) -> None:
        """Deleting the bridge must never delete what it bridged."""
        assert (_REPO_ROOT / _MIGRATION).is_file(), (
            f"{_MIGRATION} is gone. The vendored migration is what issues the "
            "local_environment grant the instances declare; retiring the "
            "OMN-19988 bridge at pin 39c422c1a325 retires the DECLARATION "
            "SOURCE, never the migration"
        )


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
class TestTheShippedGrantSurvivedTheRetirement:
    """The half that actually protects the runtime, on every shipped instance.

    This is the assertion the replaced module made by derivation. Making it
    against the committed file instead is strictly stronger: the deploy applies
    this artifact, whatever derived it.
    """

    def test_the_projection_writer_can_still_write_the_projection(
        self, instance: str
    ) -> None:
        granted = _committed_privileges(instance)

        missing = sorted(_REQUIRED_PRIVILEGES - granted)
        assert not missing, (
            f"instance {instance!r} no longer grants {missing} on "
            f"{_SCHEMA}.{_RELATION} to {_PRINCIPAL}. Retiring the OMN-19988 "
            "supplemental bridge must leave the shipped grant byte-identical, "
            "because the pinned contract (39c422c1a325, OMN-17292) now derives "
            "what the entry used to. The vendored migration still issues this "
            "grant, so a projection writer booted from this topology would "
            "fail on first write rather than at deploy (OMN-18768)"
        )
