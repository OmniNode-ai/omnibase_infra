# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ci_attempt_outcome supplemental bridge was retired (OMN-18903 / OMN-18863).

``ci_attempt_outcome`` reached the shipped topology instances through the
infra-first supplemental-bridge mechanism: this repository vendored the
migrations that create and grant the relation before omnimarket#2730, the node
package that declares it in its contract, could merge. A hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry carried it in the interim
(``table_grant_derivation.py``, tracked for expiry in ``_INTERIM_ENTRIES`` in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py``).

The omnimarket contract pin advancing to fcc374d5908f (OMN-17292) makes that
bridge redundant: ``node_projection_ci_attempt_outcome`` now declares the
relation in its ``db_io.db_tables`` with the same schema, access and role, and
``test_no_interim_entry_is_redundant`` in the expiry module named the remedy --
delete the entry from both files, since the regenerated grants come back
byte-identical.

This module asserts, on committed repository state alone, that the deletion
happened and that it did not take the shipped grant with it. The
pin-versus-declared-set comparison needs the omnimarket checkout the CI
enforcement job provides, so it stays in the expiry module rather than here.
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
_RELATION = "ci_attempt_outcome"
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"

# The three instances the generator renders and the deploy ships. Named rather
# than derived from the profile map so that dropping an instance from the
# deploy is a visible edit here too.
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

# The contract declares access read_write: the upsert's conflict arm reads the
# stored row to refuse a stale redelivery, so SELECT alone would read green
# while the writer still could not write.
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

_MIGRATIONS = (
    Path(
        "docker/migrations/forward/nodes/node_projection_ci_attempt_outcome/"
        "0000_create_ci_attempt_outcome.sql"
    ),
    Path(
        "docker/migrations/forward/nodes/node_projection_ci_attempt_outcome/"
        "0001_grant_omninode_runtime_ci_attempt_outcome.sql"
    ),
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
            "omnimarket contracts (fcc374d5908f, OMN-17292, carrying "
            "omnimarket#2730) declare it. A redundant bridge contributes "
            "byte-identical output and nothing else will tell you it is there."
        )

    @pytest.mark.parametrize("migration", _MIGRATIONS, ids=lambda p: p.name)
    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self, migration: Path
    ) -> None:
        """Deleting the bridge must never delete what it bridged."""
        assert (_REPO_ROOT / migration).is_file(), (
            f"{migration} is gone. The vendored migrations are what create the "
            "relation and issue the grant the instances declare; retiring the "
            "supplemental entry retires the DECLARATION SOURCE, never the "
            "migration"
        )


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
class TestTheShippedGrantSurvivedTheRetirement:
    """The half that protects the runtime, asserted on every shipped instance."""

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
