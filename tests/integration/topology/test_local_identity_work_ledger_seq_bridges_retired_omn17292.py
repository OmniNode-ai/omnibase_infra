# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The local_identity and work_ledger_seq_conflicts bridges were retired (OMN-17292).

The omnimarket contract pin advances to a dev commit that carries the
local-identity reader opt-out (omnimarket#5766, merged 2026-11-11T01:17Z) on top
of the contracts that declare ``local_identity`` (node_projection_local_identity)
and ``work_ledger_seq_conflicts`` (node_projection_work_ledger). The pinned
contracts now derive the grants the two hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries used to supply, so the entries
are redundant and are deleted by the pin advance that made them so. This module
mirrors ``test_prod_promotion_gate_decisions_bridge_retired_omn18999.py``.

The privilege assertion is made against the COMMITTED instance files, the
artifact the deploy applies, so it holds whichever declaration source supplied
the grant. Needs no database and no foreign checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.application_database import (
    TOPOLOGY_PROFILE_INSTANCE_MAP,
)
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Every instance the generator renders and the deploy ships, read from the
# generator's own profile map so dropping one from the deploy is a visible edit.
_SHIPPED_INSTANCES = tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))

# A write projection needs all three; SELECT alone reads green while the writer
# still cannot write (OMN-18768).
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

# relation -> (schema, principal, migration that creates or grants it)
_RETIRED: dict[str, tuple[str, str, str]] = {
    "local_identity": (
        "public",
        "tenant_projection_writer",
        "docker/migrations/forward/nodes/node_projection_local_identity/"
        "0001_grant_tenant_projection_writer_local_identity.sql",
    ),
    "work_ledger_seq_conflicts": (
        "omninode_internal",
        "omninode_runtime",
        "docker/migrations/forward/nodes/node_projection_work_ledger/"
        "0010_grant_work_ledger_seq_conflicts.sql",
    ),
}


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


def _committed_privileges(
    profile: str, relation: str, schema: str, principal: str
) -> frozenset[str]:
    """Privileges the SHIPPED instance file grants ``principal`` on ``relation``."""
    granted: set[str] = set()
    for database in _instance_document(profile).get("databases", {}).values():
        entry = database.get("principals", {}).get(principal)
        if entry is None:
            continue
        for grant in entry.get("grants", ()):
            if grant.get("object_type") != "TABLE" or grant.get("schema") != schema:
                continue
            if relation in tuple(grant.get("objects") or ()):
                granted.update(grant.get("privileges", ()))
    return frozenset(granted)


@pytest.mark.parametrize("relation", sorted(_RETIRED))
class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self, relation: str) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry in table_grant_derivation.py, but the pinned "
            "omnimarket contracts declare it. A redundant bridge contributes "
            "byte-identical output and nothing else will tell you it is there."
        )

    def test_the_vendored_migration_is_still_in_the_tree(self, relation: str) -> None:
        """Retiring the declaration source never deletes the migration."""
        migration = _RETIRED[relation][2]
        assert (_REPO_ROOT / migration).is_file(), (
            f"{migration} is gone. The vendored migration issues the grant the "
            "instances declare; retiring the supplemental entry retires the "
            "DECLARATION SOURCE only."
        )


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
@pytest.mark.parametrize("relation", sorted(_RETIRED))
def test_the_shipped_grant_survived_the_retirement(
    instance: str, relation: str
) -> None:
    schema, principal, _migration = _RETIRED[relation]
    granted = _committed_privileges(instance, relation, schema, principal)

    missing = sorted(_REQUIRED_PRIVILEGES - granted)
    assert not missing, (
        f"instance {instance!r} no longer grants {missing} on {schema}.{relation} "
        f"to {principal}. Retiring the supplemental bridge must leave the shipped "
        "grant byte-identical, because the pinned contract now derives what the "
        "entry used to; a runtime booted from this topology would fail on first "
        "write rather than at deploy (OMN-18768)."
    )
