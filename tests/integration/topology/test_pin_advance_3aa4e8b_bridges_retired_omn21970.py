# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Six more supplemental table bridges were retired by the pin advance (OMN-21970, OMN-17292).

The omnimarket contract pin advancing to 3aa4e8b26af6 carries the node contracts
that declare these relations, so the hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries that stood in for them are
deleted from ``table_grant_derivation``:

- ``delegation_capability_cells`` (OMN-20169)
- ``delegation_budget_readings`` (OMN-17427)
- ``delegation_lab_jobs`` and ``delegation_lab_job_outbox`` (OMN-17427)
- ``compaction_receipts`` (OMN-21336)
- ``local_environment`` (OMN-19988)

Deleting a bridge must not delete the grant it carried. For each relation this
proves, from the committed repository state alone, that the derivation no longer
carries a bridge for it, that the vendored migration lineage is still in the
tree, and that every shipped instance still grants the principal exactly
SELECT, INSERT and UPDATE on it. The grants are read from the committed
``topology/instances/*.yaml``.

What this does NOT prove: that the pinned omnimarket contracts derive those
grants. The instances are generated files, so a bridge removed ahead of the
contract that replaces it would still pass here. That proof is the OMN-15361
Application Database Domain Enforcement job's: it regenerates every grant from
the pinned contracts (``.proof-dependencies/omnimarket-pin``) and fails when
the result differs from these committed instances by a byte. This repo may not
import omnimarket (OMN-19998), which is why the derivation proof lives there
and not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS_ROOT = Path("docker/migrations/forward/nodes")
_INSTANCES_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in _INSTANCES_ROOT.glob("*.yaml")))
_RUNTIME_WRITE = frozenset({"INSERT", "SELECT", "UPDATE"})

_LAB_JOB_EXECUTION = (
    "node_llm_delegation_call_effect",
    ("0000_delegation_lab_job_execution.sql",),
)

# (relation, schema, principal, node package, migrations)
_RETIRED_BRIDGES = (
    (
        "delegation_capability_cells",
        "public",
        "tenant_projection_writer",
        "node_projection_delegation_capability_map",
        (
            "0000_create_delegation_capability_cells.sql",
            "0001_grant_tenant_projection_writer_delegation_capability_cells.sql",
        ),
    ),
    (
        "delegation_budget_readings",
        "omninode_internal",
        "omninode_runtime",
        "node_projection_delegation_budget_readings",
        (
            "0000_create_delegation_budget_readings.sql",
            "0001_grant_omninode_runtime_delegation_budget_readings.sql",
        ),
    ),
    (
        "delegation_lab_jobs",
        "omninode_internal",
        "omninode_runtime",
        *_LAB_JOB_EXECUTION,
    ),
    (
        "delegation_lab_job_outbox",
        "omninode_internal",
        "omninode_runtime",
        *_LAB_JOB_EXECUTION,
    ),
    (
        "compaction_receipts",
        "omninode_internal",
        "omninode_runtime",
        "node_projection_compaction_receipts",
        (
            "0000_create_compaction_receipts.sql",
            "0001_grant_omninode_runtime_compaction_receipts.sql",
        ),
    ),
    (
        "local_environment",
        "public",
        "tenant_projection_writer",
        "node_projection_local_environment",
        (
            "0000_create_local_environment.sql",
            "0001_grant_tenant_projection_writer_local_environment.sql",
        ),
    ),
)

_IDS = [bridge[0] for bridge in _RETIRED_BRIDGES]


def _committed_privileges(
    profile: str, relation: str, schema: str, principal: str
) -> frozenset[str]:
    path = _INSTANCES_ROOT / f"{profile}.yaml"
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    granted: set[str] = set()
    for database in document.get("databases", {}).values():
        entry = database.get("principals", {}).get(principal)
        if entry is None:
            continue
        for grant in entry.get("grants", ()):
            if (
                grant.get("object_type") == "TABLE"
                and grant.get("schema") == schema
                and relation in tuple(grant.get("objects") or ())
            ):
                granted.update(grant.get("privileges", ()))
    return frozenset(granted)


def test_every_shipped_instance_is_checked() -> None:
    # Every instance a topology profile resolves to is one this module reads, so
    # an instance file that goes missing fails here rather than passing unread.
    routed = set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())
    assert routed
    assert routed <= set(_SHIPPED_INSTANCES)


@pytest.mark.parametrize(
    ("relation", "schema", "principal", "package", "migrations"),
    _RETIRED_BRIDGES,
    ids=_IDS,
)
class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried

    def test_the_migration_lineage_is_still_in_the_tree(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        for name in migrations:
            assert (_REPO_ROOT / _MIGRATIONS_ROOT / package / name).is_file()

    @pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
    def test_the_shipped_grant_survived(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
        instance: str,
    ) -> None:
        granted = _committed_privileges(instance, relation, schema, principal)
        assert granted == _RUNTIME_WRITE, (
            f"{principal} on {schema}.{relation} in {instance}: {sorted(granted)}"
        )
