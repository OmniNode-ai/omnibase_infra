# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned omnimarket contracts replaced two delegation bridges.

The pin advance to e0f9332e7ad4 brought contracts that declare
delegation_routing_feedback (OMN-20578) and delegation_budget_applied_events
(OMN-20613), so both hand-authored declarations were retired. Retiring a bridge
removes only its declaration: the vendored migration lineage and the exact
grants in every shipped instance must survive. This proof uses committed
repository state and needs no foreign checkout.
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
_MIGRATION_ROOT = Path("docker/migrations/forward/nodes")
_TABLE_MIGRATIONS: dict[str, tuple[str, ...]] = {
    "delegation_routing_feedback": (
        "node_projection_routing_feedback/0000_create_delegation_routing_feedback.sql",
        "node_projection_routing_feedback/"
        "0001_grant_tenant_projection_writer_delegation_routing_feedback.sql",
    ),
    "delegation_budget_applied_events": (
        "node_projection_delegation/0056_delegation_budget_applied_events.sql",
        "node_projection_delegation/0057_delegation_budget_applied_events_rls.sql",
    ),
}
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
_SCHEMA = "public"
_PRINCIPAL = "tenant_projection_writer"
_REQUIRED_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE"})


@pytest.mark.parametrize("table", sorted(_TABLE_MIGRATIONS))
def test_no_supplemental_bridge_remains(table: str) -> None:
    carried = {
        declaration.table.name for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
    }
    assert table not in carried, (
        f"{table} still has a supplemental declaration, although the pinned "
        "omnimarket contract now declares it"
    )


@pytest.mark.parametrize(
    ("table", "migration"),
    [
        (table, migration)
        for table, migrations in sorted(_TABLE_MIGRATIONS.items())
        for migration in migrations
    ],
)
def test_vendored_migration_lineage_remains(table: str, migration: str) -> None:
    path = _MIGRATION_ROOT / migration
    assert (_REPO_ROOT / path).is_file(), (
        f"{path} is missing; retiring {table}'s bridge must leave its vendored "
        "migration in the tree"
    )


@pytest.mark.parametrize("table", sorted(_TABLE_MIGRATIONS))
@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
def test_shipped_instance_retains_exact_writer_privileges(
    profile: str, table: str
) -> None:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    grants = document["databases"]["application"]["principals"][_PRINCIPAL]["grants"]
    matching = [
        grant
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant["schema"] == _SCHEMA
        and table in grant["objects"]
    ]
    assert len(matching) == 1, (
        f"{profile} ships {len(matching)} TABLE grants for {_SCHEMA}.{table} "
        f"to {_PRINCIPAL}; expected exactly one"
    )
    assert frozenset(matching[0]["privileges"]) == _REQUIRED_PRIVILEGES
