# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Retiring the local_environment bridge preserves migrations and shipped grants.

The advanced omnimarket contract pin owns this declaration. The supplemental
declaration must disappear while every topology and rendered catalog retains
the writer privileges needed by the projection's upserts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INSTANCES_ROOT = _REPO_ROOT / "src/omnibase_infra/topology/instances"
_CATALOG_ROOT = _REPO_ROOT / "docker/catalog/database-topology"
_MIGRATIONS_ROOT = (
    _REPO_ROOT / "docker/migrations/forward/nodes/node_projection_local_environment"
)
_SHIPPED_TOPOLOGIES = tuple(sorted(_INSTANCES_ROOT.glob("*.yaml"))) + tuple(
    sorted(_CATALOG_ROOT.glob("*.yaml"))
)


def test_no_supplemental_bridge_remains() -> None:
    carried = {
        declaration.table.name for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
    }
    assert "local_environment" not in carried


@pytest.mark.parametrize(
    ("filename", "statement"),
    [
        (
            "0000_create_local_environment.sql",
            "CREATE TABLE IF NOT EXISTS public.local_environment",
        ),
        (
            "0001_grant_tenant_projection_writer_local_environment.sql",
            "GRANT SELECT, INSERT, UPDATE ON public.local_environment "
            "TO tenant_projection_writer;",
        ),
        (
            "0002_force_rls_local_environment.sql",
            "ALTER TABLE public.local_environment FORCE ROW LEVEL SECURITY;",
        ),
    ],
)
def test_vendored_migration_behavior_remains(filename: str, statement: str) -> None:
    sql = (_MIGRATIONS_ROOT / filename).read_text(encoding="utf-8")
    assert statement in " ".join(sql.split())


def test_shipped_topologies_are_found() -> None:
    assert len(tuple(_INSTANCES_ROOT.glob("*.yaml"))) >= 3
    assert len(tuple(_CATALOG_ROOT.glob("*.yaml"))) >= 3


@pytest.mark.parametrize(
    "path",
    _SHIPPED_TOPOLOGIES,
    ids=[str(path.relative_to(_REPO_ROOT)) for path in _SHIPPED_TOPOLOGIES],
)
def test_shipped_topology_retains_exact_writer_privileges(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    grants = document["databases"]["application"]["principals"][
        "tenant_projection_writer"
    ]["grants"]
    matching = [
        grant
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant["schema"] == "public"
        and "local_environment" in grant["objects"]
    ]
    assert len(matching) == 1, f"{path} must carry exactly one local_environment grant"
    assert frozenset(matching[0]["privileges"]) == {"SELECT", "INSERT", "UPDATE"}
