# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The skill_executions supplemental bridge was retired (OMN-17292 / OMN-21933).

The omnimarket contract pin advance (OMN-17292) brings
``node_projection_skill_lifecycle`` into the pinned contracts, so the
hand-authored ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry for
``skill_executions`` is redundant. This module asserts, on committed repository
state alone, that the entry is gone and that the shipped grant to the
``tenant_projection_writer`` principal survived the deletion.
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
_RELATION = "skill_executions"
_PRINCIPAL = "tenant_projection_writer"
_INSTANCES_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_SHIPPED_INSTANCES = tuple(sorted(p.stem for p in _INSTANCES_DIR.glob("*.yaml")))
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})
_VENDOR = Path("docker/migrations/forward/nodes/node_projection_skill_lifecycle")
_MIGRATIONS = (
    _VENDOR / "0001_create_skill_executions.sql",
    _VENDOR / "0002_grant_tenant_projection_writer_skill_executions.sql",
)


def _committed_privileges(profile: str) -> frozenset[str]:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    granted: set[str] = set()
    for database in document.get("databases", {}).values():
        principal = database.get("principals", {}).get(_PRINCIPAL)
        if principal is None:
            continue
        for grant in principal.get("grants", ()):
            if grant.get("object_type") != "TABLE":
                continue
            if _RELATION not in tuple(grant.get("objects") or ()):
                continue
            granted.update(grant.get("privileges", ()))
    return frozenset(granted)


def test_no_supplemental_bridge_remains() -> None:
    carried = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert _RELATION not in carried


@pytest.mark.parametrize("migration", _MIGRATIONS, ids=lambda p: p.name)
def test_the_vendored_migration_is_still_in_the_tree(migration: Path) -> None:
    assert (_REPO_ROOT / migration).is_file()


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
def test_the_writer_principal_can_still_write_the_projection(instance: str) -> None:
    missing = sorted(_REQUIRED_PRIVILEGES - _committed_privileges(instance))
    assert not missing, f"{instance} no longer grants {missing} on {_RELATION}"
