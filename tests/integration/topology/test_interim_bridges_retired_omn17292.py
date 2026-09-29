# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Three interim bridges were retired by the omnimarket pin advance (OMN-17292).

The advance to ``341957de4ffc`` carries omnimarket#2905 for
``session_content`` and omnimarket#3000 for ``pr_landing_state`` and
``pr_landing_transitions``. Those contracts now declare the relations that
their interim ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries once supplied.

The expiry module went red on the commit that advanced the pin and named all
three entries; their deletion rides that same commit. The vendored migrations
remain in this repository, and the committed topology artifacts must continue
to grant the runtime principal the privileges those migrations require.

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
_RELATIONS = ("session_content", "pr_landing_state", "pr_landing_transitions")
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT"})
_MIGRATIONS = (
    Path(
        "docker/migrations/forward/nodes/node_projection_session_content/"
        "0001_create_session_content.sql"
    ),
    Path(
        "docker/migrations/forward/nodes/node_projection_pr_landing/"
        "0000_create_pr_landing.sql"
    ),
    Path(
        "docker/migrations/forward/nodes/node_projection_pr_landing/"
        "0001_grant_omninode_runtime_pr_landing.sql"
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


def _committed_privileges(profile: str, relation: str) -> frozenset[str]:
    """Privileges the shipped instance grants on one retired bridge relation."""
    granted: set[str] = set()
    for database in _instance_document(profile).get("databases", {}).values():
        principal = database.get("principals", {}).get(_PRINCIPAL)
        if principal is None:
            continue
        for grant in principal.get("grants", ()):
            if grant.get("object_type") != "TABLE":
                continue
            if grant.get("schema") != _SCHEMA:
                continue
            if relation not in tuple(grant.get("objects") or ()):
                continue
            granted.update(grant.get("privileges", ()))
    return frozenset(granted)


class TestTheBridgesWereRetired:
    def test_no_supplemental_bridges_remain(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        remaining = sorted(set(_RELATIONS) & carried)
        assert not remaining, (
            f"{remaining} still have supplemental "
            "LEGACY_MIGRATION_TABLE_DECLARATIONS entries, but the pinned "
            "omnimarket contracts at 341957de4ffc declare them."
        )

    def test_the_vendored_migration_lineage_remains_in_the_tree(self) -> None:
        """Retiring declaration bridges must not delete their migrations."""
        missing = [
            str(migration)
            for migration in _MIGRATIONS
            if not (_REPO_ROOT / migration).is_file()
        ]
        assert not missing, f"vendored migrations disappeared: {missing}"


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
@pytest.mark.parametrize("relation", _RELATIONS)
class TestTheShippedGrantsSurvivedTheRetirement:
    def test_runtime_retains_the_required_table_privileges(
        self, instance: str, relation: str
    ) -> None:
        granted = _committed_privileges(instance, relation)
        missing = sorted(_REQUIRED_PRIVILEGES - granted)
        assert not missing, (
            f"instance {instance!r} no longer grants {missing} on "
            f"{_SCHEMA}.{relation} to {_PRINCIPAL}; the committed topology "
            "must retain the migration's runtime access after the contract "
            "pin replaces its interim declaration bridge."
        )
