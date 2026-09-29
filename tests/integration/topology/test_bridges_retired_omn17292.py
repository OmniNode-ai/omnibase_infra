# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Five supplemental bridges were retired by the PR 4278 pin advance (OMN-17292).

The omnimarket contract pin advance in omnibase_infra PR 4278 (OMN-17292) is
the moment these five bridges became redundant. The expiry test in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py`` named them.
Deleting a bridge retires its declaration source and never its migration.

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
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"

# The three instances the generator renders and the deploy ships. Named rather
# than derived from the profile map so that dropping an instance from the
# deploy is a visible edit here too.
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

_RETIRED_BRIDGES = (
    (
        "session_content",
        Path(
            "docker/migrations/forward/nodes/node_projection_session_content/"
            "0001_create_session_content.sql"
        ),
    ),
    (
        "pr_landing_state",
        Path(
            "docker/migrations/forward/nodes/node_projection_pr_landing/"
            "0000_create_pr_landing.sql"
        ),
    ),
    (
        "pr_landing_transitions",
        Path(
            "docker/migrations/forward/nodes/node_projection_pr_landing/"
            "0000_create_pr_landing.sql"
        ),
    ),
    (
        "claude_agent_spans",
        Path(
            "docker/migrations/forward/nodes/"
            "node_projection_claude_hook_events/0000_create_claude_hook_events.sql"
        ),
    ),
    (
        "claude_hook_events",
        Path(
            "docker/migrations/forward/nodes/"
            "node_projection_claude_hook_events/0000_create_claude_hook_events.sql"
        ),
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
    """Privileges the shipped instance grants _PRINCIPAL on _RELATION."""
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
            if relation not in tuple(grant.get("objects") or ()):
                continue
            granted.update(grant.get("privileges", ()))
    return frozenset(granted)


@pytest.mark.parametrize(("relation", "migration"), _RETIRED_BRIDGES)
class TestTheBridgesWereRetired:
    def test_no_supplemental_bridge_remains(
        self, relation: str, migration: Path
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry in table_grant_derivation.py, but the pinned "
            "omnimarket contracts now declare it. A redundant bridge "
            "contributes byte-identical output and nothing else will tell you "
            "it is there."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self, relation: str, migration: Path
    ) -> None:
        """Deleting the bridge must never delete what it bridged."""
        assert (_REPO_ROOT / migration).is_file(), (
            f"{migration} is gone. The vendored migration is what creates "
            f"{relation}; retiring the supplemental entry retires the "
            "DECLARATION SOURCE, never the migration."
        )


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
@pytest.mark.parametrize(("relation", "migration"), _RETIRED_BRIDGES)
class TestTheShippedGrantsSurvivedTheRetirement:
    def test_committed_instance_grants_exact_writer_privileges(
        self, relation: str, migration: Path, instance: str
    ) -> None:
        granted = _committed_privileges(instance, relation)
        assert granted == _REQUIRED_PRIVILEGES, (
            f"instance {instance!r} grants {sorted(granted)} on "
            f"{_SCHEMA}.{relation} to {_PRINCIPAL}, expected exactly "
            f"{sorted(_REQUIRED_PRIVILEGES)}. Retiring the supplemental "
            "bridge must leave the shipped grant byte-identical; a stray "
            "DELETE or missing writer privilege is a regression."
        )
