# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned omnimarket contracts replaced seven supplemental bridges.

Retiring a bridge removes only its hand-authored declaration. The vendored
migration lineage and the exact grants in every shipped instance must survive.
This proof uses committed repository state and needs no foreign checkout.
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
_MIGRATIONS_ROOT = Path("docker/migrations/forward/nodes")
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
_SCHEMA = "omninode_internal"
_PRINCIPAL = "omninode_runtime"
_REQUIRED_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE"})

# Migration filenames come from the pinned db_tables declarations and the
# vendored lineage; grant files are included where distinct from the create.
_RETIRED_BRIDGES = (
    (
        "claude_agent_spans",
        "node_projection_claude_hook_events",
        (
            "0000_create_claude_hook_events.sql",
            "0001_grant_omninode_runtime_claude_hook_events.sql",
        ),
    ),
    (
        "claude_hook_events",
        "node_projection_claude_hook_events",
        (
            "0000_create_claude_hook_events.sql",
            "0001_grant_omninode_runtime_claude_hook_events.sql",
        ),
    ),
    (
        "lab_container_memory_window",
        "node_projection_lab_container_memory",
        (
            "0000_create_lab_container_memory_window.sql",
            "0001_grant_omninode_runtime_lab_container_memory_window.sql",
        ),
    ),
    (
        "pr_landing_state",
        "node_projection_pr_landing",
        ("0000_create_pr_landing.sql", "0001_grant_omninode_runtime_pr_landing.sql"),
    ),
    (
        "pr_landing_transitions",
        "node_projection_pr_landing",
        ("0000_create_pr_landing.sql", "0001_grant_omninode_runtime_pr_landing.sql"),
    ),
    (
        "session_content",
        "node_projection_session_content",
        ("0001_create_session_content.sql",),
    ),
    (
        "worktree_reconcile_hosts",
        "node_projection_worktree_reconcile",
        (
            "0000_create_worktree_reconcile_hosts.sql",
            "0001_grant_runtime_worktree_reconcile.sql",
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
    assert isinstance(document, dict), f"{path} is not a topology mapping"
    return document


def _shipped_privileges(profile: str, relation: str) -> frozenset[str]:
    application = _instance_document(profile)["databases"]["application"]
    grants = application["principals"][_PRINCIPAL]["grants"]
    matching = [
        grant
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant["schema"] == _SCHEMA
        and relation in grant["objects"]
    ]
    assert len(matching) == 1, (
        f"{profile} ships {len(matching)} TABLE grants for {_SCHEMA}.{relation} "
        f"to {_PRINCIPAL}; expected exactly one"
    )
    return frozenset(matching[0]["privileges"])


@pytest.mark.parametrize(("relation", "node", "migrations"), _RETIRED_BRIDGES)
class TestTheBridgesWereRetired:
    def test_no_supplemental_bridge_remains(
        self, relation: str, node: str, migrations: tuple[str, ...]
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental declaration, although the "
            "pinned omnimarket contract now declares it"
        )

    def test_vendored_migration_lineage_remains(
        self, relation: str, node: str, migrations: tuple[str, ...]
    ) -> None:
        for filename in migrations:
            migration = _MIGRATIONS_ROOT / node / filename
            assert (_REPO_ROOT / migration).is_file(), (
                f"{migration} is missing; retiring {relation}'s bridge must "
                "leave its vendored migration in the tree"
            )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
@pytest.mark.parametrize(("relation", "node", "migrations"), _RETIRED_BRIDGES)
def test_shipped_instance_retains_exact_writer_privileges(
    profile: str, relation: str, node: str, migrations: tuple[str, ...]
) -> None:
    granted = _shipped_privileges(profile, relation)
    assert granted == _REQUIRED_PRIVILEGES, (
        f"{profile} grants {sorted(granted)} on {_SCHEMA}.{relation} to "
        f"{_PRINCIPAL}; expected exactly {sorted(_REQUIRED_PRIVILEGES)}"
    )
