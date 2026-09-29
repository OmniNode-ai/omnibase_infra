# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The Claude hook-event bridge was retired and its grants still ship (OMN-19513).

OMN-19513 declared ``claude_agent_spans`` and ``claude_hook_events`` by hand in
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` until omnimarket#2956 landed. The pin
advance to ``791c3b89970b`` (OMN-19566, carrying omnimarket#2956) makes the
pinned contracts declare both relations, so the bridge entries were deleted
with the commit that made them redundant. The privilege assertions are made
against the COMMITTED instance files, which is what the deploy applies.
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
_RELATIONS = ("claude_agent_spans", "claude_hook_events")
_SCHEMA = "omninode_internal"
_DATABASE_REF = "application"
_PRINCIPAL = "omninode_runtime"
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
_VENDORED_MIGRATIONS = (
    Path(
        "docker/migrations/forward/nodes/node_projection_claude_hook_events/"
        "0000_create_claude_hook_events.sql"
    ),
    Path(
        "docker/migrations/forward/nodes/node_projection_claude_hook_events/"
        "0001_grant_omninode_runtime_claude_hook_events.sql"
    ),
)


def _bridge_entries(relation: str) -> tuple[Any, ...]:
    return tuple(
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == relation
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
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), (
        f"{path} is not a topology mapping, so this test cannot verify the "
        "shipped grant. Restore the committed topology document structure."
    )
    return document


@pytest.mark.parametrize("relation", _RELATIONS)
class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self, relation: str) -> None:
        entries = _bridge_entries(relation)
        assert not entries, (
            f"{relation} still has {len(entries)} supplemental "
            "LEGACY_MIGRATION_TABLE_DECLARATIONS entries, but the pinned "
            "omnimarket contracts (791c3b89970b, carrying omnimarket#2956) "
            "declare it. A redundant bridge contributes byte-identical output "
            "and nothing else will tell you it is there."
        )

    @pytest.mark.parametrize("migration", _VENDORED_MIGRATIONS, ids=lambda p: p.name)
    def test_both_vendored_migrations_remain_in_the_tree(
        self, relation: str, migration: Path
    ) -> None:
        assert (_REPO_ROOT / migration).is_file(), (
            f"{migration} is missing while {relation} is "
            "declared by the pinned contract. Restore both vendored Claude "
            "hook-event migration files; the instances grant relations that "
            "this migration lineage creates and grants."
        )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
@pytest.mark.parametrize("relation", _RELATIONS)
class TestTheShippedInstancesGrantTheWriter:
    def test_shipped_instance_grants_exact_writer_privileges(
        self, profile: str, relation: str
    ) -> None:
        document = _instance_document(profile)
        databases = document.get("databases")
        assert isinstance(databases, dict), (
            f"{profile} has no databases mapping, so the shipped application "
            "grant cannot be verified. Restore the topology instance structure."
        )
        application = databases.get(_DATABASE_REF)
        assert isinstance(application, dict), (
            f"{profile} has no {_DATABASE_REF!r} database entry. Restore the "
            "application database topology and regenerate its grants."
        )
        principals = application.get("principals")
        assert isinstance(principals, dict), (
            f"{profile} has no principals mapping for {_DATABASE_REF}. Restore "
            f"the {_PRINCIPAL} principal and regenerate topology grants."
        )
        runtime = principals.get(_PRINCIPAL)
        assert isinstance(runtime, dict), (
            f"{profile} has no {_PRINCIPAL} application principal. Restore it "
            "and regenerate the active bridge's TABLE grant."
        )
        grants = runtime.get("grants")
        assert isinstance(grants, list), (
            f"{profile}'s {_PRINCIPAL} has no grants list. Restore the shipped "
            "TABLE grant generated from the active bridge."
        )
        matching_grants = [
            grant
            for grant in grants
            if isinstance(grant, dict)
            and grant.get("object_type") == "TABLE"
            and grant.get("schema") == _SCHEMA
            and relation in tuple(grant.get("objects") or ())
        ]
        assert len(matching_grants) == 1, (
            f"{profile} ships {len(matching_grants)} {_PRINCIPAL} TABLE grants "
            f"for {_SCHEMA}.{relation}, not one. Regenerate the application "
            "topology from the active bridge so it carries one exact grant."
        )
        privileges = frozenset(matching_grants[0].get("privileges") or ())
        assert privileges == _REQUIRED_PRIVILEGES, (
            f"{profile} ships {sorted(privileges)} for {_PRINCIPAL} on "
            f"{_SCHEMA}.{relation}, expected {sorted(_REQUIRED_PRIVILEGES)} "
            "and no DELETE. Regenerate the instance from the active read_write "
            "bridge to restore the exact runtime writer grant."
        )
