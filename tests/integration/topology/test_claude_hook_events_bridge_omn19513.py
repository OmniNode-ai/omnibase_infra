# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The active Claude hook-event bridge derives the grants it ships (OMN-19513)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    derive_table_grants,
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
class TestTheActiveBridgeIsCommitted:
    def test_exactly_one_bridge_carries_the_relation(self, relation: str) -> None:
        entries = _bridge_entries(relation)
        assert len(entries) == 1, (
            f"{relation} is carried by {len(entries)} "
            "LEGACY_MIGRATION_TABLE_DECLARATIONS entries; the active interim "
            "window requires exactly one bridge. Add the missing declaration or "
            "remove the duplicate before regenerating topology grants."
        )
        table = entries[0].table
        actual = (table.schema, table.access, table.database_ref)
        expected = (_SCHEMA, "read_write", _DATABASE_REF)
        assert actual == expected, (
            f"{relation}'s active bridge has schema/access/database_ref "
            f"{actual!r}, expected {expected!r}. Restore the bridge to the "
            "omninode_internal application read_write declaration so the "
            "runtime writer derives its required grant."
        )

    @pytest.mark.parametrize("migration", _VENDORED_MIGRATIONS, ids=lambda p: p.name)
    def test_both_vendored_migrations_remain_in_the_tree(
        self, relation: str, migration: Path
    ) -> None:
        assert (_REPO_ROOT / migration).is_file(), (
            f"{migration} is missing while {relation}'s interim bridge is "
            "active. Restore both vendored Claude hook-event migration files; "
            "the bridge declares relations that this migration lineage creates "
            "and grants."
        )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
@pytest.mark.parametrize("relation", _RELATIONS)
class TestTheActiveBridgeProducesTheShippedGrant:
    def test_real_derivation_grants_exact_writer_privileges(
        self, profile: str, relation: str
    ) -> None:
        derived = derive_table_grants(
            load_topology_profile(profile), LEGACY_MIGRATION_TABLE_DECLARATIONS
        )
        matching_grants = tuple(
            grant
            for grant in derived.grants.get(_PRINCIPAL, ())
            if grant.object_type is EnumDatabaseGrantObjectType.TABLE
            and grant.schema == _SCHEMA
            and relation in grant.objects
        )
        assert len(matching_grants) == 1, (
            f"real derivation for {profile} produced {len(matching_grants)} "
            f"{_PRINCIPAL} TABLE grants for {_SCHEMA}.{relation}, not one. "
            "Keep exactly one active bridge with the required declaration "
            "shape, then regenerate the shipped topology grants."
        )
        privileges = frozenset(
            privilege.value for privilege in matching_grants[0].privileges
        )
        assert privileges == _REQUIRED_PRIVILEGES, (
            f"real derivation for {profile} gives {_PRINCIPAL} {sorted(privileges)} "
            f"on {_SCHEMA}.{relation}, expected {sorted(_REQUIRED_PRIVILEGES)} "
            "and no DELETE. Restore read_write access on the active bridge and "
            "regenerate the topology grant."
        )

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
