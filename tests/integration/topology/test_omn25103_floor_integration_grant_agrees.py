# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The floor_integration_rows grant agrees end to end (OMN-25103).

The writer upserts each integration pass and deletes the rows of an older pass
the newest no longer names, so the runtime needs DELETE; hosted CI's live ACL
gate failed when the topology omitted it. The grant has to say the same thing in
three places: the node-owned migration that issues it, the shipped topology
instances that declare it, and every rendered database-topology catalog the
lanes deploy from. Each lists the relation in exactly two grants, the WRITE and PRUNE shapes,
whose union is what the migration issues.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    PRUNE_PRIVILEGES,
    WRITE_PRIVILEGES,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA = "omninode_internal"
_RELATION = "floor_integration_rows"
_PRINCIPAL = "omninode_runtime"
_WRITE = frozenset(privilege.value for privilege in WRITE_PRIVILEGES)
_PRUNE = frozenset(privilege.value for privilege in PRUNE_PRIVILEGES)
_INSTANCE_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in _INSTANCE_DIR.glob("*.yaml")))
_CATALOG_DIR = _REPO_ROOT / "docker" / "catalog" / "database-topology"
_RENDERED_CATALOGS = tuple(sorted(path.stem for path in _CATALOG_DIR.glob("*.yaml")))
_GRANT_FILE = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_floor_integration"
    / "0001_grant_omninode_runtime_floor_integration_rows.sql"
)
_TABLE_GRANT = re.compile(
    r"GRANT\s+(?P<privileges>[A-Z,\s]+?)\s+ON\s+(?P<schema>\w+)\.(?P<relation>\w+)"
    r"\s+TO\s+(?P<role>\w+)",
    re.IGNORECASE,
)


def _relation_grants(document: dict[str, Any]) -> list[dict[str, Any]]:
    principal = document["databases"]["application"]["principals"][_PRINCIPAL]
    return [
        grant
        for grant in principal["grants"]
        if grant.get("object_type") == "TABLE"
        and grant.get("schema") == _SCHEMA
        and _RELATION in tuple(grant.get("objects") or ())
    ]


def _migration_privileges() -> frozenset[str]:
    sql = "\n".join(
        line
        for line in _GRANT_FILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    matches = [
        match
        for match in _TABLE_GRANT.finditer(sql)
        if (match["schema"], match["relation"], match["role"])
        == (_SCHEMA, _RELATION, _PRINCIPAL)
    ]
    assert len(matches) == 1, f"{_GRANT_FILE.name} issues {len(matches)} grants"
    return frozenset(p.strip().upper() for p in matches[0]["privileges"].split(","))


def test_the_write_shape_is_select_insert_update() -> None:
    assert frozenset({"SELECT", "INSERT", "UPDATE"}) == _WRITE


def test_the_prune_shape_is_select_and_delete() -> None:
    assert frozenset({"SELECT", "DELETE"}) == _PRUNE


def test_the_migration_issues_exactly_select_insert_update_delete() -> None:
    assert _migration_privileges() == frozenset(
        {"SELECT", "INSERT", "UPDATE", "DELETE"}
    )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
def test_each_shipped_instance_declares_what_the_migration_issues(
    profile: str,
) -> None:
    document = yaml.safe_load(
        (_INSTANCE_DIR / f"{profile}.yaml").read_text(encoding="utf-8")
    )
    matching = _relation_grants(document)
    assert len(matching) == 2, f"{profile} ships {len(matching)} grants"
    actual = {frozenset(grant["privileges"]) for grant in matching}
    assert actual == {_WRITE, _PRUNE}, f"{profile} grants {actual}"
    union = frozenset(p for grant in matching for p in grant["privileges"])
    assert union == _migration_privileges()


def test_every_instance_and_catalog_is_read() -> None:
    assert len(_SHIPPED_INSTANCES) == 3, _SHIPPED_INSTANCES
    assert len(_RENDERED_CATALOGS) == 9, _RENDERED_CATALOGS


@pytest.mark.parametrize("catalog", _RENDERED_CATALOGS)
def test_each_rendered_catalog_deploys_what_the_migration_issues(
    catalog: str,
) -> None:
    document = yaml.safe_load(
        (_CATALOG_DIR / f"{catalog}.yaml").read_text(encoding="utf-8")
    )
    matching = _relation_grants(document)
    assert len(matching) == 2, f"{catalog} renders {len(matching)} grants"
    actual = {frozenset(grant["privileges"]) for grant in matching}
    assert actual == {_WRITE, _PRUNE}, f"{catalog} grants {actual}"
    union = frozenset(p for grant in matching for p in grant["privileges"])
    assert union == _migration_privileges()
