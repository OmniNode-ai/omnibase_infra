# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""tenant_projection_writer holds write on public.context_audit_events (OMN-21934).

The omnimarket node_projection_context_audit declares ``context_audit_events``
in ``db_io.db_tables`` with ``access: read_write`` against the application
database. The wiring validator requires SELECT, INSERT and UPDATE for the
tenant projection principal on every shipped topology instance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.application_database import render_database_projection
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    STATE_IO_TABLE_DECLARATIONS,
)

_ROOT = Path(__file__).resolve().parents[3]
_INSTANCE_DIR = _ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_CATALOG_DIR = _ROOT / "docker" / "catalog" / "database-topology"
_INSTANCES = tuple(sorted(path.stem for path in _INSTANCE_DIR.glob("*.yaml")))
_CATALOGS = tuple(sorted(path.stem for path in _CATALOG_DIR.glob("*.yaml")))
_LEDGER = (
    _ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "_ledger"
    / "application-migrations.tsv"
)
_NODE_MIGRATION_DIR = (
    _ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_context_audit"
)
_TABLE = "context_audit_events"
_WRITE = {"INSERT", "SELECT", "UPDATE"}


def _writer_table_privileges(document: dict[str, Any]) -> set[str]:
    privileges: set[str] = set()
    for database in document["databases"].values():
        principal = database.get("principals", {}).get("tenant_projection_writer")
        if principal is None:
            continue
        for grant in principal["grants"]:
            if (
                grant["object_type"] == "TABLE"
                and grant.get("schema") == "public"
                and _TABLE in grant.get("objects", [])
            ):
                privileges.update(grant["privileges"])
    return privileges


@pytest.mark.unit
@pytest.mark.parametrize("instance", _INSTANCES)
def test_tenant_projection_writer_can_write_context_audit_events(
    instance: str,
) -> None:
    document = yaml.safe_load(
        (_INSTANCE_DIR / f"{instance}.yaml").read_text(encoding="utf-8")
    )
    assert _writer_table_privileges(document) == _WRITE


@pytest.mark.unit
def test_every_rendered_catalog_is_found() -> None:
    assert _CATALOGS
    assert {path.stem for path in _CATALOG_DIR.glob("*.yaml")} >= set(_INSTANCES)


@pytest.mark.unit
@pytest.mark.parametrize("catalog", _CATALOGS)
def test_the_rendered_catalog_grants_the_writer_on_the_table(catalog: str) -> None:
    document = yaml.safe_load(
        (_CATALOG_DIR / f"{catalog}.yaml").read_text(encoding="utf-8")
    )
    assert _writer_table_privileges(document) == _WRITE


@pytest.mark.unit
@pytest.mark.parametrize("catalog", _CATALOGS)
def test_the_rendered_catalog_matches_a_fresh_projection(catalog: str) -> None:
    document = yaml.safe_load(
        (_CATALOG_DIR / f"{catalog}.yaml").read_text(encoding="utf-8")
    )
    source = document["source"]
    instance_bytes = (_ROOT / source["path"]).read_bytes()
    assert source["sha256"] == hashlib.sha256(instance_bytes).hexdigest()
    assert document == render_database_projection(catalog)


@pytest.mark.unit
def test_the_ledger_hashes_match_the_node_migration_files() -> None:
    rows: dict[str, str] = {}
    for line in _LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        assert len(columns) == 6, line
        rows[columns[0]] = columns[-1]
    files = sorted(_NODE_MIGRATION_DIR.glob("*.sql"))
    assert [path.name for path in files] == [
        "0000_add_context_audit_event_key.sql",
        "0001_grant_tenant_projection_writer_context_audit_events.sql",
    ]
    for path in files:
        key = f"nodes/node_projection_context_audit/{path.name}"
        assert rows[key] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.unit
def test_the_bridge_declaration_names_the_existing_table_migration() -> None:
    declarations = [
        d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == _TABLE
    ]
    assert len(declarations) == 1
    assert declarations[0].table.schema == "public"
    assert declarations[0].table.access == "read_write"
    assert declarations[0].table.database_ref == "application"
    assert declarations[0].table.migration == (
        "docker/migrations/forward/nodes/node_projection_context_audit/"
        "0000_add_context_audit_event_key.sql"
    )
    assert declarations[0].contract_path == Path(declarations[0].table.migration)
    migration = _ROOT / declarations[0].table.migration
    assert migration.is_file()
    assert f"CREATE TABLE IF NOT EXISTS public.{_TABLE}" in migration.read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_the_shared_declaration_helper_keeps_each_entry_shape() -> None:
    by_node = {
        d.node: d
        for d in (*STATE_IO_TABLE_DECLARATIONS, *LEGACY_MIGRATION_TABLE_DECLARATIONS)
    }
    credentials = by_node["legacy_migration:tenant_inference_credentials"]
    assert credentials.table.access == "write"
    assert credentials.table.database_ref == "application"
    state_io = [d for node, d in by_node.items() if node.startswith("state_io:")]
    assert {
        "state_io:delegation_workflow_state",
        "state_io:session_phase_state",
        "state_io:pr_landing_workflow_state",
    } <= {d.node for d in state_io}
    for declaration in state_io:
        assert declaration.table.database_ref == "omnibase_infra"
        assert declaration.table.access == "read_write"
    for declaration in (credentials, *state_io):
        assert declaration.contract_path == Path(declaration.table.migration)
