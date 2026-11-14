# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 116 creates the branch deploy orchestrator's state_io table (OMN-24118).

omnimarket's ``node_branch_deploy_orchestrator`` declares
``state_io: {database: omnibase_infra, table: branch_deploy_state,
key: correlation_id}``. The runtime's state_io seam loads that row before every
leg and CAS-writes it after, through the same ``StateStoreAdapter`` that reads
``delegation_workflow_state`` (090 + 093), ``session_phase_state`` (102) and
``pr_landing_workflow_state`` (108).
This module pins:

* the migration exists, creates the table in the flat omnibase_infra stream,
  keys it on ``correlation_id`` and carries every column the adapter's SQL names,
  in the same shape as 102 and 108 with the key column renamed;
* the adapter accepts the table and key names (its identifier guard);
* the table participates in grant derivation through
  ``STATE_IO_TABLE_DECLARATIONS``, and every shipped instance grants it to the
  runtime, so ``--check`` cannot report green while the relation is ungranted;
* the migration grants SELECT, INSERT, UPDATE to role_omnibase_infra on
  public.branch_deploy_state and does not grant DELETE;
* the omnimarket ownership manifest strings: table branch_deploy_state,
  function refresh_branch_deploy_state_updated_at, and the migration path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.state_io.state_store_adapter import StateStoreAdapter
from omnibase_infra.topology.table_grant_derivation import (
    STATE_IO_TABLE_DECLARATIONS,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
MIGRATION = FORWARD_DIR / "116_create_branch_deploy_state.sql"
SIBLING = FORWARD_DIR / "102_create_session_phase_state.sql"
SIBLING_108 = FORWARD_DIR / "108_create_pr_landing_workflow_state.sql"
INSTANCES_DIR = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
TABLE = "branch_deploy_state"
KEY = "correlation_id"
SHIPPED_INSTANCES = tuple(sorted(path.stem for path in INSTANCES_DIR.glob("*.yaml")))
assert len(SHIPPED_INSTANCES) == 3, SHIPPED_INSTANCES

_COLUMN_RE = re.compile(r"^\s{4}([a-z_]+)\s+([A-Z]+)", re.MULTILINE)


def _columns(sql: str, table: str) -> dict[str, str]:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS public\.{table} \((.*?)\n\);", sql, re.DOTALL
    )
    assert match is not None, f"no CREATE TABLE for public.{table}"
    return dict(_COLUMN_RE.findall(match.group(1)))


@pytest.mark.unit
def test_the_migration_creates_the_table_keyed_on_correlation_id() -> None:
    assert MIGRATION.is_file(), f"{MIGRATION.name} is missing"
    sql = MIGRATION.read_text(encoding="utf-8")
    assert re.search(rf"^\s{{4}}{KEY}\s+TEXT PRIMARY KEY,", sql, re.MULTILINE)
    assert f"trg_{TABLE}_updated_at" in sql


@pytest.mark.unit
def test_the_columns_are_the_adapter_shape_of_migrations_102_and_108() -> None:
    ours = _columns(MIGRATION.read_text(encoding="utf-8"), TABLE)
    sibling = _columns(SIBLING.read_text(encoding="utf-8"), "session_phase_state")
    # Positive control: the parser reads the sibling's real column set.
    assert {"session_id", "payload", "pending_emissions"} <= set(sibling)
    sibling[KEY] = sibling.pop("session_id")
    assert ours == sibling
    # The 108 sibling, keyed landing_key, has the same shape under its own key.
    landing = _columns(
        SIBLING_108.read_text(encoding="utf-8"), "pr_landing_workflow_state"
    )
    assert {"landing_key", "payload", "pending_emissions"} <= set(landing)
    landing[KEY] = landing.pop("landing_key")
    assert ours == landing


@pytest.mark.unit
def test_the_adapter_accepts_the_table_and_key() -> None:
    # The adapter validates both identifiers before it ever interpolates them
    # into SQL; construction opens no connection.
    adapter = StateStoreAdapter(
        "postgresql://unused.invalid/omnibase_infra", table=TABLE, key_column=KEY
    )
    assert adapter is not None


@pytest.mark.unit
def test_the_table_is_a_state_io_grant_declaration() -> None:
    (declaration,) = (d for d in STATE_IO_TABLE_DECLARATIONS if d.table.name == TABLE)
    assert declaration.table.database_ref == "omnibase_infra"
    assert declaration.table.schema == "public"
    assert declaration.table.access == "read_write"
    assert declaration.table.migration == str(MIGRATION.relative_to(REPO_ROOT))


@pytest.mark.parametrize("instance", SHIPPED_INSTANCES)
@pytest.mark.unit
def test_every_shipped_instance_grants_the_table(instance: str) -> None:
    raw = yaml.safe_load((INSTANCES_DIR / f"{instance}.yaml").read_text("utf-8"))
    text = yaml.safe_dump(raw)
    # Positive control: the sibling state_io table is granted in the same file.
    assert "session_phase_state" in text
    assert TABLE in text


@pytest.mark.unit
def test_the_migration_grants_select_insert_update_but_not_delete() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert re.search(
        r"GRANT\s+SELECT,\s*INSERT,\s*UPDATE\s+ON\s+TABLE\s+public\.branch_deploy_state\s+TO\s+role_omnibase_infra",
        sql,
    )
    assert not re.search(r"GRANT[^;]*\bDELETE\b", sql)


@pytest.mark.unit
def test_the_omnimarket_ownership_manifest_strings_are_pinned() -> None:
    sql = MIGRATION.read_text(encoding="utf-8")
    assert TABLE in sql
    assert "refresh_branch_deploy_state_updated_at" in sql
    assert "docker/migrations/forward/116_create_branch_deploy_state.sql" in str(
        MIGRATION.relative_to(REPO_ROOT)
    )
