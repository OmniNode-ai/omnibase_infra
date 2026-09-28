# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 108 creates the PR landing workflow's state_io table (OMN-19829).

omnimarket's ``node_pr_landing_orchestrator`` declares
``state_io: {database: omnibase_infra, table: pr_landing_workflow_state,
key: landing_key}``. The runtime's state_io seam loads that row before every
leg and CAS-writes it after, through the same ``StateStoreAdapter`` that reads
``delegation_workflow_state`` (090 + 093) and ``session_phase_state`` (102).
This module pins:

* the migration exists, creates the table in the flat omnibase_infra stream,
  keys it on ``landing_key`` and carries every column the adapter's SQL names,
  in the same shape as 102 with the key column renamed;
* the adapter accepts the table and key names (its identifier guard);
* the table participates in grant derivation through
  ``STATE_IO_TABLE_DECLARATIONS``, and every shipped instance grants it to the
  runtime, so ``--check`` cannot report green while the relation is ungranted;
* ordinal 108, not the burned 107.
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
MIGRATION = FORWARD_DIR / "108_create_pr_landing_workflow_state.sql"
SIBLING = FORWARD_DIR / "102_create_session_phase_state.sql"
INSTANCES_DIR = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
TABLE = "pr_landing_workflow_state"
KEY = "landing_key"
SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

_COLUMN_RE = re.compile(r"^\s{4}([a-z_]+)\s+([A-Z]+)", re.MULTILINE)


def _columns(sql: str, table: str) -> dict[str, str]:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS public\.{table} \((.*?)\n\);", sql, re.DOTALL
    )
    assert match is not None, f"no CREATE TABLE for public.{table}"
    return dict(_COLUMN_RE.findall(match.group(1)))


@pytest.mark.unit
def test_the_migration_creates_the_table_keyed_on_landing_key() -> None:
    assert MIGRATION.is_file(), f"{MIGRATION.name} is missing"
    sql = MIGRATION.read_text(encoding="utf-8")
    assert re.search(rf"^\s{{4}}{KEY}\s+TEXT PRIMARY KEY,", sql, re.MULTILINE)
    assert f"trg_{TABLE}_updated_at" in sql


@pytest.mark.unit
def test_the_columns_are_the_adapter_shape_of_migration_102() -> None:
    ours = _columns(MIGRATION.read_text(encoding="utf-8"), TABLE)
    sibling = _columns(SIBLING.read_text(encoding="utf-8"), "session_phase_state")
    # Positive control: the parser reads the sibling's real column set.
    assert {"session_id", "payload", "pending_emissions"} <= set(sibling)
    sibling[KEY] = sibling.pop("session_id")
    assert ours == sibling


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
def test_the_burned_ordinal_107_is_not_reused() -> None:
    assert not any(path.name.startswith("107_") for path in FORWARD_DIR.glob("*.sql"))
