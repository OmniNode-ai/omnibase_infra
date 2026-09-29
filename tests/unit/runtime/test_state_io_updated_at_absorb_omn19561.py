# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Duplicate absorbs preserve the delegation completion-bound clock (OMN-19561)."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

FORWARD_DIR = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
MIGRATION = (
    FORWARD_DIR / "109_delegation_workflow_state_updated_at_only_on_transition.sql"
)
PREREQUISITES = (
    "090_create_delegation_workflow_state.sql",
    "093_add_delegation_workflow_state_outbox_columns.sql",
    "106_add_delegation_workflow_state_traffic_class.sql",
)


@pytest.mark.unit
def test_migration_preserves_updated_at_when_transition_fields_are_unchanged() -> None:
    sql = re.sub(r"--[^\n]*", "", MIGRATION.read_text(encoding="utf-8"))
    sql = " ".join(sql.upper().split())
    assert (
        "CREATE OR REPLACE FUNCTION REFRESH_DELEGATION_WORKFLOW_STATE_UPDATED_AT()"
        in sql
    )
    for column in ("STATE", "PAYLOAD", "TENANT_ID"):
        assert f"NEW.{column} IS NOT DISTINCT FROM OLD.{column}" in sql
    assert "THEN NEW.UPDATED_AT = OLD.UPDATED_AT;" in sql
    assert "ELSE NEW.UPDATED_AT = NOW();" in sql


@pytest.mark.integration
class TestUpdatedAtAbsorb:
    """Exercise the shipped trigger in an isolated schema on an opt-in database."""

    @pytest.mark.asyncio
    async def test_absorb_preserves_timestamp_and_transition_advances_it(self) -> None:
        dsn = os.environ.get("OMNIBASE_INFRA_DB_URL")
        if not dsn:
            pytest.skip("OMNIBASE_INFRA_DB_URL is not set")

        schema = f"test_updated_at_omn19561_{uuid4().hex}"
        conn = await asyncpg.connect(dsn, timeout=10)
        try:
            await conn.execute(f'CREATE SCHEMA "{schema}"')
            try:
                # No public fallback: migrations must only touch scratch objects.
                await conn.execute(f'SET search_path TO "{schema}"')
                for filename in PREREQUISITES:
                    await conn.execute((FORWARD_DIR / filename).read_text("utf-8"))
                sql = MIGRATION.read_text(encoding="utf-8")
                await conn.execute(sql)
                await conn.execute(sql)  # Reapplying the migration is idempotent.

                # Each statement commits separately: NOW() is transaction-scoped.
                seeded_at = await conn.fetchval(
                    """
                    INSERT INTO delegation_workflow_state
                        (correlation_id, tenant_id, state, payload)
                    VALUES ('absorb', 'tenant', 'ROUTED', '{}'::jsonb)
                    RETURNING updated_at
                    """
                )
                assert seeded_at is not None
                await asyncio.sleep(0.05)
                absorbed = await conn.fetchrow(
                    """
                    UPDATE delegation_workflow_state
                    SET version = version + 1, state = 'ROUTED',
                        payload = '{}'::jsonb, tenant_id = 'tenant'
                    WHERE correlation_id = 'absorb' AND version = 0
                    RETURNING updated_at, version
                    """
                )
                assert absorbed is not None
                assert absorbed["version"] == 1
                assert absorbed["updated_at"] == seeded_at

                transitioned_at = await conn.fetchval(
                    """
                    UPDATE delegation_workflow_state
                    SET version = version + 1, state = 'COMPLETED'
                    WHERE correlation_id = 'absorb' AND version = 1
                    RETURNING updated_at
                    """
                )
                assert transitioned_at is not None
                assert transitioned_at > seeded_at
            finally:
                await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        finally:
            await conn.close()
