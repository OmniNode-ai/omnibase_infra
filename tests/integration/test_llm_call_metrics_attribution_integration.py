# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration checks for LLM call metric attribution wiring (OMN-10333)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_infra.services.observability.llm_cost_aggregation.writer_postgres import (
    WriterLlmCostAggregationPostgres,
)

pytestmark = [pytest.mark.integration]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORWARD_MIGRATION = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "072_add_llm_call_metrics_attribution.sql"
)


def _mock_pool() -> MagicMock:
    pool = MagicMock()
    conn = AsyncMock()
    transaction_cm = MagicMock()
    transaction_cm.__aenter__ = AsyncMock(return_value=None)
    transaction_cm.__aexit__ = AsyncMock(return_value=False)
    conn.transaction = MagicMock(return_value=transaction_cm)
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool


@pytest.mark.asyncio
async def test_attribution_migration_and_writer_bind_same_columns() -> None:
    """Migration columns are populated by the raw metric writer insert."""
    migration_sql = _FORWARD_MIGRATION.read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS repo_name VARCHAR(255)" in migration_sql
    assert "ADD COLUMN IF NOT EXISTS machine_id VARCHAR(255)" in migration_sql

    pool = _mock_pool()
    writer = WriterLlmCostAggregationPostgres(pool=pool)
    result = await writer.write_call_metrics(
        [
            {
                "model_id": "gpt-4o",
                "session_id": "session-omni-10333",
                "input_hash": "sha256-omni-10333-attribution-proof",
                "repo_name": "omnibase_infra",
                "machine_id": "devbox-201",
                "reporting_source": "integration-test",
            }
        ]
    )

    assert result == 1
    conn = pool.acquire.return_value.__aenter__.return_value
    insert_call = next(
        call
        for call in conn.execute.call_args_list
        if "INSERT INTO llm_call_metrics" in call.args[0]
    )
    assert "repo_name, machine_id, source" in insert_call.args[0]
    assert insert_call.args[16] == "omnibase_infra"
    assert insert_call.args[17] == "devbox-201"
