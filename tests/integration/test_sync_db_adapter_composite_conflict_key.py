# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration contract for sync DB adapter composite conflict key support (OMN-11299).

Verifies that the projection upsert adapter correctly generates SQL with
composite ON CONFLICT clauses when multiple conflict columns are provided,
matching the savings_estimates projection schema.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
)
from tests.helpers.application_db_topology import (
    projection_database_target,
    projection_database_urls,
)

pytestmark = [pytest.mark.integration]


def _internal_adapter(table: str) -> object:
    target = projection_database_target(table, schema="omninode_internal")
    return _build_projection_db_adapter(
        projection_database_urls(target, "postgresql://user:pass@host/db"),
        target,
        None,
        None,
    )


def test_sync_db_adapter_composite_conflict_key_sql_shape() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = ("omninode_runtime", "omnidash_analytics")
    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor_context

    with patch("psycopg2.connect", return_value=conn):
        adapter = _internal_adapter("cost_by_repo_snapshots")
        result = adapter.upsert(
            "cost_by_repo_snapshots",
            "session_id,event_timestamp,model_local,model_cloud_baseline",
            {
                "session_id": "sess-1",
                "event_timestamp": "2026-05-20T20:00:00+00:00",
                "model_local": "local-model",
                "model_cloud_baseline": "cloud-model",
                "savings_usd": "0.001",
            },
        )

    assert result is True
    sql = cursor.execute.call_args.args[0]
    assert (
        'ON CONFLICT ("session_id", "event_timestamp", '
        '"model_local", "model_cloud_baseline") DO UPDATE SET'
    ) in sql
    assert '"savings_usd" = EXCLUDED."savings_usd"' in sql
    assert '"session_id" = EXCLUDED."session_id"' not in sql


def test_sync_db_adapter_single_conflict_key_unchanged() -> None:
    cursor = MagicMock()
    cursor.fetchone.return_value = ("omninode_runtime", "omnidash_analytics")
    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor_context

    with patch("psycopg2.connect", return_value=conn):
        adapter = _internal_adapter("node_service_registry")
        result = adapter.upsert(
            "node_service_registry",
            "node_id",
            {"node_id": "n1", "status": "active"},
        )

    assert result is True
    sql = cursor.execute.call_args.args[0]
    assert 'ON CONFLICT ("node_id") DO UPDATE SET' in sql
    assert '"status" = EXCLUDED."status"' in sql
