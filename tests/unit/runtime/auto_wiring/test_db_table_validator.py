# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the warning-based preflight db_table_validator.

This is Phase 1 — warning only, does NOT block wiring.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_table_validator_warns_on_missing_table() -> None:
    """Preflight must emit a warning when a declared table doesn't exist."""
    from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)  # table not found

    contracts = [
        {
            "name": "node_projection_test",
            "db_io": {
                "db_tables": [
                    {
                        "name": "nonexistent_table",
                        "migration": "0099.sql",
                        "database": "omnidash_analytics",
                        "role": "events",
                    }
                ]
            },
        }
    ]
    warnings = await validate_db_tables(contracts, db_conn=mock_conn)

    assert len(warnings) == 1
    assert warnings[0]["reason"] == "missing_db_table"
    assert warnings[0]["severity"] == "warning"
    assert warnings[0]["details"]["table"] == "nonexistent_table"
    assert warnings[0]["details"]["database"] == "omnidash_analytics"
    assert warnings[0]["details"]["node"] == "node_projection_test"


@pytest.mark.asyncio
async def test_table_validator_no_warnings_when_table_exists() -> None:
    """No warnings produced when the declared table exists."""
    from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value="delegation_events")  # table found

    contracts = [
        {
            "name": "node_projection_delegation",
            "db_io": {
                "db_tables": [
                    {
                        "name": "delegation_events",
                        "migration": "0007_delegation_events.sql",
                        "database": "omnidash_analytics",
                        "role": "events",
                    }
                ]
            },
        }
    ]
    warnings = await validate_db_tables(contracts, db_conn=mock_conn)

    assert warnings == []


@pytest.mark.asyncio
async def test_table_validator_backwards_compatible_no_db_io() -> None:
    """Contracts with no db_io section produce zero warnings."""
    from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)

    contracts = [
        {
            "name": "node_no_db",
            # no db_io key at all
        }
    ]
    warnings = await validate_db_tables(contracts, db_conn=mock_conn)

    assert warnings == []
    mock_conn.fetchval.assert_not_called()


@pytest.mark.asyncio
async def test_table_validator_multiple_tables_partial_missing() -> None:
    """Only missing tables produce warnings; present tables are silent."""
    from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables

    # First call: table found; second call: table missing
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(side_effect=["existing_table", None])

    contracts = [
        {
            "name": "node_projection_baselines",
            "db_io": {
                "db_tables": [
                    {
                        "name": "baselines_snapshots",
                        "migration": "0001.sql",
                        "database": "omnidash_analytics",
                        "role": "snapshots",
                    },
                    {
                        "name": "baselines_missing",
                        "migration": "0001.sql",
                        "database": "omnidash_analytics",
                        "role": "missing",
                    },
                ]
            },
        }
    ]
    warnings = await validate_db_tables(contracts, db_conn=mock_conn)

    assert len(warnings) == 1
    assert warnings[0]["details"]["table"] == "baselines_missing"


@pytest.mark.asyncio
async def test_table_validator_default_database_name() -> None:
    """When no database field is declared, defaults to omnidash_analytics."""
    from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)

    contracts = [
        {
            "name": "node_projection_minimal",
            "db_io": {
                "db_tables": [
                    {
                        "name": "some_table",
                        "migration": "0001.sql",
                        "role": "data",
                        # no database key
                    }
                ]
            },
        }
    ]
    warnings = await validate_db_tables(contracts, db_conn=mock_conn)

    assert len(warnings) == 1
    assert warnings[0]["details"]["database"] == "omnidash_analytics"


@pytest.mark.asyncio
async def test_table_validator_does_not_raise_on_missing() -> None:
    """validate_db_tables must return warnings, never raise, even for all-missing tables."""
    from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables

    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)

    contracts = [
        {
            "name": "node_a",
            "db_io": {
                "db_tables": [
                    {"name": "t1", "migration": "x.sql", "database": "db", "role": "r1"}
                ]
            },
        },
        {
            "name": "node_b",
            "db_io": {
                "db_tables": [
                    {"name": "t2", "migration": "x.sql", "database": "db", "role": "r2"}
                ]
            },
        },
    ]
    # Must not raise
    warnings = await validate_db_tables(contracts, db_conn=mock_conn)

    assert len(warnings) == 2
    nodes = {w["details"]["node"] for w in warnings}
    assert nodes == {"node_a", "node_b"}
