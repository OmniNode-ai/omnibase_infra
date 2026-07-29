# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed physical-table preflight tests (OMN-15418)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.db_table_validator import validate_db_tables
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
)

pytestmark = pytest.mark.unit


def _contract(*tables: ModelDbTableDeclaration) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_projection_fixture",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/typed/contract.yaml"),
        entry_point_name="node_projection_fixture",
        package_name="fixture",
        db_io=ModelDbOwnershipSubcontract(db_tables=list(tables)) if tables else None,
    )


def _table(name: str, *, schema: str = "tenant", role: str = "events"):
    return ModelDbTableDeclaration(
        name=name,
        database_ref="application",
        schema=schema,
        migration=f"tests/{name}.sql",
        access="read_write",
        role=role,
    )


@pytest.mark.asyncio
async def test_typed_table_present_returns_no_warning() -> None:
    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value="delegation_events")

    warnings = await validate_db_tables(
        (_contract(_table("delegation_events")),), connection
    )

    assert warnings == ()
    connection.fetchval.assert_awaited_once_with(
        "SELECT tablename FROM pg_tables WHERE schemaname = $1 AND tablename = $2",
        "tenant",
        "delegation_events",
    )


@pytest.mark.asyncio
async def test_missing_table_warning_preserves_typed_location() -> None:
    connection = MagicMock()
    connection.fetchval = AsyncMock(return_value=None)

    warnings = await validate_db_tables(
        (_contract(_table("delegation_events")),), connection
    )

    assert len(warnings) == 1
    assert warnings[0].table == "delegation_events"
    assert warnings[0].database_ref == "application"
    assert warnings[0].schema == "tenant"
    assert warnings[0].node == "node_projection_fixture"


@pytest.mark.asyncio
async def test_multiple_typed_schemas_are_queried_exactly() -> None:
    connection = MagicMock()
    connection.fetchval = AsyncMock(side_effect=["delegation_events", None])
    contract = _contract(
        _table("delegation_events", role="events"),
        _table(
            "schema_migrations",
            schema="omninode_internal",
            role="migration_ledger",
        ),
    )

    warnings = await validate_db_tables((contract,), connection)

    assert [warning.table for warning in warnings] == ["schema_migrations"]
    assert connection.fetchval.await_args_list[1].args[1:] == (
        "omninode_internal",
        "schema_migrations",
    )


@pytest.mark.asyncio
async def test_contract_without_db_io_does_not_query_catalog() -> None:
    connection = MagicMock()
    connection.fetchval = AsyncMock()

    warnings = await validate_db_tables((_contract(),), connection)

    assert warnings == ()
    connection.fetchval.assert_not_awaited()
