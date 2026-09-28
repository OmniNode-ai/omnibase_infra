# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The source witness reads full-current evidence without a synthetic cursor."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

import pytest

from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    _OWNER_PROOF_MINT,
    DelegationOwnerProof,
)
from omnibase_infra.runtime.db.sim_archive_source_ledger_reader import (
    PostgresSimArchiveSourceLedgerReader,
)
from tests.unit.runtime.test_sim_archive_source_receipt_omn19728 import (
    _fixture,
    _topology,
)


class _Connection:
    def __init__(self, rows: tuple[object, ...]) -> None:
        self.rows = rows
        self.sql = ""
        self.readonly = False

    @asynccontextmanager
    async def transaction(self, *, isolation: str, readonly: bool) -> Any:
        assert isolation == "repeatable_read"
        self.readonly = readonly
        yield self

    async def fetch(self, sql: str, *_args: object) -> list[dict[str, object]]:
        self.sql = sql
        result = [asdict(row) for row in self.rows]
        for row in result:
            row.pop("ingest_watermark", None)
        return result


class _Pool:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self) -> Any:
        yield self.connection


@pytest.mark.unit
@pytest.mark.asyncio
async def test_source_witness_is_readonly_and_does_not_select_watermark() -> None:
    authority, _verifier, _raw, rows, _source, _calls = _fixture()
    owner = DelegationOwnerProof(
        correlation_id=authority.correlation_id,
        tenant_id=authority.tenant_id,
        _mint=_OWNER_PROOF_MINT,
    )
    connection = _Connection(rows)
    reader = PostgresSimArchiveSourceLedgerReader(_Pool(connection))

    observed = await reader.read_full_current(authority, owner, _topology().read_set)

    assert connection.readonly is True
    assert "ingest_watermark" not in connection.sql
    assert len(observed) == 5
    assert all(row.ingest_watermark is None for row in observed)
