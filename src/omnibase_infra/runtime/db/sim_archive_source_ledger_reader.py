# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Full-current source witness for the pre-watermark archive database.

The sim source on .201 predates migration 109. This read is deliberately
separate from the live graph reader, whose replay bounds require a real ingest
watermark. A source witness has no replay bound and never invents one.
"""

from __future__ import annotations

import asyncpg

from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    DelegationOwnerProof,
    ExecutionGraphLedgerRecord,
    PinnedExecutionGraphReadSet,
    _ledger_record,
    _require_matching_owner,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)

_SQL_READ_SOURCE_FULL_CURRENT = """
SELECT
    ledger_entry_id, topic, partition, kafka_offset, event_key, event_value,
    onex_headers::text AS onex_headers, envelope_id, correlation_id, event_type,
    source, event_timestamp, ledger_written_at
FROM public.event_ledger
WHERE correlation_id = $1::uuid
  AND topic = ANY($2::text[])
ORDER BY topic, partition, kafka_offset
"""


class PostgresSimArchiveSourceLedgerReader:
    """Read every source-chain row, without assuming migration 109 exists."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def read_full_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        owner: DelegationOwnerProof,
        read_set: PinnedExecutionGraphReadSet,
    ) -> tuple[ExecutionGraphLedgerRecord, ...]:
        _require_matching_owner(authority, owner)
        if type(read_set) is not PinnedExecutionGraphReadSet:
            raise TypeError("Sim source witness requires a pinned read set")
        async with self._pool.acquire() as connection:
            async with connection.transaction(
                isolation="repeatable_read", readonly=True
            ):
                rows = await connection.fetch(
                    _SQL_READ_SOURCE_FULL_CURRENT,
                    authority.correlation_id,
                    sorted(read_set.topics),
                )
        return tuple(_ledger_record(row) for row in rows)


__all__ = ["PostgresSimArchiveSourceLedgerReader"]
