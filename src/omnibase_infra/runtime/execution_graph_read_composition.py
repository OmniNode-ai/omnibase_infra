# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose graph reads across the authoritative analytics and ledger databases."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import asyncpg

from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    ExecutionGraphCurrentEvidenceReader,
    PostgresDelegationOwnerReader,
    PostgresExecutionGraphLedgerReader,
)
from omnibase_infra.runtime.db.protocol_execution_graph_stored_chain_reader import (
    ProtocolExecutionGraphStoredChainReader,
)
from omnibase_infra.runtime.execution_graph_read_command_handler import (
    ExecutionGraphReadCommandExecutor,
    ExecutionGraphReadFold,
    ExecutionGraphTerminalPublisher,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PinnedExecutionGraphTopology,
)
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)


@asynccontextmanager
async def open_execution_graph_read_pools(
    config: ModelExecutionGraphReadDatabases,
) -> AsyncIterator[tuple[asyncpg.Pool, asyncpg.Pool]]:
    """Open and close the analytics owner pool and ledger evidence pool separately."""
    analytics_pool = await asyncpg.create_pool(
        dsn=config.analytics_dsn.get_secret_value()
    )
    try:
        ledger_pool = await asyncpg.create_pool(
            dsn=config.ledger_dsn.get_secret_value()
        )
        try:
            yield analytics_pool, ledger_pool
        finally:
            await ledger_pool.close()
    finally:
        await analytics_pool.close()


def compose_execution_graph_read_executor(
    *,
    analytics_pool: asyncpg.Pool,
    ledger_pool: asyncpg.Pool,
    topology: PinnedExecutionGraphTopology,
    workflow_type: str,
    fold: ExecutionGraphReadFold,
    stored_chain_reader_factory: Callable[
        [asyncpg.Pool], ProtocolExecutionGraphStoredChainReader
    ],
    publish_terminal: ExecutionGraphTerminalPublisher,
) -> ExecutionGraphReadCommandExecutor:
    """Compose chain reads only; verdict joins require a separate analytics reader.

    Owner admission uses analytics, while event-ledger and stored-chain evidence
    use the ledger database. No DoD verdict candidate reader is installed here.
    """
    if analytics_pool is ledger_pool:
        raise ValueError("graph owner and ledger reads require distinct database pools")
    return ExecutionGraphReadCommandExecutor(
        evidence_reader=ExecutionGraphCurrentEvidenceReader(
            PostgresDelegationOwnerReader(analytics_pool),
            PostgresExecutionGraphLedgerReader(ledger_pool),
        ),
        stored_chain_reader=stored_chain_reader_factory(ledger_pool),
        topology=topology,
        workflow_type=workflow_type,
        fold=fold,
        publish_terminal=publish_terminal,
    )


__all__ = ["compose_execution_graph_read_executor", "open_execution_graph_read_pools"]
