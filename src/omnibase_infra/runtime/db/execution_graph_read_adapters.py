# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request-time owner and exhaustive current ledger reads for graph replay.

These readers are IO boundaries, not graph semantics. They take only a sealed
gateway-verified authority and a pinned topology read set; neither a naked
tenant nor a caller-selected replay cursor can narrow the current scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import asyncpg

from omnibase_core.models.execution_graph_replay.model_execution_graph_stored_chain_annotation import (
    ModelExecutionGraphStoredChainAnnotation,
)
from omnibase_infra.runtime.db.protocol_delegation_owner_reader import (
    ProtocolDelegationOwnerReader,
)
from omnibase_infra.runtime.db.protocol_execution_graph_ledger_reader import (
    ProtocolExecutionGraphLedgerReader,
)
from omnibase_infra.runtime.db.protocol_execution_graph_stored_chain_reader import (
    ProtocolExecutionGraphStoredChainReader,
)
from omnibase_infra.runtime.db.protocol_execution_graph_verdict_candidate_reader import (
    ProtocolExecutionGraphVerdictCandidateReader,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    VerifiedExecutionGraphReadAuthority,
)

_SQL_SET_LOCAL_TENANT = "SELECT set_config('app.tenant_id', $1::text, true)"
_SQL_READ_OWNER = """
SELECT correlation_id, tenant_id::text AS tenant_id
FROM public.delegation_events
WHERE correlation_id = $1::text
  AND tenant_id::text = $2::text
"""
_SQL_READ_FULL_CURRENT = """
SELECT
    ledger_entry_id, topic, partition, kafka_offset, event_key, event_value,
    onex_headers::text AS onex_headers, envelope_id, correlation_id, event_type,
    source, event_timestamp, ledger_written_at
FROM public.event_ledger
WHERE correlation_id = $1::uuid
  AND topic = ANY($2::text[])
ORDER BY topic, partition, kafka_offset
"""
_SQL_READ_VERIFICATION_IDS = """
SELECT DISTINCT correlation_id
FROM omninode_internal.dod_verify_runs
WHERE delegation_correlation_id = $1::uuid
  AND correlation_id IS NOT NULL
ORDER BY correlation_id
"""
_SQL_READ_VERDICT_TERMINALS = """
SELECT
    ledger_entry_id, topic, partition, kafka_offset, event_key, event_value,
    onex_headers::text AS onex_headers, envelope_id, correlation_id, event_type,
    source, event_timestamp, ledger_written_at
FROM public.event_ledger
WHERE correlation_id = ANY($1::uuid[])
  AND topic = $2::text
ORDER BY topic, partition, kafka_offset
"""
_SQL_READ_STORED_CHAIN = """
SELECT envelope_id, hop_index, replay_green, verifier_verdict
FROM public.ledger_chain
WHERE correlation_id = $1::text
  AND envelope_id = ANY($2::text[])
ORDER BY hop_index, envelope_id
"""
_OWNER_PROOF_MINT = object()
_PINNED_READ_SET_MINT = object()


class ExecutionGraphOwnerNotFoundError(LookupError):
    """Uniform denial for absent, foreign, or ambiguous delegation ownership."""


@dataclass(frozen=True, slots=True, init=False)
class PinnedExecutionGraphReadSet:
    """Immutable topic boundary supplied by a versioned topology/contract registry."""

    topology_version: str
    topics: frozenset[str]
    head_topic: str
    verdict_topic: str

    def __init__(
        self,
        *,
        topology_version: str,
        topics: frozenset[str],
        head_topic: str,
        verdict_topic: str,
        _mint: object = None,
    ) -> None:
        if _mint is not _PINNED_READ_SET_MINT:
            raise TypeError("Pinned read set requires a verified topology artifact")
        object.__setattr__(self, "topology_version", topology_version)
        object.__setattr__(self, "topics", topics)
        object.__setattr__(self, "head_topic", head_topic)
        object.__setattr__(self, "verdict_topic", verdict_topic)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            not self.topology_version
            or not self.topics
            or not self.head_topic
            or not self.verdict_topic
            or self.head_topic == self.verdict_topic
            or self.head_topic not in self.topics
            or self.verdict_topic not in self.topics
            or any(not topic or topic != topic.strip() for topic in self.topics)
        ):
            raise ValueError("Invalid pinned execution graph read set")


@dataclass(frozen=True, slots=True, init=False)
class DelegationOwnerProof:
    """Current analytics binding for the signed request correlation."""

    correlation_id: UUID
    tenant_id: UUID

    def __init__(self, *, correlation_id: UUID, tenant_id: UUID, _mint: object) -> None:
        if _mint is not _OWNER_PROOF_MINT:
            raise TypeError("Delegation owner proof requires an analytics read")
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "tenant_id", tenant_id)


@dataclass(frozen=True, slots=True)
class ExecutionGraphLedgerRecord:
    """One unredacted raw ledger row; never serialized directly to clients."""

    ledger_entry_id: UUID
    topic: str
    partition: int
    kafka_offset: int
    event_key: bytes | None
    event_value: bytes
    onex_headers: str
    envelope_id: UUID | None
    correlation_id: UUID
    event_type: str | None
    source: str | None
    event_timestamp: datetime | None
    ledger_written_at: datetime


@dataclass(frozen=True, slots=True)
class ExecutionGraphCurrentEvidence:
    """Owner-first current evidence; authorization must inspect all records."""

    owner: DelegationOwnerProof
    ledger_rows: tuple[ExecutionGraphLedgerRecord, ...]


@dataclass(frozen=True, slots=True)
class ExecutionGraphVerdictCandidates:
    """Index-discovered verification IDs and raw terminal evidence only."""

    verification_correlation_ids: tuple[UUID, ...]
    ledger_rows: tuple[ExecutionGraphLedgerRecord, ...]


def _require_authority(authority: VerifiedExecutionGraphReadAuthority) -> None:
    if type(authority) is not VerifiedExecutionGraphReadAuthority:
        raise TypeError("Graph evidence read requires sealed gateway authority")


def _require_matching_owner(
    authority: VerifiedExecutionGraphReadAuthority, owner: DelegationOwnerProof
) -> None:
    _require_authority(authority)
    if (
        type(owner) is not DelegationOwnerProof
        or owner.tenant_id != authority.tenant_id
        or owner.correlation_id != authority.correlation_id
    ):
        raise ExecutionGraphOwnerNotFoundError(
            "Execution graph correlation was not found"
        )


class PostgresDelegationOwnerReader:
    """Analytics ownership read with explicit predicate and tenant-local GUC."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def require_owner(
        self, authority: VerifiedExecutionGraphReadAuthority
    ) -> DelegationOwnerProof:
        _require_authority(authority)
        async with self._pool.acquire() as connection:
            async with connection.transaction(
                isolation="repeatable_read", readonly=True
            ):
                # Parameterized set_config(..., true) is PostgreSQL's SET LOCAL
                # equivalent and is scoped to this same explicit transaction.
                await connection.execute(
                    _SQL_SET_LOCAL_TENANT, str(authority.tenant_id)
                )
                rows = await connection.fetch(
                    _SQL_READ_OWNER,
                    str(authority.correlation_id),
                    str(authority.tenant_id),
                )
        if len(rows) != 1:
            raise ExecutionGraphOwnerNotFoundError(
                "Execution graph correlation was not found"
            )
        row = rows[0]
        if row["correlation_id"] != str(authority.correlation_id) or row[
            "tenant_id"
        ] != str(authority.tenant_id):
            raise ExecutionGraphOwnerNotFoundError(
                "Execution graph correlation was not found"
            )
        return DelegationOwnerProof(
            correlation_id=authority.correlation_id,
            tenant_id=authority.tenant_id,
            _mint=_OWNER_PROOF_MINT,
        )


class PostgresExecutionGraphLedgerReader:
    """Unbounded correlation snapshot across pinned topics and all partitions."""

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
            raise TypeError("Graph evidence read requires a pinned read set")
        async with self._pool.acquire() as connection:
            async with connection.transaction(
                isolation="repeatable_read", readonly=True
            ):
                rows = await connection.fetch(
                    _SQL_READ_FULL_CURRENT,
                    authority.correlation_id,
                    sorted(read_set.topics),
                )
        return tuple(ExecutionGraphLedgerRecord(**dict(row)) for row in rows)


class PostgresExecutionGraphVerdictCandidateReader:
    """Use the projection only as a verification-ID index, never verdict truth."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def read_full_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        owner: DelegationOwnerProof,
        read_set: PinnedExecutionGraphReadSet,
    ) -> ExecutionGraphVerdictCandidates:
        _require_matching_owner(authority, owner)
        if type(read_set) is not PinnedExecutionGraphReadSet:
            raise TypeError("Verdict evidence read requires a pinned read set")
        async with self._pool.acquire() as connection:
            async with connection.transaction(
                isolation="repeatable_read", readonly=True
            ):
                index_rows = await connection.fetch(
                    _SQL_READ_VERIFICATION_IDS, owner.correlation_id
                )
                verification_ids = tuple(row["correlation_id"] for row in index_rows)
                rows = (
                    await connection.fetch(
                        _SQL_READ_VERDICT_TERMINALS,
                        verification_ids,
                        read_set.verdict_topic,
                    )
                    if verification_ids
                    else []
                )
        return ExecutionGraphVerdictCandidates(
            verification_correlation_ids=verification_ids,
            ledger_rows=tuple(ExecutionGraphLedgerRecord(**dict(row)) for row in rows),
        )


class PostgresExecutionGraphStoredChainReader:
    """Current comparison rows, filtered to fully admitted envelope identities."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def read_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        owner: DelegationOwnerProof,
        owned_envelope_ids: tuple[UUID, ...],
    ) -> tuple[ModelExecutionGraphStoredChainAnnotation, ...]:
        _require_matching_owner(authority, owner)
        if not owned_envelope_ids or any(
            type(envelope_id) is not UUID or envelope_id.int == 0
            for envelope_id in owned_envelope_ids
        ):
            raise ValueError("Stored chain read requires admitted envelope identities")
        owned = {str(envelope_id) for envelope_id in owned_envelope_ids}
        async with self._pool.acquire() as connection:
            async with connection.transaction(
                isolation="repeatable_read", readonly=True
            ):
                rows = await connection.fetch(
                    _SQL_READ_STORED_CHAIN,
                    str(owner.correlation_id),
                    sorted(owned),
                )
        return tuple(
            ModelExecutionGraphStoredChainAnnotation(
                node_id=UUID(row["envelope_id"]),
                hop_index=row["hop_index"],
                replay_green=row["replay_green"],
                verifier_verdict=row["verifier_verdict"],
            )
            for row in rows
            if row["envelope_id"] in owned
        )


class ExecutionGraphCurrentEvidenceReader:
    """Orchestrate the non-negotiable owner-first IO ordering."""

    def __init__(
        self,
        owner_reader: ProtocolDelegationOwnerReader,
        ledger_reader: ProtocolExecutionGraphLedgerReader,
    ) -> None:
        self._owner_reader = owner_reader
        self._ledger_reader = ledger_reader

    async def read_authorized_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        read_set: PinnedExecutionGraphReadSet,
    ) -> ExecutionGraphCurrentEvidence:
        _require_authority(authority)
        owner = await self._owner_reader.require_owner(authority)
        ledger_rows = await self._ledger_reader.read_full_current(
            authority, owner, read_set
        )
        return ExecutionGraphCurrentEvidence(owner=owner, ledger_rows=ledger_rows)


__all__ = [
    "DelegationOwnerProof",
    "ExecutionGraphCurrentEvidence",
    "ExecutionGraphCurrentEvidenceReader",
    "ExecutionGraphLedgerRecord",
    "ExecutionGraphOwnerNotFoundError",
    "ExecutionGraphVerdictCandidates",
    "PostgresDelegationOwnerReader",
    "PostgresExecutionGraphLedgerReader",
    "PostgresExecutionGraphStoredChainReader",
    "PostgresExecutionGraphVerdictCandidateReader",
    "PinnedExecutionGraphReadSet",
    "ProtocolDelegationOwnerReader",
    "ProtocolExecutionGraphLedgerReader",
    "ProtocolExecutionGraphStoredChainReader",
    "ProtocolExecutionGraphVerdictCandidateReader",
]
