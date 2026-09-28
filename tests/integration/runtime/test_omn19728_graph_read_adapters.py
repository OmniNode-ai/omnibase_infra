# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Full-current graph reads refuse foreign owners and never page evidence."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_core.crypto.crypto_ed25519_signer import generate_keypair
from omnibase_core.models.envelope.model_message_envelope import ModelMessageEnvelope
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.execution_graph_read_fold import (
    ExecutionGraphReadFold,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    DelegationOwnerProof,
    ExecutionGraphCurrentEvidenceReader,
    ExecutionGraphOwnerNotFoundError,
    PinnedExecutionGraphReadSet,
    PostgresDelegationOwnerReader,
    PostgresExecutionGraphLedgerReader,
    PostgresExecutionGraphStoredChainReader,
    PostgresExecutionGraphVerdictCandidateReader,
)
from omnibase_infra.runtime.execution_graph_ownership import (
    ExecutionGraphOwnershipRefusalError,
    admit_current_ownership,
    admit_verdict_candidates,
)
from omnibase_infra.runtime.execution_graph_read_authority import (
    TrustedExecutionGraphGatewayPolicy,
    TrustedGatewaySignerScope,
    VerifiedExecutionGraphReadAuthority,
    verify_signed_execution_graph_read_authority,
)
from omnibase_infra.runtime.execution_graph_topology_registry import (
    PackagedExecutionGraphTopologyContract,
)
from tests.helpers.projection_tenant_authority import InMemoryKeyProvider

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_MIGRATION = REPO_ROOT / "docker/migrations/forward/044_create_event_ledger.sql"
WATERMARK_MIGRATION = (
    REPO_ROOT / "docker/migrations/forward/109_add_event_ledger_ingest_watermark.sql"
)
VERDICT_BASE_MIGRATION = (
    REPO_ROOT
    / "docker/migrations/forward/nodes/node_projection_dod_verdict/0000_create_dod_verify_runs.sql"
)
VERDICT_LINK_MIGRATION = (
    REPO_ROOT
    / "docker/migrations/forward/nodes/node_projection_dod_verdict/0002_dod_verify_runs_delegation_correlation_id.sql"
)
CHAIN_MIGRATION = REPO_ROOT / "docker/migrations/forward/105_create_ledger_chain.sql"


def _test_read_set(
    *, topics: frozenset[str], head_topic: str, verdict_topic: str
) -> PinnedExecutionGraphReadSet:
    """Supply a synthetic read scope only to isolated PostgreSQL tests."""
    read_set = object.__new__(PinnedExecutionGraphReadSet)
    object.__setattr__(read_set, "topology_version", "test-topology-v1")
    object.__setattr__(read_set, "topics", topics)
    object.__setattr__(read_set, "head_topic", head_topic)
    object.__setattr__(read_set, "verdict_topic", verdict_topic)
    return read_set


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
async def local_pool(tmp_path: Path) -> AsyncGenerator[asyncpg.Pool, None]:
    """Isolated Postgres only; the explicit DSN requires a disposable marker."""
    test_dsn = os.environ.get("OMN19803_TEST_DSN")
    if test_dsn is not None:
        parsed = urlparse(test_dsn)
        if (
            parsed.scheme not in {"postgres", "postgresql"}
            or parsed.hostname != "127.0.0.1"
            or parsed.username != "postgres"
            or parsed.password is None
            or parsed.path != "/postgres"
            or parsed.port is None
        ):
            raise ValueError(
                "OMN19803_TEST_DSN must target dedicated loopback postgres"
            )
        pool = await asyncpg.create_pool(test_dsn, min_size=1, max_size=1)
        try:
            async with pool.acquire() as conn:
                marker = await conn.fetchval(
                    "SELECT to_regclass('omn19803_test_guard.disposable_instance') "
                    "IS NOT NULL"
                )
                if marker is not True:
                    raise ValueError("OMN19803_TEST_DSN lacks disposable DB marker")
                await conn.execute("DROP SCHEMA IF EXISTS omninode_internal CASCADE")
                await conn.execute("DROP SCHEMA public CASCADE")
                await conn.execute("CREATE SCHEMA public")
            await _initialize_schema(pool)
            yield pool
        finally:
            await pool.close()
        return

    initdb = shutil.which("initdb")
    pg_ctl = shutil.which("pg_ctl")
    if initdb is None or pg_ctl is None:
        pytest.skip("Local PostgreSQL binaries are unavailable")
    data_dir = tmp_path / "pgdata"
    socket_dir = Path(tempfile.mkdtemp(prefix="omn19728-read-", dir="/tmp"))
    port = _free_port()
    subprocess.run(
        [initdb, "-D", str(data_dir), "--auth=trust", "--no-instructions"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            pg_ctl,
            "-D",
            str(data_dir),
            "-l",
            str(tmp_path / "postgres.log"),
            "-o",
            f"-h '' -k {socket_dir} -p {port}",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pool = await asyncpg.create_pool(
        database="postgres",
        user=getpass.getuser(),
        host=str(socket_dir),
        port=port,
        min_size=1,
        max_size=1,
    )
    try:
        await _initialize_schema(pool)
        yield pool
    finally:
        await pool.close()
        subprocess.run(
            [pg_ctl, "-D", str(data_dir), "-m", "immediate", "-w", "stop"],
            check=True,
            capture_output=True,
            text=True,
        )
        socket_dir.rmdir()


async def _initialize_schema(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(BASE_MIGRATION.read_text(encoding="utf-8"))
        await conn.execute(WATERMARK_MIGRATION.read_text(encoding="utf-8"))
        await conn.execute("CREATE SCHEMA omninode_internal")
        await conn.execute(VERDICT_BASE_MIGRATION.read_text(encoding="utf-8"))
        await conn.execute(VERDICT_LINK_MIGRATION.read_text(encoding="utf-8"))
        await conn.execute(CHAIN_MIGRATION.read_text(encoding="utf-8"))
        await conn.execute(
            "CREATE TABLE public.delegation_events ("
            "correlation_id TEXT UNIQUE NOT NULL, tenant_id UUID NOT NULL)"
        )


def _authority(
    tenant_id: UUID, correlation_id: UUID
) -> VerifiedExecutionGraphReadAuthority:
    scope = TrustedGatewaySignerScope(
        runtime_id="trusted-api-gateway", realm="test", bus_id="graph-read"
    )
    keypair = generate_keypair()
    inner = ModelEventEnvelope[dict[str, object]](
        tenant_id=str(tenant_id),
        correlation_id=correlation_id,
        metadata={"tags": {"workflow_id": str(uuid4())}},
        payload={"correlation_id": str(correlation_id), "cursor_mode": "latest"},
    ).model_dump(mode="json")
    signed = ModelMessageEnvelope[dict[str, object]].create_signed(
        realm=scope.realm,
        runtime_id=scope.runtime_id,
        bus_id=scope.bus_id,
        trace_id=correlation_id,
        tenant_id=str(tenant_id),
        payload=inner,
        private_key=keypair.private_key_bytes,
    )
    return verify_signed_execution_graph_read_authority(
        signed,
        InMemoryKeyProvider({scope.runtime_id: keypair.public_key_bytes}),
        TrustedExecutionGraphGatewayPolicy(scopes=frozenset({scope})),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_owner_is_checked_with_explicit_tenant_predicate_and_local_guc(
    local_pool: asyncpg.Pool,
) -> None:
    owner_id = uuid4()
    foreign_id = uuid4()
    correlation_id = uuid4()
    async with local_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2)",
            str(correlation_id),
            owner_id,
        )

    reader = PostgresDelegationOwnerReader(local_pool)
    owner = await reader.require_owner(_authority(owner_id, correlation_id))
    assert owner.tenant_id == owner_id
    assert owner.correlation_id == correlation_id
    with pytest.raises(ExecutionGraphOwnerNotFoundError):
        await reader.require_owner(_authority(foreign_id, correlation_id))
    async with local_pool.acquire() as conn:
        current_guc = await conn.fetchval(
            "SELECT current_setting('app.tenant_id', true)"
        )
    assert current_guc != str(owner_id)
    assert current_guc != str(foreign_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_ledger_reader_returns_all_topics_partitions_and_old_rows(
    local_pool: asyncpg.Pool,
) -> None:
    tenant_id = uuid4()
    correlation_id = uuid4()
    async with local_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2)",
            str(correlation_id),
            tenant_id,
        )
        for topic, partition, offset in (
            ("onex.cmd.test.head.v1", 0, 1),
            ("onex.evt.test.reroute.v1", 1, 999),
            ("onex.evt.test.verdict.v1", 2, 4),
            ("onex.evt.test.outside-v1", 3, 5),
        ):
            await conn.execute(
                "INSERT INTO public.event_ledger "
                "(topic, partition, kafka_offset, event_value, correlation_id) "
                "VALUES ($1, $2, $3, $4, $5)",
                topic,
                partition,
                offset,
                b"{}",
                correlation_id,
            )
    authority = _authority(tenant_id, correlation_id)
    read_set = _test_read_set(
        topics=frozenset(
            {
                "onex.cmd.test.head.v1",
                "onex.evt.test.reroute.v1",
                "onex.evt.test.verdict.v1",
            }
        ),
        head_topic="onex.cmd.test.head.v1",
        verdict_topic="onex.evt.test.verdict.v1",
    )
    evidence = await ExecutionGraphCurrentEvidenceReader(
        PostgresDelegationOwnerReader(local_pool),
        PostgresExecutionGraphLedgerReader(local_pool),
    ).read_authorized_current(authority, read_set)
    records = evidence.ledger_rows

    with pytest.raises(TypeError):
        DelegationOwnerProof(  # type: ignore[call-arg]
            correlation_id=correlation_id,
            tenant_id=tenant_id,
        )
    other_correlation = uuid4()
    async with local_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2)",
            str(other_correlation),
            tenant_id,
        )
    other_proof = await PostgresDelegationOwnerReader(local_pool).require_owner(
        _authority(tenant_id, other_correlation)
    )
    with pytest.raises(ExecutionGraphOwnerNotFoundError):
        await PostgresExecutionGraphLedgerReader(local_pool).read_full_current(
            authority,
            other_proof,
            read_set,
        )

    assert evidence.owner.tenant_id == tenant_id
    assert evidence.owner.correlation_id == correlation_id

    assert {(row.topic, row.partition, row.kafka_offset) for row in records} == {
        ("onex.cmd.test.head.v1", 0, 1),
        ("onex.evt.test.reroute.v1", 1, 999),
        ("onex.evt.test.verdict.v1", 2, 4),
    }
    assert all(row.correlation_id == correlation_id for row in records)
    assert all(row.event_value == b"{}" for row in records)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_verdict_discovery_uses_index_only_then_raw_terminal_evidence(
    local_pool: asyncpg.Pool,
) -> None:
    tenant = uuid4()
    delegation = uuid4()
    verification = uuid4()
    not_indexed = uuid4()
    other_delegation = uuid4()
    verdict_topic = "onex.evt.test.verdict.v1"
    head_topic = "onex.cmd.test.head.v1"
    read_set = _test_read_set(
        topics=frozenset({head_topic, verdict_topic}),
        head_topic=head_topic,
        verdict_topic=verdict_topic,
    )
    async with local_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2), ($3, $2)",
            str(delegation),
            tenant,
            str(other_delegation),
        )
        for verify_id, delegated_id in (
            (verification, delegation),
            (not_indexed, other_delegation),
        ):
            await conn.execute(
                "INSERT INTO omninode_internal.dod_verify_runs "
                "(ticket_id, correlation_id, delegation_correlation_id, "
                "completed_at, started_at, status, total_checks, verified_count, "
                "failed_count, skipped_count, superseded_count, non_probative_count, "
                "behavior_proving_count, outcome, projected_at) "
                "VALUES ($1, $2, $3, NOW(), NOW(), 'failed', 1, 0, 1, 0, 0, 0, "
                "0, 'refused', NOW())",
                "OMN-19728",
                verify_id,
                delegated_id,
            )
            await conn.execute(
                "INSERT INTO public.event_ledger "
                "(topic, partition, kafka_offset, event_value, correlation_id) "
                "VALUES ($1, 2, $2, $3, $4)",
                verdict_topic,
                9 if verify_id == verification else 10,
                json.dumps(
                    {
                        "tenant_id": str(tenant),
                        "payload": {"delegation_correlation_id": str(delegation)},
                    }
                ).encode(),
                verify_id,
            )
        await conn.execute(
            "INSERT INTO public.event_ledger "
            "(topic, partition, kafka_offset, event_value, correlation_id, envelope_id) "
            "VALUES ($1, 0, 1, $2, $3, $4)",
            head_topic,
            json.dumps({"tenant_id": str(tenant), "payload": {}}).encode(),
            delegation,
            uuid4(),
        )
    authority = _authority(tenant, delegation)
    owner = await PostgresDelegationOwnerReader(local_pool).require_owner(authority)
    result = await PostgresExecutionGraphVerdictCandidateReader(
        local_pool
    ).read_full_current(authority, owner, read_set)

    assert result.verification_correlation_ids == (verification,)
    assert len(result.ledger_rows) == 1
    assert result.ledger_rows[0].correlation_id == verification
    assert result.ledger_rows[0].kafka_offset == 9
    # The index's stale `failed/refused` fields are deliberately absent from
    # returned evidence; only the raw terminal can be admitted/reduced.
    current = await ExecutionGraphCurrentEvidenceReader(
        PostgresDelegationOwnerReader(local_pool),
        PostgresExecutionGraphLedgerReader(local_pool),
    ).read_authorized_current(authority, read_set)
    admission = admit_current_ownership(current, read_set)
    assert (
        admit_verdict_candidates(
            admission, result.ledger_rows, read_set
        ).admitted_verdict_rows
        == result.ledger_rows
    )

    with pytest.raises(ExecutionGraphOwnerNotFoundError):
        await PostgresExecutionGraphVerdictCandidateReader(
            local_pool
        ).read_full_current(
            authority,
            await PostgresDelegationOwnerReader(local_pool).require_owner(
                _authority(tenant, other_delegation)
            ),
            read_set,
        )
    conflicting = replace(
        result.ledger_rows[0],
        event_value=json.dumps(
            {
                "tenant_id": str(tenant),
                "payload": {"delegation_correlation_id": str(other_delegation)},
            }
        ).encode(),
    )
    with pytest.raises(ExecutionGraphOwnershipRefusalError):
        admit_verdict_candidates(admission, (conflicting,), read_set)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_owned_stored_chain_rewrite_changes_only_current_annotations(
    local_pool: asyncpg.Pool,
) -> None:
    tenant = uuid4()
    correlation = uuid4()
    head = uuid4()
    stray = uuid4()
    head_topic = "onex.cmd.omnimarket.delegate-skill.v1"
    verdict_topic = "onex.evt.omnimarket.dod-verify-completed.v1"
    read_set = _test_read_set(
        topics=frozenset({head_topic, verdict_topic}),
        head_topic=head_topic,
        verdict_topic=verdict_topic,
    )
    async with local_pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO public.delegation_events (correlation_id, tenant_id) "
            "VALUES ($1, $2)",
            str(correlation),
            tenant,
        )
        await conn.execute(
            "INSERT INTO public.event_ledger "
            "(topic, partition, kafka_offset, event_value, correlation_id, "
            "envelope_id, onex_headers, ingest_watermark) "
            "VALUES ($1, 0, 1, $2, $3, $4, '{}'::jsonb, 1)",
            head_topic,
            json.dumps({"tenant_id": str(tenant)}).encode(),
            correlation,
            head,
        )
        for index, envelope in enumerate((head, stray)):
            await conn.execute(
                "INSERT INTO public.ledger_chain "
                "(correlation_id, hop_index, hop, replay_green, verifier_verdict, "
                "observed_topic, envelope_id) VALUES ($1, $2, 'delegate-skill', "
                "TRUE, 'pass', $3, $4)",
                str(correlation),
                index,
                head_topic,
                str(envelope),
            )
    authority = _authority(tenant, correlation)
    current = await ExecutionGraphCurrentEvidenceReader(
        PostgresDelegationOwnerReader(local_pool),
        PostgresExecutionGraphLedgerReader(local_pool),
    ).read_authorized_current(authority, read_set)
    admission = admit_current_ownership(current, read_set)
    assert len(admission.owned_envelope_ids) == 1
    assert type(admission.owned_envelope_ids[0]) is UUID
    reader = PostgresExecutionGraphStoredChainReader(local_pool)
    first = await reader.read_current(
        authority, admission.owner, admission.owned_envelope_ids
    )
    assert len(first) == 1
    assert first[0].node_id == head
    assert first[0].replay_green is True
    with pytest.raises(ExecutionGraphOwnerNotFoundError):
        await reader.read_current(
            _authority(uuid4(), correlation),
            admission.owner,
            admission.owned_envelope_ids,
        )

    topology = PackagedExecutionGraphTopologyContract().resolve(
        ModelExecutionGraphTopologyVersion(
            contract_version=ModelSemVer(major=1, minor=3, patch=0),
            topology_sha256="0505ab0b163492380739a15646c0442a3ecfb54efb0acb53bc23d232640fbfd3",
        )
    )
    fold = ExecutionGraphReadFold(
        workflow_type="delegation_execution_graph_read",
        read_clock=lambda: datetime(2026, 9, 27, tzinfo=UTC),
    )
    before = await fold(authority.request, authority, topology, admission, first)
    async with local_pool.acquire() as conn:
        await conn.execute(
            "UPDATE public.ledger_chain SET replay_green = FALSE, "
            "verifier_verdict = 'fail' WHERE correlation_id = $1 AND envelope_id = $2",
            str(correlation),
            str(head),
        )
    changed = await reader.read_current(
        authority, admission.owner, admission.owned_envelope_ids
    )
    after = await fold(authority.request, authority, topology, admission, changed)
    assert before.result is not None and after.result is not None
    assert before.result.replay.model_dump(
        mode="json"
    ) == after.result.replay.model_dump(mode="json")
    assert before.result.annotations.stored_chain == first
    assert after.result.annotations.stored_chain == changed
    assert first != changed
    assert before.result.annotations.stored_verdicts == ()
    assert after.result.replay.verdicts == ()
