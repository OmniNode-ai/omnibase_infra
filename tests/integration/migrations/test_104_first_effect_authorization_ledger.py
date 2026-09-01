# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real ephemeral-Postgres proof for migration 104 and adapter recovery.

This test never discovers a DSN or touches a shared database. The shared
``ephemeral_postgres`` fixture starts an isolated Unix-socket cluster using
``initdb`` and applies the migration through ``psql -f``. If PostgreSQL tools
are unavailable, pytest reports an explicit skip rather than downgrading this
to a fake concurrency proof.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import asyncpg
import pytest

from omnibase_infra.runtime.first_effect_ledger import (
    FirstEffectLedgerConflictError,
    ModelFirstEffectAuthorizationRequest,
    PostgresFirstEffectLedger,
)
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "104_create_first_effect_authorization_ledger.sql"
)
ROLLBACK = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_104_create_first_effect_authorization_ledger.sql"
)

_EVIDENCE = "e" * 64
_RECEIPT = "f" * 64


def _apply(ephemeral_postgres: EphemeralPostgres, migration: Path) -> None:
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(migration))
    assert result.returncode == 0, result.stderr


def _digest(value: int) -> str:
    return f"{value:064x}"


def _request(identity: int) -> ModelFirstEffectAuthorizationRequest:
    return ModelFirstEffectAuthorizationRequest(
        authorization_digest=_digest(identity * 10 + 1),
        nonce_digest=_digest(identity * 10 + 2),
        correlation_id=UUID(int=identity),
        request_digest=_digest(identity * 10 + 3),
        manifest_hash=_digest(identity * 10 + 4),
    )


class _SynchronizedAcquire:
    def __init__(self, acquire: object, barrier: asyncio.Barrier) -> None:
        self._acquire = acquire
        self._barrier = barrier
        self._transaction: asyncpg.Transaction | None = None

    async def __aenter__(self) -> asyncpg.Connection:
        connection = await self._acquire.__aenter__()  # type: ignore[union-attr]
        self._transaction = connection.transaction()
        await self._transaction.start()
        await self._barrier.wait()
        return connection

    async def __aexit__(self, *args: object) -> bool:
        if self._transaction is not None:
            if args[0] is None:
                await self._transaction.commit()
            else:
                await self._transaction.rollback()
        return await self._acquire.__aexit__(*args)  # type: ignore[union-attr]


class _SynchronizedPool:
    def __init__(self, pool: asyncpg.Pool, barrier: asyncio.Barrier) -> None:
        self._pool = pool
        self._barrier = barrier

    def acquire(self) -> _SynchronizedAcquire:
        return _SynchronizedAcquire(self._pool.acquire(), self._barrier)

    async def close(self) -> None:
        await self._pool.close()


@pytest.mark.integration
def test_104_applies_real_schema_and_rollback_removes_it(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, FORWARD)
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.first_effect_authorization_ledger') IS NOT NULL, "
                "to_regprocedure('public.enforce_first_effect_authorization_transition()') IS NOT NULL"
            )
            assert cursor.fetchone() == (True, True)
    finally:
        connection.close()

    _apply(ephemeral_postgres, ROLLBACK)
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.first_effect_authorization_ledger'), "
                "to_regprocedure('public.enforce_first_effect_authorization_transition()')"
            )
            assert cursor.fetchone() == (None, None)
    finally:
        connection.close()


@pytest.mark.integration
def test_104_refuses_an_unverified_preexisting_relation(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    connection = ephemeral_postgres.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TABLE public.first_effect_authorization_ledger (id INTEGER)"
            )
        connection.commit()
    finally:
        connection.close()

    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(FORWARD))
    assert result.returncode != 0
    assert "already exists" in result.stderr


@pytest.mark.integration
def test_104_real_cas_and_lost_response_recovery(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, FORWARD)

    async def scenario() -> None:
        async def create_pool() -> asyncpg.Pool:
            return await asyncpg.create_pool(
                host=ephemeral_postgres.socket_dir,
                port=ephemeral_postgres.port,
                user="postgres",
                database="postgres",
                min_size=1,
                max_size=2,
            )

        pool = await create_pool()

        async def pool_factory() -> asyncpg.Pool:
            return pool

        ledger = PostgresFirstEffectLedger(pool_factory=pool_factory)
        try:
            issued = await ledger.issue(_request(1))

            race_pool_one, race_pool_two = await asyncio.gather(
                create_pool(), create_pool()
            )
            try:
                async with race_pool_one.acquire() as connection_one:
                    backend_one = await connection_one.fetchrow(
                        "SELECT pg_backend_pid() AS pid, txid_current() AS transaction_id"
                    )
                async with race_pool_two.acquire() as connection_two:
                    backend_two = await connection_two.fetchrow(
                        "SELECT pg_backend_pid() AS pid, txid_current() AS transaction_id"
                    )
                assert backend_one is not None
                assert backend_two is not None
                assert backend_one["pid"] != backend_two["pid"]
                assert backend_one["transaction_id"] != backend_two["transaction_id"]

                barrier = asyncio.Barrier(2)

                async def race_pool_one_factory() -> _SynchronizedPool:
                    return _SynchronizedPool(race_pool_one, barrier)

                async def race_pool_two_factory() -> _SynchronizedPool:
                    return _SynchronizedPool(race_pool_two, barrier)

                race_one = PostgresFirstEffectLedger(
                    pool_factory=race_pool_one_factory  # type: ignore[arg-type]
                )
                race_two = PostgresFirstEffectLedger(
                    pool_factory=race_pool_two_factory  # type: ignore[arg-type]
                )
                raced = await asyncio.gather(
                    race_one.consume_preflight(
                        authorization_digest=issued.authorization_digest,
                        expected_version=issued.version,
                    ),
                    race_two.consume_preflight(
                        authorization_digest=issued.authorization_digest,
                        expected_version=issued.version,
                    ),
                )
            finally:
                await race_pool_one.close()
                await race_pool_two.close()

            winners = [record for record in raced if record is not None]
            assert len(winners) == 1
            assert winners[0].state.value == "PREFLIGHT_CONSUMED"

            async def published_unknown(identity: int) -> tuple[str, int]:
                request = _request(identity)
                issued_record = await ledger.issue(request)
                preflight = await ledger.consume_preflight(
                    authorization_digest=request.authorization_digest,
                    expected_version=issued_record.version,
                )
                assert preflight is not None
                publishing = await ledger.record_publishing_started(
                    authorization_digest=request.authorization_digest,
                    expected_version=preflight.version,
                )
                assert publishing is not None
                unknown = await ledger.record_published_unknown(
                    authorization_digest=request.authorization_digest,
                    expected_version=publishing.version,
                    publish_evidence_hash=_EVIDENCE,
                )
                assert unknown is not None
                return request.authorization_digest, unknown.version

            terminal_digest, unknown_version = await published_unknown(2)
            observed = await ledger.observe_terminal(
                authorization_digest=terminal_digest,
                expected_version=unknown_version,
                publish_evidence_hash=_EVIDENCE,
                terminal_receipt_hash=_RECEIPT,
            )
            assert observed is not None
            assert observed.state.value == "TERMINAL_OBSERVED"

            recovered = await ledger.observe_terminal(
                authorization_digest=terminal_digest,
                expected_version=unknown_version,
                publish_evidence_hash=_EVIDENCE,
                terminal_receipt_hash=_RECEIPT,
            )
            assert recovered == observed

            with pytest.raises(
                FirstEffectLedgerConflictError, match="committed evidence"
            ):
                await ledger.observe_terminal(
                    authorization_digest=terminal_digest,
                    expected_version=unknown_version,
                    publish_evidence_hash=_EVIDENCE,
                    terminal_receipt_hash="0" * 64,
                )

            conflict_digest, conflict_version = await published_unknown(3)
            with pytest.raises(
                FirstEffectLedgerConflictError, match="publish evidence"
            ):
                await ledger.observe_terminal(
                    authorization_digest=conflict_digest,
                    expected_version=conflict_version,
                    publish_evidence_hash="0" * 64,
                    terminal_receipt_hash=_RECEIPT,
                )

            blocked_digest, blocked_version = await published_unknown(4)
            blocked = await ledger.block(
                authorization_digest=blocked_digest,
                expected_version=blocked_version,
            )
            assert blocked is not None
            assert blocked.state.value == "BLOCKED"

            direct_request = _request(5)
            direct_issued = await ledger.issue(direct_request)
            direct_preflight = await ledger.consume_preflight(
                authorization_digest=direct_request.authorization_digest,
                expected_version=direct_issued.version,
            )
            assert direct_preflight is not None
            async with pool.acquire() as connection:
                with pytest.raises(asyncpg.RaiseError, match="illegal first-effect"):
                    await connection.execute(
                        """
                        UPDATE public.first_effect_authorization_ledger
                           SET state = 'BLOCKED',
                               publishing_at = NOW(),
                               blocked_at = NOW(),
                               updated_at = NOW(),
                               version = version + 1
                         WHERE authorization_digest = $1
                        """,
                        direct_request.authorization_digest,
                    )

            direct_unknown_digest, direct_unknown_version = await published_unknown(6)
            async with pool.acquire() as connection:
                with pytest.raises(asyncpg.RaiseError, match="evidence and transition"):
                    await connection.execute(
                        """
                        UPDATE public.first_effect_authorization_ledger
                           SET publish_evidence_hash = $1,
                               updated_at = NOW(),
                               version = version + 1
                         WHERE authorization_digest = $2
                           AND version = $3
                        """,
                        "0" * 64,
                        direct_unknown_digest,
                        direct_unknown_version,
                    )

            with pytest.raises(FirstEffectLedgerConflictError, match="committed block"):
                await ledger.observe_terminal(
                    authorization_digest=blocked_digest,
                    expected_version=blocked_version,
                    publish_evidence_hash=_EVIDENCE,
                    terminal_receipt_hash=_RECEIPT,
                )
        finally:
            await ledger.close()

    asyncio.run(scenario())
