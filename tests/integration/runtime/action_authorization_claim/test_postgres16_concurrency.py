# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real PostgreSQL proof that one canonical nonce has exactly one claimant."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import asyncpg
import pytest

from omnibase_infra.runtime.action_authorization_claim import (
    EnumActionAuthorizationClaimOutcome,
    ModelActionAuthorizationClaimRequest,
    PostgresActionAuthorizationClaim,
)
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_ROOT = Path(__file__).resolve().parents[4]
_MIGRATION = (
    _ROOT / "docker/migrations/forward/107_create_action_authorization_nonce_claim.sql"
)


def _apply(ephemeral_postgres: EphemeralPostgres) -> None:
    result = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(_MIGRATION))
    assert result.returncode == 0, result.stderr


def _request(identity: int = 1) -> ModelActionAuthorizationClaimRequest:
    return ModelActionAuthorizationClaimRequest(
        authorization_id=(f"action-auth-12345678-1234-1234-1234-{identity:012x}"),
        ticket_id="OMN-17462",
        contract_path="contracts/OMN-17462.yaml",
        contract_commit_sha="a" * 40,
        contract_sha256="sha256:" + "b" * 64,
        action_id="postgres-push-lane-bootstrap",
        source_sha="c" * 40,
        artifact_sha256="sha256:" + "d" * 64,
        target_database="rsd_push_lanes",
        target_schema="push_lanes",
        target_service="rsd_push_lane_broker",
        target_principal="rsd_push_lane_broker",
        execute_enabled=False,
        issuer="operator-governance",
        nonce=f"{identity:064x}",
        issued_at=datetime(2040, 1, 1, 10, tzinfo=UTC),
        expires_at=datetime(2040, 1, 1, 11, tzinfo=UTC),
        one_time_use=True,
        reason="bounded execute-disabled bootstrap verification",
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
def test_postgres16_concurrent_claims_have_one_winner(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres)

    async def scenario() -> None:
        async def create_pool() -> asyncpg.Pool:
            return await asyncpg.create_pool(
                host=ephemeral_postgres.socket_dir,
                port=ephemeral_postgres.port,
                user="postgres",
                database="postgres",
                min_size=1,
                max_size=1,
            )

        request = _request()
        first_pool, second_pool = await asyncio.gather(create_pool(), create_pool())
        try:
            barrier = asyncio.Barrier(2)

            async def first_factory() -> _SynchronizedPool:
                return _SynchronizedPool(first_pool, barrier)

            async def second_factory() -> _SynchronizedPool:
                return _SynchronizedPool(second_pool, barrier)

            first = PostgresActionAuthorizationClaim(
                pool_factory=first_factory  # type: ignore[arg-type]
            )
            second = PostgresActionAuthorizationClaim(
                pool_factory=second_factory  # type: ignore[arg-type]
            )
            outcomes = await asyncio.gather(first.claim(request), second.claim(request))
            assert [outcome.outcome for outcome in outcomes].count(
                EnumActionAuthorizationClaimOutcome.CLAIMED
            ) == 1
            assert [outcome.outcome for outcome in outcomes].count(
                EnumActionAuthorizationClaimOutcome.ALREADY_CONSUMED
            ) == 1
        finally:
            await first_pool.close()
            await second_pool.close()

        verification_pool = await create_pool()
        async with verification_pool.acquire() as connection:
            persisted = await connection.fetchrow(
                "SELECT state, version, claimed_at IS NOT NULL AS claimed "
                "FROM action_authorization_claim.nonce_claims "
                "WHERE authorization_id = $1 AND nonce_digest = $2",
                request.authorization_id,
                request.nonce_digest,
            )
        assert persisted is not None
        assert dict(persisted) == {"state": "CLAIMED", "version": 1, "claimed": True}
        await verification_pool.close()

    asyncio.run(scenario())
