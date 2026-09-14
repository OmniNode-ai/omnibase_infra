# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real PostgreSQL proof that one canonical nonce has exactly one claimant.

Every pool here sets ``statement_timeout``. The claim function is reached by
concurrent callers and by repeat callers, and a defect in its conflict path
shows up as a call that never returns rather than as a wrong answer. Without
a server-side bound such a defect hangs the test run instead of failing it,
and a hung run is indistinguishable from a slow one until someone kills it.
"""

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

_STATEMENT_TIMEOUT_MS = "10000"
_CLAIM_SQL = """
SELECT outcome, state, version, redacted_receipt_digest
  FROM action_authorization_claim.claim_action_authorization(
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
    $16, $17, $18, $19, $20, $21
  )
"""
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


def _expired_request(identity: int) -> ModelActionAuthorizationClaimRequest:
    """The same canonical request, already past its expiry on arrival."""
    return _request(identity).model_copy(
        update={
            "issued_at": datetime(2001, 1, 1, 10, tzinfo=UTC),
            "expires_at": datetime(2001, 1, 1, 11, tzinfo=UTC),
        }
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
                server_settings={"statement_timeout": _STATEMENT_TIMEOUT_MS},
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


@pytest.mark.integration
def test_postgres16_repeat_claim_is_bounded_and_returns_exactly_one_row(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    """A second claim of the same canonical request answers once and stops.

    This is the sequential form of the concurrent case and needs no second
    session: whatever decides that a conflicting claim already exists has to
    reach a terminal answer on the first look at it. Row COUNT is asserted, not
    just the outcome, because a conflict path that answers and then carries on
    still answers correctly on its first row while never returning.
    """
    _apply(ephemeral_postgres)

    async def scenario() -> None:
        pool = await asyncpg.create_pool(
            host=ephemeral_postgres.socket_dir,
            port=ephemeral_postgres.port,
            user="postgres",
            database="postgres",
            min_size=1,
            max_size=1,
            server_settings={"statement_timeout": _STATEMENT_TIMEOUT_MS},
        )
        try:
            async with pool.acquire() as connection:
                live = _request(7)
                first = await connection.fetch(_CLAIM_SQL, *live.sql_arguments())
                assert [row["outcome"] for row in first] == ["CLAIMED"]
                second = await connection.fetch(_CLAIM_SQL, *live.sql_arguments())
                assert [row["outcome"] for row in second] == ["ALREADY_CONSUMED"]
                assert second[0]["state"] == "CLAIMED"
                assert second[0]["version"] == 1

                expired = _expired_request(8)
                first_expired = await connection.fetch(
                    _CLAIM_SQL, *expired.sql_arguments()
                )
                assert [row["outcome"] for row in first_expired] == ["EXPIRED"]
                second_expired = await connection.fetch(
                    _CLAIM_SQL, *expired.sql_arguments()
                )
                assert [row["outcome"] for row in second_expired] == ["EXPIRED"]
                assert second_expired[0]["state"] == "EXPIRED"
        finally:
            await pool.close()

    asyncio.run(scenario())
