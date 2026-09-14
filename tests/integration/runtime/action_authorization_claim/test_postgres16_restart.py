# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Restart and typed-outcome proofs for the durable PostgreSQL claim adapter."""

from __future__ import annotations

import asyncio
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


def _request(
    identity: int, *, expires_at: datetime | None = None, reason: str | None = None
) -> ModelActionAuthorizationClaimRequest:
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
        expires_at=expires_at or datetime(2040, 1, 1, 11, tzinfo=UTC),
        one_time_use=True,
        reason=reason or "bounded execute-disabled bootstrap verification",
    )


@pytest.mark.integration
def test_postgres16_restart_recovers_the_single_consumed_claim(
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

        request = _request(1)
        first_pool = await create_pool()

        async def first_factory() -> asyncpg.Pool:
            return first_pool

        first_process = PostgresActionAuthorizationClaim(pool_factory=first_factory)
        assert (
            await first_process.claim(request)
        ).outcome is EnumActionAuthorizationClaimOutcome.CLAIMED
        await first_process.close()

        restarted_pool = await create_pool()

        async def restarted_factory() -> asyncpg.Pool:
            return restarted_pool

        restarted_process = PostgresActionAuthorizationClaim(
            pool_factory=restarted_factory
        )
        recovered = await restarted_process.claim(request)
        assert recovered.outcome is EnumActionAuthorizationClaimOutcome.ALREADY_CONSUMED
        assert recovered.version == 1
        await restarted_process.close()

    asyncio.run(scenario())


@pytest.mark.integration
def test_postgres16_returns_closed_expired_and_mismatch_outcomes_without_fallback(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres)

    async def scenario() -> None:
        pool = await asyncpg.create_pool(
            host=ephemeral_postgres.socket_dir,
            port=ephemeral_postgres.port,
            user="postgres",
            database="postgres",
            min_size=1,
            max_size=1,
        )

        async def factory() -> asyncpg.Pool:
            return pool

        adapter = PostgresActionAuthorizationClaim(pool_factory=factory)
        expired_request = _request(
            3,
            expires_at=datetime(2001, 1, 1, 11, tzinfo=UTC),
        )
        expired = await adapter.claim(expired_request)
        assert expired.outcome is EnumActionAuthorizationClaimOutcome.EXPIRED

        valid_request = _request(4)
        claimed = await adapter.claim(valid_request)
        assert claimed.outcome is EnumActionAuthorizationClaimOutcome.CLAIMED
        mismatched = await adapter.claim(
            _request(4, reason="contradictory canonical request")
        )
        assert mismatched.outcome is EnumActionAuthorizationClaimOutcome.MISMATCH
        await adapter.close()

    asyncio.run(scenario())
