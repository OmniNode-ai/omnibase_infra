# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected asyncpg adapter for the separate nonce-claim PostgreSQL boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import asyncpg

from omnibase_infra.runtime.action_authorization_claim.enum_action_authorization_claim_outcome import (
    EnumActionAuthorizationClaimOutcome,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_request import (
    ModelActionAuthorizationClaimRequest,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_result import (
    ModelActionAuthorizationClaimResult,
)

PoolFactory = Callable[[], Awaitable[asyncpg.Pool]]
_TABLE = "action_authorization_claim.nonce_claims"
_CLAIM_SQL = """
SELECT outcome, state, version, redacted_receipt_digest
  FROM action_authorization_claim.claim_action_authorization(
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
    $16, $17, $18, $19, $20, $21
  )
"""


class PostgresActionAuthorizationClaim:
    """One injected PostgreSQL port; it has no old-ledger fallback or authority.

    PostgreSQL stored procedures validate and serialize claims. This adapter
    returns a redacted ``ERROR`` result on database failure so callers can
    default-deny without observing transport or database details.
    """

    def __init__(self, *, pool_factory: PoolFactory) -> None:
        self._pool_factory = pool_factory
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool_locked(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        self._pool = await self._pool_factory()
        return self._pool

    async def close(self) -> None:
        async with self._pool_lock:
            if self._pool is not None:
                pool = self._pool
                self._pool = None
                await pool.close()

    @staticmethod
    def _claim_result(row: Mapping[str, object]) -> ModelActionAuthorizationClaimResult:
        return ModelActionAuthorizationClaimResult.model_validate(dict(row))

    async def _fetchrow(
        self, sql: str, request: ModelActionAuthorizationClaimRequest
    ) -> asyncpg.Record | None:
        async with self._pool_lock:
            pool = await self._get_pool_locked()
            async with pool.acquire() as connection:
                return await connection.fetchrow(sql, *request.sql_arguments())

    async def claim(
        self, request: ModelActionAuthorizationClaimRequest
    ) -> ModelActionAuthorizationClaimResult:
        try:
            row = await self._fetchrow(_CLAIM_SQL, request)
        except (asyncpg.PostgresError, OSError):
            return ModelActionAuthorizationClaimResult(
                outcome=EnumActionAuthorizationClaimOutcome.ERROR
            )
        if row is None:
            return ModelActionAuthorizationClaimResult(
                outcome=EnumActionAuthorizationClaimOutcome.ERROR
            )
        return self._claim_result(row)


__all__ = ["PostgresActionAuthorizationClaim"]
