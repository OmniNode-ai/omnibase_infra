# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Injected asyncpg adapter for the separate nonce-claim PostgreSQL boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import asyncpg
from pydantic import ValidationError

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
        self._pool_idle = asyncio.Condition(self._pool_lock)
        self._active_claims = 0

    async def _borrow_pool(self) -> asyncpg.Pool:
        async with self._pool_lock:
            if self._pool is None:
                self._pool = await self._pool_factory()
            self._active_claims += 1
            return self._pool

    async def _release_pool(self) -> None:
        async with self._pool_idle:
            self._active_claims -= 1
            if self._active_claims == 0:
                self._pool_idle.notify_all()

    async def close(self) -> None:
        async with self._pool_idle:
            while self._active_claims:
                await self._pool_idle.wait()
            pool = self._pool
            self._pool = None
        if pool is not None:
            await pool.close()

    @staticmethod
    def _claim_result(row: Mapping[str, object]) -> ModelActionAuthorizationClaimResult:
        return ModelActionAuthorizationClaimResult.model_validate(dict(row))

    async def _fetchrow(
        self, sql: str, request: ModelActionAuthorizationClaimRequest
    ) -> asyncpg.Record | None:
        pool = await self._borrow_pool()
        try:
            async with pool.acquire() as connection:
                return await connection.fetchrow(sql, *request.sql_arguments())
        finally:
            await self._release_pool()

    async def claim(
        self, request: ModelActionAuthorizationClaimRequest
    ) -> ModelActionAuthorizationClaimResult:
        try:
            row = await self._fetchrow(_CLAIM_SQL, request)
            if row is None:
                return ModelActionAuthorizationClaimResult(
                    outcome=EnumActionAuthorizationClaimOutcome.ERROR
                )
            return self._claim_result(row)
        except (asyncpg.PostgresError, OSError, TypeError, ValidationError):
            return ModelActionAuthorizationClaimResult(
                outcome=EnumActionAuthorizationClaimOutcome.ERROR
            )


__all__ = ["PostgresActionAuthorizationClaim"]
