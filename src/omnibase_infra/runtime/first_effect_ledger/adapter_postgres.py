# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# ruff: noqa: S608
"""PostgreSQL observation/recording scaffold for the first-effect ledger.

The scaffold records lifecycle observations; it confers no effect permission
and has no publisher. The canonical in-row outbox is owned by the state-IO
dispatch seam and cannot be safely staged from this payload-free adapter. The
non-authorizing composition guard in :mod:`.composition` rejects this adapter
where an external transactional-outbox implementation is required.

The database migration owns lifecycle validation. Each mutation uses one CAS
statement. Recovery reads a committed terminal or blocked observation only when
its immutable evidence matches the retry request.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping

import asyncpg

from omnibase_infra.errors.repository import RepositoryExecutionError
from omnibase_infra.runtime.first_effect_ledger.enum_first_effect_ledger_state import (
    EnumFirstEffectLedgerState,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    ModelFirstEffectAuthorizationRequest,
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_ledger_record import (
    ModelFirstEffectLedgerRecord,
)

PoolFactory = Callable[[], Awaitable[asyncpg.Pool]]
_TABLE = "public.first_effect_authorization_ledger"
_IMMUTABLE_EVIDENCE_TRIGGER_SQLSTATE = "P0001"
_IMMUTABLE_EVIDENCE_TRIGGER_MESSAGE = (
    "first-effect authorization evidence and transition timestamps are immutable"
)
_RETURNING = """
authorization_digest, nonce_digest, correlation_id, request_digest, manifest_hash,
state, publish_evidence_hash, terminal_receipt_hash, issued_at,
preflight_consumed_at, publishing_at, published_unknown_at,
terminal_observed_at, blocked_at, updated_at, version
"""


class FirstEffectLedgerConflictError(RuntimeError):
    """Raised when a retry conflicts with immutable ledger evidence."""


class PostgresFirstEffectLedger:
    """Injected-pool adapter for non-authorizing lifecycle observations."""

    def __init__(self, *, pool_factory: PoolFactory) -> None:
        self._pool_factory = pool_factory
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                self._pool = await self._pool_factory()
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @staticmethod
    def _record(row: Mapping[str, object]) -> ModelFirstEffectLedgerRecord:
        return ModelFirstEffectLedgerRecord.model_validate(dict(row))

    async def _fetchrow(
        self,
        operation: str,
        sql: str,
        *values: object,
        map_immutable_evidence_conflict: bool = False,
    ) -> asyncpg.Record | None:
        try:
            pool = await self._get_pool()
            async with pool.acquire() as connection:
                return await connection.fetchrow(sql, *values)
        except asyncpg.PostgresError as exc:
            if (
                map_immutable_evidence_conflict
                and exc.sqlstate == _IMMUTABLE_EVIDENCE_TRIGGER_SQLSTATE
                and str(exc) == _IMMUTABLE_EVIDENCE_TRIGGER_MESSAGE
            ):
                raise FirstEffectLedgerConflictError(
                    "terminal observation conflicts with committed publish evidence"
                ) from exc
            raise RepositoryExecutionError(
                f"first-effect ledger {operation} failed: {type(exc).__name__}",
                op_name=operation,
                table=_TABLE,
            ) from exc

    async def issue(
        self, request: ModelFirstEffectAuthorizationRequest
    ) -> ModelFirstEffectLedgerRecord:
        row = await self._fetchrow(
            "issue",
            f"""
            INSERT INTO {_TABLE} (
                authorization_digest, nonce_digest, correlation_id, request_digest,
                manifest_hash, state
            ) VALUES ($1, $2, $3, $4, $5, 'ISSUED')
            ON CONFLICT DO NOTHING
            RETURNING {_RETURNING}
            """,
            request.authorization_digest,
            request.nonce_digest,
            request.correlation_id,
            request.request_digest,
            request.manifest_hash,
        )
        if row is None:
            raise FirstEffectLedgerConflictError(
                "first-effect recording identity is already bound"
            )
        return self._record(row)

    async def get(
        self, *, authorization_digest: Sha256Digest
    ) -> ModelFirstEffectLedgerRecord | None:
        """Return one redacted ledger record without changing its lifecycle."""
        row = await self._fetchrow(
            "get",
            f"""
            SELECT {_RETURNING}
              FROM {_TABLE}
             WHERE authorization_digest = $1
            """,
            authorization_digest,
        )
        return None if row is None else self._record(row)

    async def consume_preflight(
        self, *, authorization_digest: Sha256Digest, expected_version: int
    ) -> ModelFirstEffectLedgerRecord | None:
        return await self._transition(
            operation="consume_preflight",
            authorization_digest=authorization_digest,
            expected_version=expected_version,
            from_states=(EnumFirstEffectLedgerState.ISSUED,),
            to_state=EnumFirstEffectLedgerState.PREFLIGHT_CONSUMED,
        )

    async def record_publishing_started(
        self, *, authorization_digest: Sha256Digest, expected_version: int
    ) -> ModelFirstEffectLedgerRecord | None:
        """Record that publishing started; this confers no effect permission.

        The caller must have already committed an actual publish intent through
        a transactional outbox. This adapter cannot create that intent because
        it intentionally never receives effect payload data.
        """
        return await self._transition(
            operation="record_publishing_started",
            authorization_digest=authorization_digest,
            expected_version=expected_version,
            from_states=(EnumFirstEffectLedgerState.PREFLIGHT_CONSUMED,),
            to_state=EnumFirstEffectLedgerState.PUBLISHING,
        )

    async def record_published_unknown(
        self,
        *,
        authorization_digest: Sha256Digest,
        expected_version: int,
        publish_evidence_hash: Sha256Digest,
    ) -> ModelFirstEffectLedgerRecord | None:
        return await self._transition(
            operation="record_published_unknown",
            authorization_digest=authorization_digest,
            expected_version=expected_version,
            from_states=(EnumFirstEffectLedgerState.PUBLISHING,),
            to_state=EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN,
            publish_evidence_hash=publish_evidence_hash,
        )

    async def observe_terminal(
        self,
        *,
        authorization_digest: Sha256Digest,
        expected_version: int,
        publish_evidence_hash: Sha256Digest,
        terminal_receipt_hash: Sha256Digest,
    ) -> ModelFirstEffectLedgerRecord | None:
        """Observe a terminal result with evidence-bound lost-response recovery.

        Retrying a committed observation returns that record even when the
        supplied version is stale, but only when both evidence digests match.
        A different terminal result, or a prior block, raises a conflict.
        """
        current = await self.get(authorization_digest=authorization_digest)
        if (
            current is not None
            and current.state is EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN
            and current.publish_evidence_hash != publish_evidence_hash
        ):
            raise FirstEffectLedgerConflictError(
                "terminal observation conflicts with committed publish evidence"
            )

        transitioned = await self._transition(
            operation="observe_terminal",
            authorization_digest=authorization_digest,
            expected_version=expected_version,
            from_states=(
                EnumFirstEffectLedgerState.PUBLISHING,
                EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN,
            ),
            to_state=EnumFirstEffectLedgerState.TERMINAL_OBSERVED,
            publish_evidence_hash=publish_evidence_hash,
            terminal_receipt_hash=terminal_receipt_hash,
            map_immutable_evidence_conflict=True,
        )
        if transitioned is not None:
            return transitioned

        current = await self.get(authorization_digest=authorization_digest)
        if current is None:
            return None
        if current.state is EnumFirstEffectLedgerState.TERMINAL_OBSERVED:
            if (
                current.publish_evidence_hash == publish_evidence_hash
                and current.terminal_receipt_hash == terminal_receipt_hash
            ):
                return current
            raise FirstEffectLedgerConflictError(
                "terminal observation conflicts with committed evidence"
            )
        if current.state is EnumFirstEffectLedgerState.BLOCKED:
            raise FirstEffectLedgerConflictError(
                "terminal observation conflicts with committed block"
            )
        return None

    async def block(
        self, *, authorization_digest: Sha256Digest, expected_version: int
    ) -> ModelFirstEffectLedgerRecord | None:
        """Record a block; retrying a committed block returns that record.

        A block cannot overwrite a committed terminal observation and raises a
        conflict in that case. A row still in another non-terminal state is a
        normal CAS miss and returns ``None``.
        """
        transitioned = await self._transition(
            operation="block",
            authorization_digest=authorization_digest,
            expected_version=expected_version,
            from_states=(
                EnumFirstEffectLedgerState.PUBLISHING,
                EnumFirstEffectLedgerState.PUBLISHED_UNKNOWN,
            ),
            to_state=EnumFirstEffectLedgerState.BLOCKED,
        )
        if transitioned is not None:
            return transitioned

        current = await self.get(authorization_digest=authorization_digest)
        if current is None:
            return None
        if current.state is EnumFirstEffectLedgerState.BLOCKED:
            return current
        if current.state is EnumFirstEffectLedgerState.TERMINAL_OBSERVED:
            raise FirstEffectLedgerConflictError(
                "block conflicts with committed terminal observation"
            )
        return None

    async def _transition(
        self,
        *,
        operation: str,
        authorization_digest: Sha256Digest,
        expected_version: int,
        from_states: tuple[EnumFirstEffectLedgerState, ...],
        to_state: EnumFirstEffectLedgerState,
        publish_evidence_hash: Sha256Digest | None = None,
        terminal_receipt_hash: Sha256Digest | None = None,
        map_immutable_evidence_conflict: bool = False,
    ) -> ModelFirstEffectLedgerRecord | None:
        row = await self._fetchrow(
            operation,
            f"""
            UPDATE {_TABLE}
               SET state = $1,
                   publish_evidence_hash = COALESCE($2, publish_evidence_hash),
                   terminal_receipt_hash = COALESCE($3, terminal_receipt_hash),
                   preflight_consumed_at = CASE WHEN $1 = 'PREFLIGHT_CONSUMED' THEN NOW() ELSE preflight_consumed_at END,
                   publishing_at = CASE WHEN $1 = 'PUBLISHING' THEN NOW() ELSE publishing_at END,
                   published_unknown_at = CASE WHEN $1 = 'PUBLISHED_UNKNOWN' THEN NOW() ELSE published_unknown_at END,
                   terminal_observed_at = CASE WHEN $1 = 'TERMINAL_OBSERVED' THEN NOW() ELSE terminal_observed_at END,
                   blocked_at = CASE WHEN $1 = 'BLOCKED' THEN NOW() ELSE blocked_at END,
                   updated_at = NOW(),
                   version = version + 1
             WHERE authorization_digest = $4
               AND version = $5
               AND state = ANY($6::text[])
            RETURNING {_RETURNING}
            """,
            to_state.value,
            publish_evidence_hash,
            terminal_receipt_hash,
            authorization_digest,
            expected_version,
            [state.value for state in from_states],
            map_immutable_evidence_conflict=map_immutable_evidence_conflict,
        )
        return None if row is None else self._record(row)


__all__ = ["FirstEffectLedgerConflictError", "PostgresFirstEffectLedger"]
