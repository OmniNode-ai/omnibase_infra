# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""PostgreSQL persistence for remote agent task state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import UUID, uuid4

import asyncpg

from omnibase_core.models.delegation.model_remote_task_state import ModelRemoteTaskState
from omnibase_infra.enums import EnumInfraTransportType, EnumPostgresErrorCode
from omnibase_infra.errors import ModelInfraErrorContext, RepositoryExecutionError
from omnibase_infra.mixins.mixin_postgres_op_executor import MixinPostgresOpExecutor

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")


SQL_UPSERT_REMOTE_TASK_STATE = """
INSERT INTO remote_task_state (
    task_id,
    invocation_kind,
    protocol,
    target_ref,
    remote_task_handle,
    correlation_id,
    status,
    last_remote_status,
    last_emitted_event_type,
    submitted_at,
    updated_at,
    completed_at,
    error
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
)
ON CONFLICT (task_id) DO UPDATE SET
    invocation_kind = EXCLUDED.invocation_kind,
    protocol = EXCLUDED.protocol,
    target_ref = EXCLUDED.target_ref,
    remote_task_handle = EXCLUDED.remote_task_handle,
    correlation_id = EXCLUDED.correlation_id,
    status = EXCLUDED.status,
    last_remote_status = EXCLUDED.last_remote_status,
    last_emitted_event_type = EXCLUDED.last_emitted_event_type,
    submitted_at = EXCLUDED.submitted_at,
    updated_at = EXCLUDED.updated_at,
    completed_at = EXCLUDED.completed_at,
    error = EXCLUDED.error
RETURNING task_id
"""

SQL_GET_REMOTE_TASK_STATE = """
SELECT
    task_id,
    invocation_kind,
    protocol,
    target_ref,
    remote_task_handle,
    correlation_id,
    status,
    last_remote_status,
    last_emitted_event_type,
    submitted_at,
    updated_at,
    completed_at,
    error
FROM remote_task_state
WHERE task_id = $1
"""

SQL_GET_REMOTE_TASK_STATE_BY_HANDLE = """
SELECT
    task_id,
    invocation_kind,
    protocol,
    target_ref,
    remote_task_handle,
    correlation_id,
    status,
    last_remote_status,
    last_emitted_event_type,
    submitted_at,
    updated_at,
    completed_at,
    error
FROM remote_task_state
WHERE remote_task_handle = $1
"""

SQL_LOAD_UNFINISHED_REMOTE_TASK_STATES = """
SELECT
    task_id,
    invocation_kind,
    protocol,
    target_ref,
    remote_task_handle,
    correlation_id,
    status,
    last_remote_status,
    last_emitted_event_type,
    submitted_at,
    updated_at,
    completed_at,
    error
FROM remote_task_state
WHERE status NOT IN ('COMPLETED', 'FAILED', 'TIMED_OUT', 'CANCELED')
ORDER BY submitted_at ASC
"""


class RemoteTaskStateRepository(MixinPostgresOpExecutor):
    """Persistence adapter for restart-safe remote task resumption."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def upsert(
        self,
        state: ModelRemoteTaskState,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        op_correlation_id = correlation_id or state.correlation_id

        async def _op() -> None:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    SQL_UPSERT_REMOTE_TASK_STATE,
                    state.task_id,
                    state.invocation_kind.value,
                    state.protocol.value if state.protocol is not None else None,
                    state.target_ref,
                    state.remote_task_handle,
                    state.correlation_id,
                    state.status.value,
                    state.last_remote_status,
                    (
                        state.last_emitted_event_type.value
                        if state.last_emitted_event_type is not None
                        else None
                    ),
                    state.submitted_at,
                    state.updated_at,
                    state.completed_at,
                    state.error,
                )
            if row is None:
                raise RepositoryExecutionError(
                    "remote_task_state upsert returned no row",
                    op_name="upsert_remote_task_state",
                    table="remote_task_state",
                    sql_fingerprint="INSERT INTO remote_task_state ... ON CONFLICT (task_id) DO UPDATE",
                    context=self._context(
                        op_correlation_id, "upsert_remote_task_state"
                    ),
                )

        await self._run_checked(
            correlation_id=op_correlation_id,
            op_name="upsert_remote_task_state",
            log_context={"task_id": str(state.task_id)},
            fn=_op,
        )

    async def get(
        self,
        task_id: UUID,
        *,
        correlation_id: UUID | None = None,
    ) -> ModelRemoteTaskState | None:
        row = await self._run_checked(
            correlation_id=correlation_id or uuid4(),
            op_name="get_remote_task_state",
            log_context={"task_id": str(task_id)},
            fn=lambda: self._fetchrow(SQL_GET_REMOTE_TASK_STATE, task_id),
        )
        if row is None:
            return None
        return self._row_to_model(row)

    async def get_by_remote_task_handle(
        self,
        remote_task_handle: str,
        *,
        correlation_id: UUID | None = None,
    ) -> ModelRemoteTaskState | None:
        row = await self._run_checked(
            correlation_id=correlation_id or uuid4(),
            op_name="get_remote_task_state_by_handle",
            log_context={"remote_task_handle": remote_task_handle},
            fn=lambda: self._fetchrow(
                SQL_GET_REMOTE_TASK_STATE_BY_HANDLE,
                remote_task_handle,
            ),
        )
        if row is None:
            return None
        return self._row_to_model(row)

    async def load_unfinished(
        self,
        *,
        correlation_id: UUID | None = None,
    ) -> list[ModelRemoteTaskState]:
        rows = await self._run_checked(
            correlation_id=correlation_id or uuid4(),
            op_name="load_unfinished_remote_task_states",
            log_context={"table": "remote_task_state"},
            fn=lambda: self._fetch(SQL_LOAD_UNFINISHED_REMOTE_TASK_STATES),
        )
        return [self._row_to_model(row) for row in rows]

    async def _fetchrow(
        self,
        sql: str,
        *params: object,
    ) -> asyncpg.Record | None:
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(sql, *params)

    async def _fetch(
        self,
        sql: str,
        *params: object,
    ) -> Sequence[asyncpg.Record]:
        async with self._pool.acquire() as conn:
            return cast("Sequence[asyncpg.Record]", await conn.fetch(sql, *params))

    async def _run_checked(
        self,
        *,
        correlation_id: UUID,
        op_name: str,
        log_context: dict[str, object],
        fn: Callable[[], Awaitable[T]],
    ) -> T:
        result_holder: dict[str, T] = {}

        async def _wrapped() -> None:
            result_holder["value"] = await fn()

        backend_result = await self._execute_postgres_op(
            op_error_code=EnumPostgresErrorCode.UPSERT_ERROR,
            correlation_id=correlation_id,
            log_context=log_context,
            fn=_wrapped,
        )
        if backend_result.success:
            return result_holder["value"]

        retriable: bool | None = None
        if backend_result.error_code is not None:
            try:
                retriable = EnumPostgresErrorCode(
                    backend_result.error_code
                ).is_retriable
            except ValueError:
                retriable = None

        raise RepositoryExecutionError(
            backend_result.error or f"{op_name} failed",
            op_name=op_name,
            table="remote_task_state",
            retriable=retriable,
            context=self._context(correlation_id, op_name),
        )

    def _context(
        self,
        correlation_id: UUID,
        operation: str,
    ) -> ModelInfraErrorContext:
        return ModelInfraErrorContext(
            transport_type=EnumInfraTransportType.DATABASE,
            operation=operation,
            correlation_id=correlation_id,
            target_name="remote_task_state",
        )

    @staticmethod
    def _row_to_model(row: asyncpg.Record) -> ModelRemoteTaskState:
        return ModelRemoteTaskState.model_validate(dict(row))


__all__ = ["RemoteTaskStateRepository"]
