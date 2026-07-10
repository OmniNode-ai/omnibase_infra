# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Async CAS-backed adapter for the state_io runtime dispatch seam (OMN-14208).

Backs the opt-in ``state_io`` contract binding wired in
``omnibase_infra.runtime.auto_wiring.handler_wiring``. Stores one durable row
per ``correlation_id`` — the only key every leg of a multi-leg orchestrator
workflow carries on the wire — with an integer ``version`` column for
optimistic concurrency (see ``omnibase_infra.utils.util_retry_optimistic``).

Opaque by design: ``payload`` is passed through as raw JSON text end-to-end.
This adapter never decodes its business shape — the contract-declared codec
that understands the payload's structure lives with the orchestrator handler
(omnimarket), not here. ``tenant_id`` / ``state`` / ``in_flight`` are
denormalized top-level columns the wiring seam extracts from well-known
top-level keys on the opaque payload so staleness sweeps can filter/index
without decoding it.

``CONTEXTVAR_STATE_IO_ROWS`` is the cross-repo seam: the omnimarket-side
workflow-state proxy imports this SAME ContextVar object to read the row the
wiring loaded before ``handle()`` and to hand back the mutated payload after.
Do not rename or re-instantiate it — a second ``ContextVar`` instance with the
same name is a different object and breaks the seam silently.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from contextvars import ContextVar
from uuid import UUID

import asyncpg

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.errors.repository import (
    RepositoryContractError,
    RepositoryExecutionError,
)

logger = logging.getLogger(__name__)

# Defense-in-depth: table name comes from contract YAML (state_io.table), not
# user input, but is still interpolated into SQL — validate it the same way
# idempotency/store_postgres.py validates its configured table name.
_TABLE_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_DEFAULT_STALE_TTL_SECONDS = 900
_STALE_TTL_ENV = "DELEGATION_STALE_TTL_SECONDS"

# Cross-repo seam ContextVar. Value is unconditionally set (never left as the
# default None) for the duration of a state_io-wired dispatch, keyed by the
# envelope's correlation_id, to `(payload_json, version)` — `payload_json` is
# `None` (not a missing key) when no row exists yet, so "no row for this
# correlation_id" is distinguishable from "state_io is not active on this
# dispatch" (ContextVar still at its unset default).
CONTEXTVAR_STATE_IO_ROWS: ContextVar[dict[str, tuple[str | None, int]] | None] = (
    ContextVar("onex_state_io_rows", default=None)
)


class StateStoreAdapter:
    """Async load/seed/CAS-update adapter over one state_io-declared table.

    Table shape (see
    ``docker/migrations/forward/090_create_delegation_workflow_state.sql``)::

        correlation_id TEXT PRIMARY KEY
        tenant_id      TEXT NOT NULL
        state          TEXT NOT NULL
        in_flight      BOOLEAN NOT NULL DEFAULT FALSE
        payload        JSONB NOT NULL
        version        INTEGER NOT NULL DEFAULT 0
        created_at / updated_at TIMESTAMPTZ

    The asyncpg pool is created lazily on first use (guarded by an
    ``asyncio.Lock`` so concurrent first-dispatches don't race to open two
    pools) rather than eagerly at wiring time, matching the opt-in nature of
    the seam — no connection is attempted until a state_io-declaring contract
    actually dispatches.
    """

    def __init__(self, dsn: str, *, table: str) -> None:
        """Initialize the adapter.

        Args:
            dsn: PostgreSQL connection string (resolved by the caller from
                the contract-declared database's env var — never defaulted
                here; a missing DSN is a wiring-time fail-closed error, not
                an adapter-level concern).
            table: Table name (contract-declared ``state_io.table``).

        Raises:
            RepositoryContractError: If ``table`` is not a valid identifier.
        """
        if not _TABLE_NAME_PATTERN.match(table):
            raise RepositoryContractError(
                f"Invalid state_io table name: {table!r} — must match "
                r"^[a-zA-Z_][a-zA-Z0-9_]*$",
                op_name="__init__",
                table=table,
            )
        self._dsn = dsn
        self._table = table
        self._pool: asyncpg.Pool | None = None
        self._pool_lock = asyncio.Lock()

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is not None:
            return self._pool
        async with self._pool_lock:
            if self._pool is None:
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn, min_size=1, max_size=5
                )
        return self._pool

    async def close(self) -> None:
        """Close the underlying pool, if one was ever created."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    def _context(
        self, operation: str, correlation_id: str | None
    ) -> ModelInfraErrorContext:
        parsed_correlation_id: UUID | None = None
        if correlation_id is not None:
            try:
                parsed_correlation_id = UUID(correlation_id)
            except ValueError:
                parsed_correlation_id = None
        return ModelInfraErrorContext.with_correlation(
            correlation_id=parsed_correlation_id,
            transport_type=EnumInfraTransportType.DATABASE,
            operation=operation,
            target_name=self._table,
        )

    async def load(self, correlation_id: str) -> tuple[str, int] | None:
        """Load the current opaque payload + version for a correlation_id.

        Returns:
            ``(payload_json, version)`` if a row exists, else ``None``.
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT payload::text AS payload_json, version "  # noqa: S608 — table validated in __init__
                    f"FROM {self._table} WHERE correlation_id = $1",
                    correlation_id,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"state_io load failed: {type(exc).__name__}",
                op_name="load",
                table=self._table,
                context=self._context("load", correlation_id),
            ) from exc
        if row is None:
            return None
        return row["payload_json"], row["version"]

    async def seed(
        self,
        correlation_id: str,
        *,
        tenant_id: str,
        state: str,
        in_flight: bool,
        payload_json: str,
    ) -> bool:
        """Insert the first row for a correlation_id (leg-1 creation).

        Uses ``INSERT ... ON CONFLICT (correlation_id) DO NOTHING RETURNING``
        so a losing racer against a concurrent leg-1 dispatch for the same
        correlation_id can tell it lost (no row returned) and reload +
        retry against the winner's row rather than silently dropping its own
        ``handle()`` output.

        Returns:
            True if this call's INSERT won (row created), False if a
            concurrent seed already created the row first (conflict — the
            caller must reload and retry).
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"INSERT INTO {self._table} "  # noqa: S608 — table validated in __init__
                    f"(correlation_id, tenant_id, state, in_flight, payload) "
                    f"VALUES ($1, $2, $3, $4, $5::jsonb) "
                    f"ON CONFLICT (correlation_id) DO NOTHING "
                    f"RETURNING correlation_id",
                    correlation_id,
                    tenant_id,
                    state,
                    in_flight,
                    payload_json,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"state_io seed failed: {type(exc).__name__}",
                op_name="seed",
                table=self._table,
                context=self._context("seed", correlation_id),
            ) from exc
        return row is not None

    async def cas_update(
        self,
        correlation_id: str,
        *,
        tenant_id: str,
        state: str,
        in_flight: bool,
        payload_json: str,
        expected_version: int,
    ) -> int:
        """Compare-and-swap update gated on the row's current version.

        Returns:
            Row count affected by the UPDATE (0 = conflict — a concurrent
            writer advanced ``version`` since this call's ``load()``).
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"UPDATE {self._table} "  # noqa: S608 — table validated in __init__
                    f"SET payload = $1::jsonb, state = $2, in_flight = $3, "
                    f"tenant_id = $4, version = version + 1 "
                    f"WHERE correlation_id = $5 AND version = $6",
                    payload_json,
                    state,
                    in_flight,
                    tenant_id,
                    correlation_id,
                    expected_version,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"state_io cas_update failed: {type(exc).__name__}",
                op_name="cas_update",
                table=self._table,
                context=self._context("cas_update", correlation_id),
            ) from exc
        # asyncpg execute() returns a tag string like "UPDATE 1" / "UPDATE 0".
        return int(result.split()[-1])

    async def recover_stale_rows(self, ttl_seconds: int | None = None) -> int:
        """Fail closed on abandoned in-flight rows past their TTL.

        Marks any row that is still ``in_flight`` and NOT terminal
        (``COMPLETED``/``FAILED``) whose ``updated_at`` is older than the TTL
        as ``FAILED`` — recovery from a process crash / redeploy that left a
        row wedged mid-workflow with no further leg ever arriving to advance
        it. Documented limitation (OMN-14208 slice-1): this is row-only
        recovery — no terminal event is emitted to the bus, since this
        adapter has no bus access (the full event-sourced end-state that
        emits a terminal FAILED event is OMN-14107).

        Args:
            ttl_seconds: Override for testing. Defaults to
                ``DELEGATION_STALE_TTL_SECONDS`` env var, else 900s.

        Returns:
            Number of rows recovered.
        """
        ttl = (
            ttl_seconds
            if ttl_seconds is not None
            else int(os.environ.get(_STALE_TTL_ENV, _DEFAULT_STALE_TTL_SECONDS))
        )
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    f"UPDATE {self._table} "  # noqa: S608 — table validated in __init__
                    f"SET state = 'FAILED', "
                    f"    in_flight = FALSE, "
                    f"    payload = jsonb_set("
                    f"        payload, '{{failure_reason}}', "
                    f"        '\"stale_in_flight_recovery\"'"
                    f"    ), "
                    f"    version = version + 1 "
                    f"WHERE state NOT IN ('COMPLETED', 'FAILED') "
                    f"  AND in_flight "
                    f"  AND updated_at < NOW() - make_interval(secs => $1)",
                    ttl,
                )
        except asyncpg.PostgresError as exc:
            raise RepositoryExecutionError(
                f"state_io recover_stale_rows failed: {type(exc).__name__}",
                op_name="recover_stale_rows",
                table=self._table,
                context=self._context("recover_stale_rows", None),
            ) from exc
        recovered = int(result.split()[-1])
        if recovered:
            logger.warning(
                "state_io recovered %d stale in-flight row(s) in %s "
                "(ttl_seconds=%d) — row-only recovery, no terminal event "
                "emitted (OMN-14107 tracks full event-sourced recovery)",
                recovered,
                self._table,
                ttl,
            )
        return recovered


class StateIoUnconfiguredError(RuntimeHostError):
    """Raised at wiring time when a contract declares state_io but its DSN env var is unset.

    Unlike the optional db_io projection path (which logs and returns None
    when its DSN is unset), state_io is a REQUIRED durability seam — a
    contract that opts in without a working DSN is a startup-fatal
    configuration error, not a degradable condition.
    """


__all__ = [
    "CONTEXTVAR_STATE_IO_ROWS",
    "StateIoUnconfiguredError",
    "StateStoreAdapter",
]
