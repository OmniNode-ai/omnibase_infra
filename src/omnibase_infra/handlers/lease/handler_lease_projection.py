# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease projection writer (OMN-21424).

Upserts on ``(kind, resource)``; the ``WHERE`` clause keeps a row whose fence
is above the incoming one, so an out-of-order replay never lowers a fence.
At one fence a ``held`` row never overwrites a ``released`` or ``expired`` one,
the same rank the reducer applies, so a replayed grant or late renewal cannot
revive an ended lease. The table is created by
``docker/migrations/forward/nodes/node_projection_lease/0001_create_lease_projection.sql``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.lease import (
    ModelLeaseProjectionRow,
    ModelPayloadLeaseProjectionUpsert,
)

UPSERT_LEASE_PROJECTION_SQL = """
INSERT INTO lease_projection (kind, resource, lease_id, fence, holder, state)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (kind, resource) DO UPDATE SET
    lease_id = EXCLUDED.lease_id,
    fence = EXCLUDED.fence,
    holder = CASE
        WHEN lease_projection.fence = EXCLUDED.fence AND EXCLUDED.holder = ''
        THEN lease_projection.holder
        ELSE EXCLUDED.holder
    END,
    state = EXCLUDED.state,
    updated_at = CURRENT_TIMESTAMP
WHERE lease_projection.fence < EXCLUDED.fence
   OR (
       lease_projection.fence = EXCLUDED.fence
       AND (lease_projection.state = 'held' OR EXCLUDED.state <> 'held')
   )
""".strip()


# The bound ``execute(sql, *args)`` of an asyncpg connection or pool.
LeaseProjectionExecute = Callable[..., Awaitable[str]]


def lease_projection_params(row: ModelLeaseProjectionRow) -> tuple[object, ...]:
    """Positional parameters of ``UPSERT_LEASE_PROJECTION_SQL`` for ``row``."""
    return (
        row.kind,
        row.resource,
        str(row.lease_id),
        row.fence,
        row.holder,
        row.state,
    )


class HandlerLeaseProjection:
    """Applies the reducer's upsert intents to the lease projection table."""

    def __init__(self, execute: LeaseProjectionExecute) -> None:
        self._execute = execute

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, intent: ModelPayloadLeaseProjectionUpsert) -> None:
        await self.apply(intent)

    async def apply(self, intent: ModelPayloadLeaseProjectionUpsert) -> None:
        await self._execute(
            UPSERT_LEASE_PROJECTION_SQL, *lease_projection_params(intent.row)
        )


__all__ = [
    "UPSERT_LEASE_PROJECTION_SQL",
    "HandlerLeaseProjection",
    "LeaseProjectionExecute",
    "lease_projection_params",
]
