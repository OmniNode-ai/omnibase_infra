# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lease projection writer (OMN-21424).

Upserts on ``(kind, resource)``; the ``WHERE`` clause keeps a row whose fence
is above the incoming one, so an out-of-order replay never lowers a fence. At an
equal fence the statement applies the reducer's join (state by the order
``held < released < expired``, the lease ID of the higher-ranked state with
the greater UUID breaking equal-rank ties, the greater holder), so a replay
that reaches the table in any order, including
after the reducer restarts with empty state, lands the same row.
The table is created by
``docker/migrations/forward/nodes/node_projection_lease/0000_create_lease_projection.sql``.
"""

from __future__ import annotations

import asyncpg

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.lease import (
    ModelLeaseProjectionRow,
    ModelPayloadLeaseProjectionUpsert,
)

UPSERT_LEASE_PROJECTION_SQL = """
INSERT INTO lease_projection (kind, resource, lease_id, fence, holder, state)
VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (kind, resource) DO UPDATE SET
    lease_id = CASE
        WHEN EXCLUDED.fence > lease_projection.fence
            OR CASE EXCLUDED.state WHEN 'released' THEN 1 WHEN 'expired' THEN 2 ELSE 0 END
            > CASE lease_projection.state WHEN 'released' THEN 1 WHEN 'expired' THEN 2 ELSE 0 END
            OR (
                CASE EXCLUDED.state WHEN 'released' THEN 1 WHEN 'expired' THEN 2 ELSE 0 END
                = CASE lease_projection.state WHEN 'released' THEN 1 WHEN 'expired' THEN 2 ELSE 0 END
                AND EXCLUDED.lease_id > lease_projection.lease_id
            )
        THEN EXCLUDED.lease_id ELSE lease_projection.lease_id END,
    fence = EXCLUDED.fence,
    holder = CASE
        WHEN EXCLUDED.fence > lease_projection.fence THEN EXCLUDED.holder
        WHEN EXCLUDED.holder > lease_projection.holder THEN EXCLUDED.holder
        ELSE lease_projection.holder END,
    state = CASE
        WHEN EXCLUDED.fence > lease_projection.fence
            OR CASE EXCLUDED.state WHEN 'released' THEN 1 WHEN 'expired' THEN 2 ELSE 0 END
            >= CASE lease_projection.state WHEN 'released' THEN 1 WHEN 'expired' THEN 2 ELSE 0 END
        THEN EXCLUDED.state ELSE lease_projection.state END,
    updated_at = now()
WHERE lease_projection.fence <= EXCLUDED.fence
""".strip()


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

    def __init__(self, executor: asyncpg.Pool) -> None:
        self._executor = executor

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, intent: ModelPayloadLeaseProjectionUpsert) -> None:
        await self.apply(intent)

    async def apply(self, intent: ModelPayloadLeaseProjectionUpsert) -> None:
        await self._executor.execute(
            UPSERT_LEASE_PROJECTION_SQL, *lease_projection_params(intent.row)
        )


__all__ = [
    "UPSERT_LEASE_PROJECTION_SQL",
    "HandlerLeaseProjection",
    "lease_projection_params",
]
