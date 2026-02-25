# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler for cursor-paginated decision store queries.

Executes PostgreSQL SELECT queries against the decision_store table with
optional filters for domain, layer, decision_type, tags, epic_id, status,
and scope_services (ANY/ALL/EXACT modes). Returns a paginated result with
an opaque cursor for page continuation.

Cursor encoding:
    Encode: base64(f"{created_at.isoformat()}|{decision_id}")
    Decode: split on first "|", parse ISO datetime and UUID
    Pagination predicate: (created_at, decision_id) < (cursor_ts, cursor_id)
    Ordering: created_at DESC, decision_id DESC

Cursor is stable under concurrent writes — new inserts do not shift existing
page boundaries because the predicate references immutable column values.

scope_services_mode semantics:
    ANY  — JSONB array overlaps with filter: scope_services ?| $filter_array
    ALL  — JSONB array is a superset: all filter items present in entry
    EXACT — JSONB array equals filter set (after sort+lower normalization)
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_decision_store_query_compute.models.model_payload_query_decisions import (
    ModelPayloadQueryDecisions,
)
from omnibase_infra.nodes.node_decision_store_query_compute.models.model_result_decision_list import (
    ModelResultDecisionList,
)

if TYPE_CHECKING:
    import asyncpg

    from omnibase_core.models.store.model_decision_store_entry import (
        ModelDecisionStoreEntry,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

# Base column list matches decision_store schema from OMN-2764 migration.
_SQL_COLUMNS = """
    decision_id,
    correlation_id,
    title,
    decision_type,
    status,
    scope_domain,
    scope_services,
    scope_layer,
    rationale,
    alternatives,
    tags,
    epic_id,
    supersedes,
    superseded_by,
    source,
    created_at,
    db_written_at,
    created_by
"""

# ---------------------------------------------------------------------------
# Cursor helpers
# ---------------------------------------------------------------------------


def encode_cursor(created_at: datetime, decision_id: UUID) -> str:
    """Encode a pagination cursor from (created_at, decision_id).

    The cursor is base64-encoded to be opaque to callers.

    Args:
        created_at: Timezone-aware creation timestamp of the last row on page.
        decision_id: UUID of the last row on the current page.

    Returns:
        Base64-encoded opaque cursor string.
    """
    raw = f"{created_at.isoformat()}|{decision_id}"
    return base64.b64encode(raw.encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Decode a pagination cursor into (created_at, decision_id).

    Args:
        cursor: Base64-encoded cursor from a previous result's next_cursor.

    Returns:
        Tuple of (created_at datetime, decision_id UUID).

    Raises:
        ValueError: If the cursor is malformed or cannot be decoded.
    """
    try:
        raw = base64.b64decode(cursor.encode()).decode()
        ts_str, id_str = raw.split("|", 1)
        created_at = datetime.fromisoformat(ts_str)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        decision_id = UUID(id_str)
        return created_at, decision_id
    except Exception as exc:
        msg = f"Invalid cursor format: {exc}"
        raise ValueError(msg) from exc


# ---------------------------------------------------------------------------
# Row-to-model conversion helper
# ---------------------------------------------------------------------------


def _row_to_entry(row: Mapping[str, object]) -> ModelDecisionStoreEntry:
    """Convert an asyncpg Record to a ModelDecisionStoreEntry.

    Args:
        row: asyncpg Record (Mapping[str, object]) from decision_store SELECT.

    Returns:
        ModelDecisionStoreEntry populated from the row.

    Raises:
        ImportError: If omnibase_core.models.store is not yet available
            (dependency on OMN-2763 PR #562).
    """
    from omnibase_core.models.store.model_decision_store_entry import (
        ModelDecisionStoreEntry,
    )

    # JSONB columns come back as strings from asyncpg; parse and cast to list[object].
    _ss = row["scope_services"]
    scope_services_list = cast(
        "list[object]", json.loads(_ss) if isinstance(_ss, str) else _ss
    )
    _tags = row["tags"]
    tags_list = cast(
        "list[object]", json.loads(_tags) if isinstance(_tags, str) else _tags
    )
    _alts = row["alternatives"]
    alternatives_list = cast(
        "list[object]", json.loads(_alts) if isinstance(_alts, str) else _alts
    )
    _sup = row["supersedes"]
    supersedes_list = cast(
        "list[object]", json.loads(_sup) if isinstance(_sup, str) else _sup
    )

    return ModelDecisionStoreEntry.model_validate(
        {
            "decision_id": row["decision_id"],
            "correlation_id": row["correlation_id"],
            "title": row["title"],
            "decision_type": row["decision_type"],
            "status": row["status"],
            "scope_domain": row["scope_domain"],
            "scope_services": tuple(scope_services_list),
            "scope_layer": row["scope_layer"],
            "rationale": row["rationale"],
            "alternatives": alternatives_list,
            "tags": tuple(tags_list),
            "epic_id": row["epic_id"],
            "supersedes": tuple(UUID(str(s)) for s in supersedes_list),
            "superseded_by": row["superseded_by"],
            "source": row["source"],
            "created_at": row["created_at"],
        }
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class HandlerQueryDecisions:
    """Cursor-paginated query handler for the decision store.

    Executes a SELECT against decision_store with filter parameters from
    ModelPayloadQueryDecisions and returns a ModelResultDecisionList with
    an opaque cursor for page continuation.

    Cursor stability:
        The cursor encodes (created_at, decision_id) of the last row
        returned. The pagination predicate is:
            WHERE (created_at, decision_id) < (cursor_ts, cursor_id)
        This is stable under concurrent writes — new inserts after the
        cursor position do not shift existing page boundaries.

    scope_services_mode semantics:
        ANY  — at least one filter service is present in the entry
               (JSONB overlap: scope_services ?| $array)
        ALL  — all filter services are present in the entry
               (superset: checked via containment for each element)
        EXACT — entry scope_services set equals filter set exactly
                (normalized: sorted lowercase before comparison)

    Empty scope_services filter (None or []):
        No scope filter applied — returns all decisions including
        platform-wide ones (entries with empty scope_services).

    Attributes:
        _pool: asyncpg connection pool.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """Initialise with an asyncpg connection pool.

        Args:
            pool: Pre-configured asyncpg connection pool.
        """
        self._pool = pool

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-query-decisions"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: EFFECT (performs DB I/O).

        Returns:
            EnumHandlerTypeCategory.EFFECT
        """
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        payload: ModelPayloadQueryDecisions,
    ) -> ModelResultDecisionList:
        """Execute the cursor-paginated decision store query.

        Builds a dynamic WHERE clause from the payload filters, applies
        cursor-based pagination, and returns a ModelResultDecisionList
        containing the page of decisions, next_cursor, and total_active count.

        Args:
            payload: Query filter parameters including cursor and limit.

        Returns:
            ModelResultDecisionList with decisions, next_cursor, and total_active.

        Raises:
            ValueError: If the cursor is malformed.
            asyncpg.PostgresError: On database errors.
        """
        # Build dynamic WHERE clause
        conditions: list[str] = []
        params: list[object] = []
        idx = 1  # asyncpg uses $1, $2, ... placeholders

        # --- Static filters ---
        if payload.status:
            conditions.append(f"status = ${idx}")
            params.append(payload.status)
            idx += 1

        if payload.domain is not None:
            conditions.append(f"scope_domain = ${idx}")
            params.append(payload.domain)
            idx += 1

        if payload.layer is not None:
            conditions.append(f"scope_layer = ${idx}")
            params.append(payload.layer)
            idx += 1

        if payload.epic_id is not None:
            conditions.append(f"epic_id = ${idx}")
            params.append(payload.epic_id)
            idx += 1

        # --- decision_type: OR semantics (any of the listed types) ---
        if payload.decision_type:
            placeholders = ", ".join(
                f"${idx + i}" for i in range(len(payload.decision_type))
            )
            conditions.append(f"decision_type IN ({placeholders})")
            params.extend(payload.decision_type)
            idx += len(payload.decision_type)

        # --- tags: ALL must be present (each tag contained in JSONB array) ---
        if payload.tags:
            for tag in payload.tags:
                # JSONB array containment: tags @> '["tag"]'::jsonb
                conditions.append(f"tags @> ${idx}::jsonb")
                params.append(json.dumps([tag]))
                idx += 1

        # --- scope_services filter ---
        if payload.scope_services:
            scope_svc_normalized = sorted(s.lower() for s in payload.scope_services)
            mode = payload.scope_services_mode

            if mode == "ANY":
                # Overlap: entry has at least one of the filter services
                # JSONB ?| operator: scope_services ?| array[...]
                conditions.append(f"scope_services ?| ${idx}::text[]")
                params.append(scope_svc_normalized)
                idx += 1

            elif mode == "ALL":
                # Superset: entry contains all filter services
                # Each filter service must be present: scope_services @> '["svc"]'
                for svc in scope_svc_normalized:
                    conditions.append(f"scope_services @> ${idx}::jsonb")
                    params.append(json.dumps([svc]))
                    idx += 1

            elif mode == "EXACT":
                # Exact match: normalize both sides and compare as JSONB array
                # We store scope_services as sorted JSON array, so this works
                # when entries are normalized on write (OMN-2765 handler does this).
                conditions.append(f"scope_services = ${idx}::jsonb")
                params.append(json.dumps(scope_svc_normalized))
                idx += 1

        # --- Cursor predicate ---
        cursor_ts: datetime | None = None
        cursor_id: UUID | None = None
        if payload.cursor is not None:
            cursor_ts, cursor_id = decode_cursor(payload.cursor)
            # Tuple comparison for stable cursor: (created_at, decision_id) < (ts, id)
            conditions.append(f"(created_at, decision_id) < (${idx}, ${idx + 1})")
            params.append(cursor_ts)
            params.append(cursor_id)
            idx += 2

        # --- Assemble WHERE clause ---
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + "\n    AND ".join(conditions)

        # --- COUNT query (same WHERE, no LIMIT) ---

        # all user-supplied values are bound via $N asyncpg parameters, never interpolated.
        count_sql = f"SELECT COUNT(*) FROM decision_store\n{where_clause}"  # noqa: S608

        # --- SELECT query ---
        select_sql = (
            f"SELECT {_SQL_COLUMNS}"
            f"FROM decision_store\n"
            f"{where_clause}\n"
            f"ORDER BY created_at DESC, decision_id DESC\n"
            f"LIMIT ${idx}"
        )
        params_with_limit = params + [payload.limit]

        async with self._pool.acquire() as conn:
            total_active = await conn.fetchval(count_sql, *params)
            rows = await conn.fetch(select_sql, *params_with_limit)

        # Convert rows to ModelDecisionStoreEntry instances
        decisions = tuple(_row_to_entry(row) for row in rows)

        # Compute next_cursor: None if we got fewer rows than limit
        next_cursor: str | None = None
        if len(decisions) == payload.limit and len(decisions) > 0:
            last = decisions[-1]
            next_cursor = encode_cursor(last.created_at, last.decision_id)

        logger.debug(
            "QueryDecisions: filters=%s, cursor=%s, limit=%d -> "
            "returned=%d, total_active=%d, next_cursor=%s",
            {
                "domain": payload.domain,
                "layer": payload.layer,
                "status": payload.status,
                "epic_id": payload.epic_id,
                "scope_services_mode": payload.scope_services_mode,
            },
            payload.cursor is not None,
            payload.limit,
            len(decisions),
            total_active,
            next_cursor is not None,
        )

        return ModelResultDecisionList(
            decisions=decisions,
            next_cursor=next_cursor,
            total_active=total_active,
        )


__all__: list[str] = [
    "HandlerQueryDecisions",
    "encode_cursor",
    "decode_cursor",
]
