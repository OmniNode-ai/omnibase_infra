# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for NodeDecisionStoreQueryCompute.

Tests:
- ModelPayloadQueryDecisions construction, defaults, and validation
- ModelResultDecisionList construction and defaults
- Cursor encoding/decoding (deterministic round-trip)
- decode_cursor raises ValueError on malformed input
- next_cursor is None when result count < limit
- next_cursor is set when result count == limit
- HandlerQueryDecisions handler properties
- HandlerQueryDecisions.handle — scope_services_mode ANY (mock pool)
- HandlerQueryDecisions.handle — scope_services_mode ALL (mock pool)
- HandlerQueryDecisions.handle — scope_services_mode EXACT (mock pool)
- HandlerQueryDecisions.handle — no scope_services filter (mock pool)
- HandlerQueryDecisions.handle — cursor pagination predicate applied
- HandlerQueryDecisions.handle — total_active matches count query
- HandlerQueryDecisions.handle — next_cursor None on last page
- SQL WHERE clause builder: domain + layer + status filters
- SQL WHERE clause builder: decision_type OR semantics
- SQL WHERE clause builder: tags ALL semantics
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions import (
    HandlerQueryDecisions,
    decode_cursor,
    encode_cursor,
)
from omnibase_infra.nodes.node_decision_store_query_compute.models.model_payload_query_decisions import (
    ModelPayloadQueryDecisions,
)
from omnibase_infra.nodes.node_decision_store_query_compute.models.model_result_decision_list import (
    ModelResultDecisionList,
)

pytestmark = pytest.mark.unit

# ============================================================================
# Helpers
# ============================================================================

_FIXED_TS = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)
_FIXED_ID = UUID("00000000-0000-0000-0000-000000000001")
_FIXED_ID_2 = UUID("00000000-0000-0000-0000-000000000002")


def _make_mock_row(
    decision_id: UUID | None = None,
    created_at: datetime | None = None,
    scope_services: list[str] | None = None,
    status: str = "ACTIVE",
    decision_type: str = "TECH_STACK_CHOICE",
    scope_domain: str = "data-model",
    scope_layer: str = "architecture",
) -> dict[str, Any]:
    """Build a minimal mock asyncpg row dict."""
    return {
        "decision_id": decision_id or uuid4(),
        "correlation_id": uuid4(),
        "title": "Test Decision",
        "decision_type": decision_type,
        "status": status,
        "scope_domain": scope_domain,
        "scope_services": json.dumps(scope_services or []),
        "scope_layer": scope_layer,
        "rationale": "Test rationale",
        "alternatives": json.dumps([]),
        "tags": json.dumps([]),
        "epic_id": None,
        "supersedes": json.dumps([]),
        "superseded_by": None,
        "source": "manual",
        "created_at": created_at or _FIXED_TS,
        "db_written_at": _FIXED_TS,
        "created_by": "test-agent",
    }


class _MockEntry:
    """Lightweight stand-in for ModelDecisionStoreEntry in handler tests."""

    def __init__(self, decision_id: UUID, created_at: datetime) -> None:
        self.decision_id = decision_id
        self.created_at = created_at


# ============================================================================
# ModelPayloadQueryDecisions — construction and defaults
# ============================================================================


def test_payload_defaults() -> None:
    payload = ModelPayloadQueryDecisions()
    assert payload.domain is None
    assert payload.layer is None
    assert payload.decision_type is None
    assert payload.tags is None
    assert payload.epic_id is None
    assert payload.status == "ACTIVE"
    assert payload.scope_services is None
    assert payload.scope_services_mode == "ANY"
    assert payload.cursor is None
    assert payload.limit == 50


def test_payload_custom_values() -> None:
    payload = ModelPayloadQueryDecisions(
        domain="data-model",
        layer="architecture",
        decision_type=["TECH_STACK_CHOICE", "API_DESIGN"],
        tags=["database", "storage"],
        epic_id="OMN-2762",
        status="PROPOSED",
        scope_services=["omnibase_core", "omnibase_infra"],
        scope_services_mode="ALL",
        cursor="dGVzdA==",
        limit=25,
    )
    assert payload.domain == "data-model"
    assert payload.layer == "architecture"
    assert payload.decision_type == ["TECH_STACK_CHOICE", "API_DESIGN"]
    assert payload.tags == ["database", "storage"]
    assert payload.epic_id == "OMN-2762"
    assert payload.status == "PROPOSED"
    assert payload.scope_services == ["omnibase_core", "omnibase_infra"]
    assert payload.scope_services_mode == "ALL"
    assert payload.cursor == "dGVzdA=="
    assert payload.limit == 25


def test_payload_frozen() -> None:
    payload = ModelPayloadQueryDecisions(domain="x")
    with pytest.raises(Exception):  # ValidationError or TypeError for frozen
        payload.domain = "y"  # type: ignore[misc]


def test_payload_limit_bounds() -> None:
    with pytest.raises(Exception):
        ModelPayloadQueryDecisions(limit=0)
    with pytest.raises(Exception):
        ModelPayloadQueryDecisions(limit=1001)
    # Boundary values should work
    p1 = ModelPayloadQueryDecisions(limit=1)
    p2 = ModelPayloadQueryDecisions(limit=1000)
    assert p1.limit == 1
    assert p2.limit == 1000


def test_payload_scope_services_mode_values() -> None:
    for mode in ("ANY", "ALL", "EXACT"):
        p = ModelPayloadQueryDecisions(
            scope_services=["svc"],
            scope_services_mode=mode,  # type: ignore[arg-type]
        )
        assert p.scope_services_mode == mode


def test_payload_invalid_scope_services_mode() -> None:
    with pytest.raises(Exception):
        ModelPayloadQueryDecisions(
            scope_services=["svc"],
            scope_services_mode="INVALID",  # type: ignore[arg-type]
        )


# ============================================================================
# ModelResultDecisionList — construction and defaults
# ============================================================================


def test_result_defaults() -> None:
    result = ModelResultDecisionList(total_active=0)
    assert result.decisions == ()
    assert result.next_cursor is None
    assert result.total_active == 0


def test_result_with_decisions() -> None:
    entry = _MockEntry(decision_id=_FIXED_ID, created_at=_FIXED_TS)
    result = ModelResultDecisionList(
        decisions=(entry,),  # type: ignore[arg-type]
        next_cursor="abc123",
        total_active=42,
    )
    assert len(result.decisions) == 1
    assert result.next_cursor == "abc123"
    assert result.total_active == 42


def test_result_negative_total_active_rejected() -> None:
    with pytest.raises(Exception):
        ModelResultDecisionList(total_active=-1)


# ============================================================================
# Cursor encoding / decoding
# ============================================================================


def test_encode_cursor_deterministic() -> None:
    cursor1 = encode_cursor(_FIXED_TS, _FIXED_ID)
    cursor2 = encode_cursor(_FIXED_TS, _FIXED_ID)
    assert cursor1 == cursor2


def test_encode_cursor_is_base64() -> None:
    cursor = encode_cursor(_FIXED_TS, _FIXED_ID)
    # Should decode without error
    decoded = base64.b64decode(cursor.encode()).decode()
    assert "|" in decoded


def test_decode_cursor_round_trip() -> None:
    original_ts = _FIXED_TS
    original_id = _FIXED_ID
    cursor = encode_cursor(original_ts, original_id)
    decoded_ts, decoded_id = decode_cursor(cursor)
    assert decoded_id == original_id
    # Timestamps compare equal (ISO roundtrip preserves timezone)
    assert decoded_ts == original_ts


def test_decode_cursor_timezone_aware() -> None:
    cursor = encode_cursor(_FIXED_TS, _FIXED_ID)
    ts, _ = decode_cursor(cursor)
    assert ts.tzinfo is not None


def test_decode_cursor_malformed_raises() -> None:
    with pytest.raises(ValueError):
        decode_cursor("not-valid-base64!!!")


def test_decode_cursor_missing_pipe_raises() -> None:
    # Valid base64 but no pipe separator
    bad = base64.b64encode(b"nodivider").decode()
    with pytest.raises(ValueError):
        decode_cursor(bad)


def test_decode_cursor_invalid_uuid_raises() -> None:
    bad = base64.b64encode(b"2026-01-01T00:00:00+00:00|not-a-uuid").decode()
    with pytest.raises(ValueError):
        decode_cursor(bad)


def test_cursor_different_inputs_differ() -> None:
    c1 = encode_cursor(_FIXED_TS, _FIXED_ID)
    c2 = encode_cursor(_FIXED_TS, _FIXED_ID_2)
    assert c1 != c2


# ============================================================================
# HandlerQueryDecisions — handler properties
# ============================================================================


def test_handler_properties() -> None:
    mock_pool = MagicMock()
    handler = HandlerQueryDecisions(pool=mock_pool)
    assert handler.handler_id == "handler-query-decisions"
    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# ============================================================================
# HandlerQueryDecisions.handle — mocked pool, no ModelDecisionStoreEntry
# ============================================================================
# We patch _row_to_entry to return lightweight mock entries so these tests
# do not depend on omnibase_core.models.store being available.


def _make_mock_conn(
    rows: list[dict[str, Any]],
    count: int,
) -> MagicMock:
    """Create a mock asyncpg connection with fetch/fetchval results."""
    conn = AsyncMock()
    conn.fetchval = AsyncMock(return_value=count)
    conn.fetch = AsyncMock(return_value=rows)
    return conn


def _make_mock_pool(rows: list[dict[str, Any]], count: int) -> MagicMock:
    """Create a mock asyncpg pool that yields the mock connection."""
    pool = MagicMock()
    mock_conn = _make_mock_conn(rows, count)
    pool.acquire = MagicMock(
        return_value=MagicMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None),
        )
    )
    return pool


@pytest.mark.asyncio
async def test_handle_no_filters_empty_result() -> None:
    pool = _make_mock_pool(rows=[], count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(limit=50)

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ) as mock_convert:
        mock_convert.side_effect = lambda r: _MockEntry(
            decision_id=r["decision_id"], created_at=r["created_at"]
        )
        result = await handler.handle(payload)

    assert result.decisions == ()
    assert result.next_cursor is None
    assert result.total_active == 0


@pytest.mark.asyncio
async def test_handle_next_cursor_none_when_less_than_limit() -> None:
    """next_cursor must be None when returned rows < limit (last page)."""
    rows = [_make_mock_row(decision_id=_FIXED_ID, created_at=_FIXED_TS)]
    pool = _make_mock_pool(rows=rows, count=1)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(limit=50)  # 1 row < 50 limit

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ) as mock_convert:
        mock_convert.side_effect = lambda r: _MockEntry(
            decision_id=r["decision_id"], created_at=r["created_at"]
        )
        result = await handler.handle(payload)

    assert result.next_cursor is None
    assert result.total_active == 1


@pytest.mark.asyncio
async def test_handle_next_cursor_set_when_full_page() -> None:
    """next_cursor must be set when returned rows == limit."""
    rows = [
        _make_mock_row(
            decision_id=UUID(f"0000000{i}-0000-0000-0000-000000000000"),
            created_at=_FIXED_TS,
        )
        for i in range(1, 4)  # 3 rows
    ]
    pool = _make_mock_pool(rows=rows, count=10)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(limit=3)  # 3 rows == 3 limit

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ) as mock_convert:
        mock_convert.side_effect = lambda r: _MockEntry(
            decision_id=r["decision_id"], created_at=r["created_at"]
        )
        result = await handler.handle(payload)

    assert result.next_cursor is not None
    # Verify cursor can be decoded
    ts, _uid = decode_cursor(result.next_cursor)
    assert ts == _FIXED_TS


@pytest.mark.asyncio
async def test_handle_total_active_from_count_query() -> None:
    """total_active must use the count result, not len(decisions)."""
    rows = [_make_mock_row(decision_id=_FIXED_ID, created_at=_FIXED_TS)]
    pool = _make_mock_pool(rows=rows, count=999)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(limit=50)

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ) as mock_convert:
        mock_convert.side_effect = lambda r: _MockEntry(
            decision_id=r["decision_id"], created_at=r["created_at"]
        )
        result = await handler.handle(payload)

    assert result.total_active == 999


@pytest.mark.asyncio
async def test_handle_scope_services_mode_any_sends_array_param() -> None:
    """ANY mode should pass the filter as a text[] parameter."""
    rows: list[dict[str, Any]] = []
    pool = _make_mock_pool(rows=rows, count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(
        scope_services=["omnibase_core", "omnibase_infra"],
        scope_services_mode="ANY",
        limit=10,
    )

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        result = await handler.handle(payload)

    # Verify the count query was called (implies WHERE clause was built)
    conn = pool.acquire.return_value.__aenter__.return_value
    count_call_args = conn.fetchval.call_args
    assert count_call_args is not None
    # The param passed should be the normalized sorted service list
    sql_args = count_call_args[0]
    # For ANY mode: params include a list for the ?| operator
    found_list = any(isinstance(a, list) and "omnibase_core" in a for a in sql_args[1:])
    assert found_list, f"Expected service list in params, got: {sql_args}"

    assert result.total_active == 0


@pytest.mark.asyncio
async def test_handle_scope_services_mode_all_uses_containment() -> None:
    """ALL mode: each filter service generates a separate @> condition."""
    rows: list[dict[str, Any]] = []
    pool = _make_mock_pool(rows=rows, count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(
        scope_services=["svc_a", "svc_b"],
        scope_services_mode="ALL",
        limit=10,
    )

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        result = await handler.handle(payload)

    # Count query called
    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    # Both services should appear as JSON containment params
    assert "@>" in count_sql, f"Expected @> in SQL, got: {count_sql}"
    count_args = conn.fetchval.call_args[0]
    json_params = [a for a in count_args[1:] if isinstance(a, str) and "svc" in a]
    assert len(json_params) >= 2, (
        f"Expected 2 JSON params for ALL mode, got: {json_params}"
    )
    assert result.total_active == 0


@pytest.mark.asyncio
async def test_handle_scope_services_mode_exact_uses_equality() -> None:
    """EXACT mode: filter as normalized JSONB equality."""
    rows: list[dict[str, Any]] = []
    pool = _make_mock_pool(rows=rows, count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(
        scope_services=["OmniBase_Core"],
        scope_services_mode="EXACT",
        limit=10,
    )

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        result = await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    # EXACT uses = operator (not @> or ?|)
    assert "= $" in count_sql, f"Expected = in SQL for EXACT mode, got: {count_sql}"
    # The JSON param should be the lowercased sorted version
    count_args = conn.fetchval.call_args[0]
    json_params = [
        a for a in count_args[1:] if isinstance(a, str) and "omnibase_core" in a
    ]
    assert len(json_params) >= 1, f"Expected normalized param, got: {count_args}"
    assert result.total_active == 0


@pytest.mark.asyncio
async def test_handle_empty_scope_services_no_filter() -> None:
    """Empty scope_services list should not add scope filter to WHERE."""
    rows: list[dict[str, Any]] = []
    pool = _make_mock_pool(rows=rows, count=5)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(
        scope_services=[],  # Empty = no filter
        scope_services_mode="ANY",
        limit=10,
    )

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        result = await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    # No scope filter in WHERE
    assert "?|" not in count_sql
    assert (
        "@>" not in count_sql or "tags" in count_sql
    )  # @> could be in tags but not scope
    assert result.total_active == 5


@pytest.mark.asyncio
async def test_handle_cursor_adds_pagination_predicate() -> None:
    """Cursor should add (created_at, decision_id) < (ts, id) predicate."""
    rows: list[dict[str, Any]] = []
    pool = _make_mock_pool(rows=rows, count=0)
    handler = HandlerQueryDecisions(pool=pool)

    cursor = encode_cursor(_FIXED_TS, _FIXED_ID)
    payload = ModelPayloadQueryDecisions(cursor=cursor, limit=10)

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    assert "(created_at, decision_id)" in count_sql, (
        f"Expected cursor predicate in SQL, got: {count_sql}"
    )


@pytest.mark.asyncio
async def test_handle_domain_filter_in_where() -> None:
    pool = _make_mock_pool(rows=[], count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(domain="data-model", limit=10)

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    assert "scope_domain" in count_sql


@pytest.mark.asyncio
async def test_handle_decision_type_or_semantics() -> None:
    """decision_type filter uses IN (OR) semantics."""
    pool = _make_mock_pool(rows=[], count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(
        decision_type=["TECH_STACK_CHOICE", "API_DESIGN"],
        limit=10,
    )

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    assert "decision_type IN" in count_sql


@pytest.mark.asyncio
async def test_handle_tags_all_semantics() -> None:
    """tags filter: each tag generates a separate @> condition (ALL required)."""
    pool = _make_mock_pool(rows=[], count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(
        tags=["database", "storage"],
        limit=10,
    )

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    count_sql = conn.fetchval.call_args[0][0]
    # Both tags need @> containment
    assert count_sql.count("tags @>") == 2


@pytest.mark.asyncio
async def test_handle_order_by_created_at_desc() -> None:
    """SELECT must ORDER BY created_at DESC, decision_id DESC."""
    pool = _make_mock_pool(rows=[], count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(limit=10)

    with patch(
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions._row_to_entry",
    ):
        await handler.handle(payload)

    conn = pool.acquire.return_value.__aenter__.return_value
    fetch_sql = conn.fetch.call_args[0][0]
    assert "ORDER BY created_at DESC, decision_id DESC" in fetch_sql


@pytest.mark.asyncio
async def test_handle_malformed_cursor_raises() -> None:
    pool = _make_mock_pool(rows=[], count=0)
    handler = HandlerQueryDecisions(pool=pool)
    payload = ModelPayloadQueryDecisions(cursor="not-valid-base64!!!", limit=10)

    with pytest.raises(ValueError):
        await handler.handle(payload)
