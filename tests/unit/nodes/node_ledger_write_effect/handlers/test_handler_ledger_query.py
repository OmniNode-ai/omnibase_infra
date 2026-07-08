# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLedgerQuery.

Tests validate:
- Pagination boundaries (_normalize_limit edge cases)
- Query builder (_build_time_range_query with filter combinations)
- Protocol compliance via isinstance() check
- Lazy DB connection guard (missing DSN raises RuntimeHostError)
- handle() auto-wiring adapter

Related Tickets:
    - OMN-1686: Add unit tests and minor fixes for NodeLedgerWriteEffect handlers
    - OMN-1647: Add PostgreSQL handlers for event ledger persistence
    - OMN-14140: HandlerDb composed internally from `container` (single-arg
      constructor) instead of accepted as a `db_handler` constructor argument,
      so the auto-wiring resolver can construct this handler.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.handler_db import HandlerDb
from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_query import (
    _DEFAULT_LIMIT,
    _MAX_LIMIT,
    HandlerLedgerQuery,
)
from omnibase_infra.nodes.node_ledger_write_effect.models.model_ledger_query import (
    ModelLedgerQuery,
)

# =============================================================================
# Fixtures
# =============================================================================


def make_mock_container() -> MagicMock:
    """Create a minimal mock ModelONEXContainer."""
    return MagicMock(spec=ModelONEXContainer)


def make_handler(
    initialized: bool = True,
) -> tuple[HandlerLedgerQuery, AsyncMock]:
    """Create a HandlerLedgerQuery with its composed HandlerDb replaced by a mock.

    HandlerLedgerQuery composes HandlerDb internally from `container`
    (OMN-14140) -- construction never takes a db_handler argument. Tests that
    need to control DB behavior replace `_db_handler` post-construction and
    set `_initialized` directly to bypass the real (env-DSN-driven) lazy
    connect in `_ensure_db_ready`.
    """
    container = make_mock_container()
    handler = HandlerLedgerQuery(container)
    db_handler = AsyncMock()
    handler._db_handler = db_handler
    handler._initialized = initialized
    return handler, db_handler


# =============================================================================
# Constructor / Composition Tests (OMN-14140)
# =============================================================================


class TestHandlerLedgerQueryComposition:
    """Tests that HandlerLedgerQuery composes HandlerDb from `container` alone."""

    @pytest.mark.unit
    def test_constructor_takes_only_container(self) -> None:
        """HandlerLedgerQuery(container) constructs without a db_handler arg."""
        container = make_mock_container()

        handler = HandlerLedgerQuery(container)

        assert isinstance(handler._db_handler, HandlerDb)
        assert handler._initialized is False


# =============================================================================
# Initialization / Lazy DB Connection Tests
# =============================================================================


class TestHandlerLedgerQueryInitialization:
    """Tests for HandlerLedgerQuery's lazy HandlerDb connection lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_raises_when_dsn_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """initialize() raises RuntimeHostError when no DSN is supplied.

        The auto-wiring resolver never calls initialize() on constructed
        handlers, so the composed HandlerDb only connects lazily -- and lazily
        connecting with no configured DSN must fail loudly, not silently no-op
        (OMN-14140). Handler-level code does not read environment directly.
        """
        monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
        monkeypatch.delenv("DATABASE_URL", raising=False)
        container = make_mock_container()
        handler = HandlerLedgerQuery(container)

        with pytest.raises(RuntimeHostError, match="Missing PostgreSQL DSN"):
            await handler.initialize({})

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_connects_composed_db_handler_when_dsn_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """initialize() connects the composed HandlerDb using config DSN."""
        container = make_mock_container()
        handler = HandlerLedgerQuery(container)
        handler._db_handler.initialize = AsyncMock()  # type: ignore[method-assign]

        await handler.initialize({"dsn": "postgresql://test-dsn"})

        handler._db_handler.initialize.assert_awaited_once_with(
            {"dsn": "postgresql://test-dsn"}
        )
        assert handler._initialized is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_ensure_db_ready_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repeated _ensure_db_ready calls connect at most once."""
        container = make_mock_container()
        handler = HandlerLedgerQuery(container, db_dsn="postgresql://test-dsn")
        handler._db_handler.initialize = AsyncMock()  # type: ignore[method-assign]

        await handler._ensure_db_ready()
        await handler._ensure_db_ready()

        handler._db_handler.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_shutdown_sets_initialized_false(self) -> None:
        """shutdown() marks the handler as not initialized and shuts down HandlerDb."""
        handler, db_handler = make_handler(initialized=True)
        assert handler._initialized is True

        await handler.shutdown()

        assert handler._initialized is False
        db_handler.shutdown.assert_awaited_once()


# =============================================================================
# Pagination Boundary Tests
# =============================================================================


class TestHandlerLedgerQueryNormalizeLimit:
    """Tests for _normalize_limit with edge cases."""

    @pytest.mark.unit
    def test_normalize_limit_zero_returns_default(self) -> None:
        """limit=0 is treated as 'not specified', returns _DEFAULT_LIMIT."""
        handler, _db_handler = make_handler()

        result = handler._normalize_limit(0)

        assert result == _DEFAULT_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_negative_returns_default(self) -> None:
        """Negative limit returns _DEFAULT_LIMIT."""
        handler, _db_handler = make_handler()

        assert handler._normalize_limit(-1) == _DEFAULT_LIMIT
        assert handler._normalize_limit(-100) == _DEFAULT_LIMIT
        assert handler._normalize_limit(-99999) == _DEFAULT_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_exceeds_max_returns_max(self) -> None:
        """limit > _MAX_LIMIT is clamped to _MAX_LIMIT."""
        handler, _db_handler = make_handler()

        assert handler._normalize_limit(_MAX_LIMIT + 1) == _MAX_LIMIT
        assert handler._normalize_limit(99999999) == _MAX_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_at_max_returns_max(self) -> None:
        """limit == _MAX_LIMIT is returned as-is."""
        handler, _db_handler = make_handler()

        result = handler._normalize_limit(_MAX_LIMIT)

        assert result == _MAX_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_at_one_returns_one(self) -> None:
        """limit=1 (minimum valid) is returned as-is."""
        handler, _db_handler = make_handler()

        result = handler._normalize_limit(1)

        assert result == 1

    @pytest.mark.unit
    def test_normalize_limit_normal_value_returned_unchanged(self) -> None:
        """Normal limit values within range are returned unchanged."""
        handler, _db_handler = make_handler()

        assert handler._normalize_limit(50) == 50
        assert handler._normalize_limit(100) == 100
        assert handler._normalize_limit(500) == 500
        assert handler._normalize_limit(5000) == 5000


# =============================================================================
# Query Builder Tests
# =============================================================================


class TestHandlerLedgerQueryBuilder:
    """Tests for _build_time_range_query with various filter combinations."""

    _START = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    _END = datetime(2026, 1, 31, 23, 59, 59, tzinfo=UTC)

    def _make_query(self, **kwargs: object) -> ModelLedgerQuery:
        """Create a ModelLedgerQuery with start/end time and optional kwargs."""
        defaults: dict[str, object] = {
            "start_time": self._START,
            "end_time": self._END,
            "limit": 100,
            "offset": 0,
        }
        defaults.update(kwargs)
        return ModelLedgerQuery(**defaults)

    @pytest.mark.unit
    def test_no_optional_filters(self) -> None:
        """Base query has no additional WHERE clauses."""
        handler, _db_handler = make_handler()
        query = self._make_query()

        sql, _count_sql, params = handler._build_time_range_query(query)

        # No AND filters - just the base time range
        assert "AND event_type" not in sql
        assert "AND topic" not in sql
        # Parameters: [start, end, limit, offset]
        assert params[0] == self._START
        assert params[1] == self._END
        assert 100 in params  # limit
        assert 0 in params  # offset

    @pytest.mark.unit
    def test_event_type_filter_appended(self) -> None:
        """event_type filter is appended to WHERE clause."""
        handler, _db_handler = make_handler()
        query = self._make_query(event_type="user.created")

        sql, _count_sql, params = handler._build_time_range_query(query)

        assert "AND event_type = $3" in sql
        assert "user.created" in params

    @pytest.mark.unit
    def test_topic_filter_appended(self) -> None:
        """topic filter is appended to WHERE clause."""
        handler, _db_handler = make_handler()
        query = self._make_query(topic="prod.orders.v1")

        sql, _count_sql, params = handler._build_time_range_query(query)

        assert "AND topic = $3" in sql
        assert "prod.orders.v1" in params

    @pytest.mark.unit
    def test_both_filters_appended(self) -> None:
        """Both event_type and topic filters use sequential parameter indices."""
        handler, _db_handler = make_handler()
        query = self._make_query(event_type="order.created", topic="orders.v2")

        sql, _count_sql, params = handler._build_time_range_query(query)

        assert "AND event_type = $3" in sql
        assert "AND topic = $4" in sql
        assert "order.created" in params
        assert "orders.v2" in params

    @pytest.mark.unit
    def test_count_only_excludes_limit_offset(self) -> None:
        """count_only=True does not add limit/offset to parameters."""
        handler, _db_handler = make_handler()
        query = self._make_query(limit=50, offset=200)

        _, _count_sql, params = handler._build_time_range_query(query, count_only=True)

        # Limit (50) and offset (200) should NOT be in params for count_only
        assert 50 not in params
        assert 200 not in params
        # Only start and end
        assert len(params) == 2

    @pytest.mark.unit
    def test_count_only_with_filters_excludes_limit_offset(self) -> None:
        """count_only=True with filters excludes limit/offset from params."""
        handler, _db_handler = make_handler()
        query = self._make_query(event_type="test.event", limit=50, offset=100)

        _, _count_sql, params = handler._build_time_range_query(query, count_only=True)

        assert 50 not in params
        assert 100 not in params
        # start, end, event_type
        assert len(params) == 3
        assert "test.event" in params

    @pytest.mark.unit
    def test_count_sql_excludes_order_by_and_pagination(self) -> None:
        """Count SQL doesn't include ORDER BY, LIMIT, or OFFSET."""
        handler, _db_handler = make_handler()
        query = self._make_query()

        _, count_sql, _ = handler._build_time_range_query(query)

        assert "ORDER BY" not in count_sql
        assert "LIMIT" not in count_sql
        assert "OFFSET" not in count_sql

    @pytest.mark.unit
    def test_query_sql_includes_order_by_and_pagination(self) -> None:
        """Query SQL includes ORDER BY, LIMIT, and OFFSET."""
        handler, _db_handler = make_handler()
        query = self._make_query(limit=25, offset=50)

        query_sql, _, _ = handler._build_time_range_query(query)

        assert "ORDER BY" in query_sql
        assert "LIMIT" in query_sql
        assert "OFFSET" in query_sql

    @pytest.mark.unit
    def test_parameter_order_no_filters(self) -> None:
        """Parameters without filters: [start, end, limit, offset]."""
        handler, _db_handler = make_handler()
        query = self._make_query(limit=25, offset=10)

        _, _, params = handler._build_time_range_query(query)

        assert params[0] == self._START
        assert params[1] == self._END
        assert params[2] == 25  # limit
        assert params[3] == 10  # offset

    @pytest.mark.unit
    def test_parameter_order_with_both_filters(self) -> None:
        """Parameters with both filters: [start, end, event_type, topic, limit, offset]."""
        handler, _db_handler = make_handler()
        query = self._make_query(
            event_type="node.registered",
            topic="infra.nodes.v1",
            limit=50,
            offset=100,
        )

        _, _, params = handler._build_time_range_query(query)

        assert params[0] == self._START
        assert params[1] == self._END
        assert params[2] == "node.registered"
        assert params[3] == "infra.nodes.v1"
        assert params[4] == 50  # limit
        assert params[5] == 100  # offset


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestHandlerLedgerQueryProtocolCompliance:
    """Tests for ProtocolLedgerPersistence partial compliance.

    HandlerLedgerQuery implements the query methods of ProtocolLedgerPersistence.
    HandlerLedgerAppend implements append(). Together they form a full implementation.
    The isinstance() check requires all protocol methods on a single object, which
    does not apply to this split handler design.
    """

    @pytest.mark.unit
    def test_handler_has_query_methods_matching_protocol(self) -> None:
        """HandlerLedgerQuery implements query_by_correlation_id and query_by_time_range from the protocol."""
        import inspect

        handler, _db_handler = make_handler()

        # HandlerLedgerQuery implements the query slices of the protocol
        assert hasattr(handler, "query_by_correlation_id")
        assert inspect.iscoroutinefunction(handler.query_by_correlation_id)

        assert hasattr(handler, "query_by_time_range")
        assert inspect.iscoroutinefunction(handler.query_by_time_range)

    @pytest.mark.unit
    def test_handler_does_not_implement_append(self) -> None:
        """HandlerLedgerQuery correctly does not implement append() - HandlerLedgerAppend owns that."""
        handler, _db_handler = make_handler()

        # append() belongs to HandlerLedgerAppend
        assert not hasattr(handler, "append")

    @pytest.mark.unit
    def test_handler_type_is_infra_handler(self) -> None:
        """handler_type returns INFRA_HANDLER."""
        from omnibase_infra.enums import EnumHandlerType

        handler, _db_handler = make_handler()

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    @pytest.mark.unit
    def test_handler_category_is_effect(self) -> None:
        """handler_category returns EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        handler, _db_handler = make_handler()

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


# =============================================================================
# handle() Auto-Wiring Adapter Tests
# =============================================================================


class TestHandlerLedgerQueryHandle:
    """Tests for the handle() auto-wiring entry point.

    node_ledger_write_effect's contract declares operation_match routing with
    no event_model, so handler_wiring._make_dispatch_callback dispatches via
    handle(envelope) directly rather than execute(). Without handle(), the
    auto-wiring binds _missing_handle and every dispatched ledger-query
    command raises -- the same gap HandlerLedgerAppend.handle() closed for
    the append side (OMN-14134).
    """

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_routes_to_query_and_returns_handler_output(self) -> None:
        """handle() extracts the payload, calls query(), and wraps the result."""
        handler, db_handler = make_handler()
        db_handler.execute = AsyncMock(return_value=MagicMock(result=None))

        correlation_id = uuid4()
        query = ModelLedgerQuery(correlation_id=correlation_id, limit=10, offset=0)
        envelope = MagicMock()
        envelope.payload = query
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert isinstance(output, ModelHandlerOutput)
        assert output.handler_id == "ledger-query-handler"
        assert output.result is not None
        assert output.result.entries == []
        assert output.result.total_count == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_coerces_dict_payload(self) -> None:
        """handle() validates a raw dict payload into ModelLedgerQuery."""
        handler, db_handler = make_handler()
        db_handler.execute = AsyncMock(return_value=MagicMock(result=None))

        envelope = MagicMock()
        envelope.payload = {
            "correlation_id": str(uuid4()),
            "limit": 10,
            "offset": 0,
        }
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.total_count == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_propagates_envelope_correlation_id(self) -> None:
        """handle() copies correlation_id from the envelope onto the output."""
        handler, db_handler = make_handler()
        db_handler.execute = AsyncMock(return_value=MagicMock(result=None))

        correlation_id = uuid4()
        envelope = MagicMock()
        envelope.payload = ModelLedgerQuery(correlation_id=uuid4(), limit=10, offset=0)
        envelope.correlation_id = correlation_id

        output = await handler.handle(envelope)

        assert output.correlation_id == correlation_id

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_falls_back_to_fresh_uuid_when_no_correlation_id(
        self,
    ) -> None:
        """handle() never leaves correlation_id unset — it defaults to a fresh UUID."""
        handler, db_handler = make_handler()
        db_handler.execute = AsyncMock(return_value=MagicMock(result=None))

        envelope = MagicMock()
        envelope.payload = ModelLedgerQuery(
            correlation_id=None,
            start_time=datetime(2026, 1, 1, tzinfo=UTC),
            end_time=datetime(2026, 1, 2, tzinfo=UTC),
            limit=10,
            offset=0,
        )
        envelope.correlation_id = None

        output = await handler.handle(envelope)

        assert output.correlation_id is not None

    @pytest.mark.unit
    def test_safe_correlation_id_parses_string_uuid(self) -> None:
        """_safe_correlation_id parses a string UUID into a UUID instance."""
        correlation_id = uuid4()
        assert (
            HandlerLedgerQuery._safe_correlation_id(str(correlation_id))
            == correlation_id
        )

    @pytest.mark.unit
    def test_safe_correlation_id_falls_back_on_garbage(self) -> None:
        """_safe_correlation_id never raises — bad input yields a fresh UUID."""
        result = HandlerLedgerQuery._safe_correlation_id("not-a-uuid")
        assert isinstance(result, UUID)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handle_accepts_dict_shaped_envelope(self) -> None:
        """handle() extracts payload/correlation_id from a dict envelope.

        This is the shape MessageDispatchEngine._materialize_envelope_with_bindings
        actually delivers on the real dispatch path: a dict with a "payload" key,
        not an attribute-bearing object. A MagicMock-only test suite would miss
        this — getattr(dict, "payload", default) silently falls through to the
        default instead of doing a key lookup.
        """
        handler, db_handler = make_handler()
        db_handler.execute = AsyncMock(return_value=MagicMock(result=None))

        correlation_id = uuid4()
        envelope = {
            "payload": {
                "correlation_id": str(uuid4()),
                "limit": 10,
                "offset": 0,
            },
            "correlation_id": str(correlation_id),
            "partition_key": None,
        }

        output = await handler.handle(envelope)

        assert output.result is not None
        assert output.result.total_count == 0
        assert output.correlation_id == correlation_id

    @pytest.mark.unit
    def test_extract_envelope_field_reads_dict_key(self) -> None:
        """_extract_envelope_field does a key lookup for dict envelopes."""
        envelope = {"payload": {"limit": 10}, "correlation_id": "abc"}

        assert HandlerLedgerQuery._extract_envelope_field(envelope, "payload") == {
            "limit": 10
        }
        assert (
            HandlerLedgerQuery._extract_envelope_field(envelope, "correlation_id")
            == "abc"
        )
        assert HandlerLedgerQuery._extract_envelope_field(envelope, "missing") is None

    @pytest.mark.unit
    def test_extract_envelope_field_reads_object_attribute(self) -> None:
        """_extract_envelope_field falls back to attribute access for non-dict envelopes."""
        envelope = MagicMock()
        envelope.payload = {"limit": 10}

        assert HandlerLedgerQuery._extract_envelope_field(envelope, "payload") == {
            "limit": 10
        }


__all__ = [
    "TestHandlerLedgerQueryComposition",
    "TestHandlerLedgerQueryInitialization",
    "TestHandlerLedgerQueryNormalizeLimit",
    "TestHandlerLedgerQueryBuilder",
    "TestHandlerLedgerQueryProtocolCompliance",
    "TestHandlerLedgerQueryHandle",
]
