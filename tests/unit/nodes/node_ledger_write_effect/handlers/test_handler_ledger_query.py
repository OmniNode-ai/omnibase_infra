# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for HandlerLedgerQuery.

Tests validate:
- Pagination boundaries (_normalize_limit edge cases)
- Query builder (_build_time_range_query with filter combinations)
- Protocol compliance via isinstance() check
- Initialization guard
- _ensure_initialized raises when not ready

Related Tickets:
    - OMN-1686: Add unit tests and minor fixes for NodeLedgerWriteEffect handlers
    - OMN-1647: Add PostgreSQL handlers for event ledger persistence
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_ledger_write_effect.handlers.handler_ledger_query import (
    _DEFAULT_LIMIT,
    _MAX_LIMIT,
    HandlerLedgerQuery,
)
from omnibase_infra.nodes.node_ledger_write_effect.models.model_ledger_query import (
    ModelLedgerQuery,
)
from omnibase_infra.nodes.node_ledger_write_effect.protocols.protocol_ledger_persistence import (
    ProtocolLedgerPersistence,
)

# =============================================================================
# Fixtures
# =============================================================================


def make_mock_container() -> MagicMock:
    """Create a minimal mock ModelONEXContainer."""
    return MagicMock(spec=ModelONEXContainer)


def make_mock_db_handler(initialized: bool = True) -> MagicMock:
    """Create a mock HandlerDb with _initialized attribute."""
    mock = MagicMock()
    mock._initialized = initialized
    return mock


def make_handler(initialized: bool = True) -> HandlerLedgerQuery:
    """Create a HandlerLedgerQuery with a mock db handler."""
    container = make_mock_container()
    db_handler = make_mock_db_handler(initialized=initialized)
    return HandlerLedgerQuery(container, db_handler)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestHandlerLedgerQueryInitialization:
    """Tests for HandlerLedgerQuery initialization lifecycle."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_succeeds_when_db_handler_ready(self) -> None:
        """initialize() completes when HandlerDb._initialized is True."""
        handler = make_handler()
        await handler.initialize({})

        assert handler._initialized is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_initialize_raises_when_db_not_initialized(self) -> None:
        """initialize() raises RuntimeHostError if HandlerDb is not yet initialized."""
        handler = make_handler(initialized=False)

        with pytest.raises(RuntimeHostError, match="HandlerDb must be initialized"):
            await handler.initialize({})

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_shutdown_sets_initialized_false(self) -> None:
        """shutdown() marks handler as not initialized."""
        handler = make_handler()
        await handler.initialize({})
        assert handler._initialized is True

        await handler.shutdown()
        assert handler._initialized is False

    @pytest.mark.unit
    def test_ensure_initialized_raises_when_not_initialized(self) -> None:
        """_ensure_initialized raises RuntimeHostError when not initialized."""
        handler = make_handler()
        # Do NOT call initialize()

        with pytest.raises(RuntimeHostError, match="not initialized"):
            handler._ensure_initialized("test.op")


# =============================================================================
# Pagination Boundary Tests
# =============================================================================


class TestHandlerLedgerQueryNormalizeLimit:
    """Tests for _normalize_limit with edge cases."""

    @pytest.mark.unit
    def test_normalize_limit_zero_returns_default(self) -> None:
        """limit=0 is treated as 'not specified', returns _DEFAULT_LIMIT."""
        handler = make_handler()

        result = handler._normalize_limit(0)

        assert result == _DEFAULT_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_negative_returns_default(self) -> None:
        """Negative limit returns _DEFAULT_LIMIT."""
        handler = make_handler()

        assert handler._normalize_limit(-1) == _DEFAULT_LIMIT
        assert handler._normalize_limit(-100) == _DEFAULT_LIMIT
        assert handler._normalize_limit(-99999) == _DEFAULT_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_exceeds_max_returns_max(self) -> None:
        """limit > _MAX_LIMIT is clamped to _MAX_LIMIT."""
        handler = make_handler()

        assert handler._normalize_limit(_MAX_LIMIT + 1) == _MAX_LIMIT
        assert handler._normalize_limit(99999999) == _MAX_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_at_max_returns_max(self) -> None:
        """limit == _MAX_LIMIT is returned as-is."""
        handler = make_handler()

        result = handler._normalize_limit(_MAX_LIMIT)

        assert result == _MAX_LIMIT

    @pytest.mark.unit
    def test_normalize_limit_at_one_returns_one(self) -> None:
        """limit=1 (minimum valid) is returned as-is."""
        handler = make_handler()

        result = handler._normalize_limit(1)

        assert result == 1

    @pytest.mark.unit
    def test_normalize_limit_normal_value_returned_unchanged(self) -> None:
        """Normal limit values within range are returned unchanged."""
        handler = make_handler()

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
        handler = make_handler()
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
        handler = make_handler()
        query = self._make_query(event_type="user.created")

        sql, _count_sql, params = handler._build_time_range_query(query)

        assert "AND event_type = $3" in sql
        assert "user.created" in params

    @pytest.mark.unit
    def test_topic_filter_appended(self) -> None:
        """topic filter is appended to WHERE clause."""
        handler = make_handler()
        query = self._make_query(topic="prod.orders.v1")

        sql, _count_sql, params = handler._build_time_range_query(query)

        assert "AND topic = $3" in sql
        assert "prod.orders.v1" in params

    @pytest.mark.unit
    def test_both_filters_appended(self) -> None:
        """Both event_type and topic filters use sequential parameter indices."""
        handler = make_handler()
        query = self._make_query(event_type="order.created", topic="orders.v2")

        sql, _count_sql, params = handler._build_time_range_query(query)

        assert "AND event_type = $3" in sql
        assert "AND topic = $4" in sql
        assert "order.created" in params
        assert "orders.v2" in params

    @pytest.mark.unit
    def test_count_only_excludes_limit_offset(self) -> None:
        """count_only=True does not add limit/offset to parameters."""
        handler = make_handler()
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
        handler = make_handler()
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
        handler = make_handler()
        query = self._make_query()

        _, count_sql, _ = handler._build_time_range_query(query)

        assert "ORDER BY" not in count_sql
        assert "LIMIT" not in count_sql
        assert "OFFSET" not in count_sql

    @pytest.mark.unit
    def test_query_sql_includes_order_by_and_pagination(self) -> None:
        """Query SQL includes ORDER BY, LIMIT, and OFFSET."""
        handler = make_handler()
        query = self._make_query(limit=25, offset=50)

        query_sql, _, _ = handler._build_time_range_query(query)

        assert "ORDER BY" in query_sql
        assert "LIMIT" in query_sql
        assert "OFFSET" in query_sql

    @pytest.mark.unit
    def test_parameter_order_no_filters(self) -> None:
        """Parameters without filters: [start, end, limit, offset]."""
        handler = make_handler()
        query = self._make_query(limit=25, offset=10)

        _, _, params = handler._build_time_range_query(query)

        assert params[0] == self._START
        assert params[1] == self._END
        assert params[2] == 25  # limit
        assert params[3] == 10  # offset

    @pytest.mark.unit
    def test_parameter_order_with_both_filters(self) -> None:
        """Parameters with both filters: [start, end, event_type, topic, limit, offset]."""
        handler = make_handler()
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

        handler = make_handler()

        # HandlerLedgerQuery implements the query slices of the protocol
        assert hasattr(handler, "query_by_correlation_id")
        assert inspect.iscoroutinefunction(handler.query_by_correlation_id)

        assert hasattr(handler, "query_by_time_range")
        assert inspect.iscoroutinefunction(handler.query_by_time_range)

    @pytest.mark.unit
    def test_handler_does_not_implement_append(self) -> None:
        """HandlerLedgerQuery correctly does not implement append() - HandlerLedgerAppend owns that."""
        handler = make_handler()

        # append() belongs to HandlerLedgerAppend
        assert not hasattr(handler, "append")

    @pytest.mark.unit
    def test_handler_type_is_infra_handler(self) -> None:
        """handler_type returns INFRA_HANDLER."""
        from omnibase_infra.enums import EnumHandlerType

        handler = make_handler()

        assert handler.handler_type == EnumHandlerType.INFRA_HANDLER

    @pytest.mark.unit
    def test_handler_category_is_effect(self) -> None:
        """handler_category returns EFFECT."""
        from omnibase_infra.enums import EnumHandlerTypeCategory

        handler = make_handler()

        assert handler.handler_category == EnumHandlerTypeCategory.EFFECT


__all__ = [
    "TestHandlerLedgerQueryInitialization",
    "TestHandlerLedgerQueryNormalizeLimit",
    "TestHandlerLedgerQueryBuilder",
    "TestHandlerLedgerQueryProtocolCompliance",
]
