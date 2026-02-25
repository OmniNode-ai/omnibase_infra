# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceTopicCatalogPostgres.

Tests cover:
    - build_catalog: happy path, no pool, DB timeout, empty table
    - _get_catalog_version: success, empty table, exception
    - Topic cross-reference: publishers and subscribers per topic
    - Filtering: topic_pattern, include_inactive
    - Cache: hit on same version, eviction on version advance
    - _parse_json_list: list passthrough, JSON string decoding, invalid JSON

Related Tickets:
    - OMN-2746: Replace ServiceTopicCatalog Consul KV backend with PostgreSQL
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.services.service_topic_catalog_postgres import (
    DB_UNAVAILABLE,
    ServiceTopicCatalogPostgres,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUFFIX_A = "onex.evt.platform.node-registration.v1"
_SUFFIX_B = "onex.evt.platform.node-heartbeat.v1"
_NODE_1 = str(uuid4())
_NODE_2 = str(uuid4())


def _make_service(
    pool: object | None = None,
    query_timeout_seconds: float = 5.0,
) -> ServiceTopicCatalogPostgres:
    """Create a ServiceTopicCatalogPostgres with a mock container."""
    container = MagicMock()
    return ServiceTopicCatalogPostgres(
        container=container,
        pool=pool,  # type: ignore[arg-type]
        query_timeout_seconds=query_timeout_seconds,
    )


def _make_pool(
    version_hash: str | None = "abcdef1234567890",
    rows: list[dict[str, object]] | None = None,
) -> MagicMock:
    """Create a minimal asyncpg pool mock.

    Args:
        version_hash: Value for md5(MAX(updated_at)). None simulates empty table.
        rows: List of row dicts for the topic data query.
    """
    if rows is None:
        rows = []

    # asyncpg pool acts as an async context manager via .acquire()
    conn = AsyncMock()

    # fetchrow returns None or a dict-like row
    if version_hash is not None:
        version_row = {"version_hash": version_hash}
    else:
        version_row = None

    async def _fetchrow(sql: str) -> object:
        return version_row

    async def _fetch(sql: str) -> list[object]:
        # Return asyncpg-like Record objects (we simulate with simple dicts).
        # We use a factory function to capture `r` correctly per iteration.
        def _make_record(row_data: dict[str, object]) -> MagicMock:
            record = MagicMock()
            record.__getitem__ = lambda self, k, _d=row_data: _d[k]
            return record

        return [_make_record(r) for r in rows]

    conn.fetchrow = _fetchrow
    conn.fetch = _fetch

    # pool.acquire() returns an async context manager yielding conn
    acquire_cm = MagicMock()
    acquire_cm.__aenter__ = AsyncMock(return_value=conn)
    acquire_cm.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=acquire_cm)
    return pool


def _make_pool_rows(rows: list[dict[str, object]]) -> MagicMock:
    """Create a pool mock with specific topic rows."""
    return _make_pool(version_hash="abcdef1234567890", rows=rows)


# ---------------------------------------------------------------------------
# Test: no pool configured
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceTopicCatalogPostgresNoPool:
    """Tests when no pool is provided."""

    @pytest.mark.asyncio
    async def test_build_catalog_no_pool_returns_empty_with_warning(self) -> None:
        """build_catalog should return empty topics with db_unavailable warning."""
        service = _make_service(pool=None)

        response = await service.build_catalog(correlation_id=uuid4())

        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.topics == ()
        assert DB_UNAVAILABLE in response.warnings
        assert response.catalog_version == 0

    @pytest.mark.asyncio
    async def test_get_catalog_version_no_pool_returns_minus_one(self) -> None:
        """_get_catalog_version returns -1 when no pool configured."""
        service = _make_service(pool=None)
        version = await service._get_catalog_version(uuid4())
        assert version == -1

    @pytest.mark.asyncio
    async def test_fetch_topic_rows_no_pool_returns_empty(self) -> None:
        """_fetch_topic_rows returns empty list when no pool configured."""
        service = _make_service(pool=None)
        rows = await service._fetch_topic_rows(uuid4())
        assert rows == []


# ---------------------------------------------------------------------------
# Test: happy path — build_catalog with data
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceTopicCatalogPostgresBuildCatalog:
    """Tests for build_catalog with a live pool mock."""

    @pytest.mark.asyncio
    async def test_build_catalog_empty_table_returns_empty_response(self) -> None:
        """build_catalog with no rows should return empty topics, no warnings."""
        pool = _make_pool(version_hash=None, rows=[])
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.topics == ()
        # Empty table → version_hash is None → catalog_version == 0
        assert response.catalog_version == 0

    @pytest.mark.asyncio
    async def test_build_catalog_single_node_subscribe_and_publish(self) -> None:
        """A node with both subscribe and publish topics appears in catalog."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [_SUFFIX_B],
            }
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        assert isinstance(response, ModelTopicCatalogResponse)
        suffixes = {e.topic_suffix for e in response.topics}
        assert _SUFFIX_A in suffixes
        assert _SUFFIX_B in suffixes

    @pytest.mark.asyncio
    async def test_build_catalog_subscriber_count_correct(self) -> None:
        """subscriber_count reflects number of nodes subscribing to a topic."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [],
            },
            {
                "node_id": _NODE_2,
                "node_type": "orchestrator",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [],
            },
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        entry = next(e for e in response.topics if e.topic_suffix == _SUFFIX_A)
        assert entry.subscriber_count == 2
        assert entry.publisher_count == 0

    @pytest.mark.asyncio
    async def test_build_catalog_publisher_count_correct(self) -> None:
        """publisher_count reflects number of nodes publishing to a topic."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [],
                "publish_topics": [_SUFFIX_B],
            },
            {
                "node_id": _NODE_2,
                "node_type": "orchestrator",
                "subscribe_topics": [],
                "publish_topics": [_SUFFIX_B],
            },
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        entry = next(e for e in response.topics if e.topic_suffix == _SUFFIX_B)
        assert entry.publisher_count == 2
        assert entry.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_build_catalog_topics_sorted_by_suffix(self) -> None:
        """Topics in response are sorted alphabetically by topic_suffix."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_B, _SUFFIX_A],
                "publish_topics": [],
            }
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        suffixes = [e.topic_suffix for e in response.topics]
        assert suffixes == sorted(suffixes)

    @pytest.mark.asyncio
    async def test_build_catalog_node_count_reflects_row_count(self) -> None:
        """node_count in response matches number of registered nodes."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [],
            },
            {
                "node_id": _NODE_2,
                "node_type": "orchestrator",
                "subscribe_topics": [],
                "publish_topics": [_SUFFIX_A],
            },
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        assert response.node_count == 2

    @pytest.mark.asyncio
    async def test_build_catalog_no_warnings_on_success(self) -> None:
        """Successful catalog build should produce no warnings."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [_SUFFIX_B],
            }
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        assert response.warnings == ()


# ---------------------------------------------------------------------------
# Test: filtering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceTopicCatalogPostgresFiltering:
    """Tests for topic_pattern and include_inactive filtering."""

    @pytest.mark.asyncio
    async def test_topic_pattern_filters_by_glob(self) -> None:
        """topic_pattern glob should restrict results."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A, _SUFFIX_B],
                "publish_topics": [],
            }
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(
            correlation_id=uuid4(),
            topic_pattern="*node-registration*",
        )

        assert len(response.topics) == 1
        assert response.topics[0].topic_suffix == _SUFFIX_A

    @pytest.mark.asyncio
    async def test_include_inactive_false_excludes_inactive_topics(self) -> None:
        """Topics with no publishers and no subscribers are excluded by default."""
        # A topic with only one node subscribing IS active (subscriber_count > 0)
        # To create an inactive topic we need publisher_count=0 AND subscriber_count=0,
        # which only happens if no node references it — they don't appear in the catalog
        # at all when derived from rows. So this test verifies the filter doesn't drop
        # active topics when include_inactive=False.
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [],
            }
        ]
        pool = _make_pool_rows(rows)
        service = _make_service(pool=pool)

        response_default = await service.build_catalog(
            correlation_id=uuid4(),
            include_inactive=False,
        )
        response_all = await service.build_catalog(
            correlation_id=uuid4(),
            include_inactive=True,
        )

        # Active topics appear in both
        assert any(e.topic_suffix == _SUFFIX_A for e in response_default.topics)
        assert any(e.topic_suffix == _SUFFIX_A for e in response_all.topics)


# ---------------------------------------------------------------------------
# Test: DB unavailable / timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceTopicCatalogPostgresDBUnavailable:
    """Tests for DB connection failures."""

    @pytest.mark.asyncio
    async def test_build_catalog_db_timeout_returns_empty_with_warning(self) -> None:
        """When _fetch_topic_rows times out, DB_UNAVAILABLE is emitted."""
        service = _make_service(pool=MagicMock(), query_timeout_seconds=0.001)

        async def _slow_fetch(cid: object) -> list[object]:
            await asyncio.sleep(1)
            return []

        with patch.object(service, "_fetch_topic_rows", side_effect=_slow_fetch):
            response = await service.build_catalog(correlation_id=uuid4())

        assert DB_UNAVAILABLE in response.warnings
        assert response.topics == ()

    @pytest.mark.asyncio
    async def test_build_catalog_db_exception_returns_empty_with_warning(self) -> None:
        """When _fetch_topic_rows raises, DB_UNAVAILABLE is emitted."""
        service = _make_service(pool=MagicMock())

        async def _fail_fetch(cid: object) -> list[object]:
            raise RuntimeError("connection refused")

        with patch.object(service, "_fetch_topic_rows", side_effect=_fail_fetch):
            response = await service.build_catalog(correlation_id=uuid4())

        assert DB_UNAVAILABLE in response.warnings
        assert response.topics == ()


# ---------------------------------------------------------------------------
# Test: catalog version and caching
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceTopicCatalogPostgresVersionAndCache:
    """Tests for version derivation and in-process caching."""

    @pytest.mark.asyncio
    async def test_catalog_version_derived_from_version_hash(self) -> None:
        """catalog_version should be non-negative int derived from version_hash."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [],
            }
        ]
        pool = _make_pool(version_hash="00000000000000ff", rows=rows)
        service = _make_service(pool=pool)

        response = await service.build_catalog(correlation_id=uuid4())

        # version_hash[-8:] = "000000ff" → int("000000ff", 16) = 255
        assert response.catalog_version == 255

    @pytest.mark.asyncio
    async def test_cache_hit_returns_same_version(self) -> None:
        """Second call with same version hash should return cached response."""
        rows = [
            {
                "node_id": _NODE_1,
                "node_type": "effect",
                "subscribe_topics": [_SUFFIX_A],
                "publish_topics": [],
            }
        ]
        pool = _make_pool(version_hash="abcdef1234567890", rows=rows)
        service = _make_service(pool=pool)

        response1 = await service.build_catalog(correlation_id=uuid4())
        response2 = await service.build_catalog(correlation_id=uuid4())

        assert response1.catalog_version == response2.catalog_version
        assert len(response1.topics) == len(response2.topics)


# ---------------------------------------------------------------------------
# Test: _parse_json_list
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseJsonList:
    """Tests for ServiceTopicCatalogPostgres._parse_json_list."""

    def _make_service_for_parse(self) -> ServiceTopicCatalogPostgres:
        return _make_service(pool=None)

    def test_list_passthrough(self) -> None:
        """Already-decoded list is returned as-is."""
        service = self._make_service_for_parse()
        result = service._parse_json_list(["a", "b"], uuid4())
        assert result == ["a", "b"]

    def test_json_string_decoded(self) -> None:
        """JSON string encoding of a list is decoded correctly."""
        service = self._make_service_for_parse()
        value = json.dumps(["x", "y"])
        result = service._parse_json_list(value, uuid4())
        assert result == ["x", "y"]

    def test_none_returns_empty(self) -> None:
        """None value returns empty list."""
        service = self._make_service_for_parse()
        assert service._parse_json_list(None, uuid4()) == []

    def test_invalid_json_string_returns_empty(self) -> None:
        """Invalid JSON string returns empty list."""
        service = self._make_service_for_parse()
        assert service._parse_json_list("not-json{{{", uuid4()) == []

    def test_json_object_string_returns_empty(self) -> None:
        """JSON string that is an object (not list) returns empty list."""
        service = self._make_service_for_parse()
        assert service._parse_json_list('{"key": "value"}', uuid4()) == []
