# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Unit tests for ServiceTopicCatalog.

Tests cover:
    - build_catalog: happy path, timeout, cache hit/miss
    - get_catalog_version: present, absent, invalid
    - increment_version: success, CAS failure, retry exhaustion
    - KV precedence: node arrays authoritative over reverse index
    - Filtering: topic_pattern, include_inactive
    - Partial success: warnings on Consul errors
    - Warning codes: all 5 OMN-2312 warning codes triggered by their paths

Related Tickets:
    - OMN-2311: Topic Catalog: ServiceTopicCatalog + KV precedence + caching
    - OMN-2312: Topic Catalog: response warnings channel
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.models.catalog.catalog_warning_codes import (
    CONSUL_KV_MAX_KEYS_REACHED,
    CONSUL_SCAN_TIMEOUT,
    CONSUL_UNAVAILABLE,
    PARTIAL_NODE_DATA,
    VERSION_UNKNOWN,
)
from omnibase_infra.models.catalog.model_topic_catalog_response import (
    ModelTopicCatalogResponse,
)
from omnibase_infra.services.service_topic_catalog import ServiceTopicCatalog
from omnibase_infra.topics.topic_resolver import TopicResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_SUFFIX = "onex.evt.platform.node-registration.v1"
_VALID_SUFFIX_2 = "onex.evt.platform.node-heartbeat.v1"


def _make_service(
    consul_handler: object | None = None,
    topic_resolver: TopicResolver | None = None,
) -> ServiceTopicCatalog:
    """Create a ServiceTopicCatalog with a mock container."""
    container = MagicMock()
    return ServiceTopicCatalog(
        container=container,
        consul_handler=consul_handler,  # type: ignore[arg-type]
        topic_resolver=topic_resolver,
    )


def _make_consul_handler(
    kv_get_raw_side_effect: object = None,
    kv_get_raw_return: str | None = None,
) -> MagicMock:
    """Create a minimal HandlerConsul mock."""
    handler = MagicMock()
    handler._client = MagicMock()
    handler._executor = None  # use default executor
    if kv_get_raw_side_effect is not None:
        handler.kv_get_raw = AsyncMock(side_effect=kv_get_raw_side_effect)
    else:
        handler.kv_get_raw = AsyncMock(return_value=kv_get_raw_return)
    return handler


# ---------------------------------------------------------------------------
# Test: no Consul handler configured
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestServiceTopicCatalogNoHandler:
    """Tests when no consul_handler is provided."""

    @pytest.mark.asyncio
    async def test_build_catalog_no_handler_returns_empty_with_warning(self) -> None:
        """build_catalog should return empty topics with consul_unavailable warning."""
        service = _make_service(consul_handler=None)
        correlation_id = uuid4()

        response = await service.build_catalog(
            correlation_id=correlation_id,
        )

        assert isinstance(response, ModelTopicCatalogResponse)
        assert response.topics == ()
        assert "consul_unavailable" in response.warnings
        assert response.catalog_version == 0

    @pytest.mark.asyncio
    async def test_get_catalog_version_no_handler_returns_minus_one(self) -> None:
        """get_catalog_version should return -1 when no handler configured."""
        service = _make_service(consul_handler=None)
        version = await service.get_catalog_version(uuid4())
        assert version == -1

    @pytest.mark.asyncio
    async def test_increment_version_no_handler_returns_minus_one(self) -> None:
        """increment_version should return -1 when no handler configured."""
        service = _make_service(consul_handler=None)
        result = await service.increment_version(uuid4())
        assert result == -1


# ---------------------------------------------------------------------------
# Test: get_catalog_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetCatalogVersion:
    """Tests for get_catalog_version method."""

    @pytest.mark.asyncio
    async def test_returns_current_version_when_key_exists(self) -> None:
        """Should parse and return version from Consul KV."""
        handler = _make_consul_handler(kv_get_raw_return="42")
        service = _make_service(consul_handler=handler)

        version = await service.get_catalog_version(uuid4())

        assert version == 42

    @pytest.mark.asyncio
    async def test_returns_minus_one_when_key_absent(self) -> None:
        """Should return -1 when Consul key does not exist."""
        handler = _make_consul_handler(kv_get_raw_return=None)
        service = _make_service(consul_handler=handler)

        version = await service.get_catalog_version(uuid4())

        assert version == -1

    @pytest.mark.asyncio
    async def test_returns_minus_one_when_value_invalid(self) -> None:
        """Should return -1 when Consul value is not a valid integer."""
        handler = _make_consul_handler(kv_get_raw_return="not-a-number")
        service = _make_service(consul_handler=handler)

        version = await service.get_catalog_version(uuid4())

        assert version == -1

    @pytest.mark.asyncio
    async def test_returns_zero_for_negative_stored_value(self) -> None:
        """Should clamp negative stored values to 0."""
        handler = _make_consul_handler(kv_get_raw_return="-5")
        service = _make_service(consul_handler=handler)

        version = await service.get_catalog_version(uuid4())

        assert version == 0


# ---------------------------------------------------------------------------
# Test: increment_version (CAS)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIncrementVersion:
    """Tests for increment_version CAS logic."""

    @pytest.mark.asyncio
    async def test_increment_succeeds_on_first_attempt(self) -> None:
        """Should return new version when CAS succeeds immediately."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)

        # Patch internal helpers directly
        async def _mock_get(key: str, cid: object) -> tuple[str | None, int]:
            return "5", 100

        async def _mock_put(key: str, value: str, cas: int, cid: object) -> bool:
            return True

        service._kv_get_with_modify_index = _mock_get  # type: ignore[assignment]
        service._kv_put_raw_with_cas = _mock_put  # type: ignore[assignment]

        result = await service.increment_version(uuid4())

        assert result == 6

    @pytest.mark.asyncio
    async def test_increment_retries_on_cas_failure_then_succeeds(self) -> None:
        """Should retry after CAS conflict and succeed on second attempt."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)

        call_count = 0

        async def _mock_get(key: str, cid: object) -> tuple[str | None, int]:
            return "10", 200

        async def _mock_put(key: str, value: str, cas: int, cid: object) -> bool:
            nonlocal call_count
            call_count += 1
            # Fail first attempt, succeed second
            return call_count >= 2

        service._kv_get_with_modify_index = _mock_get  # type: ignore[assignment]
        service._kv_put_raw_with_cas = _mock_put  # type: ignore[assignment]

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await service.increment_version(uuid4())

        assert result == 11
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_increment_returns_minus_one_after_all_retries_fail(self) -> None:
        """Should return -1 after all CAS retries are exhausted."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)

        async def _mock_get(key: str, cid: object) -> tuple[str | None, int]:
            return "1", 50

        async def _mock_put(key: str, value: str, cas: int, cid: object) -> bool:
            return False  # Always fail

        service._kv_get_with_modify_index = _mock_get  # type: ignore[assignment]
        service._kv_put_raw_with_cas = _mock_put  # type: ignore[assignment]

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await service.increment_version(uuid4())

        assert result == -1

    @pytest.mark.asyncio
    async def test_increment_creates_key_when_absent(self) -> None:
        """Should write version=1 when key does not exist (ModifyIndex=0 means create)."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)

        written_value: list[str] = []
        written_cas: list[int] = []

        async def _mock_get(key: str, cid: object) -> tuple[str | None, int]:
            return None, 0  # Key absent

        async def _mock_put(key: str, value: str, cas: int, cid: object) -> bool:
            written_value.append(value)
            written_cas.append(cas)
            return True

        service._kv_get_with_modify_index = _mock_get  # type: ignore[assignment]
        service._kv_put_raw_with_cas = _mock_put  # type: ignore[assignment]

        result = await service.increment_version(uuid4())

        assert result == 1
        assert written_value == ["1"]
        assert written_cas == [0]


# ---------------------------------------------------------------------------
# Test: build_catalog happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCatalogHappyPath:
    """Tests for build_catalog with mocked Consul KV data."""

    def _make_kv_items(
        self,
        node_id: str,
        subscribe_topics: list[str],
        publish_topics: list[str],
        subscribe_entries: list[dict[str, object]] | None = None,
        publish_entries: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        """Build a list of fake KV items for a node."""
        base = f"onex/nodes/{node_id}/event_bus/"
        items: list[dict[str, object]] = [
            {
                "key": base + "subscribe_topics",
                "value": json.dumps(subscribe_topics),
                "modify_index": 1,
            },
            {
                "key": base + "publish_topics",
                "value": json.dumps(publish_topics),
                "modify_index": 1,
            },
        ]
        if subscribe_entries is not None:
            items.append(
                {
                    "key": base + "subscribe_entries",
                    "value": json.dumps(subscribe_entries),
                    "modify_index": 1,
                }
            )
        if publish_entries is not None:
            items.append(
                {
                    "key": base + "publish_entries",
                    "value": json.dumps(publish_entries),
                    "modify_index": 1,
                }
            )
        return items

    @pytest.mark.asyncio
    async def test_single_node_single_topic(self) -> None:
        """Should return one entry for a node with one subscribe topic."""
        node_id = "node-aaa"
        kv_items = self._make_kv_items(
            node_id,
            subscribe_topics=[_VALID_SUFFIX],
            publish_topics=[],
        )

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)

        # Patch internals
        service._kv_get_recurse = AsyncMock(return_value=kv_items)  # type: ignore[method-assign]

        async def _mock_get_version(correlation_id: object) -> int:
            return 1

        service.get_catalog_version = _mock_get_version  # type: ignore[method-assign]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        assert len(response.topics) == 1
        entry = response.topics[0]
        assert entry.topic_suffix == _VALID_SUFFIX
        assert entry.subscriber_count == 1
        assert entry.publisher_count == 0
        assert entry.is_active is True

    @pytest.mark.asyncio
    async def test_multiple_nodes_same_topic(self) -> None:
        """Multiple nodes subscribing to the same topic should aggregate counts."""
        kv_items = self._make_kv_items(
            "node-aaa",
            subscribe_topics=[_VALID_SUFFIX],
            publish_topics=[],
        ) + self._make_kv_items(
            "node-bbb",
            subscribe_topics=[_VALID_SUFFIX],
            publish_topics=[],
        )

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=kv_items)  # type: ignore[method-assign]

        async def _mock_version(cid: object) -> int:
            return 1

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        assert len(response.topics) == 1
        entry = response.topics[0]
        assert entry.subscriber_count == 2

    @pytest.mark.asyncio
    async def test_publisher_and_subscriber_counted_separately(self) -> None:
        """A node publishing and another subscribing should be counted correctly."""
        kv_items = self._make_kv_items(
            "publisher-node",
            subscribe_topics=[],
            publish_topics=[_VALID_SUFFIX],
        ) + self._make_kv_items(
            "subscriber-node",
            subscribe_topics=[_VALID_SUFFIX],
            publish_topics=[],
        )

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="2")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=kv_items)  # type: ignore[method-assign]

        async def _mock_version(cid: object) -> int:
            return 2

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        assert len(response.topics) == 1
        entry = response.topics[0]
        assert entry.publisher_count == 1
        assert entry.subscriber_count == 1
        assert response.node_count == 2

    @pytest.mark.asyncio
    async def test_enrichment_applies_description_from_entries(self) -> None:
        """Description from publish_entries should be enriched on the entry."""
        kv_items = self._make_kv_items(
            "publisher-node",
            subscribe_topics=[],
            publish_topics=[_VALID_SUFFIX],
            publish_entries=[
                {
                    "topic_suffix": _VALID_SUFFIX,
                    "description": "Node registration events",
                    "partitions": 6,
                }
            ],
        )

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=kv_items)  # type: ignore[method-assign]

        async def _mock_version(cid: object) -> int:
            return 1

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        assert len(response.topics) == 1
        entry = response.topics[0]
        assert entry.description == "Node registration events"
        assert entry.partitions == 6


# ---------------------------------------------------------------------------
# Test: cache hit / miss
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCatalogCache:
    """Tests for the in-process version-based cache."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_kv_scan(self) -> None:
        """Second call with same catalog version should skip _build_topics_from_kv."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="3")

        service = _make_service(consul_handler=handler)

        kv_items = [
            {
                "key": "onex/nodes/node-aaa/event_bus/subscribe_topics",
                "value": json.dumps([_VALID_SUFFIX]),
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-aaa/event_bus/publish_topics",
                "value": json.dumps([]),
                "modify_index": 1,
            },
        ]

        scan_count = 0

        async def _mock_recurse(
            prefix: str, correlation_id: object
        ) -> list[dict[str, object]]:
            nonlocal scan_count
            scan_count += 1
            return kv_items

        service._kv_get_recurse = _mock_recurse  # type: ignore[method-assign]

        async def _mock_version(cid: object) -> int:
            return 3

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        cid = uuid4()
        # First call - should scan
        await service.build_catalog(correlation_id=cid)
        # Second call - should hit cache
        await service.build_catalog(correlation_id=cid)

        assert scan_count == 1

    @pytest.mark.asyncio
    async def test_cache_miss_when_version_changes(self) -> None:
        """When catalog version changes, the cache must be bypassed."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)

        kv_items: list[dict[str, object]] = []
        scan_count = 0

        async def _mock_recurse(prefix: str, cid: object) -> list[dict[str, object]]:
            nonlocal scan_count
            scan_count += 1
            return kv_items

        service._kv_get_recurse = _mock_recurse  # type: ignore[assignment]

        call_count = 0

        async def _mock_version(cid: object) -> int:
            nonlocal call_count
            call_count += 1
            # Return different version on second call
            return 1 if call_count == 1 else 2

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        await service.build_catalog(correlation_id=uuid4())
        await service.build_catalog(correlation_id=uuid4())

        assert scan_count == 2

    @pytest.mark.asyncio
    async def test_cache_disabled_when_version_is_minus_one(self) -> None:
        """When catalog_version == -1, cache is disabled and every call rebuilds."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value=None)  # version absent

        service = _make_service(consul_handler=handler)

        scan_count = 0

        async def _mock_recurse(prefix: str, cid: object) -> list[dict[str, object]]:
            nonlocal scan_count
            scan_count += 1
            return []

        service._kv_get_recurse = _mock_recurse  # type: ignore[assignment]

        async def _mock_version(cid: object) -> int:
            return -1

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        await service.build_catalog(correlation_id=uuid4())
        await service.build_catalog(correlation_id=uuid4())

        assert scan_count == 2


# ---------------------------------------------------------------------------
# Test: timeout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCatalogTimeout:
    """Tests for scan timeout behaviour."""

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_with_warning(self) -> None:
        """When KV scan exceeds budget, should return partial result + warning."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)

        async def _slow_recurse(prefix: str, cid: object) -> list[dict[str, object]]:
            await asyncio.sleep(100)  # Simulate hang
            return []

        service._kv_get_recurse = _slow_recurse  # type: ignore[assignment]

        async def _mock_version(cid: object) -> int:
            return -1  # Disable cache so scan is always attempted

        service.get_catalog_version = _mock_version  # type: ignore[assignment]

        with patch(
            "omnibase_infra.services.service_topic_catalog._SCAN_BUDGET_SECONDS",
            0.01,
        ):
            response = await service.build_catalog(
                correlation_id=uuid4(),
            )

        assert "consul_scan_timeout" in response.warnings
        assert response.topics == ()


# ---------------------------------------------------------------------------
# Test: filtering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildCatalogFiltering:
    """Tests for topic_pattern and include_inactive filters."""

    def _kv_items_two_topics(self) -> list[dict[str, object]]:
        """Build KV items for two topics - one subscribed, one published."""
        return [
            {
                "key": "onex/nodes/node-aaa/event_bus/subscribe_topics",
                "value": json.dumps([_VALID_SUFFIX]),
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-aaa/event_bus/publish_topics",
                "value": json.dumps([_VALID_SUFFIX_2]),
                "modify_index": 1,
            },
        ]

    @pytest.mark.asyncio
    async def test_include_inactive_false_excludes_inactive_topics(self) -> None:
        """Default include_inactive=False should only return active topics."""
        # _VALID_SUFFIX subscribed, _VALID_SUFFIX_2 only published (active too)
        # Both have at least one participant so both are active
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=self._kv_items_two_topics())  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return 1

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
            include_inactive=False,
        )
        # Both topics have participants, both should appear
        assert all(t.is_active for t in response.topics)

    @pytest.mark.asyncio
    async def test_topic_pattern_filters_by_glob(self) -> None:
        """topic_pattern should use fnmatch to filter topic suffixes."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=self._kv_items_two_topics())  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return 1

        service.get_catalog_version = _v  # type: ignore[assignment]

        # Only match registration topic
        response = await service.build_catalog(
            correlation_id=uuid4(),
            topic_pattern="onex.evt.platform.node-registration.*",
        )

        assert len(response.topics) == 1
        assert response.topics[0].topic_suffix == _VALID_SUFFIX

    @pytest.mark.asyncio
    async def test_topic_pattern_no_match_returns_empty(self) -> None:
        """Non-matching pattern should return empty topic list."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=self._kv_items_two_topics())  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return 1

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
            topic_pattern="onex.cmd.*",  # Doesn't match evt topics
        )

        assert response.topics == ()


# ---------------------------------------------------------------------------
# Test: KV precedence rules
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKVPrecedenceRules:
    """Tests verifying node arrays are authoritative and reverse index is ignored."""

    @pytest.mark.asyncio
    async def test_node_arrays_win_over_reverse_index(self) -> None:
        """Node subscribe_topics / publish_topics are source of truth.

        The reverse index (onex/topics/.../subscribers) is intentionally
        ignored during catalog build. Only per-node arrays matter.
        """
        # KV items: node-aaa subscribes to _VALID_SUFFIX
        # Additionally include a reverse-index key (which should be ignored)
        kv_items: list[dict[str, object]] = [
            {
                "key": "onex/nodes/node-aaa/event_bus/subscribe_topics",
                "value": json.dumps([_VALID_SUFFIX]),
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-aaa/event_bus/publish_topics",
                "value": json.dumps([]),
                "modify_index": 1,
            },
            # Reverse index claims node-bbb also subscribes, but has no node entry
            {
                "key": f"onex/topics/{_VALID_SUFFIX}/subscribers",
                "value": json.dumps(["node-aaa", "node-bbb"]),
                "modify_index": 1,
            },
        ]

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=kv_items)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return 1

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        assert len(response.topics) == 1
        entry = response.topics[0]
        # Only node-aaa counted (from authoritative per-node array)
        # node-bbb in reverse index is NOT counted
        assert entry.subscriber_count == 1

    @pytest.mark.asyncio
    async def test_invalid_json_in_node_arrays_produces_warning(self) -> None:
        """Invalid JSON in node's topic array should emit a warning, not crash."""
        kv_items: list[dict[str, object]] = [
            {
                "key": "onex/nodes/node-aaa/event_bus/subscribe_topics",
                "value": "NOT_VALID_JSON",
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-aaa/event_bus/publish_topics",
                "value": json.dumps([_VALID_SUFFIX]),
                "modify_index": 1,
            },
        ]

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=kv_items)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1  # Disable cache

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        # Should still have the publish topic entry
        assert any(t.topic_suffix == _VALID_SUFFIX for t in response.topics)
        # Should have an invalid_json warning
        assert any("invalid_json_at:" in w for w in response.warnings)


# ---------------------------------------------------------------------------
# Test: consul KV unavailable
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestConsulKVUnavailable:
    """Tests for partial-success when Consul KV scan fails."""

    @pytest.mark.asyncio
    async def test_recurse_returns_none_produces_warning(self) -> None:
        """When _kv_get_recurse returns None, should warn consul_unavailable."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None
        handler.kv_get_raw = AsyncMock(return_value="1")

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=None)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(
            correlation_id=uuid4(),
        )

        assert "consul_unavailable" in response.warnings
        assert response.topics == ()


# ---------------------------------------------------------------------------
# Test: OMN-2312 warning codes - each triggered by its failure path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOMN2312WarningCodes:
    """Tests verifying all 5 OMN-2312 warning codes are emitted on their paths.

    Warning codes are defined in catalog_warning_codes and used as constants
    throughout the service. Each test triggers exactly one warning code path.

    Related Tickets:
        - OMN-2312: Topic Catalog: response warnings channel
    """

    @pytest.mark.asyncio
    async def test_consul_unavailable_when_no_handler(self) -> None:
        """CONSUL_UNAVAILABLE emitted when no consul_handler is configured."""
        service = _make_service(consul_handler=None)

        response = await service.build_catalog(correlation_id=uuid4())

        assert CONSUL_UNAVAILABLE in response.warnings
        assert response.topics == ()
        assert response.catalog_version == 0

    @pytest.mark.asyncio
    async def test_consul_unavailable_when_kv_recurse_returns_none(self) -> None:
        """CONSUL_UNAVAILABLE emitted when KV recursive scan returns None."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=None)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1  # Disable cache; ensure we attempt the scan

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        assert CONSUL_UNAVAILABLE in response.warnings

    @pytest.mark.asyncio
    async def test_consul_scan_timeout_when_kv_scan_exceeds_budget(self) -> None:
        """CONSUL_SCAN_TIMEOUT emitted when KV scan exceeds budget."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)

        async def _slow_recurse(prefix: str, cid: object) -> list[dict[str, object]]:
            await asyncio.sleep(100)  # Simulate hang
            return []

        service._kv_get_recurse = _slow_recurse  # type: ignore[assignment]

        async def _v(cid: object) -> int:
            return -1  # Disable cache

        service.get_catalog_version = _v  # type: ignore[assignment]

        with patch(
            "omnibase_infra.services.service_topic_catalog._SCAN_BUDGET_SECONDS",
            0.01,
        ):
            response = await service.build_catalog(correlation_id=uuid4())

        assert CONSUL_SCAN_TIMEOUT in response.warnings
        assert response.topics == ()

    @pytest.mark.asyncio
    async def test_consul_kv_max_keys_reached_when_scan_hits_cap(self) -> None:
        """CONSUL_KV_MAX_KEYS_REACHED emitted when scan returns >= _MAX_KV_KEYS items."""
        from omnibase_infra.services.service_topic_catalog import _MAX_KV_KEYS

        # Build exactly _MAX_KV_KEYS items so the >= threshold is hit.
        # Each item is a valid subscribe_topics entry for a unique node so that
        # the catalog can still be built (partial success) despite the warning.
        capped_items: list[dict[str, object]] = [
            {
                "key": f"onex/nodes/node-{i:05d}/event_bus/subscribe_topics",
                "value": "[]",
                "modify_index": 1,
            }
            for i in range(_MAX_KV_KEYS)
        ]

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=capped_items)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1  # Disable cache

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        assert CONSUL_KV_MAX_KEYS_REACHED in response.warnings

    @pytest.mark.asyncio
    async def test_partial_node_data_when_node_has_malformed_kv(self) -> None:
        """PARTIAL_NODE_DATA emitted when a node has at least one malformed KV entry."""
        bad_kv_items: list[dict[str, object]] = [
            {
                "key": "onex/nodes/node-bad/event_bus/subscribe_topics",
                "value": "INVALID_JSON{{{{",  # malformed
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-bad/event_bus/publish_topics",
                "value": json.dumps([_VALID_SUFFIX]),  # valid
                "modify_index": 1,
            },
        ]

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=bad_kv_items)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1  # Disable cache

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        assert PARTIAL_NODE_DATA in response.warnings
        # Fine-grained per-key warning also present
        assert any("invalid_json_at:" in w for w in response.warnings)
        # Valid publish topic is still returned
        assert any(t.topic_suffix == _VALID_SUFFIX for t in response.topics)

    @pytest.mark.asyncio
    async def test_partial_node_data_not_emitted_for_clean_data(self) -> None:
        """PARTIAL_NODE_DATA must NOT be emitted when all KV entries are valid."""
        clean_kv_items: list[dict[str, object]] = [
            {
                "key": "onex/nodes/node-ok/event_bus/subscribe_topics",
                "value": json.dumps([_VALID_SUFFIX]),
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-ok/event_bus/publish_topics",
                "value": json.dumps([]),
                "modify_index": 1,
            },
        ]

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=clean_kv_items)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        assert PARTIAL_NODE_DATA not in response.warnings

    @pytest.mark.asyncio
    async def test_version_unknown_when_catalog_version_is_minus_one(self) -> None:
        """VERSION_UNKNOWN emitted when catalog_version returns -1 (absent/corrupt)."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=[])  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1  # Simulate absent/corrupt version key

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        assert VERSION_UNKNOWN in response.warnings

    @pytest.mark.asyncio
    async def test_version_unknown_not_emitted_when_version_known(self) -> None:
        """VERSION_UNKNOWN must NOT be emitted when catalog_version is valid (>= 0)."""
        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=[])  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return 5  # Known version

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        assert VERSION_UNKNOWN not in response.warnings

    @pytest.mark.asyncio
    async def test_multiple_warnings_can_coexist(self) -> None:
        """Multiple warning codes can appear together in a single response."""
        bad_kv_items: list[dict[str, object]] = [
            {
                "key": "onex/nodes/node-x/event_bus/subscribe_topics",
                "value": "bad_json",  # malformed -> partial_node_data
                "modify_index": 1,
            },
            {
                "key": "onex/nodes/node-x/event_bus/publish_topics",
                "value": json.dumps([]),
                "modify_index": 1,
            },
        ]

        handler = MagicMock()
        handler._client = MagicMock()
        handler._executor = None

        service = _make_service(consul_handler=handler)
        service._kv_get_recurse = AsyncMock(return_value=bad_kv_items)  # type: ignore[method-assign]

        async def _v(cid: object) -> int:
            return -1  # version_unknown

        service.get_catalog_version = _v  # type: ignore[assignment]

        response = await service.build_catalog(correlation_id=uuid4())

        # Both version_unknown and partial_node_data should be present
        assert VERSION_UNKNOWN in response.warnings
        assert PARTIAL_NODE_DATA in response.warnings
