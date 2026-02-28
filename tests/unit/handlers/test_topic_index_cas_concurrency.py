# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Regression test for OMN-2345: CAS-protected topic index concurrent writes.

Verifies that concurrent node registrations on the same Consul topic subscriber
key do not lose entries due to the non-atomic read-modify-write race that
existed before the CAS fix.

The race pattern (before fix):
    Worker A: read  → {node-x}
    Worker B: read  → {node-x}
    Worker A: write → {node-x, node-a}   (succeeds)
    Worker B: write → {node-x, node-b}   (overwrites A — node-a lost)

With CAS protection, Worker B's write returns False (CAS conflict), causing a
retry that reads the current value {node-x, node-a} and then writes
{node-x, node-a, node-b} successfully.

Related:
    - OMN-2345: Fix non-atomic read-modify-write race in _update_topic_index()
    - OMN-2314: Topic Catalog change notification emission (made race observable)
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.handler_consul import HandlerConsul
from omnibase_infra.models.registration import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)

pytestmark = [pytest.mark.unit]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_container() -> MagicMock:
    """Create mock ONEX container for handler tests."""
    return MagicMock(spec=ModelONEXContainer)


@pytest.fixture
def consul_config() -> dict[str, object]:
    """Provide test Consul configuration."""
    return {
        "host": "consul.example.com",
        "port": 8500,
        "scheme": "http",
        "token": "acl-token-abc123",
        "timeout_seconds": 30.0,
        "retry": {
            "max_attempts": 3,
            "initial_delay_seconds": 0.1,
            "max_delay_seconds": 1.0,
            "exponential_base": 2.0,
        },
    }


@pytest.fixture
def cas_aware_consul_client() -> MagicMock:
    """Consul mock with a CAS-aware KV store.

    This mock tracks the ModifyIndex per key.  A ``kv.put`` with a non-None
    ``cas`` value succeeds only if the supplied index matches the stored
    ModifyIndex; otherwise it returns ``False`` (simulating a CAS conflict).

    A threading.Lock guards the in-memory store so that asyncio tasks running
    concurrently within the event loop share a single consistent state.
    """
    client = MagicMock()
    lock = threading.Lock()

    # Each key maps to (value_bytes, modify_index).
    kv_store: dict[str, tuple[bytes, int]] = {}
    # Monotonically increasing counter used as ModifyIndex.
    counter: list[int] = [0]

    def kv_get(key: str, recurse: bool = False) -> tuple[int, dict[str, object] | None]:
        with lock:
            if key not in kv_store:
                return (0, None)
            value_bytes, modify_index = kv_store[key]
            return (
                modify_index,
                {
                    "Value": value_bytes,
                    "Key": key,
                    "ModifyIndex": modify_index,
                },
            )

    def kv_put(
        key: str,
        value: str,
        flags: int | None = None,
        cas: int | None = None,
    ) -> bool:
        with lock:
            current_index = kv_store[key][1] if key in kv_store else 0
            if cas is not None and cas != current_index:
                # CAS conflict — another writer changed the key since this
                # caller last read it.
                return False
            counter[0] += 1
            kv_store[key] = (value.encode("utf-8"), counter[0])
            return True

    client.kv = MagicMock()
    client.kv.get = MagicMock(side_effect=kv_get)
    client.kv.put = MagicMock(side_effect=kv_put)

    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = MagicMock(return_value=None)
    client.agent.service.deregister = MagicMock(return_value=None)
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    # Expose internal state for assertions.
    client._cas_kv_store = kv_store

    return client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event_bus(topics: list[str]) -> ModelNodeEventBusConfig:
    return ModelNodeEventBusConfig(
        subscribe_topics=[ModelEventBusTopicEntry(topic=t) for t in topics],
        publish_topics=[],
    )


# ---------------------------------------------------------------------------
# Regression test
# ---------------------------------------------------------------------------


class TestCASConcurrentRegistrations:
    """Regression tests for OMN-2345: CAS-protected topic index writes."""

    @pytest.mark.asyncio
    async def test_concurrent_registrations_no_entries_lost(
        self,
        consul_config: dict[str, object],
        cas_aware_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Concurrent add_subscriber calls on the same topic must not lose entries.

        Spawns N concurrent tasks that each call _add_subscriber_to_topic for a
        distinct node_id on the same topic key.  After all tasks complete, every
        node_id must appear in the stored subscriber list — no entries may be
        silently dropped due to a lost CAS write (which would be retried).
        """
        topic = "onex.evt.concurrent-test.v1"
        node_ids = [f"node-{i:03d}" for i in range(10)]

        handler = HandlerConsul(mock_container)
        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = cas_aware_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()

            # Run all add_subscriber_to_topic calls concurrently.
            await asyncio.gather(
                *[
                    handler._add_subscriber_to_topic(topic, node_id, correlation_id)
                    for node_id in node_ids
                ]
            )

        # Verify every node_id is present in the stored subscriber list.
        key = f"onex/topics/{topic}/subscribers"
        assert key in cas_aware_consul_client._cas_kv_store, (
            f"Key {key!r} missing from KV store after concurrent registrations"
        )
        stored_bytes, _ = cas_aware_consul_client._cas_kv_store[key]
        stored_subscribers: list[str] = json.loads(stored_bytes.decode("utf-8"))

        missing = set(node_ids) - set(stored_subscribers)
        assert not missing, (
            f"Entries lost due to CAS race: {sorted(missing)}\n"
            f"Stored: {sorted(stored_subscribers)}"
        )
        assert len(stored_subscribers) == len(node_ids), (
            f"Expected {len(node_ids)} unique subscribers, got {len(stored_subscribers)}: "
            f"{sorted(stored_subscribers)}"
        )

    @pytest.mark.asyncio
    async def test_concurrent_update_topic_index_no_entries_lost(
        self,
        consul_config: dict[str, object],
        cas_aware_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Concurrent _update_topic_index calls for distinct nodes must preserve all entries.

        This is the end-to-end variant of the race described in OMN-2345:
        multiple nodes registering the same topic concurrently.  All node IDs
        must appear in the final subscriber list.
        """
        shared_topic = "onex.evt.shared-concurrent.v1"
        node_ids = [f"worker-{i:03d}" for i in range(8)]

        handler = HandlerConsul(mock_container)
        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = cas_aware_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()

            # Each node registers with the shared topic.  Because the per-node
            # subscribe_topics KV key is unique (per node_id), only the shared
            # topic's subscriber list is the contention point.
            async def register_node(
                node_id: str,
            ) -> tuple[frozenset[str], frozenset[str]]:
                event_bus = _make_event_bus([shared_topic])
                return await handler._update_topic_index(
                    node_id, event_bus, correlation_id
                )

            results = await asyncio.gather(*[register_node(nid) for nid in node_ids])

        # Every _update_topic_index should have reported the shared_topic as added.
        for topics_added, topics_removed in results:
            assert shared_topic in topics_added, (
                f"Expected {shared_topic!r} in topics_added but got {topics_added}"
            )
            assert len(topics_removed) == 0

        # Shared topic subscriber list must contain all node IDs.
        key = f"onex/topics/{shared_topic}/subscribers"
        assert key in cas_aware_consul_client._cas_kv_store, (
            f"Key {key!r} missing after concurrent _update_topic_index calls"
        )
        stored_bytes, _ = cas_aware_consul_client._cas_kv_store[key]
        stored_subscribers: list[str] = json.loads(stored_bytes.decode("utf-8"))

        missing = set(node_ids) - set(stored_subscribers)
        assert not missing, (
            f"Entries lost in concurrent _update_topic_index: {sorted(missing)}\n"
            f"Stored subscribers: {sorted(stored_subscribers)}"
        )

    @pytest.mark.asyncio
    async def test_cas_retry_on_conflict_eventually_succeeds(
        self,
        consul_config: dict[str, object],
        cas_aware_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """CAS retries must succeed even when the first attempt(s) conflict.

        Pre-populates the KV store so that the first CAS write will fail
        (simulating a concurrent writer that just committed), then verifies
        that the retry loop succeeds and the final state is correct.
        """
        topic = "onex.evt.retry-test.v1"
        key = f"onex/topics/{topic}/subscribers"

        # Seed the KV store with an existing subscriber at ModifyIndex=5.
        internal_store: dict[str, tuple[bytes, int]] = (
            cas_aware_consul_client._cas_kv_store
        )
        internal_store[key] = (json.dumps(["pre-existing-node"]).encode(), 5)

        # Track how many put calls are made so we can confirm at least one retry.
        put_call_log: list[dict[str, object]] = []
        original_put = cas_aware_consul_client.kv.put.side_effect
        initial_cas = 5

        def instrumented_put(
            key_arg: str,
            value: str,
            flags: int | None = None,
            cas: int | None = None,
        ) -> bool:
            # On the first attempt with the initial CAS index, simulate a
            # concurrent writer by bumping the stored ModifyIndex and returning
            # False — this forces the handler's retry loop to exercise the
            # read-again path.
            if cas == initial_cas and not put_call_log:
                # Simulate concurrent writer: bump ModifyIndex so next read gets 6
                current_val, _ = internal_store[key_arg]
                internal_store[key_arg] = (current_val, initial_cas + 1)
                put_call_log.append({"key": key_arg, "cas": cas, "result": False})
                return False
            result = original_put(key_arg, value, flags=flags, cas=cas)
            put_call_log.append({"key": key_arg, "cas": cas, "result": result})
            return result

        cas_aware_consul_client.kv.put.side_effect = instrumented_put

        handler = HandlerConsul(mock_container)
        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = cas_aware_consul_client
            await handler.initialize(consul_config)

            # Adding a new subscriber — first CAS attempt will conflict (returns False),
            # handler retries with updated ModifyIndex=6 and succeeds.
            await handler._add_subscriber_to_topic(topic, "new-node", uuid4())

        stored_bytes, _ = internal_store[key]
        stored: list[str] = json.loads(stored_bytes.decode())
        assert "new-node" in stored
        assert "pre-existing-node" in stored

        # At least 2 put calls: first conflict (cas=5), then success (cas=6).
        topic_puts = [p for p in put_call_log if key in p["key"]]
        assert len(topic_puts) >= 2, (
            f"Expected at least 2 put attempts (conflict + retry), got {len(topic_puts)}"
        )
        assert topic_puts[0]["cas"] == initial_cas, (
            f"Expected first attempt cas={initial_cas}, got {topic_puts[0]['cas']}"
        )
        assert topic_puts[0]["result"] is False, (
            "First attempt should have returned False (CAS conflict)"
        )
        assert topic_puts[1]["cas"] == initial_cas + 1, (
            f"Expected retry cas={initial_cas + 1} (updated ModifyIndex), got {topic_puts[1]['cas']}"
        )
        assert topic_puts[1]["result"] is True, "Retry should have succeeded"


__all__: list[str] = ["TestCASConcurrentRegistrations"]
