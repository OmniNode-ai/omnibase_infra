# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HandlerConsul topic index management.

These tests validate the topic index functionality added for OMN-1613,
which stores event bus configuration and maintains a reverse index from
topics to subscriber node IDs.

Consul KV Structure:
    onex/nodes/{node_id}/event_bus/subscribe_topics     # [topic strings] for routing
    onex/nodes/{node_id}/event_bus/publish_topics       # [topic strings]
    onex/nodes/{node_id}/event_bus/subscribe_entries    # [full entries] for tooling
    onex/nodes/{node_id}/event_bus/publish_entries      # [full entries]
    onex/topics/{topic}/subscribers                     # [node_ids] reverse index
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraConsulError,
    InfraTimeoutError,
    InfraUnavailableError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.handlers.handler_consul import HandlerConsul
from omnibase_infra.models.registration import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)


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
            "initial_delay_seconds": 0.1,  # Minimum allowed value
            "max_delay_seconds": 1.0,  # Minimum allowed value
            "exponential_base": 2.0,
        },
    }


@pytest.fixture
def mock_consul_client() -> MagicMock:
    """Provide mocked consul.Consul client with KV store simulation."""
    client = MagicMock()

    # Simulated KV store
    kv_store: dict[str, bytes] = {}

    def kv_get(key: str, recurse: bool = False) -> tuple[int, dict[str, object] | None]:
        if key in kv_store:
            return (0, {"Value": kv_store[key], "Key": key, "ModifyIndex": 100})
        return (0, None)

    def kv_put(
        key: str, value: str, flags: int | None = None, cas: int | None = None
    ) -> bool:
        kv_store[key] = value.encode("utf-8")
        return True

    client.kv = MagicMock()
    client.kv.get = MagicMock(side_effect=kv_get)
    client.kv.put = MagicMock(side_effect=kv_put)

    # Mock service registration
    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = MagicMock(return_value=None)
    client.agent.service.deregister = MagicMock(return_value=None)

    # Mock status for health check
    client.status = MagicMock()
    client.status.leader = MagicMock(return_value="192.168.1.1:8300")

    # Expose kv_store for test assertions
    client._test_kv_store = kv_store

    return client


@pytest.fixture
def sample_event_bus_config() -> ModelNodeEventBusConfig:
    """Create sample event bus configuration for tests."""
    return ModelNodeEventBusConfig(
        subscribe_topics=[
            ModelEventBusTopicEntry(
                topic="onex.evt.intent-classified.v1",
                event_type="ModelIntentClassified",
                message_category="EVENT",
                description="Intent classification events",
            ),
            ModelEventBusTopicEntry(
                topic="onex.evt.node-registered.v1",
                event_type="ModelNodeRegistered",
                message_category="EVENT",
            ),
        ],
        publish_topics=[
            ModelEventBusTopicEntry(
                topic="onex.evt.node-processed.v1",
                event_type="ModelNodeProcessed",
                message_category="EVENT",
            ),
        ],
    )


class TestStoreEventBusInKV:
    """Test _store_node_event_bus stores topic data correctly."""

    @pytest.mark.asyncio
    async def test_store_event_bus_creates_all_kv_entries(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        sample_event_bus_config: ModelNodeEventBusConfig,
    ) -> None:
        """Test that storing event bus config creates all 4 KV entries."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "test-node-001"

            await handler._store_node_event_bus(
                node_id, sample_event_bus_config, correlation_id
            )

            kv_store = mock_consul_client._test_kv_store

            # Verify all 4 keys were created
            assert f"onex/nodes/{node_id}/event_bus/subscribe_topics" in kv_store
            assert f"onex/nodes/{node_id}/event_bus/publish_topics" in kv_store
            assert f"onex/nodes/{node_id}/event_bus/subscribe_entries" in kv_store
            assert f"onex/nodes/{node_id}/event_bus/publish_entries" in kv_store

    @pytest.mark.asyncio
    async def test_store_event_bus_topic_strings(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        sample_event_bus_config: ModelNodeEventBusConfig,
    ) -> None:
        """Test that topic strings are stored correctly for routing."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "test-node-002"

            await handler._store_node_event_bus(
                node_id, sample_event_bus_config, correlation_id
            )

            kv_store = mock_consul_client._test_kv_store

            # Verify subscribe topics
            subscribe_topics = json.loads(
                kv_store[f"onex/nodes/{node_id}/event_bus/subscribe_topics"].decode()
            )
            assert subscribe_topics == [
                "onex.evt.intent-classified.v1",
                "onex.evt.node-registered.v1",
            ]

            # Verify publish topics
            publish_topics = json.loads(
                kv_store[f"onex/nodes/{node_id}/event_bus/publish_topics"].decode()
            )
            assert publish_topics == ["onex.evt.node-processed.v1"]

    @pytest.mark.asyncio
    async def test_store_event_bus_full_entries(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        sample_event_bus_config: ModelNodeEventBusConfig,
    ) -> None:
        """Test that full entries are stored for tooling."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "test-node-003"

            await handler._store_node_event_bus(
                node_id, sample_event_bus_config, correlation_id
            )

            kv_store = mock_consul_client._test_kv_store

            # Verify full entries contain metadata
            subscribe_entries = json.loads(
                kv_store[f"onex/nodes/{node_id}/event_bus/subscribe_entries"].decode()
            )
            assert len(subscribe_entries) == 2
            assert subscribe_entries[0]["topic"] == "onex.evt.intent-classified.v1"
            assert subscribe_entries[0]["event_type"] == "ModelIntentClassified"
            assert subscribe_entries[0]["description"] == "Intent classification events"


class TestTopicSubscriberIndex:
    """Test topic -> node_id reverse index creation and updates."""

    @pytest.mark.asyncio
    async def test_add_subscriber_to_topic(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test adding a subscriber to a topic creates the index entry."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            topic = "onex.evt.test-topic.v1"
            node_id = "subscriber-node-001"

            await handler._add_subscriber_to_topic(topic, node_id, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify index was created
            key = f"onex/topics/{topic}/subscribers"
            assert key in kv_store
            subscribers = json.loads(kv_store[key].decode())
            assert node_id in subscribers

    @pytest.mark.asyncio
    async def test_idempotent_index_update_no_duplicates(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test that adding same subscriber twice doesn't create duplicates."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            topic = "onex.evt.test-topic.v1"
            node_id = "subscriber-node-001"

            # Add same subscriber twice
            await handler._add_subscriber_to_topic(topic, node_id, correlation_id)
            await handler._add_subscriber_to_topic(topic, node_id, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify no duplicates
            key = f"onex/topics/{topic}/subscribers"
            subscribers = json.loads(kv_store[key].decode())
            assert subscribers.count(node_id) == 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers_to_same_topic(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test multiple nodes can subscribe to the same topic."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            topic = "onex.evt.shared-topic.v1"
            node_ids = ["node-001", "node-002", "node-003"]

            for node_id in node_ids:
                await handler._add_subscriber_to_topic(topic, node_id, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify all subscribers are present
            key = f"onex/topics/{topic}/subscribers"
            subscribers = json.loads(kv_store[key].decode())
            assert sorted(subscribers) == sorted(node_ids)

    @pytest.mark.asyncio
    async def test_remove_subscriber_from_topic(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test removing a subscriber from a topic."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            topic = "onex.evt.test-topic.v1"
            node_id = "subscriber-to-remove"

            # Add then remove
            await handler._add_subscriber_to_topic(topic, node_id, correlation_id)
            await handler._remove_subscriber_from_topic(topic, node_id, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify subscriber was removed
            key = f"onex/topics/{topic}/subscribers"
            subscribers = json.loads(kv_store[key].decode())
            assert node_id not in subscribers


class TestTopicDeltaUpdates:
    """Test delta computation and index updates on contract changes."""

    @pytest.mark.asyncio
    async def test_update_topic_index_adds_new_topics(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test that new topics are added to the index."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "delta-test-node"

            # Create event bus config with new topics
            event_bus = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
                ],
            )

            await handler._update_topic_index(node_id, event_bus, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify node was added to both topic indexes
            subs_a = json.loads(
                kv_store["onex/topics/onex.evt.topic-a.v1/subscribers"].decode()
            )
            subs_b = json.loads(
                kv_store["onex/topics/onex.evt.topic-b.v1/subscribers"].decode()
            )

            assert node_id in subs_a
            assert node_id in subs_b

    @pytest.mark.asyncio
    async def test_topic_removal_on_contract_change(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test that old topics are removed from index when contract changes."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "contract-change-node"

            # First registration with topic-a and topic-b
            initial_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
                ],
            )

            # First: update topic index (reads empty old, creates indexes for a and b)
            await handler._update_topic_index(node_id, initial_config, correlation_id)
            # Then: store config (so next update_topic_index can read old topics)
            await handler._store_node_event_bus(node_id, initial_config, correlation_id)

            # Now update to only topic-a (removes topic-b)
            updated_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
                    # topic-b removed
                ],
            )

            # Update with new config - this reads old topics and computes delta
            await handler._update_topic_index(node_id, updated_config, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify node still in topic-a
            subs_a = json.loads(
                kv_store["onex/topics/onex.evt.topic-a.v1/subscribers"].decode()
            )
            assert node_id in subs_a

            # Verify node removed from topic-b
            subs_b = json.loads(
                kv_store["onex/topics/onex.evt.topic-b.v1/subscribers"].decode()
            )
            assert node_id not in subs_b


class TestTopicStringsAndEntriesSeparation:
    """Test that topic strings and full entries are stored separately."""

    @pytest.mark.asyncio
    async def test_separate_routing_and_tooling_data(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test that routing data (strings) and tooling data (entries) are separate."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "separation-test-node"

            event_bus = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(
                        topic="onex.evt.test.v1",
                        event_type="ModelTest",
                        description="Test description",
                    ),
                ],
            )

            await handler._store_node_event_bus(node_id, event_bus, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Topic strings should be a simple array
            topics_key = f"onex/nodes/{node_id}/event_bus/subscribe_topics"
            topics = json.loads(kv_store[topics_key].decode())
            assert topics == ["onex.evt.test.v1"]
            assert isinstance(topics[0], str)  # Just strings, not objects

            # Full entries should contain all metadata
            entries_key = f"onex/nodes/{node_id}/event_bus/subscribe_entries"
            entries = json.loads(kv_store[entries_key].decode())
            assert len(entries) == 1
            assert entries[0]["topic"] == "onex.evt.test.v1"
            assert entries[0]["event_type"] == "ModelTest"
            assert entries[0]["description"] == "Test description"


class TestServiceRegistrationWithEventBus:
    """Test service registration with event_bus_config integration."""

    @pytest.mark.asyncio
    async def test_register_with_event_bus_config(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test that service registration with event_bus_config stores topics."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "test-service",
                    "service_id": "test-service-001",
                    "node_id": "integration-test-node",
                    "event_bus_config": {
                        "subscribe_topics": [
                            {
                                "topic": "onex.evt.integration.v1",
                                "event_type": "ModelIntegrationEvent",
                            }
                        ],
                        "publish_topics": [
                            {
                                "topic": "onex.evt.output.v1",
                            }
                        ],
                    },
                },
                "correlation_id": uuid4(),
            }

            output = await handler.execute(envelope)
            result = output.result

            # Service registration should succeed
            assert result.status == "success"
            assert result.payload.data.registered is True

            kv_store = mock_consul_client._test_kv_store

            # Event bus config should be stored
            node_id = "integration-test-node"
            assert f"onex/nodes/{node_id}/event_bus/subscribe_topics" in kv_store

            # Topic index should be updated
            assert "onex/topics/onex.evt.integration.v1/subscribers" in kv_store
            subscribers = json.loads(
                kv_store["onex/topics/onex.evt.integration.v1/subscribers"].decode()
            )
            assert node_id in subscribers

    @pytest.mark.asyncio
    async def test_register_without_event_bus_config(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test that service registration without event_bus_config works normally."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "simple-service",
                    "service_id": "simple-service-001",
                },
                "correlation_id": uuid4(),
            }

            output = await handler.execute(envelope)
            result = output.result

            # Service registration should succeed
            assert result.status == "success"
            assert result.payload.data.registered is True

            kv_store = mock_consul_client._test_kv_store

            # No event bus keys should be created
            event_bus_keys = [k for k in kv_store if "event_bus" in k]
            assert len(event_bus_keys) == 0


class TestUpdateTopicIndexSideEffects:
    """Tests verifying which topics _add_subscriber_to_topic and
    _remove_subscriber_from_topic are called with inside _update_topic_index.

    These tests replace the deleted test_topic_index_delta_return.py coverage
    for side-effect call verification.
    """

    @pytest.mark.asyncio
    async def test_update_topic_index_calls_add_for_new_topics(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When there are no previously stored topics, _add_subscriber_to_topic
        is called for every topic in the new config and
        _remove_subscriber_from_topic is not called."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "side-effect-node-001"

            event_bus = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
                ],
            )

            add_calls: list[tuple[str, str]] = []
            remove_calls: list[tuple[str, str]] = []

            # Safe to capture: each test creates a fresh handler instance, so
            # original_add and original_remove are always valid for this test's lifetime.
            original_add = handler._add_subscriber_to_topic
            original_remove = handler._remove_subscriber_from_topic

            async def spy_add(topic: str, nid: str, cid: object) -> None:
                add_calls.append((topic, nid))
                await original_add(topic, nid, cid)

            async def spy_remove(topic: str, nid: str, cid: object) -> None:
                remove_calls.append((topic, nid))
                await original_remove(topic, nid, cid)

            handler._add_subscriber_to_topic = spy_add  # type: ignore[method-assign]
            handler._remove_subscriber_from_topic = spy_remove  # type: ignore[method-assign]

            await handler._update_topic_index(node_id, event_bus, correlation_id)

            # Both new topics should have been added
            added_topics = {t for t, _ in add_calls}
            assert "onex.evt.topic-a.v1" in added_topics
            assert "onex.evt.topic-b.v1" in added_topics

            # All add calls should reference the correct node_id
            for _, nid in add_calls:
                assert nid == node_id

            # No removals when there were no previously registered topics
            assert remove_calls == []

    @pytest.mark.asyncio
    async def test_update_topic_index_calls_remove_for_dropped_topics(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When topics are removed from the config, _remove_subscriber_from_topic
        is called for each dropped topic and _add_subscriber_to_topic is only
        called for genuinely new topics."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "side-effect-node-002"

            # Seed the KV store with an existing subscription to topic-a and topic-b
            initial_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
                ],
            )
            await handler._update_topic_index(node_id, initial_config, correlation_id)
            # Store the config so next _update_topic_index reads the old topics
            await handler._store_node_event_bus(node_id, initial_config, correlation_id)

            # Now update to only topic-c (removes a+b, adds c)
            updated_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-c.v1"),
                ],
            )

            add_calls: list[tuple[str, str]] = []
            remove_calls: list[tuple[str, str]] = []

            # Safe to capture: each test creates a fresh handler instance, so
            # original_add and original_remove are always valid for this test's lifetime.
            original_add = handler._add_subscriber_to_topic
            original_remove = handler._remove_subscriber_from_topic

            async def spy_add(topic: str, nid: str, cid: object) -> None:
                add_calls.append((topic, nid))
                await original_add(topic, nid, cid)

            async def spy_remove(topic: str, nid: str, cid: object) -> None:
                remove_calls.append((topic, nid))
                await original_remove(topic, nid, cid)

            handler._add_subscriber_to_topic = spy_add  # type: ignore[method-assign]
            handler._remove_subscriber_from_topic = spy_remove  # type: ignore[method-assign]

            await handler._update_topic_index(node_id, updated_config, correlation_id)

            # topic-c is new so it should be added
            added_topics = {t for t, _ in add_calls}
            assert "onex.evt.topic-c.v1" in added_topics

            # topic-a and topic-b were dropped so they should be removed
            removed_topics = {t for t, _ in remove_calls}
            assert "onex.evt.topic-a.v1" in removed_topics
            assert "onex.evt.topic-b.v1" in removed_topics

            # topic-c was not previously registered so it must not appear in removes
            assert "onex.evt.topic-c.v1" not in removed_topics

            # All calls reference the correct node_id
            for _, nid in add_calls + remove_calls:
                assert nid == node_id

    @pytest.mark.asyncio
    async def test_update_topic_index_no_calls_when_topics_unchanged(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When the topic set is identical to the previous registration,
        neither _add_subscriber_to_topic nor _remove_subscriber_from_topic
        should be called (empty delta)."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "side-effect-node-003"

            config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.stable.v1"),
                ],
            )

            # Initial registration
            await handler._update_topic_index(node_id, config, correlation_id)
            await handler._store_node_event_bus(node_id, config, correlation_id)

            # Second call with identical config
            add_mock = AsyncMock()
            remove_mock = AsyncMock()
            handler._add_subscriber_to_topic = add_mock  # type: ignore[method-assign]
            handler._remove_subscriber_from_topic = remove_mock  # type: ignore[method-assign]

            await handler._update_topic_index(node_id, config, correlation_id)

            # Delta is empty — neither add nor remove should be called
            add_mock.assert_not_called()
            remove_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_topic_index_propagates_remove_error(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Partial failure: _add_subscriber_to_topic succeeds but
        _remove_subscriber_from_topic raises InfraConsulError.

        Verifies:
        1. The InfraConsulError from _remove_subscriber_from_topic propagates
           out of _update_topic_index unchanged.
        2. _add_subscriber_to_topic was called with the expected new topics
           before the failure occurred.
        """
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "partial-failure-node-001"

            # Seed the KV store: node previously subscribed to topic-a and topic-b
            initial_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
                ],
            )
            await handler._update_topic_index(node_id, initial_config, correlation_id)
            await handler._store_node_event_bus(node_id, initial_config, correlation_id)

            # New config: topic-c is new (to add), topic-a and topic-b are dropped (to remove)
            updated_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="onex.evt.topic-c.v1"),
                ],
            )

            add_calls: list[tuple[str, str]] = []

            # Safe to capture: each test creates a fresh handler instance, so
            # original_add is always valid for this test's lifetime.
            original_add = handler._add_subscriber_to_topic

            async def spy_add(topic: str, nid: str, cid: object) -> None:
                add_calls.append((topic, nid))
                await original_add(topic, nid, cid)

            # _remove_subscriber_from_topic always raises InfraConsulError
            remove_error = InfraConsulError(
                "Simulated KV write failure during remove",
                context=ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put_raw",
                    target_name="consul_handler",
                ),
                consul_key="onex/topics/onex.evt.topic-a.v1/subscribers",
            )

            handler._add_subscriber_to_topic = spy_add  # type: ignore[method-assign]
            handler._remove_subscriber_from_topic = AsyncMock(side_effect=remove_error)  # type: ignore[method-assign]

            # The InfraConsulError from the remove step must propagate out
            with pytest.raises(InfraConsulError):
                await handler._update_topic_index(
                    node_id, updated_config, correlation_id
                )

            # _add_subscriber_to_topic must have been called for the new topic
            # (topic-c) before the remove step raised
            added_topics = {t for t, _ in add_calls}
            assert "onex.evt.topic-c.v1" in added_topics

            # All add calls must reference the correct node_id
            for _, nid in add_calls:
                assert nid == node_id


class TestGetTopicSubscribers:
    """Test retrieving subscribers for a topic."""

    @pytest.mark.asyncio
    async def test_get_topic_subscribers(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test retrieving all subscribers for a topic."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            topic = "onex.evt.lookup-test.v1"
            node_ids = ["lookup-node-001", "lookup-node-002"]

            # Add subscribers
            for node_id in node_ids:
                await handler._add_subscriber_to_topic(topic, node_id, correlation_id)

            # Retrieve subscribers
            subscribers = await handler._get_topic_subscribers(topic, correlation_id)

            assert sorted(subscribers) == sorted(node_ids)

    @pytest.mark.asyncio
    async def test_get_subscribers_for_nonexistent_topic(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """Test retrieving subscribers for a topic with no subscribers."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            topic = "onex.evt.nonexistent-topic.v1"

            subscribers = await handler._get_topic_subscribers(topic, correlation_id)

            assert subscribers == []


class TestRegisterServicePartialFailureWrapping:
    """Test that _register_service wraps KV-write errors as InfraConsulError.

    Exercises the try/except block in _register_service (mixin_consul_service.py)
    that catches transport-level errors from _update_topic_index or
    _store_node_event_bus and re-raises them as InfraConsulError with the
    registration-level context (node_id, service name, operation).

    All four error types in the catch tuple are exercised across this test
    class: InfraConsulError, InfraTimeoutError, InfraUnavailableError, and
    InfraConnectionError.
    """

    @pytest.mark.asyncio
    async def test_update_topic_index_failure_reraised_as_infra_consul_error(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When _update_topic_index raises InfraConsulError, _register_service
        lets it propagate as-is (no double-wrapping).

        InfraConsulError is intentionally excluded from the catch tuple in
        _register_service so that a consul-level KV error is not re-wrapped in
        another InfraConsulError (which would produce a doubly-chained error).
        The exact original error must reach callers unchanged.
        """
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            inner_error = InfraConsulError(
                "Simulated KV index write failure",
                context=ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put_raw",
                    target_name="consul_handler",
                ),
                consul_key="onex/topics/onex.evt.test.v1/subscribers",
            )

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "partial-fail-service",
                    "service_id": "partial-fail-service-001",
                    "node_id": "partial-fail-node",
                    "event_bus_config": {
                        "subscribe_topics": [
                            {"topic": "onex.evt.test.v1"},
                        ],
                    },
                },
                "correlation_id": correlation_id,
            }

            with patch.object(handler, "_update_topic_index", side_effect=inner_error):
                with pytest.raises(InfraConsulError) as exc_info:
                    await handler.execute(envelope)

            raised = exc_info.value
            # The original error propagates unchanged — not wrapped in a new
            # InfraConsulError — so raised IS inner_error.
            assert raised is inner_error
            # No double-wrapping: __cause__ is None (error was not re-raised from exc).
            assert raised.__cause__ is None

    @pytest.mark.asyncio
    async def test_store_node_event_bus_failure_reraised_as_infra_consul_error(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When _store_node_event_bus raises InfraTimeoutError, _register_service
        wraps it as InfraConsulError with the registration-level context."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            inner_error = InfraTimeoutError(
                "Consul KV put timed out",
                context=ModelTimeoutErrorContext(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put_raw",
                    target_name="consul_handler",
                ),
            )

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "timeout-service",
                    "service_id": "timeout-service-001",
                    "node_id": "timeout-node",
                    "event_bus_config": {
                        "subscribe_topics": [
                            {"topic": "onex.evt.timeout-test.v1"},
                        ],
                    },
                },
                "correlation_id": correlation_id,
            }

            with patch.object(
                handler, "_store_node_event_bus", side_effect=inner_error
            ):
                with pytest.raises(InfraConsulError) as exc_info:
                    await handler.execute(envelope)

            raised = exc_info.value
            assert "timeout-node" in str(raised)
            assert raised.__cause__ is inner_error
            # Assert service_name appears in the public error message string rather
            # than testing internal context serialisation structure.
            assert "timeout-service" in raised.args[0]

    @pytest.mark.asyncio
    async def test_infra_unavailable_error_reraised_as_infra_consul_error(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When _update_topic_index raises InfraUnavailableError, _register_service
        wraps it as InfraConsulError (all four caught error types are exercised
        across this test class)."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            inner_error = InfraUnavailableError(
                "Consul unavailable during KV index update",
                context=ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put_raw",
                    target_name="consul_handler",
                ),
            )

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "unavailable-service",
                    "node_id": "unavailable-node",
                    "event_bus_config": {
                        "publish_topics": [
                            {"topic": "onex.evt.unavailable-test.v1"},
                        ],
                    },
                },
                "correlation_id": correlation_id,
            }

            with patch.object(handler, "_update_topic_index", side_effect=inner_error):
                with pytest.raises(InfraConsulError) as exc_info:
                    await handler.execute(envelope)

            raised = exc_info.value
            assert "unavailable-node" in str(raised)
            assert raised.__cause__ is inner_error

    @pytest.mark.asyncio
    async def test_infra_connection_error_reraised_as_infra_consul_error(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When _store_node_event_bus raises InfraConnectionError, _register_service
        wraps it as InfraConsulError with the registration-level context."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            inner_error = InfraConnectionError(
                "Consul KV connection lost during event bus store",
                context=ModelInfraErrorContext.with_correlation(
                    correlation_id=correlation_id,
                    transport_type=EnumInfraTransportType.CONSUL,
                    operation="consul.kv_put_raw",
                    target_name="consul_handler",
                ),
            )

            envelope = {
                "operation": "consul.register",
                "payload": {
                    "name": "connection-error-service",
                    "node_id": "connection-error-node",
                    "event_bus_config": {
                        "subscribe_topics": [
                            {"topic": "onex.evt.connection-test.v1"},
                        ],
                    },
                },
                "correlation_id": correlation_id,
            }

            with patch.object(
                handler, "_store_node_event_bus", side_effect=inner_error
            ):
                with pytest.raises(InfraConsulError) as exc_info:
                    await handler.execute(envelope)

            raised = exc_info.value
            assert "connection-error-node" in str(raised)
            assert raised.__cause__ is inner_error


__all__: list[str] = [
    "TestStoreEventBusInKV",
    "TestTopicSubscriberIndex",
    "TestTopicDeltaUpdates",
    "TestTopicStringsAndEntriesSeparation",
    "TestServiceRegistrationWithEventBus",
    "TestUpdateTopicIndexSideEffects",
    "TestGetTopicSubscribers",
    "TestRegisterServicePartialFailureWrapping",
]
