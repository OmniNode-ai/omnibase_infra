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
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
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
                topic="dev.onex.evt.intent-classified.v1",
                event_type="ModelIntentClassified",
                message_category="EVENT",
                description="Intent classification events",
            ),
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.node-registered.v1",
                event_type="ModelNodeRegistered",
                message_category="EVENT",
            ),
        ],
        publish_topics=[
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.node-processed.v1",
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
                "dev.onex.evt.intent-classified.v1",
                "dev.onex.evt.node-registered.v1",
            ]

            # Verify publish topics
            publish_topics = json.loads(
                kv_store[f"onex/nodes/{node_id}/event_bus/publish_topics"].decode()
            )
            assert publish_topics == ["dev.onex.evt.node-processed.v1"]

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
            assert subscribe_entries[0]["topic"] == "dev.onex.evt.intent-classified.v1"
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
            topic = "dev.onex.evt.test-topic.v1"
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
            topic = "dev.onex.evt.test-topic.v1"
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
            topic = "dev.onex.evt.shared-topic.v1"
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
            topic = "dev.onex.evt.test-topic.v1"
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
                    ModelEventBusTopicEntry(topic="dev.onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="dev.onex.evt.topic-b.v1"),
                ],
            )

            await handler._update_topic_index(node_id, event_bus, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify node was added to both topic indexes
            subs_a = json.loads(
                kv_store["onex/topics/dev.onex.evt.topic-a.v1/subscribers"].decode()
            )
            subs_b = json.loads(
                kv_store["onex/topics/dev.onex.evt.topic-b.v1/subscribers"].decode()
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
                    ModelEventBusTopicEntry(topic="dev.onex.evt.topic-a.v1"),
                    ModelEventBusTopicEntry(topic="dev.onex.evt.topic-b.v1"),
                ],
            )

            # First: update topic index (reads empty old, creates indexes for a and b)
            await handler._update_topic_index(node_id, initial_config, correlation_id)
            # Then: store config (so next update_topic_index can read old topics)
            await handler._store_node_event_bus(node_id, initial_config, correlation_id)

            # Now update to only topic-a (removes topic-b)
            updated_config = ModelNodeEventBusConfig(
                subscribe_topics=[
                    ModelEventBusTopicEntry(topic="dev.onex.evt.topic-a.v1"),
                    # topic-b removed
                ],
            )

            # Update with new config - this reads old topics and computes delta
            await handler._update_topic_index(node_id, updated_config, correlation_id)

            kv_store = mock_consul_client._test_kv_store

            # Verify node still in topic-a
            subs_a = json.loads(
                kv_store["onex/topics/dev.onex.evt.topic-a.v1/subscribers"].decode()
            )
            assert node_id in subs_a

            # Verify node removed from topic-b
            subs_b = json.loads(
                kv_store["onex/topics/dev.onex.evt.topic-b.v1/subscribers"].decode()
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
                        topic="dev.onex.evt.test.v1",
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
            assert topics == ["dev.onex.evt.test.v1"]
            assert isinstance(topics[0], str)  # Just strings, not objects

            # Full entries should contain all metadata
            entries_key = f"onex/nodes/{node_id}/event_bus/subscribe_entries"
            entries = json.loads(kv_store[entries_key].decode())
            assert len(entries) == 1
            assert entries[0]["topic"] == "dev.onex.evt.test.v1"
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
                                "topic": "dev.onex.evt.integration.v1",
                                "event_type": "ModelIntegrationEvent",
                            }
                        ],
                        "publish_topics": [
                            {
                                "topic": "dev.onex.evt.output.v1",
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
            assert "onex/topics/dev.onex.evt.integration.v1/subscribers" in kv_store
            subscribers = json.loads(
                kv_store["onex/topics/dev.onex.evt.integration.v1/subscribers"].decode()
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
            topic = "dev.onex.evt.lookup-test.v1"
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
            topic = "dev.onex.evt.nonexistent-topic.v1"

            subscribers = await handler._get_topic_subscribers(topic, correlation_id)

            assert subscribers == []


__all__: list[str] = [
    "TestStoreEventBusInKV",
    "TestTopicSubscriberIndex",
    "TestTopicDeltaUpdates",
    "TestTopicStringsAndEntriesSeparation",
    "TestServiceRegistrationWithEventBus",
    "TestGetTopicSubscribers",
]
