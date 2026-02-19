# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for OMN-2314: _update_topic_index() delta return value.

Tests verify that _update_topic_index() returns (topics_added, topics_removed)
as frozensets of topic suffix strings and that _register_service() captures the
delta into ModelConsulRegisterPayload.

Related:
    - OMN-2314: Topic Catalog change notification emission + CAS versioning
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.handler_consul import HandlerConsul
from omnibase_infra.models.registration import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)

pytestmark = [pytest.mark.unit]


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
def mock_consul_client() -> MagicMock:
    """Provide mocked consul.Consul client with KV store simulation."""
    client = MagicMock()

    # Simulated KV store
    kv_store: dict[str, bytes] = {}

    def kv_get(
        key: str, recurse: bool = False
    ) -> tuple[int, dict[str, object] | None]:
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

    client.agent = MagicMock()
    client.agent.service = MagicMock()
    client.agent.service.register = MagicMock(return_value=None)

    # Expose the local kv_store dict as a plain attribute so tests can pre-populate
    # it directly without relying on a private underscore attribute on MagicMock.
    client.test_kv_store = kv_store

    return client


@pytest.fixture
def event_bus_config_a() -> ModelNodeEventBusConfig:
    """Config with topics A and B."""
    return ModelNodeEventBusConfig(
        subscribe_topics=[
            ModelEventBusTopicEntry(topic="onex.evt.topic-a.v1"),
            ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
        ],
        publish_topics=[],
    )


@pytest.fixture
def event_bus_config_b() -> ModelNodeEventBusConfig:
    """Config with topics B and C (A removed, C added)."""
    return ModelNodeEventBusConfig(
        subscribe_topics=[
            ModelEventBusTopicEntry(topic="onex.evt.topic-b.v1"),
            ModelEventBusTopicEntry(topic="onex.evt.topic-c.v1"),
        ],
        publish_topics=[],
    )


class TestUpdateTopicIndexDeltaReturn:
    """Tests for the delta return value of _update_topic_index()."""

    @pytest.mark.asyncio
    async def test_fresh_node_all_topics_added(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        event_bus_config_a: ModelNodeEventBusConfig,
    ) -> None:
        """When no previous registration exists, all topics are in topics_added."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "test-node-fresh"

            topics_added, topics_removed = await handler._update_topic_index(
                node_id, event_bus_config_a, correlation_id
            )

        assert isinstance(topics_added, frozenset)
        assert isinstance(topics_removed, frozenset)
        assert "onex.evt.topic-a.v1" in topics_added
        assert "onex.evt.topic-b.v1" in topics_added
        assert len(topics_removed) == 0

    @pytest.mark.asyncio
    async def test_no_change_empty_delta(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        event_bus_config_a: ModelNodeEventBusConfig,
    ) -> None:
        """When topics are unchanged, both delta sets are empty."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "test-node-same"

            # Pre-populate KV with same topics
            existing_topics = ["onex.evt.topic-a.v1", "onex.evt.topic-b.v1"]
            kv_store = mock_consul_client.test_kv_store
            kv_store[f"onex/nodes/{node_id}/event_bus/subscribe_topics"] = (
                json.dumps(existing_topics).encode("utf-8")
            )

            topics_added, topics_removed = await handler._update_topic_index(
                node_id, event_bus_config_a, correlation_id
            )

        assert len(topics_added) == 0
        assert len(topics_removed) == 0

    @pytest.mark.asyncio
    async def test_topic_change_correct_delta(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        event_bus_config_b: ModelNodeEventBusConfig,
    ) -> None:
        """When topics change from A,B to B,C: added={C}, removed={A}."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            node_id = "test-node-change"

            # Pre-populate with A and B
            kv_store = mock_consul_client.test_kv_store
            kv_store[f"onex/nodes/{node_id}/event_bus/subscribe_topics"] = (
                json.dumps(
                    ["onex.evt.topic-a.v1", "onex.evt.topic-b.v1"]
                ).encode("utf-8")
            )

            # Update to B and C
            topics_added, topics_removed = await handler._update_topic_index(
                node_id, event_bus_config_b, correlation_id
            )

        assert topics_added == frozenset(["onex.evt.topic-c.v1"])
        assert topics_removed == frozenset(["onex.evt.topic-a.v1"])

    @pytest.mark.asyncio
    async def test_return_type_is_frozenset(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        event_bus_config_a: ModelNodeEventBusConfig,
    ) -> None:
        """Return type must be tuple[frozenset[str], frozenset[str]]."""
        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            result = await handler._update_topic_index(
                "any-node", event_bus_config_a, uuid4()
            )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], frozenset)
        assert isinstance(result[1], frozenset)


class TestRegisterServiceDeltaPropagation:
    """Tests that _register_service() propagates the topic delta into the result."""

    @pytest.mark.asyncio
    async def test_register_with_event_bus_returns_delta_in_payload(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
        event_bus_config_a: ModelNodeEventBusConfig,
    ) -> None:
        """ModelConsulRegisterPayload should contain topics_added when topics change."""
        from omnibase_infra.handlers.models.consul.model_consul_register_payload import (
            ModelConsulRegisterPayload,
        )

        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            correlation_id = uuid4()
            envelope_id = uuid4()

            # Build the payload dict as IntentEffectConsulRegister does
            register_payload: dict[str, object] = {
                "name": "onex-orchestrator",
                "service_id": "onex-orchestrator-node1",
                "tags": ["onex"],
                "node_id": "node1",
                "event_bus_config": event_bus_config_a.model_dump(),
            }

            output = await handler._register_service(
                register_payload, correlation_id, envelope_id
            )

        # Drill into the nested payload
        assert output is not None
        response = output.result
        assert response is not None
        data = response.payload.data
        assert isinstance(data, ModelConsulRegisterPayload)
        # Fresh node - all topics added
        assert "onex.evt.topic-a.v1" in data.topics_added
        assert "onex.evt.topic-b.v1" in data.topics_added
        assert len(data.topics_removed) == 0

    @pytest.mark.asyncio
    async def test_register_without_event_bus_empty_delta(
        self,
        consul_config: dict[str, object],
        mock_consul_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When no event_bus_config, topics_added and topics_removed are both empty."""
        from omnibase_infra.handlers.models.consul.model_consul_register_payload import (
            ModelConsulRegisterPayload,
        )

        handler = HandlerConsul(mock_container)

        with patch(
            "omnibase_infra.handlers.handler_consul.consul.Consul"
        ) as MockClient:
            MockClient.return_value = mock_consul_client
            await handler.initialize(consul_config)

            register_payload: dict[str, object] = {
                "name": "onex-orchestrator",
                "service_id": "onex-orchestrator-node1",
                "tags": ["onex"],
            }

            output = await handler._register_service(
                register_payload, uuid4(), uuid4()
            )

        response = output.result
        assert response is not None
        data = response.payload.data
        assert isinstance(data, ModelConsulRegisterPayload)
        assert len(data.topics_added) == 0
        assert len(data.topics_removed) == 0
