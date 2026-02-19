# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelPayloadConsulRegister event_bus_config field.

This module tests the event_bus_config field added to ModelPayloadConsulRegister
as part of OMN-1613 (Phase 4).

Test Coverage:
    - Payload accepts event_bus_config with valid ModelNodeEventBusConfig
    - Payload works correctly when event_bus_config is None
    - JSON serialization includes event_bus_config field
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)
from omnibase_infra.nodes.reducers.models.model_payload_consul_register import (
    ModelPayloadConsulRegister,
)


class TestModelPayloadConsulRegisterEventBus:
    """Tests for event_bus_config field in ModelPayloadConsulRegister."""

    def test_payload_with_event_bus_config(self) -> None:
        """Payload should accept event_bus_config with valid ModelNodeEventBusConfig."""
        # Arrange
        correlation_id = uuid4()
        event_bus_config = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(
                    topic="onex.evt.input.v1",
                    event_type="ModelInputEvent",
                    message_category="EVENT",
                ),
            ],
            publish_topics=[
                ModelEventBusTopicEntry(
                    topic="onex.evt.output.v1",
                    event_type="ModelOutputEvent",
                    message_category="EVENT",
                ),
            ],
        )

        # Act
        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            node_id="test-node-id",
            service_id="onex-effect-test-123",
            service_name="onex-effect",
            tags=["node_type:effect", "node_version:1.0.0"],
            event_bus_config=event_bus_config,
        )

        # Assert
        assert payload.event_bus_config is not None
        assert payload.event_bus_config == event_bus_config
        assert payload.event_bus_config.subscribe_topic_strings == ["onex.evt.input.v1"]
        assert payload.event_bus_config.publish_topic_strings == ["onex.evt.output.v1"]

    def test_payload_without_event_bus_config(self) -> None:
        """Payload should work correctly when event_bus_config is None."""
        # Arrange
        correlation_id = uuid4()

        # Act
        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-compute-test-456",
            service_name="onex-compute",
            tags=["node_type:compute", "node_version:2.0.0"],
            event_bus_config=None,
        )

        # Assert
        assert payload.event_bus_config is None
        assert payload.correlation_id == correlation_id
        assert payload.service_id == "onex-compute-test-456"
        assert payload.service_name == "onex-compute"

    def test_payload_default_event_bus_config_is_none(self) -> None:
        """Payload should default event_bus_config to None if not provided."""
        # Arrange
        correlation_id = uuid4()

        # Act
        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-reducer-test-789",
            service_name="onex-reducer",
            tags=["node_type:reducer"],
        )

        # Assert
        assert payload.event_bus_config is None

    def test_serialization_includes_event_bus(self) -> None:
        """JSON serialization should include event_bus_config field."""
        # Arrange
        correlation_id = uuid4()
        event_bus_config = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(
                    topic="onex.evt.commands.v1",
                    event_type="ModelCommandEvent",
                    message_category="COMMAND",
                    description="Command input topic",
                ),
            ],
            publish_topics=[],
        )

        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            node_id="test-node-id",
            service_id="onex-orchestrator-test",
            service_name="onex-orchestrator",
            tags=["node_type:orchestrator"],
            event_bus_config=event_bus_config,
        )

        # Act
        json_data = payload.model_dump()
        json_str = payload.model_dump_json()

        # Assert
        assert "event_bus_config" in json_data
        assert json_data["event_bus_config"] is not None
        assert "subscribe_topics" in json_data["event_bus_config"]
        assert len(json_data["event_bus_config"]["subscribe_topics"]) == 1
        assert (
            json_data["event_bus_config"]["subscribe_topics"][0]["topic"]
            == "onex.evt.commands.v1"
        )

        # Verify JSON string contains the topic
        assert "onex.evt.commands.v1" in json_str
        assert "event_bus_config" in json_str

    def test_serialization_with_none_event_bus(self) -> None:
        """JSON serialization should include null event_bus_config when None."""
        # Arrange
        correlation_id = uuid4()
        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-test-node",
            service_name="onex-test",
            tags=["test"],
            event_bus_config=None,
        )

        # Act
        json_data = payload.model_dump()

        # Assert
        assert "event_bus_config" in json_data
        assert json_data["event_bus_config"] is None

    def test_payload_with_empty_topic_lists(self) -> None:
        """Payload should accept event_bus_config with empty topic lists."""
        # Arrange
        correlation_id = uuid4()
        event_bus_config = ModelNodeEventBusConfig(
            subscribe_topics=[],
            publish_topics=[],
        )

        # Act
        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            node_id="test-node-id",
            service_id="onex-effect-empty",
            service_name="onex-effect",
            tags=["node_type:effect"],
            event_bus_config=event_bus_config,
        )

        # Assert
        assert payload.event_bus_config is not None
        assert payload.event_bus_config.subscribe_topic_strings == []
        assert payload.event_bus_config.publish_topic_strings == []


class TestModelPayloadConsulRegisterImmutability:
    """Tests for immutability of event_bus_config in frozen model."""

    def test_event_bus_config_is_frozen(self) -> None:
        """Payload should be immutable (frozen)."""
        # Arrange
        correlation_id = uuid4()
        payload = ModelPayloadConsulRegister(
            correlation_id=correlation_id,
            service_id="onex-frozen-test",
            service_name="onex-frozen",
            tags=["test"],
        )

        # Act & Assert
        with pytest.raises(Exception):  # ValidationError for frozen model
            payload.event_bus_config = ModelNodeEventBusConfig()  # type: ignore[misc]
