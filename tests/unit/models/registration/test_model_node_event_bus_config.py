# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelNodeEventBusConfig and ModelEventBusTopicEntry.

Tests validate:
- Topic entry creation and serialization
- Event bus config with empty and populated topic lists
- Property methods for extracting topic strings
- Frozen model immutability
- Model extra="forbid" constraint
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.registration import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)


class TestModelEventBusTopicEntry:
    """Tests for ModelEventBusTopicEntry model."""

    def test_create_topic_entry_with_topic_only(self) -> None:
        """Test creating a topic entry with only the required topic field."""
        entry = ModelEventBusTopicEntry(topic="dev.onex.evt.intent-classified.v1")

        assert entry.topic == "dev.onex.evt.intent-classified.v1"
        assert entry.event_type is None
        assert entry.message_category == "EVENT"
        assert entry.description is None

    def test_create_topic_entry_with_all_fields(self) -> None:
        """Test creating a topic entry with all metadata fields."""
        entry = ModelEventBusTopicEntry(
            topic="dev.onex.cmd.register-node.v1",
            event_type="ModelNodeRegistration",
            message_category="COMMAND",
            description="Node registration command topic",
        )

        assert entry.topic == "dev.onex.cmd.register-node.v1"
        assert entry.event_type == "ModelNodeRegistration"
        assert entry.message_category == "COMMAND"
        assert entry.description == "Node registration command topic"

    def test_topic_entry_serialization(self) -> None:
        """Test that topic entry serializes correctly to dict."""
        entry = ModelEventBusTopicEntry(
            topic="dev.onex.evt.node-registered.v1",
            event_type="ModelNodeRegistered",
        )

        data = entry.model_dump()
        assert data["topic"] == "dev.onex.evt.node-registered.v1"
        assert data["event_type"] == "ModelNodeRegistered"
        assert data["message_category"] == "EVENT"
        assert data["description"] is None

    def test_topic_entry_deserialization(self) -> None:
        """Test that topic entry deserializes correctly from dict."""
        data = {
            "topic": "dev.onex.evt.heartbeat.v1",
            "event_type": "ModelHeartbeat",
            "message_category": "EVENT",
            "description": "Heartbeat event topic",
        }

        entry = ModelEventBusTopicEntry.model_validate(data)
        assert entry.topic == "dev.onex.evt.heartbeat.v1"
        assert entry.event_type == "ModelHeartbeat"
        assert entry.message_category == "EVENT"
        assert entry.description == "Heartbeat event topic"

    def test_topic_entry_is_frozen(self) -> None:
        """Test that topic entry is immutable (frozen)."""
        entry = ModelEventBusTopicEntry(topic="dev.onex.evt.test.v1")

        with pytest.raises(ValidationError):
            entry.topic = "dev.onex.evt.modified.v1"  # type: ignore[misc]

    def test_topic_entry_forbids_extra_fields(self) -> None:
        """Test that topic entry rejects unknown fields (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.test.v1",
                unknown_field="value",  # type: ignore[call-arg]
            )

        assert "extra_forbidden" in str(exc_info.value)


class TestModelNodeEventBusConfig:
    """Tests for ModelNodeEventBusConfig model."""

    def test_create_empty_config(self) -> None:
        """Test creating an event bus config with no topics."""
        config = ModelNodeEventBusConfig()

        assert config.subscribe_topics == []
        assert config.publish_topics == []

    def test_create_config_with_subscribe_topics(self) -> None:
        """Test creating config with subscribe topics only."""
        entries = [
            ModelEventBusTopicEntry(topic="dev.onex.evt.intent-classified.v1"),
            ModelEventBusTopicEntry(topic="dev.onex.evt.node-registered.v1"),
        ]

        config = ModelNodeEventBusConfig(subscribe_topics=entries)

        assert len(config.subscribe_topics) == 2
        assert config.subscribe_topics[0].topic == "dev.onex.evt.intent-classified.v1"
        assert config.subscribe_topics[1].topic == "dev.onex.evt.node-registered.v1"
        assert config.publish_topics == []

    def test_create_config_with_publish_topics(self) -> None:
        """Test creating config with publish topics only."""
        entries = [
            ModelEventBusTopicEntry(
                topic="dev.onex.cmd.execute-effect.v1",
                message_category="COMMAND",
            ),
        ]

        config = ModelNodeEventBusConfig(publish_topics=entries)

        assert config.subscribe_topics == []
        assert len(config.publish_topics) == 1
        assert config.publish_topics[0].topic == "dev.onex.cmd.execute-effect.v1"

    def test_create_config_with_both_topic_types(self) -> None:
        """Test creating config with both subscribe and publish topics."""
        subscribe_entries = [
            ModelEventBusTopicEntry(topic="dev.onex.evt.input.v1"),
        ]
        publish_entries = [
            ModelEventBusTopicEntry(topic="dev.onex.evt.output.v1"),
            ModelEventBusTopicEntry(topic="dev.onex.cmd.notify.v1"),
        ]

        config = ModelNodeEventBusConfig(
            subscribe_topics=subscribe_entries,
            publish_topics=publish_entries,
        )

        assert len(config.subscribe_topics) == 1
        assert len(config.publish_topics) == 2


class TestModelNodeEventBusConfigProperties:
    """Tests for ModelNodeEventBusConfig property methods."""

    def test_subscribe_topic_strings_empty(self) -> None:
        """Test subscribe_topic_strings returns empty list for empty config."""
        config = ModelNodeEventBusConfig()

        assert config.subscribe_topic_strings == []

    def test_subscribe_topic_strings_returns_topic_list(self) -> None:
        """Test subscribe_topic_strings returns list of topic strings."""
        entries = [
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.alpha.v1",
                event_type="ModelAlpha",
                description="Alpha events",
            ),
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.beta.v1",
                event_type="ModelBeta",
            ),
            ModelEventBusTopicEntry(topic="dev.onex.evt.gamma.v1"),
        ]

        config = ModelNodeEventBusConfig(subscribe_topics=entries)

        result = config.subscribe_topic_strings
        assert result == [
            "dev.onex.evt.alpha.v1",
            "dev.onex.evt.beta.v1",
            "dev.onex.evt.gamma.v1",
        ]

    def test_publish_topic_strings_empty(self) -> None:
        """Test publish_topic_strings returns empty list for empty config."""
        config = ModelNodeEventBusConfig()

        assert config.publish_topic_strings == []

    def test_publish_topic_strings_returns_topic_list(self) -> None:
        """Test publish_topic_strings returns list of topic strings."""
        entries = [
            ModelEventBusTopicEntry(
                topic="dev.onex.cmd.action-one.v1",
                message_category="COMMAND",
            ),
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.result-one.v1",
                message_category="EVENT",
            ),
        ]

        config = ModelNodeEventBusConfig(publish_topics=entries)

        result = config.publish_topic_strings
        assert result == [
            "dev.onex.cmd.action-one.v1",
            "dev.onex.evt.result-one.v1",
        ]

    def test_topic_strings_ignore_metadata(self) -> None:
        """Test that topic string properties ignore metadata fields."""
        # Create entries with various metadata
        entries = [
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.first.v1",
                event_type="ModelFirst",
                message_category="EVENT",
                description="First event with full metadata",
            ),
            ModelEventBusTopicEntry(
                topic="dev.onex.evt.second.v1",
                # Minimal metadata
            ),
        ]

        config = ModelNodeEventBusConfig(
            subscribe_topics=entries,
            publish_topics=entries,
        )

        # Properties return only topic strings, not metadata
        assert config.subscribe_topic_strings == [
            "dev.onex.evt.first.v1",
            "dev.onex.evt.second.v1",
        ]
        assert config.publish_topic_strings == [
            "dev.onex.evt.first.v1",
            "dev.onex.evt.second.v1",
        ]


class TestModelNodeEventBusConfigImmutability:
    """Tests for ModelNodeEventBusConfig immutability (frozen model)."""

    def test_config_is_frozen(self) -> None:
        """Test that event bus config is immutable (frozen)."""
        config = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(topic="dev.onex.evt.test.v1"),
            ],
        )

        with pytest.raises(ValidationError):
            config.subscribe_topics = []  # type: ignore[misc]

    def test_config_forbids_extra_fields(self) -> None:
        """Test that event bus config rejects unknown fields (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeEventBusConfig(
                subscribe_topics=[],
                unknown_field="value",  # type: ignore[call-arg]
            )

        assert "extra_forbidden" in str(exc_info.value)


class TestModelNodeEventBusConfigSerialization:
    """Tests for ModelNodeEventBusConfig serialization/deserialization."""

    def test_serialize_empty_config(self) -> None:
        """Test serializing empty config to dict."""
        config = ModelNodeEventBusConfig()

        data = config.model_dump()
        assert data == {
            "subscribe_topics": [],
            "publish_topics": [],
        }

    def test_serialize_populated_config(self) -> None:
        """Test serializing populated config to dict."""
        config = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(
                    topic="dev.onex.evt.in.v1",
                    event_type="ModelInput",
                ),
            ],
            publish_topics=[
                ModelEventBusTopicEntry(
                    topic="dev.onex.evt.out.v1",
                    message_category="EVENT",
                    description="Output topic",
                ),
            ],
        )

        data = config.model_dump()

        assert len(data["subscribe_topics"]) == 1
        assert data["subscribe_topics"][0]["topic"] == "dev.onex.evt.in.v1"
        assert data["subscribe_topics"][0]["event_type"] == "ModelInput"

        assert len(data["publish_topics"]) == 1
        assert data["publish_topics"][0]["topic"] == "dev.onex.evt.out.v1"
        assert data["publish_topics"][0]["description"] == "Output topic"

    def test_deserialize_from_dict(self) -> None:
        """Test deserializing config from dict."""
        data = {
            "subscribe_topics": [
                {
                    "topic": "dev.onex.evt.incoming.v1",
                    "event_type": "ModelIncoming",
                    "message_category": "EVENT",
                    "description": None,
                },
            ],
            "publish_topics": [
                {
                    "topic": "dev.onex.cmd.outgoing.v1",
                    "event_type": None,
                    "message_category": "COMMAND",
                    "description": "Outgoing command",
                },
            ],
        }

        config = ModelNodeEventBusConfig.model_validate(data)

        assert len(config.subscribe_topics) == 1
        assert config.subscribe_topics[0].topic == "dev.onex.evt.incoming.v1"
        assert config.subscribe_topics[0].event_type == "ModelIncoming"

        assert len(config.publish_topics) == 1
        assert config.publish_topics[0].topic == "dev.onex.cmd.outgoing.v1"
        assert config.publish_topics[0].message_category == "COMMAND"

    def test_roundtrip_serialization(self) -> None:
        """Test that config survives roundtrip serialization."""
        original = ModelNodeEventBusConfig(
            subscribe_topics=[
                ModelEventBusTopicEntry(
                    topic="dev.onex.evt.alpha.v1",
                    event_type="ModelAlpha",
                    message_category="EVENT",
                    description="Alpha topic",
                ),
            ],
            publish_topics=[
                ModelEventBusTopicEntry(
                    topic="dev.onex.cmd.beta.v1",
                    event_type="ModelBeta",
                    message_category="COMMAND",
                    description="Beta command",
                ),
            ],
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelNodeEventBusConfig.model_validate(data)

        # Verify equality
        assert restored.subscribe_topic_strings == original.subscribe_topic_strings
        assert restored.publish_topic_strings == original.publish_topic_strings
        assert (
            restored.subscribe_topics[0].event_type
            == original.subscribe_topics[0].event_type
        )
        assert (
            restored.publish_topics[0].description
            == original.publish_topics[0].description
        )
