# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for DLQ topic naming constants and utilities.

These tests verify that the DLQ topic naming conventions follow ONEX standards
and that the utility functions work correctly.
"""

from __future__ import annotations

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.topic_constants import (
    DLQ_CATEGORY_SUFFIXES,
    DLQ_COMMAND_TOPIC_SUFFIX,
    DLQ_DOMAIN,
    DLQ_EVENT_TOPIC_SUFFIX,
    DLQ_INTENT_TOPIC_SUFFIX,
    DLQ_TOPIC_PATTERN,
    DLQ_TOPIC_VERSION,
    build_dlq_topic,
    get_dlq_topic_for_original,
    is_dlq_topic,
    parse_dlq_topic,
)


class TestDLQTopicConstants:
    """Test DLQ topic constant values."""

    def test_dlq_topic_version(self) -> None:
        """Verify DLQ topic version follows semantic pattern."""
        assert DLQ_TOPIC_VERSION == "v1"
        assert DLQ_TOPIC_VERSION.startswith("v")

    def test_dlq_domain(self) -> None:
        """Verify DLQ domain is 'dlq'."""
        assert DLQ_DOMAIN == "dlq"

    def test_dlq_intent_topic_suffix(self) -> None:
        """Verify intent DLQ topic suffix format."""
        assert DLQ_INTENT_TOPIC_SUFFIX == "dlq.intents.v1"
        assert "intents" in DLQ_INTENT_TOPIC_SUFFIX
        assert DLQ_DOMAIN in DLQ_INTENT_TOPIC_SUFFIX

    def test_dlq_event_topic_suffix(self) -> None:
        """Verify event DLQ topic suffix format."""
        assert DLQ_EVENT_TOPIC_SUFFIX == "dlq.events.v1"
        assert "events" in DLQ_EVENT_TOPIC_SUFFIX

    def test_dlq_command_topic_suffix(self) -> None:
        """Verify command DLQ topic suffix format."""
        assert DLQ_COMMAND_TOPIC_SUFFIX == "dlq.commands.v1"
        assert "commands" in DLQ_COMMAND_TOPIC_SUFFIX

    def test_category_suffixes_mapping(self) -> None:
        """Verify category to suffix mapping supports both forms."""
        # Singular forms
        assert DLQ_CATEGORY_SUFFIXES["intent"] == DLQ_INTENT_TOPIC_SUFFIX
        assert DLQ_CATEGORY_SUFFIXES["event"] == DLQ_EVENT_TOPIC_SUFFIX
        assert DLQ_CATEGORY_SUFFIXES["command"] == DLQ_COMMAND_TOPIC_SUFFIX

        # Plural forms
        assert DLQ_CATEGORY_SUFFIXES["intents"] == DLQ_INTENT_TOPIC_SUFFIX
        assert DLQ_CATEGORY_SUFFIXES["events"] == DLQ_EVENT_TOPIC_SUFFIX
        assert DLQ_CATEGORY_SUFFIXES["commands"] == DLQ_COMMAND_TOPIC_SUFFIX


class TestBuildDLQTopic:
    """Test build_dlq_topic function."""

    def test_build_intent_dlq_topic(self) -> None:
        """Build DLQ topic for intents (realm-agnostic, OMN-5189)."""
        topic = build_dlq_topic("intents")
        assert topic == "onex.dlq.intents.v1"

    def test_build_event_dlq_topic(self) -> None:
        """Build DLQ topic for events."""
        topic = build_dlq_topic("events")
        assert topic == "onex.dlq.events.v1"

    def test_build_command_dlq_topic(self) -> None:
        """Build DLQ topic for commands."""
        topic = build_dlq_topic("commands")
        assert topic == "onex.dlq.commands.v1"

    def test_singular_category_normalized_to_plural(self) -> None:
        """Singular category forms are normalized to plural."""
        assert build_dlq_topic("intent") == "onex.dlq.intents.v1"
        assert build_dlq_topic("event") == "onex.dlq.events.v1"
        assert build_dlq_topic("command") == "onex.dlq.commands.v1"

    def test_custom_version(self) -> None:
        """Build DLQ topic with custom version."""
        topic = build_dlq_topic("intents", version="v2")
        assert topic == "onex.dlq.intents.v2"

    def test_empty_category_raises(self) -> None:
        """Empty category raises ProtocolConfigurationError."""
        with pytest.raises(
            ProtocolConfigurationError, match="category cannot be empty"
        ):
            build_dlq_topic("")

    def test_invalid_category_format_raises(self) -> None:
        """Category starting with digit raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="Invalid category"):
            build_dlq_topic("123abc")

    def test_domain_based_category_accepted(self) -> None:
        """Domain-based categories like 'intelligence' and 'platform' are accepted."""
        assert build_dlq_topic("intelligence") == "onex.dlq.intelligence.v1"
        assert build_dlq_topic("platform") == "onex.dlq.platform.v1"
        assert build_dlq_topic("agent") == "onex.dlq.agent.v1"

    def test_case_insensitive_category(self) -> None:
        """Category matching is case-insensitive."""
        assert build_dlq_topic("INTENTS") == "onex.dlq.intents.v1"
        assert build_dlq_topic("Events") == "onex.dlq.events.v1"
        assert build_dlq_topic("ComMAnDs") == "onex.dlq.commands.v1"


class TestParseDLQTopic:
    """Test parse_dlq_topic function."""

    def test_parse_valid_intent_dlq(self) -> None:
        """Parse valid intent DLQ topic."""
        result = parse_dlq_topic("onex.dlq.intents.v1")
        assert result is not None
        assert result["prefix"] == "onex"
        assert result["category"] == "intents"
        assert result["version"] == "v1"

    def test_parse_valid_event_dlq(self) -> None:
        """Parse valid event DLQ topic."""
        result = parse_dlq_topic("onex.dlq.events.v2")
        assert result is not None
        assert result["prefix"] == "onex"
        assert result["category"] == "events"
        assert result["version"] == "v2"

    def test_parse_valid_command_dlq(self) -> None:
        """Parse valid command DLQ topic."""
        result = parse_dlq_topic("onex.dlq.commands.v1")
        assert result is not None
        assert result["prefix"] == "onex"
        assert result["category"] == "commands"
        assert result["version"] == "v1"

    def test_parse_legacy_env_prefixed_topic(self) -> None:
        """Parse legacy DLQ topic with env prefix (backward compat)."""
        result = parse_dlq_topic("dev.dlq.intents.v1")
        assert result is not None
        assert result["prefix"] == "dev"
        assert result["category"] == "intents"

    def test_parse_non_dlq_topic_returns_none(self) -> None:
        """Non-DLQ topic returns None."""
        assert parse_dlq_topic("onex.user.events.v1") is None
        assert parse_dlq_topic("order.commands") is None
        assert parse_dlq_topic("invalid") is None

    def test_parse_domain_category_succeeds(self) -> None:
        """DLQ topic with domain-based category is parsed successfully."""
        result = parse_dlq_topic("onex.dlq.intelligence.v1")
        assert result is not None
        assert result["prefix"] == "onex"
        assert result["category"] == "intelligence"
        assert result["version"] == "v1"

    def test_parse_invalid_category_format_returns_none(self) -> None:
        """DLQ topic with category starting with digit returns None."""
        assert parse_dlq_topic("onex.dlq.123invalid.v1") is None

    def test_parse_missing_version_returns_none(self) -> None:
        """DLQ topic without version returns None."""
        assert parse_dlq_topic("onex.dlq.intents") is None


class TestIsDLQTopic:
    """Test is_dlq_topic function."""

    def test_valid_dlq_topics(self) -> None:
        """Valid DLQ topics return True."""
        assert is_dlq_topic("onex.dlq.intents.v1") is True
        assert is_dlq_topic("onex.dlq.events.v1") is True
        assert is_dlq_topic("onex.dlq.commands.v2") is True

    def test_non_dlq_topics(self) -> None:
        """Non-DLQ topics return False."""
        assert is_dlq_topic("onex.evt.platform.node-registered.v1") is False
        assert is_dlq_topic("order.commands") is False
        assert is_dlq_topic("dlq.intents.v1") is False  # Missing prefix


class TestGetDLQTopicForOriginal:
    """Test get_dlq_topic_for_original function."""

    def test_event_topic_produces_event_dlq(self) -> None:
        """Get realm-agnostic DLQ topic for event topic."""
        dlq = get_dlq_topic_for_original("onex.evt.platform.node-registered.v1")
        assert dlq == "onex.dlq.events.v1"

    def test_command_topic_produces_command_dlq(self) -> None:
        """Get realm-agnostic DLQ topic for command topic."""
        dlq = get_dlq_topic_for_original("onex.cmd.platform.node-shutdown.v1")
        assert dlq == "onex.dlq.commands.v1"

    def test_unknown_category_returns_none(self) -> None:
        """Topic with unknown category returns None."""
        dlq = get_dlq_topic_for_original("onex.user.unknown.v1")
        assert dlq is None


class TestModelKafkaEventBusConfigGetDLQTopic:
    """Test ModelKafkaEventBusConfig.get_dlq_topic method."""

    def test_get_dlq_topic_default_intents(self) -> None:
        """Default category is intents (realm-agnostic, OMN-5189)."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="prod",
        )
        assert config.get_dlq_topic() == "onex.dlq.intents.v1"

    def test_get_dlq_topic_events(self) -> None:
        """Get DLQ topic for events category."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="staging",
        )
        assert config.get_dlq_topic("events") == "onex.dlq.events.v1"

    def test_get_dlq_topic_commands(self) -> None:
        """Get DLQ topic for commands category."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="local",
        )
        assert config.get_dlq_topic("commands") == "onex.dlq.commands.v1"

    def test_explicit_dead_letter_topic_takes_precedence(self) -> None:
        """Explicit dead_letter_topic takes precedence over generated topic."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="prod",
            dead_letter_topic="custom-dlq",
        )
        # Explicit topic should be returned regardless of category
        assert config.get_dlq_topic() == "custom-dlq"
        assert config.get_dlq_topic("events") == "custom-dlq"
        assert config.get_dlq_topic("commands") == "custom-dlq"

    def test_get_dlq_topic_local_environment(self) -> None:
        """Get DLQ topic with local environment (realm-agnostic)."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="local",
        )
        assert config.get_dlq_topic() == "onex.dlq.intents.v1"

    def test_get_dlq_topic_empty_category_raises(self) -> None:
        """Empty category raises ProtocolConfigurationError."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="prod",
        )
        with pytest.raises(
            ProtocolConfigurationError, match="category cannot be empty"
        ):
            config.get_dlq_topic("")

    def test_get_dlq_topic_domain_category(self) -> None:
        """Domain-based category produces correct DLQ topic."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="local",
        )
        assert config.get_dlq_topic("intelligence") == "onex.dlq.intelligence.v1"


class TestDLQTopicPattern:
    """Test DLQ_TOPIC_PATTERN regex."""

    def test_pattern_matches_valid_topics(self) -> None:
        """Pattern matches valid DLQ topics."""
        assert DLQ_TOPIC_PATTERN.match("dev.dlq.intents.v1") is not None
        assert DLQ_TOPIC_PATTERN.match("prod.dlq.events.v1") is not None
        assert DLQ_TOPIC_PATTERN.match("staging.dlq.commands.v2") is not None

    def test_pattern_case_insensitive(self) -> None:
        """Pattern is case-insensitive for category."""
        assert DLQ_TOPIC_PATTERN.match("dev.dlq.INTENTS.v1") is not None
        assert DLQ_TOPIC_PATTERN.match("dev.DLQ.events.v1") is not None

    def test_pattern_rejects_missing_environment(self) -> None:
        """Pattern rejects topics without environment."""
        assert DLQ_TOPIC_PATTERN.match("dlq.intents.v1") is None

    def test_pattern_rejects_missing_version(self) -> None:
        """Pattern rejects topics without version."""
        assert DLQ_TOPIC_PATTERN.match("dev.dlq.intents") is None

    def test_pattern_matches_domain_categories(self) -> None:
        """Pattern accepts domain-based categories (OMN-2040)."""
        assert DLQ_TOPIC_PATTERN.match("onex.dlq.intelligence.v1") is not None
        assert DLQ_TOPIC_PATTERN.match("onex.dlq.platform.v1") is not None
        assert DLQ_TOPIC_PATTERN.match("dev.dlq.agent.v1") is not None

    def test_pattern_rejects_invalid_category_format(self) -> None:
        """Pattern rejects categories starting with digits."""
        assert DLQ_TOPIC_PATTERN.match("dev.dlq.123invalid.v1") is None

    def test_pattern_rejects_non_dlq_domain(self) -> None:
        """Pattern rejects topics without 'dlq' domain."""
        assert DLQ_TOPIC_PATTERN.match("dev.other.intents.v1") is None


class TestDeriveDlqTopicForEventType:
    """Test derive_dlq_topic_for_event_type function (OMN-2040)."""

    def test_intelligence_event_type_routes_to_intelligence_dlq(self) -> None:
        """intelligence.* event_type routes to onex.dlq.intelligence.v1."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "intelligence.code-analysis-completed.v1",
            "onex.evt.intelligence.code-analysis.v1",
        )
        assert result == "onex.dlq.intelligence.v1"

    def test_platform_event_type_routes_to_platform_dlq(self) -> None:
        """platform.* event_type routes to onex.dlq.platform.v1."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "platform.node-registered.v1",
            "onex.evt.platform.node-registration.v1",
        )
        assert result == "onex.dlq.platform.v1"

    def test_agent_event_type_routes_to_agent_dlq(self) -> None:
        """agent.* event_type routes to onex.dlq.agent.v1."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "agent.status-changed.v1",
            "onex.evt.omniclaude.agent-status.v1",
        )
        assert result == "onex.dlq.agent.v1"

    def test_none_event_type_falls_back_to_topic_based(self) -> None:
        """None event_type uses topic-based DLQ routing (legacy path)."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            None,
            "onex.evt.platform.node-registered.v1",
        )
        assert result == "onex.dlq.events.v1"

    def test_empty_event_type_falls_back_to_topic_based(self) -> None:
        """Empty event_type uses topic-based DLQ routing (legacy path)."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "",
            "onex.cmd.platform.node-shutdown.v1",
        )
        assert result == "onex.dlq.commands.v1"

    def test_whitespace_event_type_falls_back_to_topic_based(self) -> None:
        """Whitespace-only event_type uses topic-based DLQ routing."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "   ",
            "onex.cmd.platform.node-shutdown.v1",
        )
        assert result == "onex.dlq.commands.v1"

    def test_single_segment_event_type_uses_whole_string_as_domain(self) -> None:
        """Event type without dots uses the whole string as domain."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "intelligence",
            "some.topic.v1",
        )
        assert result == "onex.dlq.intelligence.v1"

    def test_none_event_type_with_command_topic(self) -> None:
        """Legacy path with command topic routes to commands DLQ."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            None,
            "onex.cmd.platform.node-shutdown.v1",
        )
        assert result == "onex.dlq.commands.v1"

    def test_invalid_domain_prefix_returns_none(self) -> None:
        """Event type with invalid domain prefix (e.g., digit-leading) returns None."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        # Domain starts with digit — fails _DLQ_CATEGORY_PATTERN
        result = derive_dlq_topic_for_event_type(
            "123.something.v1",
            "dev.user.events.v1",
        )
        assert result is None

    def test_dash_leading_domain_prefix_returns_none(self) -> None:
        """Event type with dash-leading domain prefix returns None."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            "-bad.prefix.v1",
            "dev.user.events.v1",
        )
        assert result is None

    def test_none_event_type_with_unknown_topic_returns_none(self) -> None:
        """Legacy path with unrecognizable topic returns None."""
        from omnibase_infra.event_bus.topic_constants import (
            derive_dlq_topic_for_event_type,
        )

        result = derive_dlq_topic_for_event_type(
            None,
            "some.random.topic",
        )
        assert result is None
