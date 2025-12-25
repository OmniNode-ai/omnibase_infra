# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for DLQ topic naming constants and utilities.

These tests verify that the DLQ topic naming conventions follow ONEX standards
and that the utility functions work correctly.
"""

from __future__ import annotations

import pytest

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
        """Build DLQ topic for intents."""
        topic = build_dlq_topic("dev", "intents")
        assert topic == "dev.dlq.intents.v1"

    def test_build_event_dlq_topic(self) -> None:
        """Build DLQ topic for events."""
        topic = build_dlq_topic("prod", "events")
        assert topic == "prod.dlq.events.v1"

    def test_build_command_dlq_topic(self) -> None:
        """Build DLQ topic for commands."""
        topic = build_dlq_topic("staging", "commands")
        assert topic == "staging.dlq.commands.v1"

    def test_singular_category_normalized_to_plural(self) -> None:
        """Singular category forms are normalized to plural."""
        assert build_dlq_topic("dev", "intent") == "dev.dlq.intents.v1"
        assert build_dlq_topic("dev", "event") == "dev.dlq.events.v1"
        assert build_dlq_topic("dev", "command") == "dev.dlq.commands.v1"

    def test_custom_version(self) -> None:
        """Build DLQ topic with custom version."""
        topic = build_dlq_topic("dev", "intents", version="v2")
        assert topic == "dev.dlq.intents.v2"

    def test_environment_with_hyphen(self) -> None:
        """Build DLQ topic with hyphenated environment."""
        topic = build_dlq_topic("test-1", "intents")
        assert topic == "test-1.dlq.intents.v1"

    def test_environment_with_underscore(self) -> None:
        """Build DLQ topic with underscored environment."""
        topic = build_dlq_topic("test_env", "intents")
        assert topic == "test_env.dlq.intents.v1"

    def test_empty_environment_raises(self) -> None:
        """Empty environment raises ValueError."""
        with pytest.raises(ValueError, match="environment cannot be empty"):
            build_dlq_topic("", "intents")

    def test_whitespace_environment_raises(self) -> None:
        """Whitespace-only environment raises ValueError."""
        with pytest.raises(ValueError, match="environment cannot be empty"):
            build_dlq_topic("   ", "intents")

    def test_invalid_category_raises(self) -> None:
        """Invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            build_dlq_topic("dev", "invalid")

    def test_case_insensitive_category(self) -> None:
        """Category matching is case-insensitive."""
        assert build_dlq_topic("dev", "INTENTS") == "dev.dlq.intents.v1"
        assert build_dlq_topic("dev", "Events") == "dev.dlq.events.v1"
        assert build_dlq_topic("dev", "ComMAnDs") == "dev.dlq.commands.v1"


class TestParseDLQTopic:
    """Test parse_dlq_topic function."""

    def test_parse_valid_intent_dlq(self) -> None:
        """Parse valid intent DLQ topic."""
        result = parse_dlq_topic("dev.dlq.intents.v1")
        assert result is not None
        assert result["environment"] == "dev"
        assert result["category"] == "intents"
        assert result["version"] == "v1"

    def test_parse_valid_event_dlq(self) -> None:
        """Parse valid event DLQ topic."""
        result = parse_dlq_topic("prod.dlq.events.v2")
        assert result is not None
        assert result["environment"] == "prod"
        assert result["category"] == "events"
        assert result["version"] == "v2"

    def test_parse_valid_command_dlq(self) -> None:
        """Parse valid command DLQ topic."""
        result = parse_dlq_topic("staging.dlq.commands.v1")
        assert result is not None
        assert result["environment"] == "staging"
        assert result["category"] == "commands"
        assert result["version"] == "v1"

    def test_parse_hyphenated_environment(self) -> None:
        """Parse DLQ topic with hyphenated environment."""
        result = parse_dlq_topic("test-env.dlq.intents.v1")
        assert result is not None
        assert result["environment"] == "test-env"

    def test_parse_non_dlq_topic_returns_none(self) -> None:
        """Non-DLQ topic returns None."""
        assert parse_dlq_topic("dev.user.events.v1") is None
        assert parse_dlq_topic("order.commands") is None
        assert parse_dlq_topic("invalid") is None

    def test_parse_invalid_category_returns_none(self) -> None:
        """DLQ topic with invalid category returns None."""
        assert parse_dlq_topic("dev.dlq.invalid.v1") is None

    def test_parse_missing_version_returns_none(self) -> None:
        """DLQ topic without version returns None."""
        assert parse_dlq_topic("dev.dlq.intents") is None


class TestIsDLQTopic:
    """Test is_dlq_topic function."""

    def test_valid_dlq_topics(self) -> None:
        """Valid DLQ topics return True."""
        assert is_dlq_topic("dev.dlq.intents.v1") is True
        assert is_dlq_topic("prod.dlq.events.v1") is True
        assert is_dlq_topic("staging.dlq.commands.v2") is True

    def test_non_dlq_topics(self) -> None:
        """Non-DLQ topics return False."""
        assert is_dlq_topic("dev.user.events.v1") is False
        assert is_dlq_topic("order.commands") is False
        assert is_dlq_topic("dlq.intents.v1") is False  # Missing environment


class TestGetDLQTopicForOriginal:
    """Test get_dlq_topic_for_original function."""

    def test_environment_aware_intent_topic(self) -> None:
        """Get DLQ topic for environment-aware intent topic."""
        dlq = get_dlq_topic_for_original("dev.checkout.intents.v1")
        assert dlq == "dev.dlq.intents.v1"

    def test_environment_aware_event_topic(self) -> None:
        """Get DLQ topic for environment-aware event topic."""
        dlq = get_dlq_topic_for_original("prod.order.events.v1")
        assert dlq == "prod.dlq.events.v1"

    def test_environment_aware_command_topic(self) -> None:
        """Get DLQ topic for environment-aware command topic."""
        dlq = get_dlq_topic_for_original("staging.user.commands.v2")
        assert dlq == "staging.dlq.commands.v1"

    def test_onex_format_without_environment_returns_none(self) -> None:
        """ONEX format topic without explicit environment returns None."""
        dlq = get_dlq_topic_for_original("onex.registration.commands")
        assert dlq is None

    def test_onex_format_with_explicit_environment(self) -> None:
        """ONEX format topic with explicit environment works."""
        dlq = get_dlq_topic_for_original(
            "onex.registration.commands",
            environment="prod",
        )
        assert dlq == "prod.dlq.commands.v1"

    def test_unknown_category_returns_none(self) -> None:
        """Topic with unknown category returns None."""
        dlq = get_dlq_topic_for_original("dev.user.unknown.v1")
        assert dlq is None


class TestModelKafkaEventBusConfigGetDLQTopic:
    """Test ModelKafkaEventBusConfig.get_dlq_topic method."""

    def test_get_dlq_topic_default_intents(self) -> None:
        """Default category is intents."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="prod",
        )
        assert config.get_dlq_topic() == "prod.dlq.intents.v1"

    def test_get_dlq_topic_events(self) -> None:
        """Get DLQ topic for events category."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="staging",
        )
        assert config.get_dlq_topic("events") == "staging.dlq.events.v1"

    def test_get_dlq_topic_commands(self) -> None:
        """Get DLQ topic for commands category."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="dev",
        )
        assert config.get_dlq_topic("commands") == "dev.dlq.commands.v1"

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
        """Get DLQ topic with local environment."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="local",
        )
        assert config.get_dlq_topic() == "local.dlq.intents.v1"

    def test_get_dlq_topic_invalid_category_raises(self) -> None:
        """Invalid category raises ValueError."""
        from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig

        config = ModelKafkaEventBusConfig(
            bootstrap_servers="localhost:9092",
            environment="prod",
        )
        with pytest.raises(ValueError, match="Invalid category"):
            config.get_dlq_topic("invalid")


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

    def test_pattern_rejects_invalid_category(self) -> None:
        """Pattern rejects topics with invalid category."""
        assert DLQ_TOPIC_PATTERN.match("dev.dlq.invalid.v1") is None

    def test_pattern_rejects_non_dlq_domain(self) -> None:
        """Pattern rejects topics without 'dlq' domain."""
        assert DLQ_TOPIC_PATTERN.match("dev.other.intents.v1") is None
