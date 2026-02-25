# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for util_topic_validation.validate_topic_name().

Tests cover:
    - Valid topic names (pass silently)
    - Empty topic (raises)
    - Topic exceeding 255 characters (raises)
    - Reserved names "." and ".." (raises)
    - Invalid characters (raises)
    - Optional correlation_id parameter
    - Export from omnibase_infra.utils package
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.utils.util_topic_validation import validate_topic_name


@pytest.mark.unit
class TestValidTopicNames:
    """validate_topic_name() passes silently for well-formed topic names."""

    def test_onex_kafka_events(self) -> None:
        """Standard ONEX Kafka events topic passes."""
        validate_topic_name("onex.registration.events")

    def test_onex_kafka_commands(self) -> None:
        """Standard ONEX Kafka commands topic passes."""
        validate_topic_name("onex.discovery.commands")

    def test_environment_aware_format(self) -> None:
        """Environment-aware topic format passes."""
        validate_topic_name("prod.order.events.v2")

    def test_single_segment(self) -> None:
        """Single-segment topic name is valid per Kafka rules."""
        validate_topic_name("topic")

    def test_hyphenated_domain(self) -> None:
        """Hyphens are valid Kafka topic name characters."""
        validate_topic_name("onex.order-fulfillment.events")

    def test_underscore_in_name(self) -> None:
        """Underscores are valid Kafka topic name characters."""
        validate_topic_name("my_topic_name")

    def test_mixed_case(self) -> None:
        """Mixed-case topic names are valid (Kafka is case-sensitive)."""
        validate_topic_name("MyTopic.Events")

    def test_exactly_255_characters(self) -> None:
        """Topic name of exactly 255 characters is valid."""
        topic = "a" * 255
        validate_topic_name(topic)

    def test_with_correlation_id(self) -> None:
        """Passing an explicit correlation_id does not affect valid topics."""
        correlation_id = uuid4()
        validate_topic_name("onex.registration.events", correlation_id=correlation_id)

    def test_with_none_correlation_id(self) -> None:
        """Explicit None correlation_id generates a new ID internally."""
        validate_topic_name("onex.registration.events", correlation_id=None)

    def test_numeric_segment(self) -> None:
        """Numeric-only segment is valid per Kafka rules."""
        validate_topic_name("123.topic.events")


@pytest.mark.unit
class TestEmptyTopic:
    """validate_topic_name() raises for empty topic strings."""

    def test_empty_string_raises(self) -> None:
        """Empty string raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="cannot be empty"):
            validate_topic_name("")

    def test_error_has_correlation_id(self) -> None:
        """Error has a correlation_id for tracing."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_topic_name("")
        assert exc_info.value.correlation_id is not None


@pytest.mark.unit
class TestTopicTooLong:
    """validate_topic_name() raises when topic exceeds 255 characters."""

    def test_256_characters_raises(self) -> None:
        """Topic of 256 characters raises ProtocolConfigurationError."""
        topic = "a" * 256
        with pytest.raises(ProtocolConfigurationError, match="exceeds maximum length"):
            validate_topic_name(topic)

    def test_very_long_topic_raises(self) -> None:
        """Very long topic (1000 chars) raises ProtocolConfigurationError."""
        topic = "a" * 1000
        with pytest.raises(ProtocolConfigurationError, match="exceeds maximum length"):
            validate_topic_name(topic)

    def test_error_mentions_255(self) -> None:
        """Error message mentions the 255-character limit."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_topic_name("a" * 256)
        assert "255" in str(exc_info.value)


@pytest.mark.unit
class TestReservedNames:
    """validate_topic_name() raises for Kafka-reserved topic names."""

    def test_single_dot_raises(self) -> None:
        """The reserved name '.' raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="reserved"):
            validate_topic_name(".")

    def test_double_dot_raises(self) -> None:
        """The reserved name '..' raises ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="reserved"):
            validate_topic_name("..")

    def test_triple_dot_is_not_reserved(self) -> None:
        """'...' is not a Kafka reserved name (but has invalid chars from empty segments)."""
        # '...' only contains dots which are valid characters, so it should pass
        validate_topic_name("...")


@pytest.mark.unit
class TestInvalidCharacters:
    """validate_topic_name() raises for topics with invalid characters."""

    @pytest.mark.parametrize(
        "invalid_topic",
        [
            "topic with spaces",
            "topic!",
            "topic@domain",
            "topic#tag",
            "topic$var",
            "topic%20",
            "topic^caret",
            "topic&amp",
            "topic*star",
            "topic(paren",
            "topic+plus",
            "topic=equals",
            "topic[bracket",
            "topic{brace",
            "topic|pipe",
            "topic\\backslash",
            "topic:colon",
            "topic;semicolon",
            'topic"quote',
            "topic'apostrophe",
            "topic<angle",
            "topic,comma",
            "topic?question",
            "topic/slash",
        ],
    )
    def test_invalid_character_raises(self, invalid_topic: str) -> None:
        """Topics with invalid characters raise ProtocolConfigurationError."""
        with pytest.raises(ProtocolConfigurationError, match="invalid characters"):
            validate_topic_name(invalid_topic)

    def test_error_lists_allowed_characters(self) -> None:
        """Error message describes which characters are allowed."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_topic_name("topic with space")
        error_msg = str(exc_info.value)
        assert "alphanumeric" in error_msg.lower() or "periods" in error_msg.lower()


@pytest.mark.unit
class TestCorrelationIdPropagation:
    """validate_topic_name() propagates correlation_id into error context."""

    def test_provided_correlation_id_preserved_in_error(self) -> None:
        """The correlation_id passed to validate_topic_name is preserved in the error."""
        correlation_id = UUID("12345678-1234-5678-1234-567812345678")
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_topic_name("", correlation_id=correlation_id)
        assert exc_info.value.correlation_id == correlation_id

    def test_auto_generated_correlation_id_when_none(self) -> None:
        """When correlation_id is None, an ID is auto-generated."""
        with pytest.raises(ProtocolConfigurationError) as exc_info:
            validate_topic_name("", correlation_id=None)
        assert exc_info.value.correlation_id is not None


@pytest.mark.unit
class TestPackageExport:
    """validate_topic_name is exported from omnibase_infra.utils."""

    def test_importable_from_utils_package(self) -> None:
        """validate_topic_name can be imported from omnibase_infra.utils."""
        from omnibase_infra.utils import validate_topic_name as vtn

        assert callable(vtn)

    def test_present_in_all(self) -> None:
        """validate_topic_name appears in omnibase_infra.utils.__all__."""
        import omnibase_infra.utils as utils_pkg

        assert "validate_topic_name" in utils_pkg.__all__
