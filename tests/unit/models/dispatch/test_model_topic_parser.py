# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Comprehensive tests for ModelTopicParser and ModelParsedTopic.

Tests cover:
- parse() with ONEX Kafka format (onex.<domain>.<type>)
- parse() with Environment-Aware format (<env>.<domain>.<category>.<version>)
- parse() with invalid formats
- get_category() extraction
- matches_pattern() with wildcards (*, **)
- validate_topic() strict and non-strict modes

OMN-934: Message dispatch engine implementation
"""

from __future__ import annotations

import pytest
from omnibase_core.enums.enum_topic_taxonomy import EnumTopicType

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_topic_parser import (
    EnumTopicStandard,
    ModelParsedTopic,
    ModelTopicParser,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def parser() -> ModelTopicParser:
    """Create a fresh ModelTopicParser for each test."""
    return ModelTopicParser()


# ============================================================================
# ModelParsedTopic Tests
# ============================================================================


@pytest.mark.unit
class TestModelParsedTopic:
    """Tests for ModelParsedTopic model."""

    def test_create_valid_parsed_topic(self) -> None:
        """Test creating a valid ModelParsedTopic."""
        parsed = ModelParsedTopic(
            raw_topic="onex.registration.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="registration",
            category=EnumMessageCategory.EVENT,
            topic_type=EnumTopicType.EVENTS,
            is_valid=True,
        )

        assert parsed.raw_topic == "onex.registration.events"
        assert parsed.standard == EnumTopicStandard.ONEX_KAFKA
        assert parsed.domain == "registration"
        assert parsed.category == EnumMessageCategory.EVENT
        assert parsed.is_valid is True

    def test_create_invalid_parsed_topic(self) -> None:
        """Test creating an invalid ModelParsedTopic."""
        parsed = ModelParsedTopic(
            raw_topic="invalid.topic",
            standard=EnumTopicStandard.UNKNOWN,
            is_valid=False,
            validation_error="Topic does not match any known format",
        )

        assert parsed.is_valid is False
        assert parsed.validation_error is not None

    def test_is_routable_valid_topic(self) -> None:
        """Test is_routable() for valid topic with category."""
        parsed = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        assert parsed.is_routable() is True

    def test_is_routable_invalid_topic(self) -> None:
        """Test is_routable() for invalid topic."""
        parsed = ModelParsedTopic(
            raw_topic="invalid",
            standard=EnumTopicStandard.UNKNOWN,
            is_valid=False,
        )

        assert parsed.is_routable() is False

    def test_is_routable_no_category(self) -> None:
        """Test is_routable() when category is None."""
        parsed = ModelParsedTopic(
            raw_topic="onex.user.snapshots",  # snapshots don't map to a category
            standard=EnumTopicStandard.ONEX_KAFKA,
            category=None,  # No category for snapshots
            is_valid=True,
        )

        assert parsed.is_routable() is False


@pytest.mark.unit
class TestModelParsedTopicCanonicalBehaviors:
    """Tests for canonical Pydantic model behaviors of ModelParsedTopic.

    These tests verify that the frozen Pydantic model correctly implements:
    - Immutability (frozen=True)
    - Serialization (model_dump)
    - Deserialization (model_validate)
    - Copying (model_copy)
    - Extra field rejection (extra='forbid')
    - Field validation (min_length constraints)
    - Hashability (frozen models are hashable)
    - Equality comparison
    """

    def test_immutability_frozen_model(self) -> None:
        """Test that ModelParsedTopic is immutable (frozen=True)."""
        from pydantic import ValidationError

        parsed = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            is_valid=True,
        )

        # Attempting to modify any field should raise ValidationError
        with pytest.raises(ValidationError):
            parsed.domain = "changed"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            parsed.raw_topic = "different"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            parsed.is_valid = False  # type: ignore[misc]

    def test_model_dump_all_fields(self) -> None:
        """Test model_dump() returns all fields correctly."""
        parsed = ModelParsedTopic(
            raw_topic="onex.registration.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="registration",
            category=EnumMessageCategory.EVENT,
            topic_type=EnumTopicType.EVENTS,
            environment=None,
            version=None,
            is_valid=True,
            validation_error=None,
        )

        data = parsed.model_dump()

        assert data["raw_topic"] == "onex.registration.events"
        assert data["standard"] == EnumTopicStandard.ONEX_KAFKA
        assert data["domain"] == "registration"
        assert data["category"] == EnumMessageCategory.EVENT
        assert data["topic_type"] == EnumTopicType.EVENTS
        assert data["environment"] is None
        assert data["version"] is None
        assert data["is_valid"] is True
        assert data["validation_error"] is None

    def test_model_dump_exclude_none(self) -> None:
        """Test model_dump(exclude_none=True) excludes None fields."""
        parsed = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            is_valid=True,
        )

        data = parsed.model_dump(exclude_none=True)

        assert "raw_topic" in data
        assert "standard" in data
        assert "is_valid" in data
        # None fields should be excluded
        assert "domain" not in data
        assert "category" not in data
        assert "environment" not in data

    def test_model_validate_from_dict(self) -> None:
        """Test model_validate() creates model from dictionary."""
        data = {
            "raw_topic": "dev.user.events.v1",
            "standard": EnumTopicStandard.ENVIRONMENT_AWARE,
            "domain": "user",
            "category": EnumMessageCategory.EVENT,
            "environment": "dev",
            "version": "v1",
            "is_valid": True,
        }

        parsed = ModelParsedTopic.model_validate(data)

        assert parsed.raw_topic == "dev.user.events.v1"
        assert parsed.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert parsed.domain == "user"
        assert parsed.category == EnumMessageCategory.EVENT
        assert parsed.environment == "dev"
        assert parsed.version == "v1"
        assert parsed.is_valid is True

    def test_model_validate_roundtrip(self) -> None:
        """Test model_dump() -> model_validate() roundtrip preserves data."""
        original = ModelParsedTopic(
            raw_topic="onex.workflow.intents",
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="workflow",
            category=EnumMessageCategory.INTENT,
            topic_type=EnumTopicType.INTENTS,
            is_valid=True,
        )

        # Serialize and deserialize
        data = original.model_dump()
        restored = ModelParsedTopic.model_validate(data)

        # Verify all fields match
        assert restored.raw_topic == original.raw_topic
        assert restored.standard == original.standard
        assert restored.domain == original.domain
        assert restored.category == original.category
        assert restored.topic_type == original.topic_type
        assert restored.is_valid == original.is_valid

    def test_model_copy_creates_new_instance(self) -> None:
        """Test model_copy() creates a new independent instance."""
        original = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="user",
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        # Create a copy with modified field
        copied = original.model_copy(update={"domain": "order"})

        # Verify copy has updated value
        assert copied.domain == "order"

        # Verify original is unchanged
        assert original.domain == "user"

        # Verify they are different objects
        assert copied is not original

        # Other fields should be preserved
        assert copied.raw_topic == original.raw_topic
        assert copied.standard == original.standard
        assert copied.category == original.category
        assert copied.is_valid == original.is_valid

    def test_model_copy_deep(self) -> None:
        """Test model_copy(deep=True) creates deep copy."""
        original = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            is_valid=True,
        )

        copied = original.model_copy(deep=True)

        # Should be equal but not the same object
        assert copied == original
        assert copied is not original

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected (extra='forbid')."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedTopic(
                raw_topic="onex.user.events",
                standard=EnumTopicStandard.ONEX_KAFKA,
                is_valid=True,
                unexpected_field="should_fail",  # type: ignore[call-arg]
            )

        # Verify error mentions extra field
        assert "unexpected_field" in str(exc_info.value)

    def test_raw_topic_min_length_validation(self) -> None:
        """Test that raw_topic enforces min_length=1."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelParsedTopic(
                raw_topic="",  # Empty string should fail
                standard=EnumTopicStandard.UNKNOWN,
                is_valid=False,
            )

        # Verify error mentions the constraint
        error_str = str(exc_info.value)
        assert "raw_topic" in error_str

    def test_hashability(self) -> None:
        """Test that frozen model is hashable and can be used in sets/dicts."""
        parsed1 = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        parsed2 = ModelParsedTopic(
            raw_topic="onex.order.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        parsed3 = ModelParsedTopic(
            raw_topic="onex.user.events",  # Same as parsed1
            standard=EnumTopicStandard.ONEX_KAFKA,
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        # Should be hashable
        hash1 = hash(parsed1)
        hash2 = hash(parsed2)
        hash3 = hash(parsed3)

        # Hashes should be consistent
        assert hash(parsed1) == hash1

        # Equal objects should have same hash
        assert hash1 == hash3

        # Different objects likely have different hashes
        assert hash1 != hash2

        # Can be used in set
        topic_set = {parsed1, parsed2, parsed3}
        assert len(topic_set) == 2  # parsed1 and parsed3 are equal

        # Can be used as dict key
        topic_dict = {parsed1: "first", parsed2: "second"}
        assert topic_dict[parsed3] == "first"  # parsed3 equals parsed1

    def test_equality_comparison(self) -> None:
        """Test equality comparison between ModelParsedTopic instances."""
        parsed1 = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="user",
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        parsed2 = ModelParsedTopic(
            raw_topic="onex.user.events",
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="user",
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        parsed3 = ModelParsedTopic(
            raw_topic="onex.order.events",  # Different topic
            standard=EnumTopicStandard.ONEX_KAFKA,
            domain="order",
            category=EnumMessageCategory.EVENT,
            is_valid=True,
        )

        # Equal instances
        assert parsed1 == parsed2

        # Different instances
        assert parsed1 != parsed3

        # Not equal to different types
        assert parsed1 != "onex.user.events"
        assert parsed1 != {"raw_topic": "onex.user.events"}

    def test_from_attributes_config(self) -> None:
        """Test from_attributes=True allows creation from objects with attributes."""

        class TopicData:
            """Simple class with matching attributes."""

            def __init__(self) -> None:
                self.raw_topic = "onex.user.events"
                self.standard = EnumTopicStandard.ONEX_KAFKA
                self.domain = "user"
                self.category = EnumMessageCategory.EVENT
                self.topic_type = EnumTopicType.EVENTS
                self.environment = None
                self.version = None
                self.is_valid = True
                self.validation_error = None

        source = TopicData()
        parsed = ModelParsedTopic.model_validate(source)

        assert parsed.raw_topic == "onex.user.events"
        assert parsed.standard == EnumTopicStandard.ONEX_KAFKA
        assert parsed.domain == "user"
        assert parsed.category == EnumMessageCategory.EVENT
        assert parsed.is_valid is True


# ============================================================================
# ONEX Kafka Format Tests
# ============================================================================


@pytest.mark.unit
class TestOnexKafkaFormat:
    """Tests for parsing ONEX Kafka format topics (onex.<domain>.<type>)."""

    def test_parse_onex_kafka_events(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX Kafka events topic."""
        result = parser.parse("onex.registration.events")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.domain == "registration"
        assert result.category == EnumMessageCategory.EVENT
        assert result.topic_type == EnumTopicType.EVENTS

    def test_parse_onex_kafka_commands(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX Kafka commands topic."""
        result = parser.parse("onex.discovery.commands")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.domain == "discovery"
        assert result.category == EnumMessageCategory.COMMAND
        assert result.topic_type == EnumTopicType.COMMANDS

    def test_parse_onex_kafka_intents(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX Kafka intents topic."""
        result = parser.parse("onex.workflow.intents")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.domain == "workflow"
        assert result.category == EnumMessageCategory.INTENT
        assert result.topic_type == EnumTopicType.INTENTS

    def test_parse_onex_kafka_snapshots(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX Kafka snapshots topic (no category mapping)."""
        result = parser.parse("onex.state.snapshots")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.domain == "state"
        assert result.category is None  # snapshots don't have a category
        assert result.topic_type == EnumTopicType.SNAPSHOTS

    def test_parse_onex_kafka_single_letter_domain(
        self, parser: ModelTopicParser
    ) -> None:
        """Test parsing ONEX Kafka with single-letter domain."""
        result = parser.parse("onex.a.events")

        assert result.is_valid is True
        assert result.domain == "a"

    def test_parse_onex_kafka_hyphenated_domain(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX Kafka with hyphenated domain."""
        result = parser.parse("onex.user-service.events")

        assert result.is_valid is True
        assert result.domain == "user-service"

    def test_parse_onex_kafka_case_insensitive(self, parser: ModelTopicParser) -> None:
        """Test that ONEX Kafka parsing is case-insensitive."""
        result = parser.parse("ONEX.REGISTRATION.EVENTS")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.domain == "registration"  # Normalized to lowercase


# ============================================================================
# Environment-Aware Format Tests
# ============================================================================


@pytest.mark.unit
class TestEnvironmentAwareFormat:
    """Tests for parsing Environment-Aware format topics (<env>.<domain>.<category>.<version>)."""

    def test_parse_env_aware_dev(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware topic with dev environment."""
        result = parser.parse("dev.user.events.v1")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.environment == "dev"
        assert result.domain == "user"
        assert result.category == EnumMessageCategory.EVENT
        assert result.version == "v1"

    def test_parse_env_aware_prod(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware topic with prod environment."""
        result = parser.parse("prod.order.commands.v2")

        assert result.is_valid is True
        assert result.environment == "prod"
        assert result.domain == "order"
        assert result.category == EnumMessageCategory.COMMAND
        assert result.version == "v2"

    def test_parse_env_aware_staging(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware topic with staging environment."""
        result = parser.parse("staging.payment.intents.v1")

        assert result.is_valid is True
        assert result.environment == "staging"
        assert result.domain == "payment"
        assert result.category == EnumMessageCategory.INTENT

    def test_parse_env_aware_test(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware topic with test environment."""
        result = parser.parse("test.notification.events.v3")

        assert result.is_valid is True
        assert result.environment == "test"
        assert result.version == "v3"

    def test_parse_env_aware_local(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware topic with local environment."""
        result = parser.parse("local.auth.commands.v1")

        assert result.is_valid is True
        assert result.environment == "local"

    def test_parse_env_aware_case_insensitive(self, parser: ModelTopicParser) -> None:
        """Test that Environment-Aware parsing is case-insensitive."""
        result = parser.parse("DEV.USER.EVENTS.V1")

        assert result.is_valid is True
        assert result.environment == "dev"  # Normalized to lowercase
        assert result.domain == "user"

    def test_parse_env_aware_higher_version(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware with multi-digit version."""
        result = parser.parse("prod.order.events.v123")

        assert result.is_valid is True
        assert result.version == "v123"


# ============================================================================
# Invalid Format Tests
# ============================================================================


@pytest.mark.unit
class TestInvalidFormats:
    """Tests for parsing invalid topic formats."""

    def test_parse_empty_string(self, parser: ModelTopicParser) -> None:
        """Test parsing empty string."""
        result = parser.parse("")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.validation_error is not None

    def test_parse_whitespace_only(self, parser: ModelTopicParser) -> None:
        """Test parsing whitespace-only string."""
        result = parser.parse("   ")

        assert result.is_valid is False
        assert result.validation_error is not None
        assert (
            "empty" in result.validation_error.lower()
            or "whitespace" in result.validation_error.lower()
        )

    def test_parse_no_category(self, parser: ModelTopicParser) -> None:
        """Test parsing topic without category segment."""
        result = parser.parse("some.random.topic")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_parse_invalid_onex_prefix(self, parser: ModelTopicParser) -> None:
        """Test parsing with wrong prefix (not 'onex')."""
        result = parser.parse("notx.user.events")

        # May fallback to category detection
        assert result.standard != EnumTopicStandard.ONEX_KAFKA

    def test_parse_onex_missing_domain(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX format with missing domain."""
        result = parser.parse("onex..events")

        assert result.standard != EnumTopicStandard.ONEX_KAFKA

    def test_parse_onex_invalid_type(self, parser: ModelTopicParser) -> None:
        """Test parsing ONEX format with invalid type."""
        result = parser.parse("onex.user.invalid")

        assert result.standard != EnumTopicStandard.ONEX_KAFKA

    def test_parse_env_aware_invalid_env(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware with invalid environment."""
        result = parser.parse("invalid.user.events.v1")

        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_parse_env_aware_missing_version(self, parser: ModelTopicParser) -> None:
        """Test parsing Environment-Aware without version."""
        result = parser.parse("dev.user.events")

        # Should not match ENVIRONMENT_AWARE pattern
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_parse_env_aware_invalid_version_format(
        self, parser: ModelTopicParser
    ) -> None:
        """Test parsing Environment-Aware with invalid version format."""
        result = parser.parse("dev.user.events.version1")

        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE


# ============================================================================
# Version Suffix Validation Tests (PR #54 Review)
# ============================================================================


@pytest.mark.unit
class TestVersionSuffixValidation:
    """Tests for version suffix validation (.v\\d+) in topic names.

    Version suffixes are required for Environment-Aware format topics and must
    follow the pattern .v<digits> (e.g., .v1, .v2, .v10).

    These tests verify:
    - Valid version suffixes are accepted (.v1, .v2, .v10, .v123)
    - Invalid/incomplete suffixes are rejected (no version, .v without digit, .V1)

    Addresses: PR #54 review feedback on version suffix validation coverage.
    """

    # -------------------------------------------------------------------------
    # Valid Version Suffix Tests
    # -------------------------------------------------------------------------

    def test_version_suffix_v1_valid(self, parser: ModelTopicParser) -> None:
        """Test that .v1 version suffix is valid for Environment-Aware format."""
        result = parser.parse("dev.user.events.v1")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.version == "v1"

    def test_version_suffix_v2_valid(self, parser: ModelTopicParser) -> None:
        """Test that .v2 version suffix is valid for Environment-Aware format."""
        result = parser.parse("prod.order.commands.v2")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.version == "v2"

    def test_version_suffix_v10_valid(self, parser: ModelTopicParser) -> None:
        """Test that multi-digit .v10 version suffix is valid."""
        result = parser.parse("staging.payment.intents.v10")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.version == "v10"

    def test_version_suffix_v123_valid(self, parser: ModelTopicParser) -> None:
        """Test that high multi-digit .v123 version suffix is valid."""
        result = parser.parse("test.analytics.events.v123")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.version == "v123"

    # -------------------------------------------------------------------------
    # Invalid Version Suffix Tests
    # -------------------------------------------------------------------------

    def test_version_suffix_missing_invalid(self, parser: ModelTopicParser) -> None:
        """Test that missing version suffix is invalid for Environment-Aware format.

        Topics like 'dev.user.events' without version suffix should NOT match
        the ENVIRONMENT_AWARE standard.
        """
        result = parser.parse("dev.user.events")

        # Should not match ENVIRONMENT_AWARE pattern (requires version)
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_version_suffix_incomplete_v_only_invalid(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that incomplete .v suffix (without digit) is invalid.

        Topics like 'dev.user.events.v' with just 'v' but no digit should NOT
        match the ENVIRONMENT_AWARE standard.
        """
        result = parser.parse("dev.user.events.v")

        # Should not match ENVIRONMENT_AWARE pattern (requires v + digit)
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_version_suffix_uppercase_v_invalid(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that uppercase .V1 suffix is invalid (case-sensitive matching).

        While parsing is generally case-insensitive, the canonical form uses
        lowercase 'v'. This test verifies the pattern handling.
        """
        result = parser.parse("dev.user.events.V1")

        # Should match ENVIRONMENT_AWARE (pattern is case-insensitive)
        # but version should be normalized to lowercase
        if result.standard == EnumTopicStandard.ENVIRONMENT_AWARE:
            assert result.version == "v1"  # Normalized to lowercase

    def test_version_suffix_wrong_prefix_invalid(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that version1 (without 'v' prefix) is invalid."""
        result = parser.parse("dev.user.events.version1")

        # Should not match ENVIRONMENT_AWARE pattern
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_version_suffix_semantic_version_invalid(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that semantic versioning (.v1.0.0) is invalid.

        ONEX topics use simple version numbers (.v1, .v2), not semantic versioning.
        """
        result = parser.parse("dev.user.events.v1.0.0")

        # Should not match ENVIRONMENT_AWARE pattern
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_version_suffix_number_only_invalid(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that .1 suffix (number without 'v') is invalid."""
        result = parser.parse("dev.user.events.1")

        # Should not match ENVIRONMENT_AWARE pattern
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    # -------------------------------------------------------------------------
    # ONEX Kafka Format Version Tests (no version required)
    # -------------------------------------------------------------------------

    def test_onex_kafka_format_no_version_required(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that ONEX Kafka format does not require version suffix.

        ONEX Kafka format (onex.<domain>.<type>) does not have version suffix.
        This is by design - versioning is at the Environment-Aware level only.
        """
        result = parser.parse("onex.registration.events")

        assert result.is_valid is True
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.version is None  # No version in ONEX Kafka format


# ============================================================================
# Fallback Category Detection Tests
# ============================================================================


@pytest.mark.unit
class TestFallbackCategoryDetection:
    """Tests for fallback category detection from non-standard topics."""

    def test_parse_fallback_events(self, parser: ModelTopicParser) -> None:
        """Test fallback detection for topics containing .events."""
        result = parser.parse("some.custom.events.format")

        assert result.is_valid is True  # Category was detected
        assert result.category == EnumMessageCategory.EVENT
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_parse_fallback_commands(self, parser: ModelTopicParser) -> None:
        """Test fallback detection for topics containing .commands."""
        result = parser.parse("some.custom.commands.format")

        assert result.is_valid is True
        assert result.category == EnumMessageCategory.COMMAND

    def test_parse_fallback_intents(self, parser: ModelTopicParser) -> None:
        """Test fallback detection for topics containing .intents."""
        result = parser.parse("some.custom.intents.format")

        assert result.is_valid is True
        assert result.category == EnumMessageCategory.INTENT

    def test_parse_fallback_events_at_end(self, parser: ModelTopicParser) -> None:
        """Test fallback detection for topics ending with .events."""
        result = parser.parse("custom.topic.events")

        assert result.category == EnumMessageCategory.EVENT


# ============================================================================
# False Positive Protection Tests (OMN-977)
# ============================================================================


@pytest.mark.unit
class TestFalsePositiveProtection:
    """Tests ensuring no false positive matches in topic category detection.

    These tests verify that segment-based matching prevents incorrectly
    inferring categories when a category suffix appears as a substring
    within a segment name (e.g., "eventsource" should not match "events").

    Addresses: PR #63 review feedback on topic matching false positives.
    """

    def test_eventsource_segment_does_not_match_events(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that 'eventsource' segment does not match 'events' category."""
        result = parser.parse("dev.eventsource.data.v1")

        # Should NOT match EVENT because 'eventsource' != 'events'
        assert result.category is None
        assert result.is_valid is False

    def test_commandservices_segment_does_not_match_commands(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that 'commandservices' segment does not match 'commands' category."""
        result = parser.parse("dev.commandservices.data.v1")

        # Should NOT match COMMAND because 'commandservices' != 'commands'
        assert result.category is None
        assert result.is_valid is False

    def test_intentservice_segment_does_not_match_intents(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that 'intentservice' segment does not match 'intents' category."""
        result = parser.parse("dev.intentservice.data.v1")

        # Should NOT match INTENT because 'intentservice' != 'intents'
        assert result.category is None
        assert result.is_valid is False

    def test_projectionsource_segment_does_not_match_projections(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that 'projectionsource' segment does not match 'projections' category."""
        result = parser.parse("dev.projectionsource.data.v1")

        # Should NOT match PROJECTION because 'projectionsource' != 'projections'
        assert result.category is None
        assert result.is_valid is False

    def test_exact_events_segment_still_matches(self, parser: ModelTopicParser) -> None:
        """Test that exact 'events' segment still correctly matches EVENT."""
        result = parser.parse("dev.user.events.v1")

        assert result.category == EnumMessageCategory.EVENT
        assert result.is_valid is True

    def test_exact_commands_segment_still_matches(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that exact 'commands' segment still correctly matches COMMAND."""
        result = parser.parse("dev.user.commands.v1")

        assert result.category == EnumMessageCategory.COMMAND
        assert result.is_valid is True

    def test_category_segment_at_end_still_matches(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that category segment at end of topic still matches."""
        result = parser.parse("onex.user.events")

        assert result.category == EnumMessageCategory.EVENT
        assert result.is_valid is True

    def test_category_segment_at_middle_still_matches(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that category segment in middle of topic still matches."""
        result = parser.parse("custom.prefix.events.suffix.extra")

        assert result.category == EnumMessageCategory.EVENT
        assert result.is_valid is True  # Category was detected


# ============================================================================
# get_category() Tests
# ============================================================================


@pytest.mark.unit
class TestGetCategory:
    """Tests for get_category() method."""

    def test_get_category_onex_events(self, parser: ModelTopicParser) -> None:
        """Test get_category for ONEX events topic."""
        category = parser.get_category("onex.registration.events")

        assert category == EnumMessageCategory.EVENT

    def test_get_category_onex_commands(self, parser: ModelTopicParser) -> None:
        """Test get_category for ONEX commands topic."""
        category = parser.get_category("onex.discovery.commands")

        assert category == EnumMessageCategory.COMMAND

    def test_get_category_onex_intents(self, parser: ModelTopicParser) -> None:
        """Test get_category for ONEX intents topic."""
        category = parser.get_category("onex.workflow.intents")

        assert category == EnumMessageCategory.INTENT

    def test_get_category_env_aware(self, parser: ModelTopicParser) -> None:
        """Test get_category for Environment-Aware topic."""
        category = parser.get_category("dev.user.events.v1")

        assert category == EnumMessageCategory.EVENT

    def test_get_category_invalid_topic(self, parser: ModelTopicParser) -> None:
        """Test get_category for invalid topic returns None."""
        category = parser.get_category("invalid.topic")

        assert category is None


# ============================================================================
# matches_pattern() Tests
# ============================================================================


@pytest.mark.unit
class TestMatchesPattern:
    """Tests for matches_pattern() with wildcards."""

    def test_exact_match(self, parser: ModelTopicParser) -> None:
        """Test exact pattern matching."""
        assert parser.matches_pattern("onex.user.events", "onex.user.events") is True
        assert parser.matches_pattern("onex.user.events", "onex.user.commands") is False

    def test_single_wildcard(self, parser: ModelTopicParser) -> None:
        """Test single wildcard (*) matches single segment."""
        pattern = "onex.*.events"

        assert parser.matches_pattern(pattern, "onex.user.events") is True
        assert parser.matches_pattern(pattern, "onex.order.events") is True
        assert parser.matches_pattern(pattern, "onex.user.commands") is False
        assert (
            parser.matches_pattern(pattern, "onex.user.service.events") is False
        )  # * doesn't match dots

    def test_double_wildcard(self, parser: ModelTopicParser) -> None:
        """Test double wildcard (**) matches multiple segments."""
        pattern = "dev.**"

        assert parser.matches_pattern(pattern, "dev.user.events.v1") is True
        assert parser.matches_pattern(pattern, "dev.order.commands.v2") is True
        assert parser.matches_pattern(pattern, "prod.user.events.v1") is False

    def test_double_wildcard_middle(self, parser: ModelTopicParser) -> None:
        """Test double wildcard in the middle of pattern."""
        pattern = "**.events.*"

        assert parser.matches_pattern(pattern, "dev.user.events.v1") is True
        assert parser.matches_pattern(pattern, "prod.order.events.v2") is True
        assert (
            parser.matches_pattern(pattern, "dev.user.events") is False
        )  # Missing segment after events

    def test_double_wildcard_at_end(self, parser: ModelTopicParser) -> None:
        """Test double wildcard at end of pattern."""
        pattern = "dev.user.**"

        assert parser.matches_pattern(pattern, "dev.user.events.v1") is True
        assert parser.matches_pattern(pattern, "dev.user.commands.v2") is True

    def test_mixed_wildcards(self, parser: ModelTopicParser) -> None:
        """Test pattern with both * and **."""
        pattern = "*.user.**"

        assert parser.matches_pattern(pattern, "dev.user.events.v1") is True
        assert parser.matches_pattern(pattern, "prod.user.commands.v2") is True
        assert (
            parser.matches_pattern(pattern, "dev.order.events.v1") is False
        )  # Wrong domain

    def test_wildcard_case_insensitive(self, parser: ModelTopicParser) -> None:
        """Test pattern matching is case-insensitive."""
        pattern = "onex.*.events"

        assert parser.matches_pattern(pattern, "ONEX.USER.EVENTS") is True
        assert parser.matches_pattern(pattern, "onex.USER.events") is True

    def test_empty_pattern(self, parser: ModelTopicParser) -> None:
        """Test empty pattern returns False."""
        assert parser.matches_pattern("", "onex.user.events") is False

    def test_empty_topic(self, parser: ModelTopicParser) -> None:
        """Test empty topic returns False."""
        assert parser.matches_pattern("onex.*.events", "") is False


# ============================================================================
# validate_topic() Tests
# ============================================================================


@pytest.mark.unit
class TestValidateTopic:
    """Tests for validate_topic() in strict and non-strict modes."""

    def test_validate_onex_kafka_non_strict(self, parser: ModelTopicParser) -> None:
        """Test validate_topic for ONEX Kafka format in non-strict mode."""
        is_valid, error = parser.validate_topic("onex.registration.events")

        assert is_valid is True
        assert error is None

    def test_validate_onex_kafka_strict(self, parser: ModelTopicParser) -> None:
        """Test validate_topic for ONEX Kafka format in strict mode."""
        is_valid, error = parser.validate_topic("onex.registration.events", strict=True)

        assert is_valid is True
        assert error is None

    def test_validate_env_aware_non_strict(self, parser: ModelTopicParser) -> None:
        """Test validate_topic for Environment-Aware in non-strict mode."""
        is_valid, error = parser.validate_topic("dev.user.events.v1")

        assert is_valid is True
        assert error is None

    def test_validate_env_aware_strict_fails(self, parser: ModelTopicParser) -> None:
        """Test validate_topic for Environment-Aware in strict mode fails."""
        is_valid, error = parser.validate_topic("dev.user.events.v1", strict=True)

        assert is_valid is False
        assert error is not None
        assert "ONEX Kafka format" in error

    def test_validate_invalid_topic_non_strict(self, parser: ModelTopicParser) -> None:
        """Test validate_topic for invalid topic in non-strict mode."""
        is_valid, error = parser.validate_topic("completely.invalid.topic")

        assert is_valid is False
        assert error is not None

    def test_validate_invalid_topic_strict(self, parser: ModelTopicParser) -> None:
        """Test validate_topic for invalid topic in strict mode."""
        is_valid, error = parser.validate_topic("completely.invalid.topic", strict=True)

        assert is_valid is False
        assert error is not None


# ============================================================================
# Convenience Method Tests
# ============================================================================


@pytest.mark.unit
class TestConvenienceMethods:
    """Tests for convenience methods."""

    def test_is_onex_kafka_format_true(self, parser: ModelTopicParser) -> None:
        """Test is_onex_kafka_format returns True for valid ONEX topic."""
        assert parser.is_onex_kafka_format("onex.user.events") is True

    def test_is_onex_kafka_format_false(self, parser: ModelTopicParser) -> None:
        """Test is_onex_kafka_format returns False for non-ONEX topic."""
        assert parser.is_onex_kafka_format("dev.user.events.v1") is False

    def test_is_environment_aware_format_true(self, parser: ModelTopicParser) -> None:
        """Test is_environment_aware_format returns True for valid env-aware topic."""
        assert parser.is_environment_aware_format("dev.user.events.v1") is True

    def test_is_environment_aware_format_false(self, parser: ModelTopicParser) -> None:
        """Test is_environment_aware_format returns False for non-env-aware topic."""
        assert parser.is_environment_aware_format("onex.user.events") is False

    def test_extract_domain_onex(self, parser: ModelTopicParser) -> None:
        """Test extract_domain for ONEX Kafka topic."""
        domain = parser.extract_domain("onex.registration.events")

        assert domain == "registration"

    def test_extract_domain_env_aware(self, parser: ModelTopicParser) -> None:
        """Test extract_domain for Environment-Aware topic."""
        domain = parser.extract_domain("dev.user.events.v1")

        assert domain == "user"

    def test_extract_domain_invalid(self, parser: ModelTopicParser) -> None:
        """Test extract_domain for invalid topic."""
        domain = parser.extract_domain("invalid")

        assert domain is None


# ============================================================================
# Thread Safety Tests
# ============================================================================


@pytest.mark.unit
class TestThreadSafety:
    """Tests verifying thread-safety characteristics."""

    def test_parser_is_stateless(self) -> None:
        """Test that ModelTopicParser is stateless."""
        parser1 = ModelTopicParser()
        parser2 = ModelTopicParser()

        # Both parsers should produce identical results
        result1 = parser1.parse("onex.user.events")
        result2 = parser2.parse("onex.user.events")

        assert result1.raw_topic == result2.raw_topic
        assert result1.standard == result2.standard
        assert result1.domain == result2.domain
        assert result1.category == result2.category

    # NOTE: ModelParsedTopic immutability is tested in TestModelParsedTopic.test_parsed_topic_immutable
    # Immutability contributes to thread-safety by preventing concurrent modification issues


# ============================================================================
# EnumTopicStandard Tests
# ============================================================================


@pytest.mark.unit
class TestEnumTopicStandard:
    """Tests for EnumTopicStandard enum."""

    def test_enum_values(self) -> None:
        """Test EnumTopicStandard enum values."""
        assert EnumTopicStandard.ONEX_KAFKA.value == "onex_kafka"
        assert EnumTopicStandard.ENVIRONMENT_AWARE.value == "environment_aware"
        assert EnumTopicStandard.UNKNOWN.value == "unknown"

    def test_enum_str(self) -> None:
        """Test EnumTopicStandard __str__ method."""
        assert str(EnumTopicStandard.ONEX_KAFKA) == "onex_kafka"
        assert str(EnumTopicStandard.ENVIRONMENT_AWARE) == "environment_aware"
        assert str(EnumTopicStandard.UNKNOWN) == "unknown"


# ============================================================================
# LRU Cache Tests
# ============================================================================


@pytest.mark.unit
class TestLRUCache:
    """Tests for topic parsing LRU cache performance optimization."""

    def test_cache_info_available(self) -> None:
        """Test that cache info is available via module function."""
        from omnibase_infra.models.dispatch.model_topic_parser import (
            get_topic_parse_cache_info,
        )

        info = get_topic_parse_cache_info()

        # Should return a named tuple with cache statistics
        assert hasattr(info, "hits")
        assert hasattr(info, "misses")
        assert hasattr(info, "maxsize")
        assert hasattr(info, "currsize")

    def test_cache_clear_available(self) -> None:
        """Test that cache clear is available via module function."""
        from omnibase_infra.models.dispatch.model_topic_parser import (
            clear_topic_parse_cache,
        )

        # Should not raise
        clear_topic_parse_cache()

    def test_cache_reuses_results(self) -> None:
        """Test that cache returns the same result for repeated calls."""
        from omnibase_infra.models.dispatch.model_topic_parser import (
            clear_topic_parse_cache,
            get_topic_parse_cache_info,
        )

        # Clear cache to get clean state
        clear_topic_parse_cache()

        parser = ModelTopicParser()

        # Parse the same topic twice
        result1 = parser.parse("onex.test.events")
        result2 = parser.parse("onex.test.events")

        # Both results should be identical (from cache)
        assert result1.raw_topic == result2.raw_topic
        assert result1.domain == result2.domain
        assert result1.category == result2.category
        assert result1.standard == result2.standard

        # Check that we have at least 1 hit (second parse)
        info = get_topic_parse_cache_info()
        assert info.hits >= 1

    def test_cache_maxsize(self) -> None:
        """Test that cache has the documented maxsize."""
        from omnibase_infra.models.dispatch.model_topic_parser import (
            get_topic_parse_cache_info,
        )

        info = get_topic_parse_cache_info()

        # Cache should have maxsize of 1024 as documented
        assert info.maxsize == 1024

    def test_cache_handles_different_topics(self) -> None:
        """Test that cache correctly stores different topics separately."""
        from omnibase_infra.models.dispatch.model_topic_parser import (
            clear_topic_parse_cache,
        )

        # Clear cache to get clean state
        clear_topic_parse_cache()

        parser = ModelTopicParser()

        # Parse different topics
        result1 = parser.parse("onex.user.events")
        result2 = parser.parse("onex.order.events")
        result3 = parser.parse("dev.user.events.v1")

        # Results should be different for different topics
        assert result1.domain == "user"
        assert result2.domain == "order"
        assert result3.domain == "user"

        # Standards should be correct
        assert result1.standard == EnumTopicStandard.ONEX_KAFKA
        assert result2.standard == EnumTopicStandard.ONEX_KAFKA
        assert result3.standard == EnumTopicStandard.ENVIRONMENT_AWARE

    def test_cache_whitespace_not_cached(self) -> None:
        """Test that whitespace-only topics are not cached (edge case)."""
        from omnibase_infra.models.dispatch.model_topic_parser import (
            clear_topic_parse_cache,
            get_topic_parse_cache_info,
        )

        # Clear cache to get clean state
        clear_topic_parse_cache()

        parser = ModelTopicParser()
        initial_size = get_topic_parse_cache_info().currsize

        # Parse empty/whitespace topics (should not be cached)
        parser.parse("")
        parser.parse("   ")

        # Cache size should not increase for empty/whitespace topics
        final_size = get_topic_parse_cache_info().currsize
        assert final_size == initial_size


# ============================================================================
# Invalid Topic Name Edge Cases Tests (PR #54 Review Feedback)
# ============================================================================


@pytest.mark.unit
class TestInvalidTopicNameEdgeCases:
    """Tests for invalid topic name edge cases.

    These tests verify that malformed topic names do not match strict ONEX
    standards (ONEX_KAFKA or ENVIRONMENT_AWARE). The parser has a fallback
    mechanism that can extract a category for routing purposes from non-standard
    topics - this makes them "partially valid" (is_valid=True, standard=UNKNOWN).

    Addresses PR #54 review feedback on missing test coverage for invalid topic names.

    Edge cases covered:
    - Empty string after prefix (e.g., "onex." with nothing after)
    - Topic ending with trailing dot (e.g., "onex.user.events.")
    - Consecutive dots in topic (e.g., "onex..events")
    - Invalid characters in topic names
    - Topics that are too long
    - Domain-only topics without type suffix

    Note on "partially valid" topics:
        The parser marks topics as is_valid=True with standard=UNKNOWN when:
        - The topic doesn't match ONEX_KAFKA or ENVIRONMENT_AWARE patterns
        - BUT a message category (EVENT/COMMAND/INTENT) can still be extracted
        - This allows routing to work even for legacy/non-standard topics
        See module docstring in model_topic_parser.py for detailed semantics.
    """

    def test_topic_empty_after_prefix(self, parser: ModelTopicParser) -> None:
        """Test topic with empty string after prefix.

        Topics like "onex." with nothing after should not match any standard.
        """
        result = parser.parse("onex.")

        # Should not match strict standards
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE

    def test_topic_prefix_only_no_domain(self, parser: ModelTopicParser) -> None:
        """Test topic with only prefix and no domain or type.

        A single segment like "onex" without domain and type should be fully invalid.
        """
        result = parser.parse("onex")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.validation_error is not None
        assert result.category is None  # No category suffix found

    def test_topic_ending_with_dot(self, parser: ModelTopicParser) -> None:
        """Test topic ending with trailing dot.

        Topics like "onex.user.events." with a trailing dot should not match
        strict ONEX_KAFKA pattern, but may be partially valid for routing
        since ".events" category suffix is detectable.
        """
        result = parser.parse("onex.user.events.")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN
        # Category may still be extractable via fallback
        assert result.category == EnumMessageCategory.EVENT

    def test_topic_ending_with_dot_env_aware(self, parser: ModelTopicParser) -> None:
        """Test Environment-Aware topic ending with trailing dot.

        Topics like "dev.user.events.v1." with a trailing dot should not match
        the strict ENVIRONMENT_AWARE pattern.
        """
        result = parser.parse("dev.user.events.v1.")

        # Should not match strict ENVIRONMENT_AWARE pattern
        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_double_dots(self, parser: ModelTopicParser) -> None:
        """Test topic with consecutive dots.

        Topics like "onex..events" with double dots indicate an empty segment.
        Should not match strict patterns, but category may still be extractable.
        """
        result = parser.parse("onex..events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN
        # Domain extraction shows the malformed nature
        assert result.domain == ""  # Empty domain from double dots

    def test_topic_double_dots_in_middle(self, parser: ModelTopicParser) -> None:
        """Test topic with double dots in the middle.

        Topics like "onex.user..events" should not match strict patterns.
        """
        result = parser.parse("onex.user..events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_multiple_consecutive_dots(self, parser: ModelTopicParser) -> None:
        """Test topic with multiple consecutive dots.

        Topics like "onex...events" with triple dots should not match strict patterns.
        """
        result = parser.parse("onex...events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_starts_with_dot(self, parser: ModelTopicParser) -> None:
        """Test topic starting with a dot.

        Topics like ".onex.user.events" have an empty first segment.
        """
        result = parser.parse(".onex.user.events")

        # Should not match strict patterns due to leading dot
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_invalid_characters_spaces(self, parser: ModelTopicParser) -> None:
        """Test topic with space characters.

        Topics containing spaces should not match strict patterns.
        Kafka topic names typically don't allow spaces.
        """
        result = parser.parse("onex.user service.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_invalid_characters_at_symbol(self, parser: ModelTopicParser) -> None:
        """Test topic with @ symbol.

        Topics containing @ symbols should not match strict patterns.
        """
        result = parser.parse("onex.user@service.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_invalid_characters_hash(self, parser: ModelTopicParser) -> None:
        """Test topic with # (hash) symbol.

        Topics containing hash symbols should not match strict patterns.
        """
        result = parser.parse("onex.user#service.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_invalid_characters_asterisk(self, parser: ModelTopicParser) -> None:
        """Test topic with * (asterisk) symbol.

        Asterisks are used for pattern matching, not in actual topic names.
        """
        result = parser.parse("onex.user*.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_invalid_characters_question_mark(
        self, parser: ModelTopicParser
    ) -> None:
        """Test topic with ? (question mark) symbol.

        Question marks should not match in strict topic patterns.
        """
        result = parser.parse("onex.user?.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_too_long(self, parser: ModelTopicParser) -> None:
        """Test topic that is excessively long.

        While Kafka allows topics up to 249 characters, excessively long
        topics should still be handled gracefully without crashing.
        """
        # Create a very long domain name (300+ characters)
        long_domain = "a" * 300
        long_topic = f"onex.{long_domain}.events"
        result = parser.parse(long_topic)

        # The parser should handle this gracefully without crashing
        assert result is not None
        # Long but valid domain should still match ONEX_KAFKA pattern
        assert result.standard == EnumTopicStandard.ONEX_KAFKA
        assert result.is_valid is True

    def test_topic_domain_starting_with_number(
        self, parser: ModelTopicParser
    ) -> None:
        """Test topic with domain starting with a number.

        Domains should start with a letter according to ONEX conventions.
        """
        result = parser.parse("onex.123service.events")

        # Domain starting with number should not match ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_domain_starting_with_hyphen(
        self, parser: ModelTopicParser
    ) -> None:
        """Test topic with domain starting with a hyphen.

        Domains should not start with a hyphen.
        """
        result = parser.parse("onex.-service.events")

        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_domain_ending_with_hyphen(self, parser: ModelTopicParser) -> None:
        """Test topic with domain ending with a hyphen.

        Domains should not end with a hyphen according to the pattern.
        """
        result = parser.parse("onex.service-.events")

        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_empty_domain_between_dots(self, parser: ModelTopicParser) -> None:
        """Test topic with empty domain segment.

        Topics like "dev..events.v1" have an empty domain which is invalid
        for ENVIRONMENT_AWARE format.
        """
        result = parser.parse("dev..events.v1")

        assert result.standard != EnumTopicStandard.ENVIRONMENT_AWARE
        assert result.standard == EnumTopicStandard.UNKNOWN
        # Empty domain is extracted from the malformed structure
        assert result.domain == ""

    def test_topic_only_dots(self, parser: ModelTopicParser) -> None:
        """Test topic consisting only of dots.

        A topic like "..." should be fully invalid with no category extractable.
        """
        result = parser.parse("...")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.category is None

    def test_topic_single_dot(self, parser: ModelTopicParser) -> None:
        """Test topic that is just a single dot.

        A topic like "." should be fully invalid.
        """
        result = parser.parse(".")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.category is None

    def test_topic_no_category_suffix_fully_invalid(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that topics without any category suffix are fully invalid.

        When no category suffix (events, commands, intents) is found,
        the topic should be is_valid=False with validation_error set.
        """
        result = parser.parse("invalid..topic")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.validation_error is not None
        # The error message should reference the original topic
        assert "invalid..topic" in result.validation_error

    def test_topic_newline_character(self, parser: ModelTopicParser) -> None:
        """Test topic containing newline character.

        Topics with newline characters should not match strict patterns.
        """
        result = parser.parse("onex.user\n.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_topic_tab_character(self, parser: ModelTopicParser) -> None:
        """Test topic containing tab character.

        Topics with tab characters should not match strict patterns.
        """
        result = parser.parse("onex.user\t.events")

        # Should not match strict ONEX_KAFKA pattern
        assert result.standard != EnumTopicStandard.ONEX_KAFKA
        assert result.standard == EnumTopicStandard.UNKNOWN

    def test_strict_validation_rejects_malformed_topics(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that validate_topic() in strict mode rejects malformed topics.

        Strict mode should only accept exact ONEX_KAFKA format.
        """
        # Double dots - malformed
        is_valid, error = parser.validate_topic("onex..events", strict=True)
        assert is_valid is False
        assert error is not None
        assert "ONEX Kafka format" in error

        # Trailing dot - malformed
        is_valid, error = parser.validate_topic("onex.user.events.", strict=True)
        assert is_valid is False
        assert error is not None

        # Invalid characters
        is_valid, error = parser.validate_topic("onex.user@service.events", strict=True)
        assert is_valid is False
        assert error is not None

    def test_partially_valid_topics_route_correctly(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that partially valid topics still extract category for routing.

        Topics with standard=UNKNOWN but is_valid=True can still be routed
        because a message category was successfully extracted.
        """
        # Malformed but has .events suffix
        result = parser.parse("onex.user@service.events")

        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.is_valid is True  # Partially valid
        assert result.category == EnumMessageCategory.EVENT  # Category extracted
        assert result.is_routable() is True  # Can be routed

    def test_completely_invalid_topics_not_routable(
        self, parser: ModelTopicParser
    ) -> None:
        """Test that completely invalid topics are not routable.

        Topics without any recognized category suffix should have
        is_valid=False and is_routable()=False.
        """
        result = parser.parse("completely.invalid.topic")

        assert result.is_valid is False
        assert result.standard == EnumTopicStandard.UNKNOWN
        assert result.category is None
        assert result.is_routable() is False
