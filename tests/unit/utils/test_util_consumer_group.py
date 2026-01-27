# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for Kafka consumer group ID utilities.

This test suite validates the consumer group derivation utilities:
- normalize_kafka_identifier: String normalization for Kafka consumer group IDs
- compute_consumer_group_id: Canonical consumer group ID computation from node identity
- ModelNodeIdentity: Typed identity model for ONEX nodes
- EnumConsumerGroupPurpose: Consumer group purpose classification enum

Test Categories:
    - TestNormalizeKafkaIdentifier: Normalization transformation tests
    - TestNormalizeKafkaIdentifierEdgeCases: Edge cases and error conditions
    - TestComputeConsumerGroupId: Canonical format and determinism tests
    - TestComputeConsumerGroupIdPurposes: Purpose-based differentiation tests
    - TestModelNodeIdentity: Identity model validation tests
    - TestEnumConsumerGroupPurpose: Enum value and conversion tests

.. versionadded:: 0.2.6
    Created as part of OMN-1602.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.utils.util_consumer_group import (
    KAFKA_CONSUMER_GROUP_MAX_LENGTH,
    compute_consumer_group_id,
    normalize_kafka_identifier,
)


class TestNormalizeKafkaIdentifier:
    """Tests for normalize_kafka_identifier() transformations.

    Verifies that the normalization function correctly applies:
    - Lowercasing
    - Invalid character replacement
    - Separator collapsing
    - Edge trimming
    - Length truncation with hash suffix
    """

    def test_lowercase_conversion(self) -> None:
        """Test that uppercase characters are converted to lowercase."""
        assert normalize_kafka_identifier("UPPER") == "upper"
        assert normalize_kafka_identifier("Lower") == "lower"
        assert normalize_kafka_identifier("MixedCase") == "mixedcase"
        assert (
            normalize_kafka_identifier("ALL_CAPS_WITH_UNDERSCORES")
            == "all_caps_with_underscores"
        )

    def test_invalid_char_replacement(self) -> None:
        """Test that invalid characters are replaced with underscore."""
        # Exclamation marks
        assert normalize_kafka_identifier("My Service!!") == "my_service"
        # Spaces
        assert normalize_kafka_identifier("hello world") == "hello_world"
        # At sign
        assert normalize_kafka_identifier("user@domain") == "user_domain"
        # Hash
        assert normalize_kafka_identifier("item#123") == "item_123"
        # Brackets
        assert normalize_kafka_identifier("array[0]") == "array_0"
        # Parentheses
        assert normalize_kafka_identifier("func()") == "func"
        # Asterisk
        assert normalize_kafka_identifier("wild*card") == "wild_card"

    def test_separator_collapsing(self) -> None:
        """Test that consecutive separators are collapsed to a single separator."""
        # Multiple periods
        assert normalize_kafka_identifier("foo..bar") == "foo.bar"
        # Multiple underscores
        assert normalize_kafka_identifier("foo__bar") == "foo_bar"
        # Multiple hyphens
        assert normalize_kafka_identifier("foo--bar") == "foo-bar"
        # Mixed consecutive separators - preserves first separator type
        assert normalize_kafka_identifier("foo..bar__baz") == "foo.bar_baz"
        assert normalize_kafka_identifier("a._-b") == "a.b"
        # Many consecutive separators
        assert normalize_kafka_identifier("a....b") == "a.b"

    def test_strip_leading_trailing_separators(self) -> None:
        """Test that leading and trailing separators are stripped."""
        # Leading period
        assert normalize_kafka_identifier(".test") == "test"
        # Trailing period
        assert normalize_kafka_identifier("test.") == "test"
        # Both leading and trailing
        assert normalize_kafka_identifier("..test..") == "test"
        # Underscores
        assert normalize_kafka_identifier("__test__") == "test"
        # Hyphens
        assert normalize_kafka_identifier("--test--") == "test"
        # Mixed
        assert normalize_kafka_identifier("._-test-_.") == "test"

    def test_mixed_case_and_special_chars(self) -> None:
        """Test combination of case conversion and special character handling."""
        assert normalize_kafka_identifier("My Service!!") == "my_service"
        assert normalize_kafka_identifier("  UPPER_Case-Test  ") == "upper_case-test"
        assert normalize_kafka_identifier("Hello@World#2024") == "hello_world_2024"
        assert normalize_kafka_identifier("NODE.Name__V1") == "node.name_v1"
        assert normalize_kafka_identifier("..ABC..DEF..") == "abc.def"

    def test_valid_identifier_unchanged(self) -> None:
        """Test that already-valid identifiers are unchanged (except case)."""
        assert (
            normalize_kafka_identifier("valid.consumer-group_id")
            == "valid.consumer-group_id"
        )
        assert normalize_kafka_identifier("dev") == "dev"
        assert normalize_kafka_identifier("node_name_123") == "node_name_123"
        assert normalize_kafka_identifier("a-b-c") == "a-b-c"
        assert normalize_kafka_identifier("a.b.c") == "a.b.c"

    def test_preserves_valid_separators(self) -> None:
        """Test that valid separators (period, underscore, hyphen) are preserved."""
        assert normalize_kafka_identifier("a.b.c") == "a.b.c"
        assert normalize_kafka_identifier("a_b_c") == "a_b_c"
        assert normalize_kafka_identifier("a-b-c") == "a-b-c"
        assert normalize_kafka_identifier("a.b_c-d") == "a.b_c-d"

    def test_numeric_identifiers(self) -> None:
        """Test identifiers with numeric components."""
        assert normalize_kafka_identifier("node123") == "node123"
        assert normalize_kafka_identifier("v1.0.0") == "v1.0.0"
        assert normalize_kafka_identifier("123") == "123"
        assert normalize_kafka_identifier("service_v2") == "service_v2"


class TestNormalizeKafkaIdentifierEdgeCases:
    """Tests for normalize_kafka_identifier() edge cases and error conditions.

    Verifies correct error handling for:
    - Empty strings
    - Whitespace-only strings
    - Strings that normalize to empty
    - Length truncation with hash suffix
    """

    def test_empty_string_raises_value_error(self) -> None:
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_kafka_identifier("")

    def test_whitespace_only_raises_value_error(self) -> None:
        """Test that whitespace-only string raises ValueError.

        Whitespace characters are invalid and replaced with underscores,
        which are then stripped, resulting in an empty string.
        """
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("   ")
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("\t\n")
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("   \t   ")

    def test_result_empty_after_normalization_raises_value_error(self) -> None:
        """Test that strings normalizing to empty raise ValueError."""
        # Only invalid characters
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("@#$%^&*()")
        # Only separators (stripped away)
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("...")
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("___")
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("---")
        # Only invalid chars that become separators then stripped
        with pytest.raises(ValueError, match="results in empty string"):
            normalize_kafka_identifier("!!!...")

    def test_max_length_truncation_with_hash_suffix(self) -> None:
        """Test that long identifiers are truncated with hash suffix."""
        # Create a string longer than 255 characters
        long_value = "a" * 300
        result = normalize_kafka_identifier(long_value)

        # Result should be exactly 255 characters
        assert len(result) == KAFKA_CONSUMER_GROUP_MAX_LENGTH

        # Should end with underscore + 8-char hash
        assert result[-9] == "_"
        assert len(result[-8:]) == 8  # 8 character hash suffix

        # Prefix should be truncated appropriately (255 - 9 = 246 chars)
        assert result[:246] == "a" * 246

    def test_max_length_truncation_deterministic(self) -> None:
        """Test that truncation produces deterministic results."""
        long_value = "x" * 300
        result1 = normalize_kafka_identifier(long_value)
        result2 = normalize_kafka_identifier(long_value)

        # Same input should produce same output
        assert result1 == result2

    def test_max_length_different_inputs_different_hashes(self) -> None:
        """Test that different long inputs produce different hash suffixes."""
        long_value_a = "a" * 300
        long_value_b = "b" * 300

        result_a = normalize_kafka_identifier(long_value_a)
        result_b = normalize_kafka_identifier(long_value_b)

        # Hash suffixes should differ
        assert result_a[-8:] != result_b[-8:]

    def test_exactly_max_length_not_truncated(self) -> None:
        """Test that strings exactly at max length are not truncated."""
        # 255 characters exactly
        exact_value = "a" * 255
        result = normalize_kafka_identifier(exact_value)

        # Should not have hash suffix
        assert result == exact_value
        assert "_" not in result[-9:]

    def test_single_character(self) -> None:
        """Test single character inputs."""
        assert normalize_kafka_identifier("a") == "a"
        assert normalize_kafka_identifier("Z") == "z"
        assert normalize_kafka_identifier("1") == "1"

    def test_unicode_replacement(self) -> None:
        """Test that Unicode characters are replaced with underscore."""
        assert normalize_kafka_identifier("cafe") == "cafe"
        # Non-ASCII characters get replaced
        result = normalize_kafka_identifier(
            "cafe\u0301"
        )  # e with combining acute accent
        assert "cafe" in result


class TestComputeConsumerGroupId:
    """Tests for compute_consumer_group_id() canonical format.

    Verifies:
    - Canonical format: {env}.{service}.{node_name}.{purpose}.{version}
    - Component normalization
    - Deterministic output
    """

    def test_basic_canonical_format(self) -> None:
        """Test basic canonical format: dev.service.node.consume.v1."""
        identity = ModelNodeIdentity(
            env="dev",
            service="omniintelligence",
            node_name="claude_hook_event_effect",
            version="v1",
        )
        result = compute_consumer_group_id(identity)

        assert result == "dev.omniintelligence.claude_hook_event_effect.consume.v1"

    def test_all_components_present(self) -> None:
        """Test that all identity components appear in the result."""
        identity = ModelNodeIdentity(
            env="prod",
            service="myservice",
            node_name="mynode",
            version="v2",
        )
        result = compute_consumer_group_id(identity)

        # All components should be present
        assert "prod" in result
        assert "myservice" in result
        assert "mynode" in result
        assert "consume" in result  # default purpose
        assert "v2" in result

        # Should have exactly 5 parts separated by periods
        parts = result.split(".")
        assert len(parts) == 5

    def test_deterministic_same_identity_same_output(self) -> None:
        """Test that same identity always produces same output."""
        identity = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

        result1 = compute_consumer_group_id(identity)
        result2 = compute_consumer_group_id(identity)
        result3 = compute_consumer_group_id(identity)

        assert result1 == result2 == result3

    def test_components_are_normalized(self) -> None:
        """Test that special characters in components are normalized."""
        # Identity with components that need normalization
        identity = ModelNodeIdentity(
            env="DEV",  # uppercase
            service="Omni Intelligence",  # space
            node_name="claude-hook-event-effect",  # hyphens (valid)
            version="V1.0.0",  # uppercase
        )
        result = compute_consumer_group_id(identity)

        # Should be lowercased and normalized
        assert result == "dev.omni_intelligence.claude-hook-event-effect.consume.v1.0.0"

    def test_version_with_semver_format(self) -> None:
        """Test version strings with semantic versioning format."""
        identity = ModelNodeIdentity(
            env="staging",
            service="api",
            node_name="handler",
            version="1.2.3",
        )
        result = compute_consumer_group_id(identity)

        assert result == "staging.api.handler.consume.1.2.3"

    def test_version_with_v_prefix(self) -> None:
        """Test version strings with v prefix."""
        identity = ModelNodeIdentity(
            env="dev",
            service="svc",
            node_name="node",
            version="v2.0.0-beta",
        )
        result = compute_consumer_group_id(identity)

        assert result == "dev.svc.node.consume.v2.0.0-beta"


class TestComputeConsumerGroupIdPurposes:
    """Tests for compute_consumer_group_id() with different purposes.

    Verifies:
    - Each purpose produces distinct output
    - No collisions between purposes
    - All enum values work correctly
    """

    @pytest.fixture
    def identity(self) -> ModelNodeIdentity:
        """Provide a standard identity for purpose tests."""
        return ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

    def test_default_purpose_is_consume(self, identity: ModelNodeIdentity) -> None:
        """Test that default purpose is CONSUME."""
        result = compute_consumer_group_id(identity)
        assert ".consume." in result

    def test_different_purposes_produce_distinct_ids(
        self, identity: ModelNodeIdentity
    ) -> None:
        """Test that different purposes produce different consumer group IDs."""
        result_consume = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.CONSUME
        )
        result_introspection = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.INTROSPECTION
        )
        result_replay = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.REPLAY
        )
        result_audit = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.AUDIT
        )
        result_backfill = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.BACKFILL
        )

        # All should be distinct
        all_results = [
            result_consume,
            result_introspection,
            result_replay,
            result_audit,
            result_backfill,
        ]
        assert len(set(all_results)) == 5, (
            "All purposes should produce unique group IDs"
        )

    def test_introspection_vs_consume_no_collision(
        self, identity: ModelNodeIdentity
    ) -> None:
        """Test that introspection and consume purposes don't collide."""
        result_consume = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.CONSUME
        )
        result_introspection = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.INTROSPECTION
        )

        assert result_consume != result_introspection
        assert ".consume." in result_consume
        assert ".introspection." in result_introspection

    def test_all_purpose_enum_values_work(self, identity: ModelNodeIdentity) -> None:
        """Test that all EnumConsumerGroupPurpose values work correctly."""
        for purpose in EnumConsumerGroupPurpose:
            result = compute_consumer_group_id(identity, purpose)

            # Should contain the purpose value
            assert f".{purpose.value}." in result

            # Should be valid (no errors raised)
            assert len(result) <= KAFKA_CONSUMER_GROUP_MAX_LENGTH

    def test_explicit_consume_purpose(self, identity: ModelNodeIdentity) -> None:
        """Test explicit CONSUME purpose matches default."""
        result_default = compute_consumer_group_id(identity)
        result_explicit = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.CONSUME
        )

        assert result_default == result_explicit

    def test_replay_purpose(self, identity: ModelNodeIdentity) -> None:
        """Test REPLAY purpose format."""
        result = compute_consumer_group_id(identity, EnumConsumerGroupPurpose.REPLAY)
        assert ".replay." in result

    def test_audit_purpose(self, identity: ModelNodeIdentity) -> None:
        """Test AUDIT purpose format."""
        result = compute_consumer_group_id(identity, EnumConsumerGroupPurpose.AUDIT)
        assert ".audit." in result

    def test_backfill_purpose(self, identity: ModelNodeIdentity) -> None:
        """Test BACKFILL purpose format."""
        result = compute_consumer_group_id(identity, EnumConsumerGroupPurpose.BACKFILL)
        assert ".backfill." in result


class TestComputeConsumerGroupIdLengthHandling:
    """Tests for compute_consumer_group_id() length handling.

    Verifies that long identities are truncated with hash suffix
    instead of raising errors, maintaining Kafka's 255 char limit.
    """

    def test_long_components_truncated_with_hash(self) -> None:
        """Test that very long components are truncated with hash suffix."""
        # Create identity with very long components
        identity = ModelNodeIdentity(
            env="development_environment_with_very_long_name",
            service="a" * 100,
            node_name="b" * 100,
            version="v1",
        )

        # Should succeed with truncation, not raise ValueError
        result = compute_consumer_group_id(identity)

        # Result should be exactly 255 characters (max length)
        assert len(result) == KAFKA_CONSUMER_GROUP_MAX_LENGTH

        # Should end with underscore + 8-char hash
        assert result[-9] == "_"
        assert len(result[-8:]) == 8

    def test_truncation_is_deterministic(self) -> None:
        """Test that truncation produces deterministic results."""
        identity = ModelNodeIdentity(
            env="x" * 100,
            service="y" * 100,
            node_name="z" * 100,
            version="v1",
        )

        result1 = compute_consumer_group_id(identity)
        result2 = compute_consumer_group_id(identity)

        # Same identity should produce same truncated output
        assert result1 == result2

    def test_different_identities_different_hashes(self) -> None:
        """Test that different long identities produce different hash suffixes."""
        identity_a = ModelNodeIdentity(
            env="a" * 100,
            service="a" * 100,
            node_name="a" * 100,
            version="v1",
        )
        identity_b = ModelNodeIdentity(
            env="b" * 100,
            service="b" * 100,
            node_name="b" * 100,
            version="v1",
        )

        result_a = compute_consumer_group_id(identity_a)
        result_b = compute_consumer_group_id(identity_b)

        # Hash suffixes should differ
        assert result_a[-8:] != result_b[-8:]

    def test_normal_length_not_truncated(self) -> None:
        """Test that normal-length identities are not truncated."""
        identity = ModelNodeIdentity(
            env="dev",
            service="myservice",
            node_name="mynode",
            version="v1",
        )

        result = compute_consumer_group_id(identity)

        # Should be the full canonical format without hash suffix
        assert result == "dev.myservice.mynode.consume.v1"
        assert "_" not in result[-9:]  # No hash suffix pattern

    def test_purpose_affects_truncation_hash(self) -> None:
        """Test that different purposes produce different hashes for long identities."""
        identity = ModelNodeIdentity(
            env="x" * 100,
            service="y" * 100,
            node_name="z" * 100,
            version="v1",
        )

        result_consume = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.CONSUME
        )
        result_introspection = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.INTROSPECTION
        )

        # Both should be truncated to max length
        assert len(result_consume) == KAFKA_CONSUMER_GROUP_MAX_LENGTH
        assert len(result_introspection) == KAFKA_CONSUMER_GROUP_MAX_LENGTH

        # Hash suffixes should differ due to different purpose in hash input
        assert result_consume[-8:] != result_introspection[-8:]


class TestModelNodeIdentity:
    """Tests for ModelNodeIdentity Pydantic model.

    Verifies:
    - Immutability (frozen)
    - Extra fields forbidden
    - Non-empty string validation
    - Whitespace-only validation
    """

    def test_valid_construction(self) -> None:
        """Test that valid inputs construct successfully."""
        identity = ModelNodeIdentity(
            env="dev",
            service="myservice",
            node_name="mynode",
            version="v1",
        )

        assert identity.env == "dev"
        assert identity.service == "myservice"
        assert identity.node_name == "mynode"
        assert identity.version == "v1"

    def test_frozen_immutable(self) -> None:
        """Test that the model is frozen (immutable)."""
        identity = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

        # Attempting to modify should raise ValidationError
        with pytest.raises(ValidationError):
            identity.env = "prod"  # type: ignore[misc]

    def test_frozen_all_fields_immutable(self) -> None:
        """Test that all fields are immutable."""
        identity = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

        with pytest.raises(ValidationError):
            identity.service = "other"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            identity.node_name = "other"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            identity.version = "v2"  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="dev",
                service="service",
                node_name="node",
                version="v1",
                extra_field="not_allowed",  # type: ignore[call-arg]
            )

        # Verify the error is about extra field
        assert "extra_field" in str(exc_info.value)

    def test_empty_string_env_raises_validation_error(self) -> None:
        """Test that empty string for env raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="",
                service="service",
                node_name="node",
                version="v1",
            )

        assert "must not be empty" in str(exc_info.value)

    def test_empty_string_service_raises_validation_error(self) -> None:
        """Test that empty string for service raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="dev",
                service="",
                node_name="node",
                version="v1",
            )

        assert "must not be empty" in str(exc_info.value)

    def test_empty_string_node_name_raises_validation_error(self) -> None:
        """Test that empty string for node_name raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="dev",
                service="service",
                node_name="",
                version="v1",
            )

        assert "must not be empty" in str(exc_info.value)

    def test_empty_string_version_raises_validation_error(self) -> None:
        """Test that empty string for version raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="dev",
                service="service",
                node_name="node",
                version="",
            )

        assert "must not be empty" in str(exc_info.value)

    def test_whitespace_only_env_raises_validation_error(self) -> None:
        """Test that whitespace-only env raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="   ",
                service="service",
                node_name="node",
                version="v1",
            )

        assert "whitespace" in str(exc_info.value).lower()

    def test_whitespace_only_service_raises_validation_error(self) -> None:
        """Test that whitespace-only service raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIdentity(
                env="dev",
                service="\t\n",
                node_name="node",
                version="v1",
            )

        assert "whitespace" in str(exc_info.value).lower()

    def test_whitespace_only_all_fields(self) -> None:
        """Test whitespace-only validation for all fields."""
        whitespace_variants = ["   ", "\t", "\n", " \t \n "]

        for ws in whitespace_variants:
            # Each field should reject whitespace-only
            with pytest.raises(ValidationError):
                ModelNodeIdentity(env=ws, service="s", node_name="n", version="v")

            with pytest.raises(ValidationError):
                ModelNodeIdentity(env="e", service=ws, node_name="n", version="v")

            with pytest.raises(ValidationError):
                ModelNodeIdentity(env="e", service="s", node_name=ws, version="v")

            with pytest.raises(ValidationError):
                ModelNodeIdentity(env="e", service="s", node_name="n", version=ws)

    def test_strict_mode_rejects_non_string(self) -> None:
        """Test that strict mode rejects non-string types."""
        with pytest.raises(ValidationError):
            ModelNodeIdentity(
                env=123,  # type: ignore[arg-type]
                service="service",
                node_name="node",
                version="v1",
            )

    def test_hashable_for_dict_key(self) -> None:
        """Test that frozen model can be used as dict key."""
        identity1 = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )
        identity2 = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

        # Should be usable as dict keys
        d = {identity1: "value1"}
        d[identity2] = "value2"

        # Same identity should overwrite
        assert len(d) == 1
        assert d[identity1] == "value2"

    def test_equality(self) -> None:
        """Test equality comparison for identical identities."""
        identity1 = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )
        identity2 = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

        assert identity1 == identity2

    def test_inequality(self) -> None:
        """Test inequality for different identities."""
        identity1 = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )
        identity2 = ModelNodeIdentity(
            env="prod",  # different
            service="service",
            node_name="node",
            version="v1",
        )

        assert identity1 != identity2


class TestEnumConsumerGroupPurpose:
    """Tests for EnumConsumerGroupPurpose enum.

    Verifies:
    - All 5 values exist
    - String conversion works
    - Values are lowercase
    """

    def test_all_five_values_exist(self) -> None:
        """Test that all 5 purpose values exist."""
        # Should have exactly 5 members
        assert len(EnumConsumerGroupPurpose) == 5

        # All expected values should exist
        assert EnumConsumerGroupPurpose.CONSUME is not None
        assert EnumConsumerGroupPurpose.INTROSPECTION is not None
        assert EnumConsumerGroupPurpose.REPLAY is not None
        assert EnumConsumerGroupPurpose.AUDIT is not None
        assert EnumConsumerGroupPurpose.BACKFILL is not None

    def test_values_are_lowercase(self) -> None:
        """Test that all enum values are lowercase strings."""
        for purpose in EnumConsumerGroupPurpose:
            assert purpose.value == purpose.value.lower(), (
                f"{purpose.name} value is not lowercase"
            )

    def test_specific_values(self) -> None:
        """Test specific enum values."""
        assert EnumConsumerGroupPurpose.CONSUME.value == "consume"
        assert EnumConsumerGroupPurpose.INTROSPECTION.value == "introspection"
        assert EnumConsumerGroupPurpose.REPLAY.value == "replay"
        assert EnumConsumerGroupPurpose.AUDIT.value == "audit"
        assert EnumConsumerGroupPurpose.BACKFILL.value == "backfill"

    def test_string_conversion_via_str(self) -> None:
        """Test that __str__ returns the value."""
        assert str(EnumConsumerGroupPurpose.CONSUME) == "consume"
        assert str(EnumConsumerGroupPurpose.INTROSPECTION) == "introspection"
        assert str(EnumConsumerGroupPurpose.REPLAY) == "replay"
        assert str(EnumConsumerGroupPurpose.AUDIT) == "audit"
        assert str(EnumConsumerGroupPurpose.BACKFILL) == "backfill"

    def test_string_conversion_in_format_string(self) -> None:
        """Test enum works correctly in format strings."""
        purpose = EnumConsumerGroupPurpose.REPLAY
        result = f"group-{purpose}"
        assert result == "group-replay"

    def test_is_str_subclass(self) -> None:
        """Test that enum is a str subclass (str, Enum pattern)."""
        purpose = EnumConsumerGroupPurpose.CONSUME
        assert isinstance(purpose, str)
        assert isinstance(purpose.value, str)

    def test_comparison_with_string(self) -> None:
        """Test that enum can be compared with string values."""
        purpose = EnumConsumerGroupPurpose.CONSUME
        # Due to str subclass, direct comparison works
        assert purpose == "consume"
        assert purpose.value == "consume"

    def test_from_string_value(self) -> None:
        """Test creating enum from string value."""
        purpose = EnumConsumerGroupPurpose("consume")
        assert purpose == EnumConsumerGroupPurpose.CONSUME

        purpose = EnumConsumerGroupPurpose("introspection")
        assert purpose == EnumConsumerGroupPurpose.INTROSPECTION

    def test_invalid_value_raises_error(self) -> None:
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            EnumConsumerGroupPurpose("invalid_purpose")

    def test_iteration(self) -> None:
        """Test that enum can be iterated."""
        values = list(EnumConsumerGroupPurpose)
        assert len(values) == 5
        assert EnumConsumerGroupPurpose.CONSUME in values
        assert EnumConsumerGroupPurpose.INTROSPECTION in values


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_consume(self) -> None:
        """Test full workflow with CONSUME purpose."""
        identity = ModelNodeIdentity(
            env="production",
            service="order-processor",
            node_name="order-validation-effect",
            version="v2.1.0",
        )

        group_id = compute_consumer_group_id(identity, EnumConsumerGroupPurpose.CONSUME)

        assert (
            group_id
            == "production.order-processor.order-validation-effect.consume.v2.1.0"
        )
        assert len(group_id) <= KAFKA_CONSUMER_GROUP_MAX_LENGTH

    def test_full_workflow_introspection(self) -> None:
        """Test full workflow with INTROSPECTION purpose."""
        identity = ModelNodeIdentity(
            env="dev",
            service="omniintelligence",
            node_name="claude_hook_event_effect",
            version="v1",
        )

        group_id = compute_consumer_group_id(
            identity, EnumConsumerGroupPurpose.INTROSPECTION
        )

        assert (
            group_id == "dev.omniintelligence.claude_hook_event_effect.introspection.v1"
        )

    def test_same_identity_different_purposes_are_distinct(self) -> None:
        """Test that same identity with different purposes produces distinct group IDs."""
        identity = ModelNodeIdentity(
            env="dev",
            service="service",
            node_name="node",
            version="v1",
        )

        group_ids = {
            compute_consumer_group_id(identity, purpose)
            for purpose in EnumConsumerGroupPurpose
        }

        # All should be distinct
        assert len(group_ids) == len(EnumConsumerGroupPurpose)

    def test_normalization_applied_in_compute(self) -> None:
        """Test that normalization is applied during compute_consumer_group_id."""
        identity = ModelNodeIdentity(
            env="UPPER_ENV",
            service="Service Name",
            node_name="node!!special",
            version="V1",
        )

        group_id = compute_consumer_group_id(identity)

        # All components should be normalized
        assert group_id == "upper_env.service_name.node_special.consume.v1"


__all__: list[str] = [
    "TestNormalizeKafkaIdentifier",
    "TestNormalizeKafkaIdentifierEdgeCases",
    "TestComputeConsumerGroupId",
    "TestComputeConsumerGroupIdPurposes",
    "TestComputeConsumerGroupIdLengthHandling",
    "TestModelNodeIdentity",
    "TestEnumConsumerGroupPurpose",
    "TestIntegration",
]
