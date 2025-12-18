# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelNodeRegistrationMetadata model.

Comprehensive test coverage for ModelNodeRegistrationMetadata including:
- Basic instantiation (minimal and full field sets)
- Tag normalization (lowercase, whitespace, dedup, empty filtering)
- Tag limits (max exceeded, boundary cases, after-dedup scenarios)
- Label validation (k8s-style keys, key normalization, invalid keys)
- Label limits (max exceeded, exactly max, value type coercion)
- Label security (sanitized error messages)
- Environment enum handling (enum and string input, required field)
- Model immutability (frozen config)
- Serialization (model_dump, model_dump_json, model_copy)
- Extra field rejection (extra='forbid' config)
- Optional field handling (release_channel, region)
- Boundary inputs (single character tags/labels)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_registry_effect.v1_0_0.models import (
    EnumEnvironment,
    ModelNodeRegistrationMetadata,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_node_registration_metadata import (
    MAX_LABELS,
    MAX_TAGS,
)


class TestModelNodeRegistrationMetadataBasicInstantiation:
    """Test basic model instantiation."""

    def test_valid_instantiation_minimal(self) -> None:
        """Test minimal valid instantiation with required fields only."""
        metadata = ModelNodeRegistrationMetadata(environment=EnumEnvironment.TESTING)
        assert metadata.environment == EnumEnvironment.TESTING
        assert metadata.tags == []
        assert metadata.labels == {}
        assert metadata.release_channel is None
        assert metadata.region is None

    def test_valid_instantiation_full(self) -> None:
        """Test full instantiation with all fields."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.PRODUCTION,
            tags=["tag1", "tag2"],
            labels={"app": "test", "version": "1.0"},
            release_channel="stable",
            region="us-east-1",
        )
        assert metadata.environment == EnumEnvironment.PRODUCTION
        assert metadata.tags == ["tag1", "tag2"]
        assert metadata.labels == {"app": "test", "version": "1.0"}
        assert metadata.release_channel == "stable"
        assert metadata.region == "us-east-1"


class TestModelNodeRegistrationMetadataTagNormalization:
    """Test tag normalization behavior."""

    def test_tags_normalized_to_lowercase(self) -> None:
        """Test that tags are normalized to lowercase."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["TAG1", "Tag2", "tAg3"],
        )
        assert metadata.tags == ["tag1", "tag2", "tag3"]

    def test_tags_stripped_of_whitespace(self) -> None:
        """Test that tags have whitespace stripped."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["  tag1  ", " tag2", "tag3 "],
        )
        assert metadata.tags == ["tag1", "tag2", "tag3"]

    def test_tags_deduplicated(self) -> None:
        """Test that duplicate tags are removed, preserving order."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["tag1", "tag2", "tag1", "tag3", "tag2"],
        )
        assert metadata.tags == ["tag1", "tag2", "tag3"]

    def test_tags_empty_strings_removed(self) -> None:
        """Test that empty strings are filtered out."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["tag1", "", "  ", "tag2"],
        )
        assert metadata.tags == ["tag1", "tag2"]

    def test_tags_empty_list_allowed(self) -> None:
        """Test that empty tag list is allowed."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=[],
        )
        assert metadata.tags == []


class TestModelNodeRegistrationMetadataTagLimits:
    """Test tag limit enforcement behavior.

    CRITICAL: Tags exceeding MAX_TAGS must raise an explicit error,
    NOT be silently truncated. This prevents unexpected data loss.
    """

    def test_tags_exceeding_max_raises_error(self) -> None:
        """Test that exceeding MAX_TAGS raises error (not silent truncation)."""
        tags = [f"tag{i}" for i in range(MAX_TAGS + 5)]

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                tags=tags,
            )

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert str(MAX_TAGS) in error_msg
        assert "tag" in error_msg.lower()

    def test_tags_exceeding_max_error_includes_count(self) -> None:
        """Test that tag limit error includes the actual count received."""
        tag_count = MAX_TAGS + 10
        tags = [f"tag{i}" for i in range(tag_count)]

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                tags=tags,
            )

        # Error should mention both the limit and received count
        error_msg = str(exc_info.value)
        assert str(MAX_TAGS) in error_msg  # Max allowed
        assert str(tag_count) in error_msg  # Received count

    def test_tags_exceeding_max_after_dedup_raises_error(self) -> None:
        """Test that limit is checked after deduplication."""
        # Create MAX_TAGS + 5 unique tags, plus some duplicates
        unique_tags = [f"tag{i}" for i in range(MAX_TAGS + 5)]
        tags_with_dups = unique_tags + ["tag0", "tag1", "tag2"]  # Add duplicates

        with pytest.raises(ValidationError):
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                tags=tags_with_dups,
            )

    def test_tags_within_limit_after_dedup_allowed(self) -> None:
        """Test that deduplication can bring count within limit."""
        # Create duplicates that will reduce to exactly MAX_TAGS
        tags = [f"tag{i % MAX_TAGS}" for i in range(MAX_TAGS + 10)]

        # This should succeed because after deduplication we have exactly MAX_TAGS
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=tags,
        )
        assert len(metadata.tags) == MAX_TAGS

    def test_exactly_max_tags_allowed(self) -> None:
        """Test that exactly MAX_TAGS is allowed without error."""
        tags = [f"tag{i}" for i in range(MAX_TAGS)]

        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=tags,
        )

        assert len(metadata.tags) == MAX_TAGS
        # All tags should be present
        for i in range(MAX_TAGS):
            assert f"tag{i}" in metadata.tags

    def test_one_below_max_tags_allowed(self) -> None:
        """Test boundary: MAX_TAGS - 1 is allowed."""
        tags = [f"tag{i}" for i in range(MAX_TAGS - 1)]

        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=tags,
        )

        assert len(metadata.tags) == MAX_TAGS - 1

    def test_one_above_max_tags_raises_error(self) -> None:
        """Test boundary: MAX_TAGS + 1 raises error."""
        tags = [f"tag{i}" for i in range(MAX_TAGS + 1)]

        with pytest.raises(ValidationError):
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                tags=tags,
            )

    def test_max_tags_constant_is_20(self) -> None:
        """Test that MAX_TAGS constant is 20 as documented."""
        assert MAX_TAGS == 20

    def test_tags_limit_error_suggests_action(self) -> None:
        """Test that error message provides actionable guidance."""
        tags = [f"tag{i}" for i in range(MAX_TAGS + 5)]

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                tags=tags,
            )

        error_msg = str(exc_info.value)
        # Error should suggest reducing tags or splitting
        assert "reduce" in error_msg.lower() or "split" in error_msg.lower()


class TestModelNodeRegistrationMetadataLabelValidation:
    """Test label validation behavior."""

    def test_labels_keys_normalized_to_lowercase(self) -> None:
        """Test that label keys are normalized to lowercase."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"APP": "test", "Version": "1.0"},
        )
        assert "app" in metadata.labels
        assert "version" in metadata.labels
        assert "APP" not in metadata.labels
        assert "Version" not in metadata.labels

    def test_labels_values_converted_to_string(self) -> None:
        """Test that label values are converted to strings."""
        # Note: Pydantic will coerce compatible types
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"count": "123", "enabled": "true"},
        )
        assert metadata.labels["count"] == "123"
        assert metadata.labels["enabled"] == "true"

    def test_labels_valid_k8s_style_keys(self) -> None:
        """Test various valid k8s-style label keys."""
        valid_keys = [
            "app",
            "version",
            "app-name",
            "my.domain.com",
            "app.kubernetes.io",
            "a1",
            "a-b",
            "a.b",
            "a-b.c-d",
        ]
        labels = dict.fromkeys(valid_keys, "value")
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels=labels,
        )
        assert len(metadata.labels) == len(valid_keys)

    def test_labels_invalid_key_raises_error(self) -> None:
        """Test that invalid label keys raise validation error.

        Note: Keys are normalized to lowercase before pattern matching,
        so pure UPPERCASE keys become valid lowercase. Only keys with
        invalid characters (spaces, underscores, etc.) raise errors.
        """
        invalid_keys = [
            "has space",
            "has_underscore",
            "-starts-with-dash",
            ".starts.with.dot",
            "ends-with-dash-",
            "ends.with.dot.",
        ]
        for key in invalid_keys:
            with pytest.raises(ValidationError):
                ModelNodeRegistrationMetadata(
                    environment=EnumEnvironment.TESTING,
                    labels={key: "value"},
                )

    def test_labels_max_limit_raises_error(self) -> None:
        """Test that exceeding MAX_LABELS raises error (not silent truncation)."""
        labels = {f"label{i}": f"value{i}" for i in range(MAX_LABELS + 1)}

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                labels=labels,
            )

        # Verify error message is informative
        error_msg = str(exc_info.value)
        assert str(MAX_LABELS) in error_msg
        assert "label" in error_msg.lower()

    def test_labels_exactly_max_allowed(self) -> None:
        """Test that exactly MAX_LABELS is allowed."""
        labels = {f"label{i}": f"value{i}" for i in range(MAX_LABELS)}
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels=labels,
        )
        assert len(metadata.labels) == MAX_LABELS

    def test_max_labels_constant_is_50(self) -> None:
        """Test that MAX_LABELS constant is 50 as documented."""
        assert MAX_LABELS == 50


class TestModelNodeRegistrationMetadataLabelSecurityValidation:
    """Test label key validation for security (sanitized error messages)."""

    def test_invalid_label_key_error_is_sanitized(self) -> None:
        """Test that invalid label key error message is sanitized.

        Error messages should not echo back potentially malicious input verbatim.
        The sanitization removes newlines and control characters.
        """
        # Try a key with underscores (invalid for k8s-style) and newlines
        malicious_key = "bad_key\nwith\rnewlines"

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                labels={malicious_key: "value"},
            )

        error_msg = str(exc_info.value)
        # The ValueError message from our validator should have sanitized the key
        # Check that if "Invalid label key" appears, the key portion doesn't have newlines
        if "Invalid label key" in error_msg:
            # Extract the portion after "Invalid label key format:"
            parts = error_msg.split("Invalid label key format:")
            if len(parts) > 1:
                key_part = parts[1].split("'")[1] if "'" in parts[1] else parts[1][:50]
                # Newlines and carriage returns should be removed
                assert "\n" not in key_part
                assert "\r" not in key_part


class TestModelNodeRegistrationMetadataEnvironment:
    """Test environment enum handling."""

    def test_all_environment_values_valid(self) -> None:
        """Test that all EnumEnvironment values can be used."""
        for env in EnumEnvironment:
            metadata = ModelNodeRegistrationMetadata(environment=env)
            assert metadata.environment == env

    def test_environment_required(self) -> None:
        """Test that environment field is required."""
        with pytest.raises(ValidationError):
            ModelNodeRegistrationMetadata()  # type: ignore[call-arg]


class TestModelNodeRegistrationMetadataImmutability:
    """Test model immutability (frozen)."""

    def test_model_is_frozen(self) -> None:
        """Test that the model is frozen (immutable)."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["tag1"],
        )

        with pytest.raises((ValidationError, TypeError)):
            metadata.environment = EnumEnvironment.PRODUCTION  # type: ignore[misc]


class TestModelNodeRegistrationMetadataSerialization:
    """Test model serialization."""

    def test_model_dump_json_roundtrip(self) -> None:
        """Test that model can be dumped and loaded."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.PRODUCTION,
            tags=["tag1", "tag2"],
            labels={"app": "test"},
            release_channel="stable",
            region="us-east-1",
        )

        # Dump to dict
        data = metadata.model_dump(mode="json")

        # Verify structure
        assert data["environment"] == "production"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["labels"] == {"app": "test"}
        assert data["release_channel"] == "stable"
        assert data["region"] == "us-east-1"

        # Round-trip
        restored = ModelNodeRegistrationMetadata(**data)
        assert restored == metadata

    def test_model_dump_json_string(self) -> None:
        """Test that model_dump_json returns valid JSON string."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.PRODUCTION,
            tags=["tag1"],
            labels={"app": "test"},
        )

        json_str = metadata.model_dump_json()

        # Verify it's a string
        assert isinstance(json_str, str)

        # Verify it can be parsed back
        import json

        data = json.loads(json_str)
        assert data["environment"] == "production"
        assert data["tags"] == ["tag1"]
        assert data["labels"] == {"app": "test"}

    def test_model_copy_creates_new_instance(self) -> None:
        """Test that model_copy creates a proper copy."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["tag1"],
        )

        # Create copy with updated field
        copied = metadata.model_copy(update={"region": "us-west-2"})

        assert copied.region == "us-west-2"
        assert metadata.region is None  # Original unchanged
        assert copied.tags == metadata.tags
        assert copied is not metadata


class TestModelNodeRegistrationMetadataExtraFields:
    """Test extra field handling (extra='forbid' config)."""

    def test_extra_fields_rejected(self) -> None:
        """Test that extra fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                unknown_field="value",  # type: ignore[call-arg]
            )

        error_msg = str(exc_info.value)
        assert "extra" in error_msg.lower() or "unknown_field" in error_msg.lower()

    def test_multiple_extra_fields_rejected(self) -> None:
        """Test that multiple extra fields are all caught."""
        with pytest.raises(ValidationError):
            ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                extra1="value1",  # type: ignore[call-arg]
                extra2="value2",  # type: ignore[call-arg]
            )


class TestModelNodeRegistrationMetadataOptionalFields:
    """Test optional field handling (release_channel, region)."""

    def test_release_channel_accepts_any_string(self) -> None:
        """Test that release_channel accepts various string values."""
        channels = ["stable", "canary", "beta", "alpha", "rc1", "custom-channel"]
        for channel in channels:
            metadata = ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                release_channel=channel,
            )
            assert metadata.release_channel == channel

    def test_release_channel_empty_string(self) -> None:
        """Test that release_channel accepts empty string."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            release_channel="",
        )
        assert metadata.release_channel == ""

    def test_region_accepts_any_string(self) -> None:
        """Test that region accepts various string values."""
        regions = ["us-east-1", "eu-west-1", "ap-southeast-2", "local", "custom"]
        for region in regions:
            metadata = ModelNodeRegistrationMetadata(
                environment=EnumEnvironment.TESTING,
                region=region,
            )
            assert metadata.region == region

    def test_region_empty_string(self) -> None:
        """Test that region accepts empty string."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            region="",
        )
        assert metadata.region == ""


class TestModelNodeRegistrationMetadataLabelValueCoercion:
    """Test label value type coercion."""

    def test_label_integer_value_coerced_to_string(self) -> None:
        """Test that integer label values are coerced to strings."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"count": 123},  # type: ignore[dict-item]
        )
        assert metadata.labels["count"] == "123"
        assert isinstance(metadata.labels["count"], str)

    def test_label_float_value_coerced_to_string(self) -> None:
        """Test that float label values are coerced to strings."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"ratio": 3.14},  # type: ignore[dict-item]
        )
        assert metadata.labels["ratio"] == "3.14"
        assert isinstance(metadata.labels["ratio"], str)

    def test_label_boolean_value_coerced_to_string(self) -> None:
        """Test that boolean label values are coerced to strings."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"enabled": True, "disabled": False},  # type: ignore[dict-item]
        )
        assert metadata.labels["enabled"] == "True"
        assert metadata.labels["disabled"] == "False"


class TestModelNodeRegistrationMetadataBoundaryInputs:
    """Test boundary inputs for tags and labels."""

    def test_single_character_tag(self) -> None:
        """Test that single character tags are valid."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            tags=["a", "b", "c"],
        )
        assert metadata.tags == ["a", "b", "c"]

    def test_single_character_label_key(self) -> None:
        """Test that single character label keys are valid."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"a": "value1", "b": "value2"},
        )
        assert metadata.labels["a"] == "value1"
        assert metadata.labels["b"] == "value2"

    def test_numeric_single_character_label_key(self) -> None:
        """Test that numeric single character label keys are valid."""
        metadata = ModelNodeRegistrationMetadata(
            environment=EnumEnvironment.TESTING,
            labels={"1": "value1", "9": "value9"},
        )
        assert metadata.labels["1"] == "value1"
        assert metadata.labels["9"] == "value9"


class TestModelNodeRegistrationMetadataEnvironmentStringInput:
    """Test environment field with string input."""

    def test_environment_accepts_string_value(self) -> None:
        """Test that environment accepts string values matching enum."""
        # Pydantic should coerce strings to enum values
        metadata = ModelNodeRegistrationMetadata(
            environment="production",  # type: ignore[arg-type]
        )
        assert metadata.environment == EnumEnvironment.PRODUCTION

    def test_environment_accepts_all_string_values(self) -> None:
        """Test that all environment string values are accepted."""
        for env in EnumEnvironment:
            metadata = ModelNodeRegistrationMetadata(
                environment=env.value,  # type: ignore[arg-type]
            )
            assert metadata.environment == env

    def test_environment_invalid_string_rejected(self) -> None:
        """Test that invalid environment string raises error."""
        with pytest.raises(ValidationError):
            ModelNodeRegistrationMetadata(
                environment="invalid_environment",  # type: ignore[arg-type]
            )
