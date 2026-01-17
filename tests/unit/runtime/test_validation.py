# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the ONEX runtime contract validation module.

Tests the contract validation functions including:
- Topic name pattern validation
- Event bus type enum validation
- Shutdown grace period range validation
- Nested object structure validation
- Error message formatting
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.util_validation import (
    MAX_GRACE_PERIOD_SECONDS,
    MIN_GRACE_PERIOD_SECONDS,
    TOPIC_NAME_PATTERN,
    VALID_EVENT_BUS_TYPES,
    load_and_validate_config,
    validate_runtime_config,
)


class TestValidateRuntimeConfig:
    """Tests for validate_runtime_config function."""

    def test_valid_config_returns_empty_errors(self) -> None:
        """Test that a fully valid config returns no errors."""
        config: dict[str, object] = {
            "input_topic": "requests",
            "output_topic": "responses",
            "consumer_group": "onex-runtime",
            "event_bus": {
                "type": "inmemory",
                "environment": "local",
                "max_history": 1000,
                "circuit_breaker_threshold": 5,
            },
            "shutdown": {
                "grace_period_seconds": 30,
            },
        }
        errors = validate_runtime_config(config)
        assert errors == []

    def test_empty_config_returns_no_errors(self) -> None:
        """Test that an empty config returns no errors (defaults apply)."""
        errors = validate_runtime_config({})
        assert errors == []

    def test_minimal_config_returns_no_errors(self) -> None:
        """Test that a minimal valid config returns no errors."""
        config: dict[str, object] = {
            "input_topic": "my-topic",
        }
        errors = validate_runtime_config(config)
        assert errors == []


class TestTopicNameValidation:
    """Tests for topic name pattern validation."""

    @pytest.mark.parametrize(
        "topic_name",
        [
            "requests",
            "responses",
            "my-topic",
            "my_topic",
            "topic123",
            "UPPERCASE",
            "CamelCase",
            "a",
            "topic-with-many-dashes",
            "topic_with_many_underscores",
            "mixed-topic_name123",
            "topic.with.dots",  # Dots are valid per Kafka conventions
            "dev.onex.evt.node-introspection.v1",  # ONEX topic naming
        ],
    )
    def test_valid_topic_names(self, topic_name: str) -> None:
        """Test that valid topic names pass validation."""
        config: dict[str, object] = {"input_topic": topic_name}
        errors = validate_runtime_config(config)
        assert errors == []

    @pytest.mark.parametrize(
        ("topic_name", "expected_error_substring"),
        [
            ("topic with spaces", "must match pattern"),
            # Note: dots are VALID per pattern ^[a-zA-Z0-9._-]+$ (Kafka convention)
            ("topic:with:colons", "must match pattern"),
            ("topic/with/slashes", "must match pattern"),
            ("topic@special", "must match pattern"),
            ("", "must match pattern"),
            ("topic\nwith\nnewlines", "must match pattern"),
        ],
    )
    def test_invalid_topic_names(
        self, topic_name: str, expected_error_substring: str
    ) -> None:
        """Test that invalid topic names fail validation with descriptive error."""
        config: dict[str, object] = {"input_topic": topic_name}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert expected_error_substring in errors[0]
        assert "input_topic" in errors[0]

    def test_topic_name_wrong_type(self) -> None:
        """Test that non-string topic names fail validation."""
        config: dict[str, object] = {"input_topic": 123}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be a string" in errors[0]
        assert "input_topic" in errors[0]

    def test_output_topic_validation(self) -> None:
        """Test that output_topic is also validated."""
        config: dict[str, object] = {"output_topic": "invalid topic name"}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "output_topic" in errors[0]


class TestConsumerGroupValidation:
    """Tests for consumer_group/group_id validation."""

    def test_valid_consumer_group(self) -> None:
        """Test that valid consumer_group passes validation."""
        config: dict[str, object] = {"consumer_group": "onex-runtime"}
        errors = validate_runtime_config(config)
        assert errors == []

    def test_valid_group_id_alias(self) -> None:
        """Test that group_id alias is also validated."""
        config: dict[str, object] = {"group_id": "onex-runtime"}
        errors = validate_runtime_config(config)
        assert errors == []

    def test_invalid_consumer_group(self) -> None:
        """Test that invalid consumer_group fails validation."""
        config: dict[str, object] = {"consumer_group": "invalid group name"}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "consumer_group" in errors[0]

    def test_consumer_group_wrong_type(self) -> None:
        """Test that non-string consumer_group fails validation."""
        config: dict[str, object] = {"consumer_group": ["list", "not", "string"]}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be a string" in errors[0]


class TestEventBusValidation:
    """Tests for event_bus configuration validation."""

    def test_valid_event_bus_inmemory(self) -> None:
        """Test that inmemory event bus type is valid."""
        config: dict[str, object] = {"event_bus": {"type": "inmemory"}}
        errors = validate_runtime_config(config)
        assert errors == []

    def test_valid_event_bus_kafka(self) -> None:
        """Test that kafka event bus type is valid."""
        config: dict[str, object] = {"event_bus": {"type": "kafka"}}
        errors = validate_runtime_config(config)
        assert errors == []

    def test_invalid_event_bus_type(self) -> None:
        """Test that invalid event bus type fails validation."""
        # NOTE: "redis" is intentionally used as an example of an invalid event bus type.
        # This is unrelated to the REDIS->VALKEY cache backend rename; event_bus only
        # supports "inmemory" and "kafka" types.
        config: dict[str, object] = {"event_bus": {"type": "redis"}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "event_bus.type" in errors[0]
        assert "inmemory" in errors[0]
        assert "kafka" in errors[0]

    def test_event_bus_type_wrong_type(self) -> None:
        """Test that non-string event bus type fails validation."""
        config: dict[str, object] = {"event_bus": {"type": 123}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be a string" in errors[0]

    def test_event_bus_not_dict(self) -> None:
        """Test that non-dict event_bus fails validation."""
        config: dict[str, object] = {"event_bus": "not-a-dict"}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        # Check for key terms without exact message coupling
        assert "event_bus" in errors[0]
        assert "object" in errors[0] or "dict" in errors[0].lower()

    def test_event_bus_environment_wrong_type(self) -> None:
        """Test that non-string environment fails validation."""
        config: dict[str, object] = {"event_bus": {"environment": 123}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "event_bus.environment" in errors[0]

    def test_event_bus_max_history_negative(self) -> None:
        """Test that negative max_history fails validation."""
        config: dict[str, object] = {"event_bus": {"max_history": -1}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        # Check for key terms (field name and constraint) without exact message coupling
        assert "max_history" in errors[0]
        assert ">= 0" in errors[0] or "non-negative" in errors[0].lower()

    def test_event_bus_max_history_wrong_type(self) -> None:
        """Test that non-integer max_history fails validation."""
        config: dict[str, object] = {"event_bus": {"max_history": "not-an-int"}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be an integer" in errors[0]

    def test_event_bus_max_history_boolean_rejected(self) -> None:
        """Test that boolean max_history is rejected (Python bool is subclass of int)."""
        config: dict[str, object] = {"event_bus": {"max_history": True}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be an integer" in errors[0]

    def test_event_bus_circuit_breaker_zero(self) -> None:
        """Test that zero circuit_breaker_threshold fails validation."""
        config: dict[str, object] = {"event_bus": {"circuit_breaker_threshold": 0}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        # Check for key terms (field name and constraint) without exact message coupling
        assert "circuit_breaker_threshold" in errors[0]
        assert ">= 1" in errors[0] or "positive" in errors[0].lower()

    def test_event_bus_circuit_breaker_wrong_type(self) -> None:
        """Test that non-integer circuit_breaker_threshold fails validation."""
        config: dict[str, object] = {"event_bus": {"circuit_breaker_threshold": 5.5}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be an integer" in errors[0]


class TestShutdownValidation:
    """Tests for shutdown configuration validation."""

    def test_valid_shutdown_config(self) -> None:
        """Test that valid shutdown config passes validation."""
        config: dict[str, object] = {"shutdown": {"grace_period_seconds": 30}}
        errors = validate_runtime_config(config)
        assert errors == []

    def test_shutdown_grace_period_zero(self) -> None:
        """Test that zero grace_period_seconds is valid."""
        config: dict[str, object] = {"shutdown": {"grace_period_seconds": 0}}
        errors = validate_runtime_config(config)
        assert errors == []

    def test_shutdown_grace_period_max(self) -> None:
        """Test that max grace_period_seconds is valid."""
        config: dict[str, object] = {
            "shutdown": {"grace_period_seconds": MAX_GRACE_PERIOD_SECONDS}
        }
        errors = validate_runtime_config(config)
        assert errors == []

    def test_shutdown_grace_period_negative(self) -> None:
        """Test that negative grace_period_seconds fails validation."""
        config: dict[str, object] = {"shutdown": {"grace_period_seconds": -1}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        # Check for key terms without exact message coupling
        assert "grace_period" in errors[0].lower()
        assert (
            f">= {MIN_GRACE_PERIOD_SECONDS}" in errors[0]
            or "non-negative" in errors[0].lower()
            or "minimum" in errors[0].lower()
        )

    def test_shutdown_grace_period_too_large(self) -> None:
        """Test that grace_period_seconds exceeding max fails validation."""
        config: dict[str, object] = {
            "shutdown": {"grace_period_seconds": MAX_GRACE_PERIOD_SECONDS + 1}
        }
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        # Check for key terms without exact message coupling
        assert "grace_period" in errors[0].lower()
        assert (
            f"<= {MAX_GRACE_PERIOD_SECONDS}" in errors[0]
            or "maximum" in errors[0].lower()
            or str(MAX_GRACE_PERIOD_SECONDS) in errors[0]
        )

    def test_shutdown_grace_period_wrong_type(self) -> None:
        """Test that non-integer grace_period_seconds fails validation."""
        config: dict[str, object] = {"shutdown": {"grace_period_seconds": "thirty"}}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        assert "must be an integer" in errors[0]

    def test_shutdown_not_dict(self) -> None:
        """Test that non-dict shutdown fails validation."""
        config: dict[str, object] = {"shutdown": "not-a-dict"}
        errors = validate_runtime_config(config)
        assert len(errors) == 1
        # Check for key terms without exact message coupling
        assert "shutdown" in errors[0]
        assert "object" in errors[0] or "dict" in errors[0].lower()


class TestMultipleErrors:
    """Tests for configs with multiple validation errors."""

    def test_multiple_errors_all_reported(self) -> None:
        """Test that multiple validation errors are all reported."""
        config: dict[str, object] = {
            "input_topic": "invalid topic",
            "output_topic": "also invalid",
            "event_bus": {"type": "unknown"},
            "shutdown": {"grace_period_seconds": -100},
        }
        errors = validate_runtime_config(config)
        assert len(errors) == 4
        assert any("input_topic" in e for e in errors)
        assert any("output_topic" in e for e in errors)
        assert any("event_bus.type" in e for e in errors)
        assert any("grace_period_seconds" in e for e in errors)


class TestLoadAndValidateConfig:
    """Tests for load_and_validate_config function."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Test loading a valid configuration file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "input_topic": "requests",
            "output_topic": "responses",
        }
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        result = load_and_validate_config(config_file)
        assert result["input_topic"] == "requests"
        assert result["output_topic"] == "responses"

    def test_load_file_not_found(self, tmp_path: Path) -> None:
        """Test that missing file raises ProtocolConfigurationError."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        assert "not found" in str(exc_info.value).lower()

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises ProtocolConfigurationError."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        assert "parse" in str(exc_info.value).lower()

    def test_load_invalid_config_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid config values raise ProtocolConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "input_topic": "invalid topic with spaces",
        }
        with config_file.open("w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ProtocolConfigurationError) as exc_info:
            load_and_validate_config(config_file)

        assert "validation failed" in str(exc_info.value).lower()

    def test_load_empty_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        """Test that empty YAML file returns empty dict."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = load_and_validate_config(config_file)
        assert result == {}


class TestConstants:
    """Tests for module-level constants."""

    def test_topic_pattern_matches_valid(self) -> None:
        """Test that TOPIC_NAME_PATTERN matches valid topic names."""
        assert TOPIC_NAME_PATTERN.match("valid-topic")
        assert TOPIC_NAME_PATTERN.match("valid_topic")
        assert TOPIC_NAME_PATTERN.match("valid123")

    def test_topic_pattern_rejects_invalid(self) -> None:
        """Test that TOPIC_NAME_PATTERN rejects invalid topic names."""
        assert not TOPIC_NAME_PATTERN.match("invalid topic")
        assert not TOPIC_NAME_PATTERN.match("")

    def test_valid_event_bus_types(self) -> None:
        """Test that VALID_EVENT_BUS_TYPES contains expected values."""
        assert "inmemory" in VALID_EVENT_BUS_TYPES
        assert "kafka" in VALID_EVENT_BUS_TYPES
        assert len(VALID_EVENT_BUS_TYPES) == 2

    def test_grace_period_bounds(self) -> None:
        """Test that grace period bounds match ModelShutdownConfig constraints."""
        assert MIN_GRACE_PERIOD_SECONDS == 0
        assert (
            MAX_GRACE_PERIOD_SECONDS == 3600
        )  # 1 hour max to match ModelShutdownConfig
        assert MIN_GRACE_PERIOD_SECONDS < MAX_GRACE_PERIOD_SECONDS
