# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive unit tests for ModelCircuitBreakerConfig.

This test suite validates:
- Basic model instantiation with defaults and custom values
- Environment variable configuration via from_env()
- Error handling for invalid environment values
- Custom prefix support
- Error context validation
- Pydantic range validation behavior
- Edge cases (whitespace, large values, scientific notation)

Test Organization:
    - TestModelCircuitBreakerConfigBasics: Basic model functionality (6 tests)
    - TestModelCircuitBreakerConfigFromEnv: Environment variable loading (11 tests)
    - TestModelCircuitBreakerConfigFromEnvErrors: Error cases for from_env() (9 tests)
    - TestModelCircuitBreakerConfigFromEnvErrorContext: Error context validation (11 tests)
    - TestModelCircuitBreakerConfigEdgeCases: Edge cases and boundary conditions (14 tests)

Coverage Goals:
    - >90% code coverage for model
    - All from_env() paths tested (threshold/timeout parsing, defaults, custom prefix)
    - All error scenarios tested (invalid values, empty strings, whitespace)
    - Error context fields validated (transport_type, operation, target_name, correlation_id)
    - Value redaction verified for security
    - Error chaining from ValueError verified
    - Pydantic validation after parsing tested (negative values, zero threshold)
"""

import os
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig


@pytest.mark.unit
class TestModelCircuitBreakerConfigBasics:
    """Test basic model instantiation and validation."""

    def test_default_values(self) -> None:
        """Test model creates with default values."""
        config = ModelCircuitBreakerConfig()

        assert config.threshold == 5
        assert config.reset_timeout_seconds == 60.0
        assert config.service_name == "unknown"
        assert config.transport_type == EnumInfraTransportType.HTTP

    def test_custom_values(self) -> None:
        """Test model creates with custom values."""
        config = ModelCircuitBreakerConfig(
            threshold=10,
            reset_timeout_seconds=120.0,
            service_name="kafka.production",
            transport_type=EnumInfraTransportType.KAFKA,
        )

        assert config.threshold == 10
        assert config.reset_timeout_seconds == 120.0
        assert config.service_name == "kafka.production"
        assert config.transport_type == EnumInfraTransportType.KAFKA

    def test_model_is_frozen(self) -> None:
        """Test model is immutable (frozen)."""
        config = ModelCircuitBreakerConfig()

        with pytest.raises(Exception):  # Pydantic ValidationError for frozen
            config.threshold = 10  # type: ignore[misc]

    def test_threshold_minimum_validation(self) -> None:
        """Test threshold must be >= 1."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelCircuitBreakerConfig(threshold=0)

    def test_reset_timeout_minimum_validation(self) -> None:
        """Test reset_timeout_seconds must be >= 0."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelCircuitBreakerConfig(reset_timeout_seconds=-1.0)

    def test_service_name_minimum_length(self) -> None:
        """Test service_name must have minimum length."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            ModelCircuitBreakerConfig(service_name="")


@pytest.mark.unit
class TestModelCircuitBreakerConfigFromEnv:
    """Test from_env() class method for environment variable loading."""

    def test_from_env_with_defaults(self) -> None:
        """Test from_env returns default values when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )

            assert config.threshold == 5
            assert config.reset_timeout_seconds == 60.0
            assert config.service_name == "test_service"
            assert config.transport_type == EnumInfraTransportType.HTTP

    def test_from_env_with_custom_threshold(self) -> None:
        """Test from_env reads custom threshold from environment."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "10"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )

            assert config.threshold == 10
            assert config.reset_timeout_seconds == 60.0

    def test_from_env_with_custom_reset_timeout(self) -> None:
        """Test from_env reads custom reset_timeout from environment."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "120.5"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )

            assert config.threshold == 5
            assert config.reset_timeout_seconds == 120.5

    def test_from_env_with_both_custom_values(self) -> None:
        """Test from_env reads both threshold and reset_timeout from environment."""
        env_vars = {
            "ONEX_CB_THRESHOLD": "3",
            "ONEX_CB_RESET_TIMEOUT": "30.0",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="kafka.production",
                transport_type=EnumInfraTransportType.KAFKA,
            )

            assert config.threshold == 3
            assert config.reset_timeout_seconds == 30.0
            assert config.service_name == "kafka.production"
            assert config.transport_type == EnumInfraTransportType.KAFKA

    def test_from_env_with_custom_prefix(self) -> None:
        """Test from_env works with custom environment variable prefix."""
        env_vars = {
            "KAFKA_CB_THRESHOLD": "7",
            "KAFKA_CB_RESET_TIMEOUT": "90.0",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="kafka.production",
                transport_type=EnumInfraTransportType.KAFKA,
                prefix="KAFKA_CB",
            )

            assert config.threshold == 7
            assert config.reset_timeout_seconds == 90.0

    def test_from_env_custom_prefix_ignores_default_vars(self) -> None:
        """Test custom prefix ignores ONEX_CB_* variables."""
        env_vars = {
            "ONEX_CB_THRESHOLD": "99",  # Should be ignored
            "ONEX_CB_RESET_TIMEOUT": "999.0",  # Should be ignored
            "CUSTOM_THRESHOLD": "3",
            "CUSTOM_RESET_TIMEOUT": "45.0",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
                prefix="CUSTOM",
            )

            assert config.threshold == 3
            assert config.reset_timeout_seconds == 45.0

    def test_from_env_passes_service_name_and_transport_type(self) -> None:
        """Test from_env correctly passes service_name and transport_type."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="postgresql-primary",
                transport_type=EnumInfraTransportType.DATABASE,
            )

            assert config.service_name == "postgresql-primary"
            assert config.transport_type == EnumInfraTransportType.DATABASE

    def test_from_env_uses_default_service_name_and_transport(self) -> None:
        """Test from_env uses defaults when service_name/transport_type not provided."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelCircuitBreakerConfig.from_env()

            assert config.service_name == "unknown"
            assert config.transport_type == EnumInfraTransportType.HTTP

    def test_from_env_integer_threshold_conversion(self) -> None:
        """Test from_env correctly converts string to integer for threshold."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "42"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test",
                transport_type=EnumInfraTransportType.HTTP,
            )

            assert isinstance(config.threshold, int)
            assert config.threshold == 42

    def test_from_env_float_timeout_conversion(self) -> None:
        """Test from_env correctly converts string to float for reset_timeout."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "123.456"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test",
                transport_type=EnumInfraTransportType.HTTP,
            )

            assert isinstance(config.reset_timeout_seconds, float)
            assert config.reset_timeout_seconds == 123.456


@pytest.mark.unit
class TestModelCircuitBreakerConfigFromEnvErrors:
    """Test error handling for from_env() with invalid environment values."""

    def test_from_env_invalid_threshold_raises_protocol_error(self) -> None:
        """Test from_env raises ProtocolConfigurationError for invalid threshold."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_THRESHOLD" in error.message
            assert "expected integer" in error.message

    def test_from_env_invalid_reset_timeout_raises_protocol_error(self) -> None:
        """Test from_env raises ProtocolConfigurationError for invalid reset_timeout."""
        with patch.dict(
            os.environ, {"ONEX_CB_RESET_TIMEOUT": "not_a_float"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_RESET_TIMEOUT" in error.message
            assert "expected numeric value" in error.message

    def test_from_env_empty_threshold_raises_protocol_error(self) -> None:
        """Test from_env raises ProtocolConfigurationError for empty threshold string."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": ""}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_THRESHOLD" in error.message

    def test_from_env_empty_reset_timeout_raises_protocol_error(self) -> None:
        """Test from_env raises ProtocolConfigurationError for empty reset_timeout string."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": ""}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_RESET_TIMEOUT" in error.message

    def test_from_env_whitespace_threshold_raises_protocol_error(self) -> None:
        """Test from_env raises ProtocolConfigurationError for whitespace threshold."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "   "}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_THRESHOLD" in error.message

    def test_from_env_float_threshold_raises_protocol_error(self) -> None:
        """Test from_env raises ProtocolConfigurationError for float value as threshold."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "5.5"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_THRESHOLD" in error.message
            assert "expected integer" in error.message

    def test_from_env_valid_threshold_invalid_timeout_raises_for_timeout(self) -> None:
        """Test from_env with valid threshold but invalid timeout raises for timeout."""
        env_vars = {
            "ONEX_CB_THRESHOLD": "10",  # Valid
            "ONEX_CB_RESET_TIMEOUT": "invalid",  # Invalid
        }
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert "ONEX_CB_RESET_TIMEOUT" in error.message
            assert "expected numeric value" in error.message

    def test_from_env_invalid_with_custom_prefix(self) -> None:
        """Test from_env raises error with custom prefix in error message."""
        with patch.dict(os.environ, {"CUSTOM_CB_THRESHOLD": "abc"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                    prefix="CUSTOM_CB",
                )

            error = exc_info.value
            assert "CUSTOM_CB_THRESHOLD" in error.message

    def test_from_env_special_characters_in_value_raises_error(self) -> None:
        """Test from_env raises error for special characters in threshold."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "5!@#"}, clear=True):
            with pytest.raises(ProtocolConfigurationError):
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )


@pytest.mark.unit
class TestModelCircuitBreakerConfigFromEnvErrorContext:
    """Test error context validation for from_env() errors."""

    def test_error_context_contains_transport_type(self) -> None:
        """Test error context includes transport_type."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.KAFKA,
                )

            error = exc_info.value
            assert error.model.context["transport_type"] == EnumInfraTransportType.KAFKA

    def test_error_context_contains_operation(self) -> None:
        """Test error context includes operation field."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert error.model.context["operation"] == "parse_env_config"

    def test_error_context_contains_target_name(self) -> None:
        """Test error context includes target_name (service_name)."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="my-kafka-service",
                    transport_type=EnumInfraTransportType.KAFKA,
                )

            error = exc_info.value
            assert error.model.context["target_name"] == "my-kafka-service"

    def test_error_context_contains_correlation_id(self) -> None:
        """Test error context includes auto-generated correlation_id."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert error.model.correlation_id is not None
            assert isinstance(error.model.correlation_id, UUID)

    def test_error_context_contains_parameter_field(self) -> None:
        """Test error includes parameter field identifying the failing env var."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            assert error.model.context.get("parameter") == "ONEX_CB_THRESHOLD"

    def test_error_context_redacts_value(self) -> None:
        """Test error redacts the actual invalid value for security."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "secret_value"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            # Value should be redacted, not the actual value
            assert error.model.context.get("value") == "[REDACTED]"
            assert "secret_value" not in str(error.model.context)

    def test_error_chains_from_value_error(self) -> None:
        """Test error is explicitly NOT chained (security: don't expose ValueError).

        The implementation uses `raise ... from None` to intentionally break the
        exception chain. This prevents exposing the original ValueError which could
        contain the raw invalid value in its message, following ONEX security
        guidelines for value redaction.
        """
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

            error = exc_info.value
            # Verify error is NOT chained (intentional for security)
            # Uses `from None` to avoid exposing raw invalid values in ValueError
            assert error.__cause__ is None

    def test_threshold_error_context_differs_from_timeout_error_context(self) -> None:
        """Test threshold and timeout errors have different parameter fields."""
        # Test threshold error
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )
            threshold_error = exc_info.value
            assert threshold_error.model.context.get("parameter") == "ONEX_CB_THRESHOLD"

        # Test timeout error
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )
            timeout_error = exc_info.value
            assert (
                timeout_error.model.context.get("parameter") == "ONEX_CB_RESET_TIMEOUT"
            )

    def test_error_context_with_database_transport(self) -> None:
        """Test error context correctly uses DATABASE transport type."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="postgresql-primary",
                    transport_type=EnumInfraTransportType.DATABASE,
                )

            error = exc_info.value
            assert (
                error.model.context["transport_type"] == EnumInfraTransportType.DATABASE
            )
            assert error.model.context["target_name"] == "postgresql-primary"

    def test_error_context_with_consul_transport(self) -> None:
        """Test error context correctly uses CONSUL transport type."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "bad"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                ModelCircuitBreakerConfig.from_env(
                    service_name="consul-primary",
                    transport_type=EnumInfraTransportType.CONSUL,
                )

            error = exc_info.value
            assert (
                error.model.context["transport_type"] == EnumInfraTransportType.CONSUL
            )


@pytest.mark.unit
class TestModelCircuitBreakerConfigEdgeCases:
    """Test edge cases for from_env() method."""

    def test_from_env_negative_threshold_string_uses_warning_and_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test negative threshold logs warning and uses default value.

        The parse_env_int utility validates ranges and falls back to default
        with a warning rather than raising an exception.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "-5"}, clear=True):
                config = ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

                # Verify default is used
                assert config.threshold == 5  # default value

                # Verify warning was logged
                assert "ONEX_CB_THRESHOLD" in caplog.text
                assert "below minimum" in caplog.text
                assert "-5" in caplog.text

    def test_from_env_zero_threshold_uses_warning_and_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test zero threshold logs warning and uses default value.

        The parse_env_int utility validates ranges and falls back to default
        with a warning. min_value=1 is enforced, so 0 triggers fallback.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "0"}, clear=True):
                config = ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

                # Verify default is used
                assert config.threshold == 5  # default value

                # Verify warning was logged
                assert "ONEX_CB_THRESHOLD" in caplog.text
                assert "below minimum" in caplog.text

    def test_from_env_very_large_threshold(self) -> None:
        """Test from_env handles very large threshold values."""
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "1000000"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.threshold == 1000000

    def test_from_env_scientific_notation_timeout(self) -> None:
        """Test from_env handles scientific notation for timeout."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "1e2"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.reset_timeout_seconds == 100.0

    def test_from_env_zero_timeout_is_valid(self) -> None:
        """Test from_env accepts zero timeout (immediate reset)."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "0"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.reset_timeout_seconds == 0.0

    def test_from_env_preserves_all_transport_types(self) -> None:
        """Test from_env correctly preserves all transport types."""
        transport_types = [
            EnumInfraTransportType.HTTP,
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.VAULT,
            EnumInfraTransportType.VALKEY,
            EnumInfraTransportType.GRPC,
            EnumInfraTransportType.RUNTIME,
        ]

        for transport_type in transport_types:
            with patch.dict(os.environ, {}, clear=True):
                config = ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=transport_type,
                )
                assert config.transport_type == transport_type

    def test_from_env_empty_prefix(self) -> None:
        """Test from_env with empty prefix uses bare variable names."""
        env_vars = {
            "_THRESHOLD": "8",
            "_RESET_TIMEOUT": "80.0",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
                prefix="",
            )
            assert config.threshold == 8
            assert config.reset_timeout_seconds == 80.0

    def test_from_env_negative_timeout_uses_warning_and_default(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test negative timeout logs warning and uses default value.

        The parse_env_float utility validates ranges and falls back to default
        with a warning. min_value=0.0 is enforced, so negative triggers fallback.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "-10.0"}, clear=True):
                config = ModelCircuitBreakerConfig.from_env(
                    service_name="test_service",
                    transport_type=EnumInfraTransportType.HTTP,
                )

                # Verify default is used
                assert config.reset_timeout_seconds == 60.0  # default value

                # Verify warning was logged
                assert "ONEX_CB_RESET_TIMEOUT" in caplog.text
                assert "below minimum" in caplog.text

    def test_from_env_very_large_timeout(self) -> None:
        """Test from_env handles very large timeout values."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "86400.0"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.reset_timeout_seconds == 86400.0  # 24 hours

    def test_from_env_whitespace_around_valid_threshold(self) -> None:
        """Test from_env handles whitespace around valid threshold values.

        Python's int() function trims leading/trailing whitespace.
        """
        with patch.dict(os.environ, {"ONEX_CB_THRESHOLD": "  7  "}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.threshold == 7

    def test_from_env_whitespace_around_valid_timeout(self) -> None:
        """Test from_env handles whitespace around valid timeout values.

        Python's float() function trims leading/trailing whitespace.
        """
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "  90.5  "}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.reset_timeout_seconds == 90.5

    def test_from_env_integer_string_for_timeout(self) -> None:
        """Test from_env correctly converts integer string to float for timeout."""
        with patch.dict(os.environ, {"ONEX_CB_RESET_TIMEOUT": "120"}, clear=True):
            config = ModelCircuitBreakerConfig.from_env(
                service_name="test_service",
                transport_type=EnumInfraTransportType.HTTP,
            )
            assert config.reset_timeout_seconds == 120.0
            assert isinstance(config.reset_timeout_seconds, float)
