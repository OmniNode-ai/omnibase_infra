# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Dedicated unit tests for util_env_parsing module.

This test suite provides comprehensive coverage of the environment variable
parsing utilities in omnibase_infra.utils.util_env_parsing:
    - parse_env_int: Integer parsing with range validation
    - parse_env_float: Float parsing with range validation

Test Organization:
    - TestParseEnvIntBasic: Core functionality for parse_env_int
    - TestParseEnvIntRangeValidation: Min/max range validation for integers
    - TestParseEnvIntErrorContext: Error context field validation
    - TestParseEnvFloatBasic: Core functionality for parse_env_float
    - TestParseEnvFloatRangeValidation: Min/max range validation for floats
    - TestParseEnvFloatSpecialFormats: Scientific notation and edge cases
    - TestParseEnvFloatErrorContext: Error context field validation
    - TestDefaultServiceName: Default behavior when service_name is not specified
    - TestDifferentTransportTypes: Verification of all supported transport types

Coverage Goals:
    - Full coverage of parse_env_int and parse_env_float functions
    - Error context fields validation (transport_type, operation, target_name, correlation_id)
    - Value redaction in error messages for security
    - Boundary value testing
    - Edge case handling

Note: These tests focus on the utility functions themselves, independent of any
specific handler context. For handler-specific tests that use these utilities,
see test_handler_http_env.py.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.utils.util_env_parsing import parse_env_float, parse_env_int


@pytest.mark.unit
class TestParseEnvIntBasic:
    """Test suite for basic parse_env_int functionality.

    Tests verify core behavior:
    - Default value when env var not set
    - Valid integer parsing
    - Invalid type raises ProtocolConfigurationError
    - Empty string handling
    """

    def test_returns_default_when_env_not_set(self) -> None:
        """Test default value is returned when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=42,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 42

    def test_parses_valid_positive_integer(self) -> None:
        """Test valid positive integer is parsed correctly."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "100"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=42,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 100

    def test_parses_valid_negative_integer(self) -> None:
        """Test valid negative integer is parsed correctly (no range constraints)."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "-50"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=42,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == -50

    def test_parses_zero(self) -> None:
        """Test zero is parsed correctly."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "0"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=42,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 0

    def test_invalid_string_raises_protocol_configuration_error(self) -> None:
        """Test non-numeric string raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "not_a_number"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            assert "TEST_INT_VAR" in error.message
            assert "expected integer" in error.message.lower()

    def test_float_string_raises_protocol_configuration_error(self) -> None:
        """Test float string raises ProtocolConfigurationError for int parsing."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "3.14"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            assert "TEST_INT_VAR" in error.message
            assert "expected integer" in error.message.lower()

    def test_empty_string_raises_protocol_configuration_error(self) -> None:
        """Test empty string raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"TEST_INT_VAR": ""}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            assert "TEST_INT_VAR" in error.message

    def test_whitespace_only_raises_protocol_configuration_error(self) -> None:
        """Test whitespace-only value raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "   "}, clear=True):
            with pytest.raises(ProtocolConfigurationError):
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )


@pytest.mark.unit
class TestParseEnvIntRangeValidation:
    """Test suite for parse_env_int range validation.

    Tests verify:
    - Value below min_value returns default with warning
    - Value above max_value returns default with warning
    - Boundary values (at min, at max) are accepted
    - No validation when min/max are None
    """

    def test_below_minimum_uses_default_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test value below minimum returns default with warning logged."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "5"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "TEST_INT_VAR",
                    default=50,
                    min_value=10,
                    max_value=100,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            assert result == 50
            assert "TEST_INT_VAR" in caplog.text
            assert "below minimum" in caplog.text
            assert "10" in caplog.text  # min value mentioned

    def test_above_maximum_uses_default_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test value above maximum returns default with warning logged."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "150"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "TEST_INT_VAR",
                    default=50,
                    min_value=10,
                    max_value=100,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            assert result == 50
            assert "TEST_INT_VAR" in caplog.text
            assert "above maximum" in caplog.text
            assert "100" in caplog.text  # max value mentioned

    def test_at_minimum_boundary_is_accepted(self) -> None:
        """Test value exactly at minimum boundary is accepted."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "10"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=50,
                min_value=10,
                max_value=100,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 10

    def test_at_maximum_boundary_is_accepted(self) -> None:
        """Test value exactly at maximum boundary is accepted."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "100"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=50,
                min_value=10,
                max_value=100,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 100

    def test_value_in_range_is_accepted(self) -> None:
        """Test value within range is accepted."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "55"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=50,
                min_value=10,
                max_value=100,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 55

    def test_no_min_validation_when_min_is_none(self) -> None:
        """Test no minimum validation when min_value is None."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "-999999"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=50,
                min_value=None,  # No minimum
                max_value=100,
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == -999999

    def test_no_max_validation_when_max_is_none(self) -> None:
        """Test no maximum validation when max_value is None."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "999999"}, clear=True):
            result = parse_env_int(
                "TEST_INT_VAR",
                default=50,
                min_value=10,
                max_value=None,  # No maximum
                transport_type=EnumInfraTransportType.DATABASE,
                service_name="test_service",
            )

            assert result == 999999


@pytest.mark.unit
class TestParseEnvIntErrorContext:
    """Test suite for parse_env_int error context validation.

    Tests verify that errors include:
    - transport_type field
    - operation field (parse_env_config)
    - target_name field (from service_name)
    - correlation_id for tracing
    - parameter field (env var name)
    - value redaction for security
    """

    def test_error_contains_transport_type(self) -> None:
        """Test error context includes specified transport type."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.KAFKA,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context["transport_type"] == EnumInfraTransportType.KAFKA

    def test_error_contains_parse_env_config_operation(self) -> None:
        """Test error context includes parse_env_config operation."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context["operation"] == "parse_env_config"

    def test_error_contains_target_name_from_service_name(self) -> None:
        """Test error context includes target_name from service_name parameter."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="my_custom_service",
                )

            error = exc_info.value
            assert error.model.context["target_name"] == "my_custom_service"

    def test_error_contains_correlation_id(self) -> None:
        """Test error context includes auto-generated correlation_id."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.correlation_id is not None
            assert isinstance(error.model.correlation_id, UUID)

    def test_error_contains_parameter_field(self) -> None:
        """Test error context includes parameter field with env var name."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context.get("parameter") == "TEST_INT_VAR"

    def test_error_redacts_value_for_security(self) -> None:
        """Test error redacts actual value for security - never exposes secrets."""
        secret_value = "my_secret_api_key_12345"
        with patch.dict(os.environ, {"TEST_INT_VAR": secret_value}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    service_name="test_service",
                )

            error = exc_info.value
            # Value should be redacted
            assert error.model.context.get("value") == "[REDACTED]"
            # Secret should NOT appear anywhere in context
            assert secret_value not in str(error.model.context)
            # Secret should NOT appear in error message
            assert secret_value not in error.message


@pytest.mark.unit
class TestParseEnvFloatBasic:
    """Test suite for basic parse_env_float functionality.

    Tests verify core behavior:
    - Default value when env var not set
    - Valid float parsing
    - Integer string parsed as float
    - Invalid type raises ProtocolConfigurationError
    - Empty string handling
    """

    def test_returns_default_when_env_not_set(self) -> None:
        """Test default value is returned when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=3.14,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 3.14

    def test_parses_valid_positive_float(self) -> None:
        """Test valid positive float is parsed correctly."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "2.718"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=3.14,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 2.718

    def test_parses_valid_negative_float(self) -> None:
        """Test valid negative float is parsed correctly (no range constraints)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "-1.5"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=3.14,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == -1.5

    def test_parses_zero_float(self) -> None:
        """Test zero (0.0) is parsed correctly."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "0.0"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=3.14,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 0.0

    def test_parses_integer_string_as_float(self) -> None:
        """Test integer string is parsed as float."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "42"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=3.14,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 42.0
            assert isinstance(result, float)

    def test_invalid_string_raises_protocol_configuration_error(self) -> None:
        """Test non-numeric string raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "not_a_number"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            error = exc_info.value
            assert "TEST_FLOAT_VAR" in error.message
            assert "expected numeric" in error.message.lower()

    def test_empty_string_raises_protocol_configuration_error(self) -> None:
        """Test empty string raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": ""}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            error = exc_info.value
            assert "TEST_FLOAT_VAR" in error.message

    def test_whitespace_only_raises_protocol_configuration_error(self) -> None:
        """Test whitespace-only value raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "   "}, clear=True):
            with pytest.raises(ProtocolConfigurationError):
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )


@pytest.mark.unit
class TestParseEnvFloatRangeValidation:
    """Test suite for parse_env_float range validation.

    Tests verify:
    - Value below min_value returns default with warning
    - Value above max_value returns default with warning
    - Boundary values (at min, at max) are accepted
    - No validation when min/max are None
    """

    def test_below_minimum_uses_default_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test value below minimum returns default with warning logged."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "0.5"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=5.0,
                    min_value=1.0,
                    max_value=10.0,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            assert result == 5.0
            assert "TEST_FLOAT_VAR" in caplog.text
            assert "below minimum" in caplog.text

    def test_above_maximum_uses_default_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test value above maximum returns default with warning logged."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "15.0"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=5.0,
                    min_value=1.0,
                    max_value=10.0,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            assert result == 5.0
            assert "TEST_FLOAT_VAR" in caplog.text
            assert "above maximum" in caplog.text

    def test_at_minimum_boundary_is_accepted(self) -> None:
        """Test value exactly at minimum boundary is accepted."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "1.0"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                min_value=1.0,
                max_value=10.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 1.0

    def test_at_maximum_boundary_is_accepted(self) -> None:
        """Test value exactly at maximum boundary is accepted."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "10.0"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                min_value=1.0,
                max_value=10.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 10.0

    def test_value_in_range_is_accepted(self) -> None:
        """Test value within range is accepted."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "5.5"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                min_value=1.0,
                max_value=10.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 5.5

    def test_no_min_validation_when_min_is_none(self) -> None:
        """Test no minimum validation when min_value is None."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "-999999.0"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                min_value=None,  # No minimum
                max_value=10.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == -999999.0

    def test_no_max_validation_when_max_is_none(self) -> None:
        """Test no maximum validation when max_value is None."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "999999.0"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                min_value=1.0,
                max_value=None,  # No maximum
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 999999.0


@pytest.mark.unit
class TestParseEnvFloatSpecialFormats:
    """Test suite for parse_env_float special format handling.

    Tests verify:
    - Scientific notation support (e.g., "1e2", "1.5e-3")
    - Very small and very large values
    - Edge cases
    """

    def test_scientific_notation_positive_exponent(self) -> None:
        """Test scientific notation with positive exponent (1e2 = 100)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "1e2"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 100.0

    def test_scientific_notation_negative_exponent(self) -> None:
        """Test scientific notation with negative exponent (1.5e-3 = 0.0015)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "1.5e-3"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == pytest.approx(0.0015)

    def test_scientific_notation_uppercase_e(self) -> None:
        """Test scientific notation with uppercase E (2.5E3 = 2500)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "2.5E3"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 2500.0

    def test_scientific_notation_with_plus_sign(self) -> None:
        """Test scientific notation with explicit plus sign (1e+2 = 100)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "1e+2"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 100.0

    def test_very_small_value(self) -> None:
        """Test very small value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "0.000001"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == pytest.approx(1e-6)

    def test_very_large_value(self) -> None:
        """Test very large value is parsed correctly."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "1000000.0"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=5.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 1000000.0

    def test_leading_decimal_point(self) -> None:
        """Test value with leading decimal point (.5 = 0.5)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": ".5"}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=1.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 0.5

    def test_trailing_decimal_point(self) -> None:
        """Test value with trailing decimal point (5. = 5.0)."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "5."}, clear=True):
            result = parse_env_float(
                "TEST_FLOAT_VAR",
                default=1.0,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="test_service",
            )

            assert result == 5.0


@pytest.mark.unit
class TestParseEnvFloatErrorContext:
    """Test suite for parse_env_float error context validation.

    Tests verify that errors include:
    - transport_type field
    - operation field (parse_env_config)
    - target_name field (from service_name)
    - correlation_id for tracing
    - parameter field (env var name)
    - value redaction for security
    """

    def test_error_contains_transport_type(self) -> None:
        """Test error context includes specified transport type."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.CONSUL,
                    service_name="test_service",
                )

            error = exc_info.value
            assert (
                error.model.context["transport_type"] == EnumInfraTransportType.CONSUL
            )

    def test_error_contains_parse_env_config_operation(self) -> None:
        """Test error context includes parse_env_config operation."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context["operation"] == "parse_env_config"

    def test_error_contains_target_name_from_service_name(self) -> None:
        """Test error context includes target_name from service_name parameter."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="my_float_service",
                )

            error = exc_info.value
            assert error.model.context["target_name"] == "my_float_service"

    def test_error_contains_correlation_id(self) -> None:
        """Test error context includes auto-generated correlation_id."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.correlation_id is not None
            assert isinstance(error.model.correlation_id, UUID)

    def test_error_contains_parameter_field(self) -> None:
        """Test error context includes parameter field with env var name."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context.get("parameter") == "TEST_FLOAT_VAR"

    def test_error_redacts_value_for_security(self) -> None:
        """Test error redacts actual value for security - never exposes secrets."""
        secret_value = "confidential_token_xyz789"
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": secret_value}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="test_service",
                )

            error = exc_info.value
            # Value should be redacted
            assert error.model.context.get("value") == "[REDACTED]"
            # Secret should NOT appear anywhere in context
            assert secret_value not in str(error.model.context)
            # Secret should NOT appear in error message
            assert secret_value not in error.message


@pytest.mark.unit
class TestDefaultServiceName:
    """Test suite for default service_name behavior.

    Tests verify that when service_name is not specified,
    "unknown" is used as the default.
    """

    def test_parse_env_int_defaults_to_unknown_service(self) -> None:
        """Test parse_env_int uses 'unknown' as default service name."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=EnumInfraTransportType.DATABASE,
                    # service_name not specified, should default to "unknown"
                )

            error = exc_info.value
            assert error.model.context["target_name"] == "unknown"

    def test_parse_env_float_defaults_to_unknown_service(self) -> None:
        """Test parse_env_float uses 'unknown' as default service name."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=EnumInfraTransportType.HTTP,
                    # service_name not specified, should default to "unknown"
                )

            error = exc_info.value
            assert error.model.context["target_name"] == "unknown"


@pytest.mark.unit
class TestDifferentTransportTypes:
    """Test suite verifying various transport types in error context.

    Ensures that all supported transport types are correctly captured in error context.
    """

    @pytest.mark.parametrize(
        "transport_type",
        [
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.HTTP,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.VALKEY,
            EnumInfraTransportType.GRPC,
            EnumInfraTransportType.RUNTIME,
        ],
    )
    def test_parse_env_int_captures_transport_type(
        self, transport_type: EnumInfraTransportType
    ) -> None:
        """Test parse_env_int captures specified transport type in error context."""
        with patch.dict(os.environ, {"TEST_INT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "TEST_INT_VAR",
                    default=42,
                    transport_type=transport_type,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context["transport_type"] == transport_type

    @pytest.mark.parametrize(
        "transport_type",
        [
            EnumInfraTransportType.DATABASE,
            EnumInfraTransportType.KAFKA,
            EnumInfraTransportType.HTTP,
            EnumInfraTransportType.CONSUL,
            EnumInfraTransportType.VALKEY,
            EnumInfraTransportType.GRPC,
            EnumInfraTransportType.RUNTIME,
        ],
    )
    def test_parse_env_float_captures_transport_type(
        self, transport_type: EnumInfraTransportType
    ) -> None:
        """Test parse_env_float captures specified transport type in error context."""
        with patch.dict(os.environ, {"TEST_FLOAT_VAR": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "TEST_FLOAT_VAR",
                    default=3.14,
                    transport_type=transport_type,
                    service_name="test_service",
                )

            error = exc_info.value
            assert error.model.context["transport_type"] == transport_type


@pytest.mark.unit
class TestGrpcAndRuntimeTransportTypes:
    """Test suite for GRPC and RUNTIME transport types.

    These tests verify that the newer GRPC and RUNTIME transport types
    work correctly with the env parsing utilities, including:
    - Valid value parsing
    - Error context inclusion
    - Range validation

    Added per PR #106 review feedback to ensure complete transport type coverage.
    """

    def test_parse_env_int_with_grpc_transport_type(self) -> None:
        """Test parse_env_int works with GRPC transport type."""
        with patch.dict(os.environ, {"GRPC_PORT": "50051"}, clear=True):
            result = parse_env_int(
                "GRPC_PORT",
                default=50050,
                transport_type=EnumInfraTransportType.GRPC,
                service_name="grpc_service",
            )

            assert result == 50051

    def test_parse_env_int_with_runtime_transport_type(self) -> None:
        """Test parse_env_int works with RUNTIME transport type."""
        with patch.dict(os.environ, {"RUNTIME_WORKERS": "4"}, clear=True):
            result = parse_env_int(
                "RUNTIME_WORKERS",
                default=2,
                transport_type=EnumInfraTransportType.RUNTIME,
                service_name="runtime_host",
            )

            assert result == 4

    def test_parse_env_float_with_grpc_transport_type(self) -> None:
        """Test parse_env_float works with GRPC transport type."""
        with patch.dict(os.environ, {"GRPC_TIMEOUT": "30.5"}, clear=True):
            result = parse_env_float(
                "GRPC_TIMEOUT",
                default=60.0,
                transport_type=EnumInfraTransportType.GRPC,
                service_name="grpc_service",
            )

            assert result == 30.5

    def test_parse_env_float_with_runtime_transport_type(self) -> None:
        """Test parse_env_float works with RUNTIME transport type."""
        with patch.dict(os.environ, {"RUNTIME_SCALE_FACTOR": "1.5"}, clear=True):
            result = parse_env_float(
                "RUNTIME_SCALE_FACTOR",
                default=1.0,
                transport_type=EnumInfraTransportType.RUNTIME,
                service_name="runtime_host",
            )

            assert result == 1.5

    def test_grpc_transport_type_in_error_context(self) -> None:
        """Test GRPC transport type is correctly included in error context."""
        with patch.dict(os.environ, {"GRPC_PORT": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "GRPC_PORT",
                    default=50050,
                    transport_type=EnumInfraTransportType.GRPC,
                    service_name="grpc_service",
                )

            error = exc_info.value
            assert error.model.context["transport_type"] == EnumInfraTransportType.GRPC
            assert error.model.context["target_name"] == "grpc_service"
            assert "GRPC_PORT" in error.message

    def test_runtime_transport_type_in_error_context(self) -> None:
        """Test RUNTIME transport type is correctly included in error context."""
        with patch.dict(os.environ, {"RUNTIME_WORKERS": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "RUNTIME_WORKERS",
                    default=2,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    service_name="runtime_host",
                )

            error = exc_info.value
            assert (
                error.model.context["transport_type"] == EnumInfraTransportType.RUNTIME
            )
            assert error.model.context["target_name"] == "runtime_host"
            assert "RUNTIME_WORKERS" in error.message

    def test_grpc_transport_type_with_range_validation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test GRPC transport type works with range validation."""
        with patch.dict(os.environ, {"GRPC_PORT": "100"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "GRPC_PORT",
                    default=50050,
                    min_value=1024,
                    max_value=65535,
                    transport_type=EnumInfraTransportType.GRPC,
                    service_name="grpc_service",
                )

            # Value below minimum should use default
            assert result == 50050
            assert "GRPC_PORT" in caplog.text
            assert "below minimum" in caplog.text

    def test_runtime_transport_type_with_range_validation(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test RUNTIME transport type works with range validation."""
        with patch.dict(os.environ, {"RUNTIME_SCALE_FACTOR": "100.0"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "RUNTIME_SCALE_FACTOR",
                    default=1.0,
                    min_value=0.1,
                    max_value=10.0,
                    transport_type=EnumInfraTransportType.RUNTIME,
                    service_name="runtime_host",
                )

            # Value above maximum should use default
            assert result == 1.0
            assert "RUNTIME_SCALE_FACTOR" in caplog.text
            assert "above maximum" in caplog.text

    def test_grpc_transport_type_string_value(self) -> None:
        """Test GRPC transport type has correct string value."""
        assert EnumInfraTransportType.GRPC.value == "grpc"

    def test_runtime_transport_type_string_value(self) -> None:
        """Test RUNTIME transport type has correct string value."""
        assert EnumInfraTransportType.RUNTIME.value == "runtime"
