# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for HTTP handler environment variable parsing.

This test suite validates environment variable handling for the HTTP handler:
- ONEX_HTTP_TIMEOUT (float, default: 30.0, range: 0.1-3600.0)
- ONEX_HTTP_MAX_REQUEST_SIZE (int, default: 10MB, range: 1-1GB)
- ONEX_HTTP_MAX_RESPONSE_SIZE (int, default: 50MB, range: 1-1GB)

Test Organization:
    - TestHttpHandlerTimeoutEnvParsing: ONEX_HTTP_TIMEOUT validation
    - TestHttpHandlerMaxRequestSizeEnvParsing: ONEX_HTTP_MAX_REQUEST_SIZE validation
    - TestHttpHandlerMaxResponseSizeEnvParsing: ONEX_HTTP_MAX_RESPONSE_SIZE validation
    - TestHttpHandlerEnvErrorContext: Error context validation

Coverage Goals:
    - All HTTP handler environment variables tested
    - Proper error types (ProtocolConfigurationError) verified
    - Range validation behavior (default used for out-of-range) verified
    - Error context fields validated

Note: Since the HTTP handler parses environment variables at module load time,
these tests call the parse_env_* utilities directly with the same parameters
used by the handler to verify correct parsing behavior.
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

# HTTP handler configuration constants (matching handler_http.py)
_HTTP_TIMEOUT_DEFAULT: float = 30.0
_HTTP_TIMEOUT_MIN: float = 0.1
_HTTP_TIMEOUT_MAX: float = 3600.0

_HTTP_MAX_REQUEST_SIZE_DEFAULT: int = 10 * 1024 * 1024  # 10 MB
_HTTP_MAX_RESPONSE_SIZE_DEFAULT: int = 50 * 1024 * 1024  # 50 MB
_HTTP_SIZE_MIN: int = 1
_HTTP_SIZE_MAX: int = 1073741824  # 1 GB


@pytest.mark.unit
class TestHttpHandlerTimeoutEnvParsing:
    """Test suite for ONEX_HTTP_TIMEOUT environment variable parsing.

    Tests verify that the HTTP handler timeout configuration:
    - Uses default value (30.0s) when env var not set
    - Accepts valid custom values from environment
    - Raises ProtocolConfigurationError for invalid (non-numeric) values
    - Returns default with warning for out-of-range values
    """

    def test_default_timeout_when_env_not_set(self) -> None:
        """Test default timeout (30.0s) is used when ONEX_HTTP_TIMEOUT not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_env_float(
                "ONEX_HTTP_TIMEOUT",
                _HTTP_TIMEOUT_DEFAULT,
                min_value=_HTTP_TIMEOUT_MIN,
                max_value=_HTTP_TIMEOUT_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 30.0

    def test_valid_custom_timeout_from_env(self) -> None:
        """Test custom valid timeout value is parsed from environment."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "60.0"}, clear=True):
            result = parse_env_float(
                "ONEX_HTTP_TIMEOUT",
                _HTTP_TIMEOUT_DEFAULT,
                min_value=_HTTP_TIMEOUT_MIN,
                max_value=_HTTP_TIMEOUT_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 60.0

    def test_invalid_timeout_raises_protocol_configuration_error(self) -> None:
        """Test invalid (non-numeric) ONEX_HTTP_TIMEOUT raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_TIMEOUT" in error.message
            assert "expected numeric" in error.message.lower()

    def test_empty_timeout_raises_protocol_configuration_error(self) -> None:
        """Test empty ONEX_HTTP_TIMEOUT value raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": ""}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_TIMEOUT" in error.message

    def test_timeout_below_minimum_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test timeout below minimum (0.1s) uses default with warning logged."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "0.05"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 30.0
            assert "ONEX_HTTP_TIMEOUT" in caplog.text
            assert "below minimum" in caplog.text

    def test_timeout_above_maximum_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test timeout above maximum (3600.0s) uses default with warning logged."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "4000.0"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 30.0
            assert "ONEX_HTTP_TIMEOUT" in caplog.text
            assert "above maximum" in caplog.text

    def test_timeout_at_minimum_boundary_is_valid(self) -> None:
        """Test timeout at exact minimum boundary (0.1s) is accepted."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "0.1"}, clear=True):
            result = parse_env_float(
                "ONEX_HTTP_TIMEOUT",
                _HTTP_TIMEOUT_DEFAULT,
                min_value=_HTTP_TIMEOUT_MIN,
                max_value=_HTTP_TIMEOUT_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 0.1

    def test_timeout_at_maximum_boundary_is_valid(self) -> None:
        """Test timeout at exact maximum boundary (3600.0s) is accepted."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "3600.0"}, clear=True):
            result = parse_env_float(
                "ONEX_HTTP_TIMEOUT",
                _HTTP_TIMEOUT_DEFAULT,
                min_value=_HTTP_TIMEOUT_MIN,
                max_value=_HTTP_TIMEOUT_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 3600.0


@pytest.mark.unit
class TestHttpHandlerMaxRequestSizeEnvParsing:
    """Test suite for ONEX_HTTP_MAX_REQUEST_SIZE environment variable parsing.

    Tests verify that the HTTP handler max request size configuration:
    - Uses default value (10MB) when env var not set
    - Accepts valid custom values from environment
    - Raises ProtocolConfigurationError for invalid (non-integer) values
    - Returns default with warning for out-of-range values
    """

    def test_default_max_request_size_when_env_not_set(self) -> None:
        """Test default max request size (10MB) is used when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_env_int(
                "ONEX_HTTP_MAX_REQUEST_SIZE",
                _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 10 * 1024 * 1024

    def test_valid_custom_max_request_size_from_env(self) -> None:
        """Test custom valid max request size is parsed from environment."""
        custom_size = 20 * 1024 * 1024  # 20 MB
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": str(custom_size)}, clear=True
        ):
            result = parse_env_int(
                "ONEX_HTTP_MAX_REQUEST_SIZE",
                _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == custom_size

    def test_invalid_max_request_size_raises_protocol_configuration_error(self) -> None:
        """Test invalid ONEX_HTTP_MAX_REQUEST_SIZE raises ProtocolConfigurationError."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": "not_a_number"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_MAX_REQUEST_SIZE" in error.message
            assert "expected integer" in error.message.lower()

    def test_float_max_request_size_raises_protocol_configuration_error(self) -> None:
        """Test float value for ONEX_HTTP_MAX_REQUEST_SIZE raises ProtocolConfigurationError."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": "1048576.5"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_MAX_REQUEST_SIZE" in error.message
            assert "expected integer" in error.message.lower()

    def test_max_request_size_below_minimum_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test max request size below minimum (1 byte) uses default with warning."""
        with patch.dict(os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": "0"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 10 * 1024 * 1024
            assert "ONEX_HTTP_MAX_REQUEST_SIZE" in caplog.text
            assert "below minimum" in caplog.text

    def test_max_request_size_above_maximum_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test max request size above maximum (1GB) uses default with warning."""
        too_large = 2 * 1073741824  # 2 GB
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": str(too_large)}, clear=True
        ):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 10 * 1024 * 1024
            assert "ONEX_HTTP_MAX_REQUEST_SIZE" in caplog.text
            assert "above maximum" in caplog.text

    def test_max_request_size_at_minimum_boundary_is_valid(self) -> None:
        """Test max request size at exact minimum boundary (1 byte) is accepted."""
        with patch.dict(os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": "1"}, clear=True):
            result = parse_env_int(
                "ONEX_HTTP_MAX_REQUEST_SIZE",
                _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 1

    def test_max_request_size_at_maximum_boundary_is_valid(self) -> None:
        """Test max request size at exact maximum boundary (1GB) is accepted."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": str(_HTTP_SIZE_MAX)}, clear=True
        ):
            result = parse_env_int(
                "ONEX_HTTP_MAX_REQUEST_SIZE",
                _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 1073741824


@pytest.mark.unit
class TestHttpHandlerMaxResponseSizeEnvParsing:
    """Test suite for ONEX_HTTP_MAX_RESPONSE_SIZE environment variable parsing.

    Tests verify that the HTTP handler max response size configuration:
    - Uses default value (50MB) when env var not set
    - Accepts valid custom values from environment
    - Raises ProtocolConfigurationError for invalid (non-integer) values
    - Returns default with warning for out-of-range values
    """

    def test_default_max_response_size_when_env_not_set(self) -> None:
        """Test default max response size (50MB) is used when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = parse_env_int(
                "ONEX_HTTP_MAX_RESPONSE_SIZE",
                _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 50 * 1024 * 1024

    def test_valid_custom_max_response_size_from_env(self) -> None:
        """Test custom valid max response size is parsed from environment."""
        custom_size = 100 * 1024 * 1024  # 100 MB
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": str(custom_size)}, clear=True
        ):
            result = parse_env_int(
                "ONEX_HTTP_MAX_RESPONSE_SIZE",
                _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == custom_size

    def test_invalid_max_response_size_raises_protocol_configuration_error(
        self,
    ) -> None:
        """Test invalid ONEX_HTTP_MAX_RESPONSE_SIZE raises ProtocolConfigurationError."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": "abc123"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_RESPONSE_SIZE",
                    _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_MAX_RESPONSE_SIZE" in error.message
            assert "expected integer" in error.message.lower()

    def test_empty_max_response_size_raises_protocol_configuration_error(self) -> None:
        """Test empty ONEX_HTTP_MAX_RESPONSE_SIZE value raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": ""}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_RESPONSE_SIZE",
                    _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_MAX_RESPONSE_SIZE" in error.message

    def test_max_response_size_below_minimum_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test max response size below minimum (1 byte) uses default with warning."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": "-100"}, clear=True
        ):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "ONEX_HTTP_MAX_RESPONSE_SIZE",
                    _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 50 * 1024 * 1024
            assert "ONEX_HTTP_MAX_RESPONSE_SIZE" in caplog.text
            assert "below minimum" in caplog.text

    def test_max_response_size_above_maximum_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test max response size above maximum (1GB) uses default with warning."""
        too_large = 5 * 1073741824  # 5 GB
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": str(too_large)}, clear=True
        ):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "ONEX_HTTP_MAX_RESPONSE_SIZE",
                    _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 50 * 1024 * 1024
            assert "ONEX_HTTP_MAX_RESPONSE_SIZE" in caplog.text
            assert "above maximum" in caplog.text

    def test_max_response_size_at_minimum_boundary_is_valid(self) -> None:
        """Test max response size at exact minimum boundary (1 byte) is accepted."""
        with patch.dict(os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": "1"}, clear=True):
            result = parse_env_int(
                "ONEX_HTTP_MAX_RESPONSE_SIZE",
                _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 1

    def test_max_response_size_at_maximum_boundary_is_valid(self) -> None:
        """Test max response size at exact maximum boundary (1GB) is accepted."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": str(_HTTP_SIZE_MAX)}, clear=True
        ):
            result = parse_env_int(
                "ONEX_HTTP_MAX_RESPONSE_SIZE",
                _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                min_value=_HTTP_SIZE_MIN,
                max_value=_HTTP_SIZE_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 1073741824


@pytest.mark.unit
class TestHttpHandlerEnvErrorContext:
    """Test suite for error context validation in HTTP handler env parsing.

    Tests verify that errors from environment variable parsing include:
    - Proper transport type (HTTP)
    - Operation field (parse_env_config)
    - Target name (http_handler)
    - Correlation ID for tracing
    - Value redaction for security
    """

    def test_timeout_error_contains_http_transport_type(self) -> None:
        """Test timeout error context includes HTTP transport type."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert error.model.context["transport_type"] == EnumInfraTransportType.HTTP

    def test_error_contains_parse_env_config_operation(self) -> None:
        """Test error context includes parse_env_config operation field."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": "invalid"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert error.model.context["operation"] == "parse_env_config"

    def test_error_contains_http_handler_target_name(self) -> None:
        """Test error context includes http_handler target name."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": "invalid"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_RESPONSE_SIZE",
                    _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert error.model.context["target_name"] == "http_handler"

    def test_error_contains_correlation_id(self) -> None:
        """Test error context includes auto-generated correlation_id for tracing."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "invalid"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert error.model.correlation_id is not None
            assert isinstance(error.model.correlation_id, UUID)

    def test_error_contains_parameter_field(self) -> None:
        """Test error includes parameter field identifying the failing env var."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "bad_value"}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert error.model.context.get("parameter") == "ONEX_HTTP_TIMEOUT"

    def test_error_redacts_actual_value_for_security(self) -> None:
        """Test error redacts the actual invalid value for security."""
        secret_value = "my_secret_password_123"
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": secret_value}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            # Value should be redacted, not the actual value
            assert error.model.context.get("value") == "[REDACTED]"
            # Actual secret value should NOT appear in error context
            assert secret_value not in str(error.model.context)


@pytest.mark.unit
class TestHttpHandlerEnvEdgeCases:
    """Test suite for edge cases in HTTP handler environment variable parsing.

    Tests verify handling of:
    - Whitespace-only values
    - Special characters in values
    - Scientific notation for floats
    - Negative values
    """

    def test_whitespace_timeout_raises_error(self) -> None:
        """Test whitespace-only ONEX_HTTP_TIMEOUT raises ProtocolConfigurationError."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "   "}, clear=True):
            with pytest.raises(ProtocolConfigurationError) as exc_info:
                parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            error = exc_info.value
            assert "ONEX_HTTP_TIMEOUT" in error.message

    def test_special_characters_in_size_raises_error(self) -> None:
        """Test special characters in ONEX_HTTP_MAX_REQUEST_SIZE raises error."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_REQUEST_SIZE": "1024!@#"}, clear=True
        ):
            with pytest.raises(ProtocolConfigurationError):
                parse_env_int(
                    "ONEX_HTTP_MAX_REQUEST_SIZE",
                    _HTTP_MAX_REQUEST_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

    def test_scientific_notation_timeout_is_valid(self) -> None:
        """Test scientific notation for ONEX_HTTP_TIMEOUT is parsed correctly."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "1e2"}, clear=True):
            result = parse_env_float(
                "ONEX_HTTP_TIMEOUT",
                _HTTP_TIMEOUT_DEFAULT,
                min_value=_HTTP_TIMEOUT_MIN,
                max_value=_HTTP_TIMEOUT_MAX,
                transport_type=EnumInfraTransportType.HTTP,
                service_name="http_handler",
            )

            assert result == 100.0

    def test_negative_timeout_uses_default(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test negative ONEX_HTTP_TIMEOUT uses default with warning."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "-5.0"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 30.0
            assert "ONEX_HTTP_TIMEOUT" in caplog.text
            assert "below minimum" in caplog.text

    def test_negative_size_uses_default(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test negative ONEX_HTTP_MAX_RESPONSE_SIZE uses default with warning."""
        with patch.dict(
            os.environ, {"ONEX_HTTP_MAX_RESPONSE_SIZE": "-1000"}, clear=True
        ):
            with caplog.at_level(logging.WARNING):
                result = parse_env_int(
                    "ONEX_HTTP_MAX_RESPONSE_SIZE",
                    _HTTP_MAX_RESPONSE_SIZE_DEFAULT,
                    min_value=_HTTP_SIZE_MIN,
                    max_value=_HTTP_SIZE_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 50 * 1024 * 1024
            assert "ONEX_HTTP_MAX_RESPONSE_SIZE" in caplog.text

    def test_very_large_timeout_exceeds_max(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test very large ONEX_HTTP_TIMEOUT (>3600s) uses default with warning."""
        with patch.dict(os.environ, {"ONEX_HTTP_TIMEOUT": "999999.0"}, clear=True):
            with caplog.at_level(logging.WARNING):
                result = parse_env_float(
                    "ONEX_HTTP_TIMEOUT",
                    _HTTP_TIMEOUT_DEFAULT,
                    min_value=_HTTP_TIMEOUT_MIN,
                    max_value=_HTTP_TIMEOUT_MAX,
                    transport_type=EnumInfraTransportType.HTTP,
                    service_name="http_handler",
                )

            assert result == 30.0
            assert "above maximum" in caplog.text
