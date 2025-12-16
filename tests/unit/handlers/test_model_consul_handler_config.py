# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for Consul handler configuration models.

These tests validate the Pydantic configuration models for ConsulHandler,
ensuring proper validation, defaults, and security handling.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from omnibase_infra.handlers.model_consul_handler_config import ModelConsulHandlerConfig
from omnibase_infra.handlers.model_consul_retry_config import ModelConsulRetryConfig


class TestModelConsulRetryConfig:
    """Tests for ModelConsulRetryConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConsulRetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 30.0
        assert config.exponential_base == 2.0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=60.0,
            exponential_base=3.0,
        )
        assert config.max_attempts == 5
        assert config.initial_delay_seconds == 2.0
        assert config.max_delay_seconds == 60.0
        assert config.exponential_base == 3.0

    def test_validation_max_attempts_minimum(self) -> None:
        """Test max_attempts validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(max_attempts=0)
        assert "max_attempts" in str(exc_info.value)

    def test_validation_max_attempts_maximum(self) -> None:
        """Test max_attempts validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(max_attempts=11)
        assert "max_attempts" in str(exc_info.value)

    def test_validation_initial_delay_minimum(self) -> None:
        """Test initial_delay_seconds validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(initial_delay_seconds=0.05)
        assert "initial_delay_seconds" in str(exc_info.value)

    def test_validation_initial_delay_maximum(self) -> None:
        """Test initial_delay_seconds validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(initial_delay_seconds=70.0)
        assert "initial_delay_seconds" in str(exc_info.value)

    def test_validation_max_delay_minimum(self) -> None:
        """Test max_delay_seconds validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(max_delay_seconds=0.5)
        assert "max_delay_seconds" in str(exc_info.value)

    def test_validation_max_delay_maximum(self) -> None:
        """Test max_delay_seconds validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(max_delay_seconds=400.0)
        assert "max_delay_seconds" in str(exc_info.value)

    def test_validation_exponential_base_minimum(self) -> None:
        """Test exponential_base validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(exponential_base=1.0)
        assert "exponential_base" in str(exc_info.value)

    def test_validation_exponential_base_maximum(self) -> None:
        """Test exponential_base validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulRetryConfig(exponential_base=5.0)
        assert "exponential_base" in str(exc_info.value)

    def test_frozen_immutability(self) -> None:
        """Test that config is immutable (frozen)."""
        config = ModelConsulRetryConfig()
        with pytest.raises(ValidationError):
            config.max_attempts = 5  # type: ignore[misc]

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelConsulRetryConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_strict_type_enforcement(self) -> None:
        """Test that strict mode enforces correct types."""
        # String instead of int for max_attempts should fail in strict mode
        with pytest.raises(ValidationError):
            ModelConsulRetryConfig(max_attempts="3")  # type: ignore[arg-type]


class TestModelConsulHandlerConfig:
    """Tests for ModelConsulHandlerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ModelConsulHandlerConfig()
        assert config.host == "localhost"
        assert config.port == 8500
        assert config.scheme == "http"
        assert config.token is None
        assert config.timeout_seconds == 30.0
        assert config.connect_timeout_seconds == 10.0
        assert config.health_check_interval_seconds == 30.0
        assert config.datacenter is None
        assert isinstance(config.retry, ModelConsulRetryConfig)

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
            token=SecretStr("my-token"),
            timeout_seconds=60.0,
            connect_timeout_seconds=15.0,
            health_check_interval_seconds=60.0,
            datacenter="dc1",
        )
        assert config.host == "consul.example.com"
        assert config.port == 8501
        assert config.scheme == "https"
        assert config.token is not None
        assert config.token.get_secret_value() == "my-token"
        assert config.timeout_seconds == 60.0
        assert config.connect_timeout_seconds == 15.0
        assert config.health_check_interval_seconds == 60.0
        assert config.datacenter == "dc1"

    def test_base_url_property_http(self) -> None:
        """Test base_url property generation with http scheme."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="http",
        )
        assert config.base_url == "http://consul.example.com:8501"

    def test_base_url_property_https(self) -> None:
        """Test base_url property generation with https scheme."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8500,
            scheme="https",
        )
        assert config.base_url == "https://consul.example.com:8500"

    def test_base_url_property_default(self) -> None:
        """Test base_url property with default values."""
        config = ModelConsulHandlerConfig()
        assert config.base_url == "http://localhost:8500"

    def test_token_secret_str(self) -> None:
        """Test that token is stored as SecretStr."""
        config = ModelConsulHandlerConfig(token=SecretStr("my-acl-token"))
        assert isinstance(config.token, SecretStr)
        assert config.token.get_secret_value() == "my-acl-token"
        # SecretStr should hide value in repr
        assert "my-acl-token" not in str(config.token)
        assert "my-acl-token" not in repr(config.token)

    def test_token_none_by_default(self) -> None:
        """Test that token is None by default."""
        config = ModelConsulHandlerConfig()
        assert config.token is None

    def test_port_validation_minimum(self) -> None:
        """Test port number validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(port=0)
        assert "port" in str(exc_info.value)

    def test_port_validation_maximum(self) -> None:
        """Test port number validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(port=65536)
        assert "port" in str(exc_info.value)

    def test_port_validation_boundary_min(self) -> None:
        """Test port number at minimum boundary."""
        config = ModelConsulHandlerConfig(port=1)
        assert config.port == 1

    def test_port_validation_boundary_max(self) -> None:
        """Test port number at maximum boundary."""
        config = ModelConsulHandlerConfig(port=65535)
        assert config.port == 65535

    def test_scheme_validation_http(self) -> None:
        """Test scheme literal validation for http."""
        config = ModelConsulHandlerConfig(scheme="http")
        assert config.scheme == "http"

    def test_scheme_validation_https(self) -> None:
        """Test scheme literal validation for https."""
        config = ModelConsulHandlerConfig(scheme="https")
        assert config.scheme == "https"

    def test_scheme_validation_invalid(self) -> None:
        """Test scheme literal validation rejects invalid values."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(scheme="ftp")  # type: ignore[arg-type]
        assert "scheme" in str(exc_info.value)

    def test_scheme_validation_uppercase_invalid(self) -> None:
        """Test scheme literal validation rejects uppercase."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(scheme="HTTP")  # type: ignore[arg-type]
        assert "scheme" in str(exc_info.value)

    def test_timeout_seconds_validation_minimum(self) -> None:
        """Test timeout_seconds validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(timeout_seconds=0.5)
        assert "timeout_seconds" in str(exc_info.value)

    def test_timeout_seconds_validation_maximum(self) -> None:
        """Test timeout_seconds validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(timeout_seconds=400.0)
        assert "timeout_seconds" in str(exc_info.value)

    def test_timeout_seconds_boundary_min(self) -> None:
        """Test timeout_seconds at minimum boundary."""
        config = ModelConsulHandlerConfig(timeout_seconds=1.0)
        assert config.timeout_seconds == 1.0

    def test_timeout_seconds_boundary_max(self) -> None:
        """Test timeout_seconds at maximum boundary."""
        config = ModelConsulHandlerConfig(timeout_seconds=300.0)
        assert config.timeout_seconds == 300.0

    def test_connect_timeout_seconds_validation_minimum(self) -> None:
        """Test connect_timeout_seconds validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(connect_timeout_seconds=0.5)
        assert "connect_timeout_seconds" in str(exc_info.value)

    def test_connect_timeout_seconds_validation_maximum(self) -> None:
        """Test connect_timeout_seconds validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(connect_timeout_seconds=70.0)
        assert "connect_timeout_seconds" in str(exc_info.value)

    def test_health_check_interval_validation_minimum(self) -> None:
        """Test health_check_interval_seconds validation below minimum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(health_check_interval_seconds=3.0)
        assert "health_check_interval_seconds" in str(exc_info.value)

    def test_health_check_interval_validation_maximum(self) -> None:
        """Test health_check_interval_seconds validation above maximum."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(health_check_interval_seconds=400.0)
        assert "health_check_interval_seconds" in str(exc_info.value)

    def test_nested_retry_config(self) -> None:
        """Test nested retry configuration."""
        config = ModelConsulHandlerConfig(retry=ModelConsulRetryConfig(max_attempts=5))
        assert config.retry.max_attempts == 5
        assert config.retry.initial_delay_seconds == 1.0  # Default

    def test_nested_retry_config_custom(self) -> None:
        """Test nested retry configuration with custom values."""
        retry_config = ModelConsulRetryConfig(
            max_attempts=7,
            initial_delay_seconds=2.0,
            max_delay_seconds=120.0,
            exponential_base=2.5,
        )
        config = ModelConsulHandlerConfig(retry=retry_config)
        assert config.retry.max_attempts == 7
        assert config.retry.initial_delay_seconds == 2.0
        assert config.retry.max_delay_seconds == 120.0
        assert config.retry.exponential_base == 2.5

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            ModelConsulHandlerConfig(unknown_field="value")  # type: ignore[call-arg]

    def test_frozen_immutability(self) -> None:
        """Test that config is immutable (frozen)."""
        config = ModelConsulHandlerConfig()
        with pytest.raises(ValidationError):
            config.host = "new-host"  # type: ignore[misc]

    def test_strict_type_enforcement(self) -> None:
        """Test that strict mode enforces correct types."""
        # String instead of int for port should fail in strict mode
        with pytest.raises(ValidationError):
            ModelConsulHandlerConfig(port="8500")  # type: ignore[arg-type]

    def test_datacenter_optional(self) -> None:
        """Test that datacenter is optional."""
        config = ModelConsulHandlerConfig(datacenter=None)
        assert config.datacenter is None

    def test_datacenter_custom(self) -> None:
        """Test custom datacenter value."""
        config = ModelConsulHandlerConfig(datacenter="us-west-1")
        assert config.datacenter == "us-west-1"

    def test_from_attributes(self) -> None:
        """Test that from_attributes mode is enabled."""

        # Create an object with attributes matching config fields
        class ConfigSource:
            host: str = "test-host"
            port: int = 9500
            scheme: str = "https"
            token: SecretStr | None = None
            timeout_seconds: float = 45.0
            connect_timeout_seconds: float = 15.0
            health_check_interval_seconds: float = 45.0
            datacenter: str | None = "dc2"
            retry: ModelConsulRetryConfig = ModelConsulRetryConfig()

        source = ConfigSource()
        config = ModelConsulHandlerConfig.model_validate(source)
        assert config.host == "test-host"
        assert config.port == 9500
        assert config.scheme == "https"

    def test_serialization_excludes_token_secret(self) -> None:
        """Test that serialization excludes token secret value."""
        config = ModelConsulHandlerConfig(token=SecretStr("secret-token"))
        model_dict = config.model_dump()
        # Token should be in the dict but as SecretStr (or serialized representation)
        assert "token" in model_dict
        # The actual secret value should not be plaintext in string representation
        str_repr = str(model_dict)
        assert "secret-token" not in str_repr


class TestModelConsulHandlerConfigEdgeCases:
    """Edge case tests for ModelConsulHandlerConfig."""

    def test_empty_host_allowed(self) -> None:
        """Test that empty host is technically allowed but not recommended."""
        # Empty string is allowed by Pydantic (no min_length constraint)
        config = ModelConsulHandlerConfig(host="")
        assert config.host == ""
        assert config.base_url == "http://:8500"

    def test_empty_datacenter_vs_none(self) -> None:
        """Test difference between empty datacenter and None."""
        config_none = ModelConsulHandlerConfig(datacenter=None)
        config_empty = ModelConsulHandlerConfig(datacenter="")
        assert config_none.datacenter is None
        assert config_empty.datacenter == ""

    def test_special_characters_in_host(self) -> None:
        """Test special characters in hostname."""
        config = ModelConsulHandlerConfig(host="consul-cluster.example.com")
        assert config.host == "consul-cluster.example.com"
        assert config.base_url == "http://consul-cluster.example.com:8500"

    def test_ipv4_host(self) -> None:
        """Test IPv4 address as hostname."""
        config = ModelConsulHandlerConfig(host="192.168.1.100")
        assert config.host == "192.168.1.100"
        assert config.base_url == "http://192.168.1.100:8500"

    def test_model_config_settings(self) -> None:
        """Test model_config settings are properly applied."""
        # Verify ConfigDict settings
        assert ModelConsulHandlerConfig.model_config.get("strict") is True
        assert ModelConsulHandlerConfig.model_config.get("frozen") is True
        assert ModelConsulHandlerConfig.model_config.get("extra") == "forbid"
        assert ModelConsulHandlerConfig.model_config.get("from_attributes") is True

    def test_retry_config_model_config_settings(self) -> None:
        """Test retry config model_config settings are properly applied."""
        assert ModelConsulRetryConfig.model_config.get("strict") is True
        assert ModelConsulRetryConfig.model_config.get("frozen") is True
        assert ModelConsulRetryConfig.model_config.get("extra") == "forbid"
        assert ModelConsulRetryConfig.model_config.get("from_attributes") is True


__all__: list[str] = [
    "TestModelConsulRetryConfig",
    "TestModelConsulHandlerConfig",
    "TestModelConsulHandlerConfigEdgeCases",
]
