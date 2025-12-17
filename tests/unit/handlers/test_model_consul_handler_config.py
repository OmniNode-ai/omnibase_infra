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

    def test_empty_host_rejected(self) -> None:
        """Test that empty host is rejected with ValidationError."""
        # Empty string is rejected by min_length=1 constraint
        with pytest.raises(ValidationError) as exc_info:
            ModelConsulHandlerConfig(host="")
        assert "host" in str(exc_info.value)

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


class TestModelConsulConfigEquality:
    """Tests for model equality, hashing, and serialization."""

    def test_retry_config_equality_same_values(self) -> None:
        """Test retry config equality with identical values."""
        config1 = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=60.0,
            exponential_base=2.5,
        )
        config2 = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=60.0,
            exponential_base=2.5,
        )
        assert config1 == config2

    def test_retry_config_equality_different_values(self) -> None:
        """Test retry config inequality with different values."""
        config1 = ModelConsulRetryConfig(max_attempts=3)
        config2 = ModelConsulRetryConfig(max_attempts=5)
        assert config1 != config2

    def test_retry_config_equality_default_values(self) -> None:
        """Test retry config equality using default values."""
        config1 = ModelConsulRetryConfig()
        config2 = ModelConsulRetryConfig()
        assert config1 == config2

    def test_handler_config_equality_same_values(self) -> None:
        """Test handler config equality with identical values."""
        config1 = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
            timeout_seconds=45.0,
        )
        config2 = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
            timeout_seconds=45.0,
        )
        assert config1 == config2

    def test_handler_config_equality_different_values(self) -> None:
        """Test handler config inequality with different values."""
        config1 = ModelConsulHandlerConfig(host="host1.example.com")
        config2 = ModelConsulHandlerConfig(host="host2.example.com")
        assert config1 != config2

    def test_handler_config_equality_default_values(self) -> None:
        """Test handler config equality using default values."""
        config1 = ModelConsulHandlerConfig()
        config2 = ModelConsulHandlerConfig()
        assert config1 == config2

    def test_handler_config_equality_with_nested_retry(self) -> None:
        """Test handler config equality considering nested retry config."""
        retry1 = ModelConsulRetryConfig(max_attempts=5)
        retry2 = ModelConsulRetryConfig(max_attempts=5)
        config1 = ModelConsulHandlerConfig(retry=retry1)
        config2 = ModelConsulHandlerConfig(retry=retry2)
        assert config1 == config2

    def test_handler_config_inequality_with_different_nested_retry(self) -> None:
        """Test handler config inequality with different nested retry config."""
        config1 = ModelConsulHandlerConfig(retry=ModelConsulRetryConfig(max_attempts=3))
        config2 = ModelConsulHandlerConfig(retry=ModelConsulRetryConfig(max_attempts=5))
        assert config1 != config2

    def test_retry_config_hash_same_values(self) -> None:
        """Test retry config hash consistency with identical values."""
        config1 = ModelConsulRetryConfig(max_attempts=5)
        config2 = ModelConsulRetryConfig(max_attempts=5)
        assert hash(config1) == hash(config2)

    def test_retry_config_hash_different_values(self) -> None:
        """Test retry config hash differs with different values."""
        config1 = ModelConsulRetryConfig(max_attempts=3)
        config2 = ModelConsulRetryConfig(max_attempts=5)
        # Different values should (usually) produce different hashes
        # Note: hash collisions are possible but unlikely for different configs
        assert hash(config1) != hash(config2)

    def test_handler_config_hash_same_values(self) -> None:
        """Test handler config hash consistency with identical values."""
        config1 = ModelConsulHandlerConfig(host="consul.example.com", port=8501)
        config2 = ModelConsulHandlerConfig(host="consul.example.com", port=8501)
        assert hash(config1) == hash(config2)

    def test_handler_config_hash_usable_in_set(self) -> None:
        """Test handler config can be used in sets (hashable)."""
        config1 = ModelConsulHandlerConfig(host="host1.example.com")
        config2 = ModelConsulHandlerConfig(host="host2.example.com")
        config3 = ModelConsulHandlerConfig(host="host1.example.com")

        config_set = {config1, config2, config3}
        # config1 and config3 are equal, so set should have 2 unique items
        assert len(config_set) == 2

    def test_retry_config_hash_usable_as_dict_key(self) -> None:
        """Test retry config can be used as dictionary key (hashable)."""
        config1 = ModelConsulRetryConfig(max_attempts=3)
        config2 = ModelConsulRetryConfig(max_attempts=5)

        config_dict = {config1: "fast_retry", config2: "slow_retry"}
        assert config_dict[config1] == "fast_retry"
        assert config_dict[config2] == "slow_retry"

    def test_retry_config_str_representation(self) -> None:
        """Test retry config string representation is meaningful."""
        config = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
        )
        str_repr = str(config)
        assert "max_attempts" in str_repr
        assert "5" in str_repr

    def test_retry_config_repr_representation(self) -> None:
        """Test retry config repr is valid and recreatable."""
        config = ModelConsulRetryConfig(max_attempts=5)
        repr_str = repr(config)
        assert "ModelConsulRetryConfig" in repr_str
        assert "max_attempts" in repr_str
        assert "5" in repr_str

    def test_handler_config_str_representation(self) -> None:
        """Test handler config string representation is meaningful."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
        )
        str_repr = str(config)
        assert "host" in str_repr
        assert "consul.example.com" in str_repr
        assert "port" in str_repr
        assert "8501" in str_repr

    def test_handler_config_repr_representation(self) -> None:
        """Test handler config repr is valid and recreatable."""
        config = ModelConsulHandlerConfig(host="consul.example.com")
        repr_str = repr(config)
        assert "ModelConsulHandlerConfig" in repr_str
        assert "host" in repr_str
        assert "consul.example.com" in repr_str

    def test_handler_config_str_excludes_token_value(self) -> None:
        """Test handler config string representation excludes token secret."""
        config = ModelConsulHandlerConfig(token=SecretStr("super-secret-token"))
        str_repr = str(config)
        # The actual token value should be hidden
        assert "super-secret-token" not in str_repr
        # But token field should be present
        assert "token" in str_repr

    def test_handler_config_repr_excludes_token_value(self) -> None:
        """Test handler config repr excludes token secret value."""
        config = ModelConsulHandlerConfig(token=SecretStr("my-secret-token"))
        repr_str = repr(config)
        # The actual token value should be hidden
        assert "my-secret-token" not in repr_str

    def test_retry_config_model_dump_serialization(self) -> None:
        """Test retry config model_dump serialization."""
        config = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=60.0,
            exponential_base=2.5,
        )
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["max_attempts"] == 5
        assert dumped["initial_delay_seconds"] == 2.0
        assert dumped["max_delay_seconds"] == 60.0
        assert dumped["exponential_base"] == 2.5

    def test_handler_config_model_dump_serialization(self) -> None:
        """Test handler config model_dump serialization."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
        )
        dumped = config.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["host"] == "consul.example.com"
        assert dumped["port"] == 8501
        assert dumped["scheme"] == "https"
        # Nested retry config should also be serialized
        assert "retry" in dumped
        assert isinstance(dumped["retry"], dict)

    def test_handler_config_model_dump_json_serialization(self) -> None:
        """Test handler config JSON serialization."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
        )
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "consul.example.com" in json_str
        assert "8501" in json_str

    def test_retry_config_model_dump_json_serialization(self) -> None:
        """Test retry config JSON serialization."""
        config = ModelConsulRetryConfig(max_attempts=7)
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "7" in json_str

    def test_handler_config_roundtrip_serialization(self) -> None:
        """Test handler config can be serialized and deserialized."""
        original = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
            timeout_seconds=45.0,
        )
        dumped = original.model_dump()
        restored = ModelConsulHandlerConfig(**dumped)
        assert original == restored

    def test_retry_config_roundtrip_serialization(self) -> None:
        """Test retry config can be serialized and deserialized."""
        original = ModelConsulRetryConfig(
            max_attempts=7,
            initial_delay_seconds=3.0,
            max_delay_seconds=90.0,
            exponential_base=2.5,
        )
        dumped = original.model_dump()
        restored = ModelConsulRetryConfig(**dumped)
        assert original == restored

    def test_handler_config_equality_with_token(self) -> None:
        """Test handler config equality when both have tokens."""
        config1 = ModelConsulHandlerConfig(token=SecretStr("same-token"))
        config2 = ModelConsulHandlerConfig(token=SecretStr("same-token"))
        # SecretStr with same value should be equal
        assert config1 == config2

    def test_handler_config_inequality_with_different_tokens(self) -> None:
        """Test handler config inequality with different token values."""
        config1 = ModelConsulHandlerConfig(token=SecretStr("token-a"))
        config2 = ModelConsulHandlerConfig(token=SecretStr("token-b"))
        assert config1 != config2

    def test_handler_config_inequality_one_token_none(self) -> None:
        """Test handler config inequality when one token is None."""
        config1 = ModelConsulHandlerConfig(token=SecretStr("some-token"))
        config2 = ModelConsulHandlerConfig(token=None)
        assert config1 != config2

    def test_handler_config_equality_both_tokens_none(self) -> None:
        """Test handler config equality when both tokens are None."""
        config1 = ModelConsulHandlerConfig(token=None)
        config2 = ModelConsulHandlerConfig(token=None)
        assert config1 == config2

    def test_hash_stability_after_roundtrip(self) -> None:
        """Test hash remains consistent after serialization roundtrip."""
        original = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
        )
        original_hash = hash(original)
        dumped = original.model_dump()
        restored = ModelConsulHandlerConfig(**dumped)
        assert hash(restored) == original_hash

    def test_retry_config_hash_stability_after_roundtrip(self) -> None:
        """Test retry config hash remains consistent after roundtrip."""
        original = ModelConsulRetryConfig(max_attempts=5, initial_delay_seconds=2.0)
        original_hash = hash(original)
        dumped = original.model_dump()
        restored = ModelConsulRetryConfig(**dumped)
        assert hash(restored) == original_hash

    def test_handler_config_model_copy(self) -> None:
        """Test handler config model_copy creates independent equal copy."""
        original = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            timeout_seconds=45.0,
        )
        copied = original.model_copy()
        assert original == copied
        assert original is not copied
        assert hash(original) == hash(copied)

    def test_handler_config_model_copy_with_update(self) -> None:
        """Test handler config model_copy with field updates."""
        original = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8500,
        )
        updated = original.model_copy(update={"port": 8501})
        assert original.port == 8500
        assert updated.port == 8501
        assert original.host == updated.host
        assert original != updated

    def test_retry_config_model_copy(self) -> None:
        """Test retry config model_copy creates independent equal copy."""
        original = ModelConsulRetryConfig(max_attempts=5)
        copied = original.model_copy()
        assert original == copied
        assert original is not copied

    def test_retry_config_model_copy_with_update(self) -> None:
        """Test retry config model_copy with field updates."""
        original = ModelConsulRetryConfig(max_attempts=3)
        updated = original.model_copy(update={"max_attempts": 5})
        assert original.max_attempts == 3
        assert updated.max_attempts == 5
        assert original != updated

    def test_handler_config_model_dump_exclude(self) -> None:
        """Test model_dump with field exclusion."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            token=SecretStr("secret"),
        )
        dumped = config.model_dump(exclude={"token", "retry"})
        assert "token" not in dumped
        assert "retry" not in dumped
        assert dumped["host"] == "consul.example.com"
        assert dumped["port"] == 8501

    def test_handler_config_model_dump_include(self) -> None:
        """Test model_dump with field inclusion."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
        )
        dumped = config.model_dump(include={"host", "port"})
        assert "host" in dumped
        assert "port" in dumped
        assert "scheme" not in dumped
        assert "retry" not in dumped

    def test_handler_config_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        import json

        original = ModelConsulHandlerConfig(
            host="consul.example.com",
            port=8501,
            scheme="https",
        )
        json_str = original.model_dump_json()
        parsed = json.loads(json_str)
        restored = ModelConsulHandlerConfig(**parsed)
        assert original == restored

    def test_retry_config_json_roundtrip(self) -> None:
        """Test retry config JSON serialization roundtrip."""
        import json

        original = ModelConsulRetryConfig(
            max_attempts=5,
            initial_delay_seconds=2.0,
        )
        json_str = original.model_dump_json()
        parsed = json.loads(json_str)
        restored = ModelConsulRetryConfig(**parsed)
        assert original == restored

    def test_equality_not_equal_to_other_types(self) -> None:
        """Test config is not equal to non-config types."""
        config = ModelConsulHandlerConfig()
        assert config != "not a config"
        assert config != 42
        assert config != {"host": "localhost", "port": 8500}
        assert config != None

    def test_retry_config_not_equal_to_other_types(self) -> None:
        """Test retry config is not equal to non-config types."""
        config = ModelConsulRetryConfig()
        assert config != "not a config"
        assert config != 42
        assert config != {"max_attempts": 3}
        assert config != None

    def test_handler_config_hash_with_token(self) -> None:
        """Test handler config hash consistency with SecretStr token."""
        config1 = ModelConsulHandlerConfig(token=SecretStr("token"))
        config2 = ModelConsulHandlerConfig(token=SecretStr("token"))
        assert hash(config1) == hash(config2)

    def test_handler_config_hash_different_tokens(self) -> None:
        """Test handler config hash differs with different tokens."""
        config1 = ModelConsulHandlerConfig(token=SecretStr("token-a"))
        config2 = ModelConsulHandlerConfig(token=SecretStr("token-b"))
        # Different tokens should (usually) produce different hashes
        assert hash(config1) != hash(config2)

    def test_handler_config_model_dump_json_indent(self) -> None:
        """Test model_dump_json with indentation for readability."""
        config = ModelConsulHandlerConfig(host="consul.example.com")
        json_str = config.model_dump_json(indent=2)
        assert isinstance(json_str, str)
        # Indented JSON should have newlines
        assert "\n" in json_str
        assert "  " in json_str  # Two-space indent

    def test_retry_config_model_dump_mode_json(self) -> None:
        """Test model_dump with mode='json' for JSON-compatible output."""
        config = ModelConsulRetryConfig(max_attempts=5)
        dumped = config.model_dump(mode="json")
        # All values should be JSON-serializable primitives
        assert isinstance(dumped["max_attempts"], int)
        assert isinstance(dumped["initial_delay_seconds"], float)

    def test_handler_config_model_dump_mode_json(self) -> None:
        """Test handler config model_dump with mode='json'."""
        config = ModelConsulHandlerConfig(
            host="consul.example.com",
            token=SecretStr("secret"),
        )
        dumped = config.model_dump(mode="json")
        # In JSON mode, SecretStr should be serialized as string
        assert isinstance(dumped["host"], str)
        # Token should be present but value depends on serialization mode
        assert "token" in dumped


__all__: list[str] = [
    "TestModelConsulRetryConfig",
    "TestModelConsulHandlerConfig",
    "TestModelConsulHandlerConfigEdgeCases",
    "TestModelConsulConfigEquality",
]
