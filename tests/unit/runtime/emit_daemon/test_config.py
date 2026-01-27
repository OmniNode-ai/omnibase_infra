# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: S108  # /tmp paths are standard for Unix socket config testing
"""Comprehensive unit tests for ModelEmitDaemonConfig.

This test suite validates:
- Basic model instantiation with default and custom values
- Required field validation (kafka_bootstrap_servers)
- Bootstrap server format validation (host:port)
- Limit validations (min/max constraints)
- Spool consistency validation
- Path validators
- spooling_enabled property
- with_env_overrides class method

Test Organization:
    - TestModelEmitDaemonConfigDefaults: Default value verification (1 test)
    - TestModelEmitDaemonConfigRequired: Required field validation (2 tests)
    - TestModelEmitDaemonConfigBootstrapServers: Bootstrap server format (14 tests)
    - TestModelEmitDaemonConfigLimits: Limit validations (16 tests)
    - TestModelEmitDaemonConfigSpoolConsistency: Spool consistency (4 tests)
    - TestModelEmitDaemonConfigPaths: Path validators (8 tests)
    - TestModelEmitDaemonConfigSpoolingEnabled: spooling_enabled property (4 tests)
    - TestModelEmitDaemonConfigEnvOverrides: with_env_overrides method (12 tests)
    - TestModelEmitDaemonConfigImmutability: Frozen model tests (2 tests)

Related Tickets:
    - OMN-1610: Hook Event Emit Daemon implementation
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.emit_daemon.config import ModelEmitDaemonConfig


@pytest.mark.unit
class TestModelEmitDaemonConfigDefaults:
    """Test default value verification."""

    def test_default_values(self) -> None:
        """Test model creates with expected default values."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:9092")

        # Path defaults
        assert config.socket_path == Path("/tmp/omniclaude-emit.sock")
        assert config.pid_path == Path("/tmp/omniclaude-emit.pid")
        assert config.spool_dir == Path.home() / ".omniclaude" / "emit-spool"

        # Limit defaults
        assert config.max_payload_bytes == 1_048_576  # 1MB
        assert config.max_memory_queue == 100
        assert config.max_spool_messages == 1000
        assert config.max_spool_bytes == 10_485_760  # 10MB

        # Kafka defaults
        assert config.kafka_client_id == "emit-daemon"

        # Timeout defaults
        assert config.socket_timeout_seconds == 5.0
        assert config.kafka_timeout_seconds == 30.0
        assert config.shutdown_drain_seconds == 10.0


@pytest.mark.unit
class TestModelEmitDaemonConfigRequired:
    """Test required field validation."""

    def test_kafka_bootstrap_servers_required(self) -> None:
        """Test kafka_bootstrap_servers is required and cannot be omitted."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig()  # type: ignore[call-arg]

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("kafka_bootstrap_servers",)
        assert errors[0]["type"] == "missing"

    def test_kafka_bootstrap_servers_cannot_be_empty(self) -> None:
        """Test kafka_bootstrap_servers cannot be an empty string."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers="")

        errors = exc_info.value.errors()
        assert any("min_length" in str(e) for e in errors)


@pytest.mark.unit
class TestModelEmitDaemonConfigBootstrapServers:
    """Test bootstrap server format validation."""

    def test_valid_single_server(self) -> None:
        """Test valid single bootstrap server passes validation."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:9092")
        assert config.kafka_bootstrap_servers == "localhost:9092"

    def test_valid_single_server_with_ip(self) -> None:
        """Test valid IP address bootstrap server passes validation."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="10.0.0.1:9092")
        assert config.kafka_bootstrap_servers == "10.0.0.1:9092"

    def test_valid_multiple_servers(self) -> None:
        """Test valid comma-separated bootstrap servers pass validation."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="host1:9092,host2:9092,host3:9092"
        )
        assert config.kafka_bootstrap_servers == "host1:9092,host2:9092,host3:9092"

    def test_valid_multiple_servers_with_whitespace(self) -> None:
        """Test whitespace around bootstrap servers is handled."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="  host1:9092 , host2:9092  "
        )
        # The validator should strip the outer whitespace
        assert "host1:9092" in config.kafka_bootstrap_servers
        assert "host2:9092" in config.kafka_bootstrap_servers

    def test_missing_port_raises_validation_error(self) -> None:
        """Test bootstrap_servers without port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost")

        assert "Invalid bootstrap server format" in str(exc_info.value)
        assert "Expected 'host:port'" in str(exc_info.value)

    def test_invalid_port_non_numeric_raises_validation_error(self) -> None:
        """Test bootstrap_servers with non-numeric port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:notaport")

        assert "Invalid port" in str(exc_info.value)
        assert "Port must be a valid integer" in str(exc_info.value)

    def test_invalid_port_too_high_raises_validation_error(self) -> None:
        """Test bootstrap_servers with port > 65535 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:99999")

        assert "Invalid port 99999" in str(exc_info.value)
        assert "Port must be between 1 and 65535" in str(exc_info.value)

    def test_invalid_port_zero_raises_validation_error(self) -> None:
        """Test bootstrap_servers with port 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:0")

        assert "Invalid port 0" in str(exc_info.value)
        assert "Port must be between 1 and 65535" in str(exc_info.value)

    def test_invalid_port_negative_raises_validation_error(self) -> None:
        """Test bootstrap_servers with negative port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:-1")

        # Negative ports are parsed as invalid integers or caught by port range check
        assert "Invalid port" in str(exc_info.value)

    def test_empty_host_raises_validation_error(self) -> None:
        """Test bootstrap_servers with empty host raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(kafka_bootstrap_servers=":9092")

        assert "Host cannot be empty" in str(exc_info.value)

    def test_empty_entry_in_list_raises_validation_error(self) -> None:
        """Test bootstrap_servers with empty entry in comma-separated list."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092,,broker2:9092"
            )

        assert "cannot contain empty entries" in str(exc_info.value)

    def test_valid_port_boundary_low(self) -> None:
        """Test bootstrap_servers with port 1 is valid."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:1")
        assert config.kafka_bootstrap_servers == "localhost:1"

    def test_valid_port_boundary_high(self) -> None:
        """Test bootstrap_servers with port 65535 is valid."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:65535")
        assert config.kafka_bootstrap_servers == "localhost:65535"


@pytest.mark.unit
class TestModelEmitDaemonConfigLimits:
    """Test limit validations."""

    # max_payload_bytes tests (min 1024, max 10MB)
    def test_max_payload_bytes_at_minimum(self) -> None:
        """Test max_payload_bytes at minimum (1024) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_payload_bytes=1024,
        )
        assert config.max_payload_bytes == 1024

    def test_max_payload_bytes_below_minimum_raises_error(self) -> None:
        """Test max_payload_bytes below 1024 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_payload_bytes=1023,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_payload_bytes",) for e in errors)

    def test_max_payload_bytes_at_maximum(self) -> None:
        """Test max_payload_bytes at maximum (10MB) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_payload_bytes=10_485_760,
        )
        assert config.max_payload_bytes == 10_485_760

    def test_max_payload_bytes_above_maximum_raises_error(self) -> None:
        """Test max_payload_bytes above 10MB raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_payload_bytes=10_485_761,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_payload_bytes",) for e in errors)

    # max_memory_queue tests (min 1, max 10000)
    def test_max_memory_queue_at_minimum(self) -> None:
        """Test max_memory_queue at minimum (1) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_memory_queue=1,
        )
        assert config.max_memory_queue == 1

    def test_max_memory_queue_below_minimum_raises_error(self) -> None:
        """Test max_memory_queue below 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_memory_queue=0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_memory_queue",) for e in errors)

    def test_max_memory_queue_at_maximum(self) -> None:
        """Test max_memory_queue at maximum (10000) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_memory_queue=10_000,
        )
        assert config.max_memory_queue == 10_000

    def test_max_memory_queue_above_maximum_raises_error(self) -> None:
        """Test max_memory_queue above 10000 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_memory_queue=10_001,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_memory_queue",) for e in errors)

    # max_spool_messages tests (min 0, max 100000)
    def test_max_spool_messages_at_zero_with_zero_bytes(self) -> None:
        """Test max_spool_messages at 0 is valid when max_spool_bytes is also 0."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=0,
            max_spool_bytes=0,
        )
        assert config.max_spool_messages == 0

    def test_max_spool_messages_at_maximum(self) -> None:
        """Test max_spool_messages at maximum (100000) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=100_000,
        )
        assert config.max_spool_messages == 100_000

    def test_max_spool_messages_above_maximum_raises_error(self) -> None:
        """Test max_spool_messages above 100000 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_spool_messages=100_001,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_spool_messages",) for e in errors)

    # max_spool_bytes tests (min 0, max 1GB)
    def test_max_spool_bytes_at_zero_with_zero_messages(self) -> None:
        """Test max_spool_bytes at 0 is valid when max_spool_messages is also 0."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=0,
            max_spool_bytes=0,
        )
        assert config.max_spool_bytes == 0

    def test_max_spool_bytes_at_maximum(self) -> None:
        """Test max_spool_bytes at maximum (1GB) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_bytes=1_073_741_824,
        )
        assert config.max_spool_bytes == 1_073_741_824

    def test_max_spool_bytes_above_maximum_raises_error(self) -> None:
        """Test max_spool_bytes above 1GB raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_spool_bytes=1_073_741_825,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_spool_bytes",) for e in errors)


@pytest.mark.unit
class TestModelEmitDaemonConfigSpoolConsistency:
    """Test spool limits consistency validation."""

    def test_spool_both_zero_is_valid(self) -> None:
        """Test both spool limits at 0 (disabled) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=0,
            max_spool_bytes=0,
        )
        assert config.max_spool_messages == 0
        assert config.max_spool_bytes == 0

    def test_spool_both_nonzero_is_valid(self) -> None:
        """Test both spool limits non-zero (enabled) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=500,
            max_spool_bytes=5_000_000,
        )
        assert config.max_spool_messages == 500
        assert config.max_spool_bytes == 5_000_000

    def test_spool_messages_zero_bytes_nonzero_raises_error(self) -> None:
        """Test max_spool_messages=0 with max_spool_bytes>0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_spool_messages=0,
                max_spool_bytes=1000,
            )

        assert "Inconsistent spool limits" in str(exc_info.value)
        assert "max_spool_messages is 0" in str(exc_info.value)

    def test_spool_bytes_zero_messages_nonzero_raises_error(self) -> None:
        """Test max_spool_bytes=0 with max_spool_messages>0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                max_spool_messages=100,
                max_spool_bytes=0,
            )

        assert "Inconsistent spool limits" in str(exc_info.value)
        assert "max_spool_bytes is 0" in str(exc_info.value)


@pytest.mark.unit
class TestModelEmitDaemonConfigPaths:
    """Test path validators."""

    def test_socket_path_with_existing_parent(self) -> None:
        """Test socket_path with existing parent directory is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            socket_path=Path("/tmp/test-emit.sock"),
        )
        assert config.socket_path == Path("/tmp/test-emit.sock")

    def test_pid_path_with_existing_parent(self) -> None:
        """Test pid_path with existing parent directory is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            pid_path=Path("/tmp/test-emit.pid"),
        )
        assert config.pid_path == Path("/tmp/test-emit.pid")

    def test_spool_dir_with_existing_ancestor(self) -> None:
        """Test spool_dir with existing ancestor directory is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            spool_dir=Path("/tmp/test-spool"),
        )
        assert config.spool_dir == Path("/tmp/test-spool")

    def test_spool_dir_nested_with_existing_ancestor(self) -> None:
        """Test nested spool_dir with existing ancestor is valid."""
        # ~/.omniclaude/emit-spool has ~ as existing ancestor
        spool_path = Path.home() / ".test-omniclaude" / "nested" / "spool"
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            spool_dir=spool_path,
        )
        assert config.spool_dir == spool_path

    def test_socket_path_invalid_parent_raises_error(self) -> None:
        """Test socket_path with invalid parent directory raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                socket_path=Path("/nonexistent/deeply/nested/emit.sock"),
            )

        assert "Parent directory does not exist" in str(exc_info.value)

    def test_pid_path_invalid_parent_raises_error(self) -> None:
        """Test pid_path with invalid parent directory raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                pid_path=Path("/nonexistent/deeply/nested/emit.pid"),
            )

        assert "Parent directory does not exist" in str(exc_info.value)

    def test_spool_dir_validator_allows_paths_with_existing_root(self) -> None:
        """Test spool_dir validator accepts paths when root exists.

        The validator walks up the path tree until it finds an existing
        directory. On Unix systems, '/' always exists, so deeply nested
        paths under root are considered valid (the directories can be created).
        """
        # This path is valid because '/' exists and is a directory
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            spool_dir=Path("/nonexistent_root/deeply/nested/spool"),
        )
        assert config.spool_dir == Path("/nonexistent_root/deeply/nested/spool")

    def test_socket_path_creatable_parent(self) -> None:
        """Test socket_path with creatable parent (grandparent exists) is valid."""
        # /tmp exists, so /tmp/newdir/emit.sock should be valid
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            socket_path=Path("/tmp/newdir/emit.sock"),
        )
        assert config.socket_path == Path("/tmp/newdir/emit.sock")


@pytest.mark.unit
class TestModelEmitDaemonConfigSpoolingEnabled:
    """Test spooling_enabled property."""

    def test_spooling_enabled_true_when_both_nonzero(self) -> None:
        """Test spooling_enabled returns True when both spool limits > 0."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=100,
            max_spool_bytes=1_000_000,
        )
        assert config.spooling_enabled is True

    def test_spooling_enabled_false_when_both_zero(self) -> None:
        """Test spooling_enabled returns False when both spool limits = 0."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            max_spool_messages=0,
            max_spool_bytes=0,
        )
        assert config.spooling_enabled is False

    def test_spooling_enabled_with_defaults(self) -> None:
        """Test spooling_enabled returns True with default values (spooling enabled)."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:9092")
        # Default: max_spool_messages=1000, max_spool_bytes=10MB
        assert config.spooling_enabled is True

    def test_spooling_enabled_property_is_read_only(self) -> None:
        """Test spooling_enabled cannot be set (frozen model).

        Note: Pydantic frozen models raise ValidationError, not AttributeError,
        when attempting to set any attribute including computed properties.
        """
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:9092")

        with pytest.raises(ValidationError):
            config.spooling_enabled = False  # type: ignore[misc]


@pytest.mark.unit
class TestModelEmitDaemonConfigEnvOverrides:
    """Test with_env_overrides class method."""

    def test_env_override_kafka_bootstrap_servers(self) -> None:
        """Test environment variable overrides kafka_bootstrap_servers."""
        with patch.dict(
            os.environ,
            {"EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "kafka.prod:9092"},
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides(
                kafka_bootstrap_servers="localhost:9092"  # Overridden by env
            )
            assert config.kafka_bootstrap_servers == "kafka.prod:9092"

    def test_env_override_socket_path(self) -> None:
        """Test environment variable overrides socket_path."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_SOCKET_PATH": "/var/run/emit.sock",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.socket_path == Path("/var/run/emit.sock")

    def test_env_override_pid_path(self) -> None:
        """Test environment variable overrides pid_path."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_PID_PATH": "/var/run/emit.pid",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.pid_path == Path("/var/run/emit.pid")

    def test_env_override_spool_dir(self) -> None:
        """Test environment variable overrides spool_dir."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_SPOOL_DIR": "/tmp/spool",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.spool_dir == Path("/tmp/spool")

    def test_env_override_kafka_client_id(self) -> None:
        """Test environment variable overrides kafka_client_id."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_KAFKA_CLIENT_ID": "my-emit-daemon",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.kafka_client_id == "my-emit-daemon"

    def test_env_override_max_payload_bytes(self) -> None:
        """Test environment variable overrides max_payload_bytes."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_MAX_PAYLOAD_BYTES": "2097152",  # 2MB
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.max_payload_bytes == 2_097_152

    def test_env_override_max_memory_queue(self) -> None:
        """Test environment variable overrides max_memory_queue."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_MAX_MEMORY_QUEUE": "500",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.max_memory_queue == 500

    def test_env_override_timeouts(self) -> None:
        """Test environment variables override timeout values."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_SOCKET_TIMEOUT_SECONDS": "10.0",
                "EMIT_DAEMON_KAFKA_TIMEOUT_SECONDS": "60.0",
                "EMIT_DAEMON_SHUTDOWN_DRAIN_SECONDS": "20.0",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.socket_timeout_seconds == 10.0
            assert config.kafka_timeout_seconds == 60.0
            assert config.shutdown_drain_seconds == 20.0

    def test_env_override_takes_precedence_over_kwargs(self) -> None:
        """Test environment variables take precedence over kwargs."""
        with patch.dict(
            os.environ,
            {"EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "env-broker:9092"},
            clear=True,
        ):
            config = ModelEmitDaemonConfig.with_env_overrides(
                kafka_bootstrap_servers="kwarg-broker:9092"
            )
            assert config.kafka_bootstrap_servers == "env-broker:9092"

    def test_kwargs_used_when_no_env_var(self) -> None:
        """Test kwargs are used when environment variables are not set."""
        with patch.dict(os.environ, {}, clear=True):
            config = ModelEmitDaemonConfig.with_env_overrides(
                kafka_bootstrap_servers="localhost:9092",
                max_memory_queue=200,
            )
            assert config.kafka_bootstrap_servers == "localhost:9092"
            assert config.max_memory_queue == 200

    def test_invalid_env_value_skipped(self) -> None:
        """Test invalid environment values are skipped (Pydantic validates)."""
        with patch.dict(
            os.environ,
            {
                "EMIT_DAEMON_MAX_MEMORY_QUEUE": "not_a_number",
                "EMIT_DAEMON_KAFKA_BOOTSTRAP_SERVERS": "localhost:9092",
            },
            clear=True,
        ):
            # Invalid int conversion is skipped, default used
            config = ModelEmitDaemonConfig.with_env_overrides()
            assert config.max_memory_queue == 100  # Default value


@pytest.mark.unit
class TestModelEmitDaemonConfigImmutability:
    """Test frozen model immutability."""

    def test_model_is_frozen(self) -> None:
        """Test model is immutable (frozen)."""
        config = ModelEmitDaemonConfig(kafka_bootstrap_servers="localhost:9092")

        with pytest.raises(ValidationError):
            config.max_memory_queue = 500  # type: ignore[misc]

    def test_model_extra_fields_forbidden(self) -> None:
        """Test extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                unknown_field="value",  # type: ignore[call-arg]
            )

        errors = exc_info.value.errors()
        assert any(e["type"] == "extra_forbidden" for e in errors)


@pytest.mark.unit
class TestModelEmitDaemonConfigTimeoutValidation:
    """Test timeout field validations."""

    def test_socket_timeout_minimum(self) -> None:
        """Test socket_timeout_seconds minimum (0.1) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            socket_timeout_seconds=0.1,
        )
        assert config.socket_timeout_seconds == 0.1

    def test_socket_timeout_below_minimum_raises_error(self) -> None:
        """Test socket_timeout_seconds below 0.1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                socket_timeout_seconds=0.05,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("socket_timeout_seconds",) for e in errors)

    def test_socket_timeout_maximum(self) -> None:
        """Test socket_timeout_seconds maximum (60.0) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            socket_timeout_seconds=60.0,
        )
        assert config.socket_timeout_seconds == 60.0

    def test_socket_timeout_above_maximum_raises_error(self) -> None:
        """Test socket_timeout_seconds above 60.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                socket_timeout_seconds=60.1,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("socket_timeout_seconds",) for e in errors)

    def test_kafka_timeout_minimum(self) -> None:
        """Test kafka_timeout_seconds minimum (1.0) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            kafka_timeout_seconds=1.0,
        )
        assert config.kafka_timeout_seconds == 1.0

    def test_kafka_timeout_below_minimum_raises_error(self) -> None:
        """Test kafka_timeout_seconds below 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                kafka_timeout_seconds=0.5,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("kafka_timeout_seconds",) for e in errors)

    def test_kafka_timeout_maximum(self) -> None:
        """Test kafka_timeout_seconds maximum (300.0) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            kafka_timeout_seconds=300.0,
        )
        assert config.kafka_timeout_seconds == 300.0

    def test_kafka_timeout_above_maximum_raises_error(self) -> None:
        """Test kafka_timeout_seconds above 300.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                kafka_timeout_seconds=300.1,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("kafka_timeout_seconds",) for e in errors)

    def test_shutdown_drain_minimum(self) -> None:
        """Test shutdown_drain_seconds minimum (0.0) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            shutdown_drain_seconds=0.0,
        )
        assert config.shutdown_drain_seconds == 0.0

    def test_shutdown_drain_below_minimum_raises_error(self) -> None:
        """Test shutdown_drain_seconds below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                shutdown_drain_seconds=-1.0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("shutdown_drain_seconds",) for e in errors)

    def test_shutdown_drain_maximum(self) -> None:
        """Test shutdown_drain_seconds maximum (300.0) is valid."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            shutdown_drain_seconds=300.0,
        )
        assert config.shutdown_drain_seconds == 300.0

    def test_shutdown_drain_above_maximum_raises_error(self) -> None:
        """Test shutdown_drain_seconds above 300.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                shutdown_drain_seconds=300.1,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("shutdown_drain_seconds",) for e in errors)


@pytest.mark.unit
class TestModelEmitDaemonConfigKafkaClientId:
    """Test kafka_client_id validation."""

    def test_kafka_client_id_custom(self) -> None:
        """Test custom kafka_client_id is accepted."""
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            kafka_client_id="my-custom-client",
        )
        assert config.kafka_client_id == "my-custom-client"

    def test_kafka_client_id_empty_raises_error(self) -> None:
        """Test empty kafka_client_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                kafka_client_id="",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("kafka_client_id",) for e in errors)

    def test_kafka_client_id_max_length(self) -> None:
        """Test kafka_client_id at max length (255) is valid."""
        client_id = "a" * 255
        config = ModelEmitDaemonConfig(
            kafka_bootstrap_servers="localhost:9092",
            kafka_client_id=client_id,
        )
        assert config.kafka_client_id == client_id

    def test_kafka_client_id_exceeds_max_length_raises_error(self) -> None:
        """Test kafka_client_id exceeding 255 chars raises ValidationError."""
        client_id = "a" * 256
        with pytest.raises(ValidationError) as exc_info:
            ModelEmitDaemonConfig(
                kafka_bootstrap_servers="localhost:9092",
                kafka_client_id=client_id,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("kafka_client_id",) for e in errors)
