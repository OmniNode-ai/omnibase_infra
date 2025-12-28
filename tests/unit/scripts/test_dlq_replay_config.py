# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for DLQ replay configuration validation.

This test suite validates:
- ModelReplayConfig bootstrap_servers validation
- Rate limit validation
- Time range validation

Related Tickets:
    - OMN-1059: Infrastructure tech debt - complete stub implementations
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add scripts directory to path for import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts"))

from dlq_replay import ModelReplayConfig


@pytest.mark.unit
class TestModelReplayConfigBootstrapServersValidation:
    """Test bootstrap_servers validation in ModelReplayConfig."""

    def test_valid_single_server(self) -> None:
        """Test valid single bootstrap server passes validation."""
        config = ModelReplayConfig(bootstrap_servers="localhost:9092")
        assert config.bootstrap_servers == "localhost:9092"

    def test_valid_multiple_servers(self) -> None:
        """Test valid comma-separated bootstrap servers pass validation."""
        config = ModelReplayConfig(
            bootstrap_servers="kafka1:9092,kafka2:9092,kafka3:9092"
        )
        assert config.bootstrap_servers == "kafka1:9092,kafka2:9092,kafka3:9092"

    def test_valid_server_with_ip(self) -> None:
        """Test valid IP address bootstrap server passes validation."""
        config = ModelReplayConfig(bootstrap_servers="192.168.1.100:29092")
        assert config.bootstrap_servers == "192.168.1.100:29092"

    def test_valid_server_with_whitespace_stripped(self) -> None:
        """Test whitespace around bootstrap servers is stripped."""
        config = ModelReplayConfig(bootstrap_servers="  localhost:9092  ")
        assert config.bootstrap_servers == "localhost:9092"

    def test_empty_string_raises_validation_error(self) -> None:
        """Test empty string bootstrap_servers raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="")

        assert "bootstrap_servers cannot be empty" in str(exc_info.value)

    def test_whitespace_only_raises_validation_error(self) -> None:
        """Test whitespace-only bootstrap_servers raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="   ")

        assert "bootstrap_servers cannot be empty" in str(exc_info.value)

    def test_none_raises_validation_error(self) -> None:
        """Test None bootstrap_servers raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers=None)  # type: ignore[arg-type]

        assert "bootstrap_servers cannot be None" in str(exc_info.value)

    def test_missing_port_raises_validation_error(self) -> None:
        """Test bootstrap_servers without port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="localhost")

        assert "Invalid bootstrap server format" in str(exc_info.value)
        assert "Expected 'host:port'" in str(exc_info.value)

    def test_invalid_port_raises_validation_error(self) -> None:
        """Test bootstrap_servers with non-numeric port raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="localhost:notaport")

        assert "Invalid port" in str(exc_info.value)
        assert "Port must be a valid integer" in str(exc_info.value)

    def test_port_out_of_range_high_raises_validation_error(self) -> None:
        """Test bootstrap_servers with port > 65535 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="localhost:99999")

        assert "Invalid port 99999" in str(exc_info.value)
        assert "Port must be between 1 and 65535" in str(exc_info.value)

    def test_port_zero_raises_validation_error(self) -> None:
        """Test bootstrap_servers with port 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="localhost:0")

        assert "Invalid port 0" in str(exc_info.value)
        assert "Port must be between 1 and 65535" in str(exc_info.value)

    def test_empty_host_raises_validation_error(self) -> None:
        """Test bootstrap_servers with empty host raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers=":9092")

        assert "Host cannot be empty" in str(exc_info.value)

    def test_empty_entry_in_list_raises_validation_error(self) -> None:
        """Test bootstrap_servers with empty entry in comma-separated list."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers="localhost:9092,,broker2:9092")

        assert "cannot contain empty entries" in str(exc_info.value)

    def test_integer_type_raises_validation_error(self) -> None:
        """Test non-string bootstrap_servers raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(bootstrap_servers=9092)  # type: ignore[arg-type]

        assert "bootstrap_servers must be a string" in str(exc_info.value)


@pytest.mark.unit
class TestModelReplayConfigRateLimitValidation:
    """Test rate_limit_per_second validation in ModelReplayConfig."""

    def test_valid_rate_limit(self) -> None:
        """Test valid rate limit passes validation."""
        config = ModelReplayConfig(
            bootstrap_servers="localhost:9092", rate_limit_per_second=50.0
        )
        assert config.rate_limit_per_second == 50.0

    def test_zero_rate_limit_raises_validation_error(self) -> None:
        """Test zero rate limit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(
                bootstrap_servers="localhost:9092", rate_limit_per_second=0.0
            )

        assert "rate_limit_per_second must be > 0" in str(exc_info.value)

    def test_negative_rate_limit_raises_validation_error(self) -> None:
        """Test negative rate limit raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelReplayConfig(
                bootstrap_servers="localhost:9092", rate_limit_per_second=-10.0
            )

        assert "rate_limit_per_second must be > 0" in str(exc_info.value)
