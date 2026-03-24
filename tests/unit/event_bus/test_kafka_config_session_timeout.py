# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelKafkaEventBusConfig session timeout fields.

Tests the session_timeout_ms, heartbeat_interval_ms, and max_poll_interval_ms
fields and the validate_session_timeout_ratio cross-field validator.
"""

from __future__ import annotations

import logging

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


class TestSessionTimeoutDefaults:
    """Test default values for session timeout fields."""

    @pytest.mark.unit
    def test_default_session_timeout_ms(self) -> None:
        config = ModelKafkaEventBusConfig()
        assert config.session_timeout_ms == 45000

    @pytest.mark.unit
    def test_default_heartbeat_interval_ms(self) -> None:
        config = ModelKafkaEventBusConfig()
        assert config.heartbeat_interval_ms == 15000

    @pytest.mark.unit
    def test_default_max_poll_interval_ms(self) -> None:
        config = ModelKafkaEventBusConfig()
        assert config.max_poll_interval_ms == 300000


class TestSessionTimeoutCustomValues:
    """Test custom values within bounds are accepted."""

    @pytest.mark.unit
    def test_custom_session_timeout_ms(self) -> None:
        config = ModelKafkaEventBusConfig(
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
            max_poll_interval_ms=300000,
        )
        assert config.session_timeout_ms == 30000

    @pytest.mark.unit
    def test_custom_heartbeat_interval_ms(self) -> None:
        config = ModelKafkaEventBusConfig(
            session_timeout_ms=60000,
            heartbeat_interval_ms=20000,
            max_poll_interval_ms=300000,
        )
        assert config.heartbeat_interval_ms == 20000

    @pytest.mark.unit
    def test_custom_max_poll_interval_ms(self) -> None:
        config = ModelKafkaEventBusConfig(
            session_timeout_ms=45000,
            heartbeat_interval_ms=15000,
            max_poll_interval_ms=600000,
        )
        assert config.max_poll_interval_ms == 600000


class TestSessionTimeoutValidator:
    """Test the validate_session_timeout_ratio cross-field validator."""

    @pytest.mark.unit
    def test_heartbeat_equal_to_session_timeout_rejected(self) -> None:
        """heartbeat_interval_ms must be strictly less than session_timeout_ms."""
        with pytest.raises(ProtocolConfigurationError, match="heartbeat_interval_ms"):
            ModelKafkaEventBusConfig(
                session_timeout_ms=30000,
                heartbeat_interval_ms=30000,
                max_poll_interval_ms=300000,
            )

    @pytest.mark.unit
    def test_heartbeat_greater_than_session_timeout_rejected(self) -> None:
        with pytest.raises(ProtocolConfigurationError, match="heartbeat_interval_ms"):
            ModelKafkaEventBusConfig(
                session_timeout_ms=30000,
                heartbeat_interval_ms=40000,
                max_poll_interval_ms=300000,
            )

    @pytest.mark.unit
    def test_max_poll_less_than_session_timeout_rejected(self) -> None:
        """max_poll_interval_ms must be >= session_timeout_ms."""
        with pytest.raises(ProtocolConfigurationError, match="max_poll_interval_ms"):
            ModelKafkaEventBusConfig(
                session_timeout_ms=45000,
                heartbeat_interval_ms=15000,
                max_poll_interval_ms=30000,
            )

    @pytest.mark.unit
    def test_valid_ratio_accepted(self) -> None:
        """Default ratio (15000 < 45000 <= 300000) should pass validation."""
        config = ModelKafkaEventBusConfig()
        assert config.heartbeat_interval_ms < config.session_timeout_ms
        assert config.max_poll_interval_ms >= config.session_timeout_ms


class TestSessionTimeoutEnvVarOverride:
    """Test environment variable overrides for session timeout fields."""

    @pytest.mark.unit
    def test_env_var_override_session_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("KAFKA_SESSION_TIMEOUT_MS", "30000")
        monkeypatch.setenv("KAFKA_HEARTBEAT_INTERVAL_MS", "10000")
        config = ModelKafkaEventBusConfig.default()
        assert config.session_timeout_ms == 30000
        assert config.heartbeat_interval_ms == 10000


class TestHeartbeatSessionRatioAdvisoryValidator:
    """Test advisory validator for heartbeat/session ratio (OMN-5445)."""

    @pytest.mark.unit
    def test_no_warning_when_ratio_is_valid(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when heartbeat <= session_timeout / 3."""
        with caplog.at_level(logging.WARNING):
            ModelKafkaEventBusConfig(
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                max_poll_interval_ms=300000,
            )
        assert "heartbeat_interval_ms" not in caplog.text

    @pytest.mark.unit
    def test_warning_when_heartbeat_exceeds_ratio(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning when heartbeat > session_timeout / 3."""
        with caplog.at_level(logging.WARNING):
            config = ModelKafkaEventBusConfig(
                session_timeout_ms=30000,
                heartbeat_interval_ms=15000,
                max_poll_interval_ms=300000,
            )
        assert "heartbeat_interval_ms" in caplog.text
        assert "exceeds" in caplog.text
        assert config.session_timeout_ms == 30000
        assert config.heartbeat_interval_ms == 15000
