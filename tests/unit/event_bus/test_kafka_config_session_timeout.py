# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ModelKafkaEventBusConfig session timeout fields (OMN-5445).

Tests the session_timeout_ms and heartbeat_interval_ms fields added to
ModelKafkaEventBusConfig, including defaults, env var overrides, advisory
validator, and field constraints.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


class TestSessionTimeoutDefaults:
    """Test default values for session timeout fields."""

    @pytest.mark.unit
    def test_default_session_timeout_ms(self) -> None:
        """Default session_timeout_ms should be 30000 ms."""
        config = ModelKafkaEventBusConfig()
        assert config.session_timeout_ms == 30000

    @pytest.mark.unit
    def test_default_heartbeat_interval_ms(self) -> None:
        """Default heartbeat_interval_ms should be 10000 ms."""
        config = ModelKafkaEventBusConfig()
        assert config.heartbeat_interval_ms == 10000

    @pytest.mark.unit
    def test_defaults_satisfy_kafka_ratio_recommendation(self) -> None:
        """Default heartbeat (10s) should be <= session_timeout (30s) / 3."""
        config = ModelKafkaEventBusConfig()
        assert config.heartbeat_interval_ms <= config.session_timeout_ms / 3


class TestSessionTimeoutCustomValues:
    """Test custom values can be set via constructor."""

    @pytest.mark.unit
    def test_custom_session_timeout_ms(self) -> None:
        """Should accept a custom session_timeout_ms value."""
        config = ModelKafkaEventBusConfig(session_timeout_ms=60000)
        assert config.session_timeout_ms == 60000

    @pytest.mark.unit
    def test_custom_heartbeat_interval_ms(self) -> None:
        """Should accept a custom heartbeat_interval_ms value."""
        config = ModelKafkaEventBusConfig(heartbeat_interval_ms=5000)
        assert config.heartbeat_interval_ms == 5000


class TestSessionTimeoutConstraints:
    """Test field constraints (ge/le bounds)."""

    @pytest.mark.unit
    def test_session_timeout_ms_below_minimum_raises(self) -> None:
        """session_timeout_ms below 6000 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelKafkaEventBusConfig(session_timeout_ms=5000)

    @pytest.mark.unit
    def test_session_timeout_ms_above_maximum_raises(self) -> None:
        """session_timeout_ms above 300000 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelKafkaEventBusConfig(session_timeout_ms=400000)

    @pytest.mark.unit
    def test_heartbeat_interval_ms_below_minimum_raises(self) -> None:
        """heartbeat_interval_ms below 1000 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelKafkaEventBusConfig(heartbeat_interval_ms=500)

    @pytest.mark.unit
    def test_heartbeat_interval_ms_above_maximum_raises(self) -> None:
        """heartbeat_interval_ms above 100000 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ModelKafkaEventBusConfig(heartbeat_interval_ms=200000)

    @pytest.mark.unit
    def test_session_timeout_ms_at_minimum(self) -> None:
        """session_timeout_ms at boundary 6000 should be accepted."""
        config = ModelKafkaEventBusConfig(session_timeout_ms=6000)
        assert config.session_timeout_ms == 6000

    @pytest.mark.unit
    def test_session_timeout_ms_at_maximum(self) -> None:
        """session_timeout_ms at boundary 300000 should be accepted."""
        config = ModelKafkaEventBusConfig(session_timeout_ms=300000)
        assert config.session_timeout_ms == 300000


class TestHeartbeatSessionRatioValidator:
    """Test advisory validator for heartbeat/session ratio."""

    @pytest.mark.unit
    def test_no_warning_when_ratio_is_valid(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when heartbeat <= session_timeout / 3."""
        with caplog.at_level(logging.WARNING):
            ModelKafkaEventBusConfig(
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
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
            )
        assert "heartbeat_interval_ms" in caplog.text
        assert "exceeds" in caplog.text
        # Validator is advisory: config is still created
        assert config.session_timeout_ms == 30000
        assert config.heartbeat_interval_ms == 15000

    @pytest.mark.unit
    def test_no_error_raised_for_bad_ratio(self) -> None:
        """Advisory validator must not raise, even with bad ratio."""
        config = ModelKafkaEventBusConfig(
            session_timeout_ms=30000,
            heartbeat_interval_ms=20000,
        )
        assert config.heartbeat_interval_ms == 20000


class TestSessionTimeoutEnvOverrides:
    """Test environment variable overrides for session timeout fields."""

    @pytest.mark.unit
    def test_kafka_session_timeout_ms_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAFKA_SESSION_TIMEOUT_MS env var should override session_timeout_ms."""
        monkeypatch.setenv("KAFKA_SESSION_TIMEOUT_MS", "45000")
        config = ModelKafkaEventBusConfig.default()
        assert config.session_timeout_ms == 45000

    @pytest.mark.unit
    def test_kafka_heartbeat_interval_ms_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAFKA_HEARTBEAT_INTERVAL_MS env var should override heartbeat_interval_ms."""
        monkeypatch.setenv("KAFKA_HEARTBEAT_INTERVAL_MS", "5000")
        config = ModelKafkaEventBusConfig.default()
        assert config.heartbeat_interval_ms == 5000


class TestSessionTimeoutDefaultFactory:
    """Test default() factory includes session timeout fields."""

    @pytest.mark.unit
    def test_default_factory_includes_session_timeout_ms(self) -> None:
        """default() factory should return expected session_timeout_ms default."""
        config = ModelKafkaEventBusConfig.default()
        assert config.session_timeout_ms == 30000

    @pytest.mark.unit
    def test_default_factory_includes_heartbeat_interval_ms(self) -> None:
        """default() factory should return expected heartbeat_interval_ms default."""
        config = ModelKafkaEventBusConfig.default()
        assert config.heartbeat_interval_ms == 10000


class TestSessionTimeoutJsonSchema:
    """Test session timeout fields appear in model JSON schema."""

    @pytest.mark.unit
    def test_fields_appear_in_json_schema(self) -> None:
        """session_timeout_ms and heartbeat_interval_ms must appear in JSON schema."""
        schema = ModelKafkaEventBusConfig.model_json_schema()
        properties = schema.get("properties", {})
        assert "session_timeout_ms" in properties
        assert "heartbeat_interval_ms" in properties
