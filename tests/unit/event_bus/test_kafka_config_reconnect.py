# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelKafkaEventBusConfig reconnect backoff fields (OMN-2916).

Tests the reconnect_backoff_ms and reconnect_backoff_max_ms fields added to
ModelKafkaEventBusConfig to prevent thundering-herd reconnection storms after
broker restarts.
"""

from __future__ import annotations

import os

import pytest

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig


class TestReconnectBackoffDefaults:
    """Test default values for reconnect backoff fields."""

    @pytest.mark.unit
    def test_default_reconnect_backoff_ms(self) -> None:
        """Default reconnect_backoff_ms should be 2000 ms."""
        config = ModelKafkaEventBusConfig()
        assert config.reconnect_backoff_ms == 2000

    @pytest.mark.unit
    def test_default_reconnect_backoff_max_ms(self) -> None:
        """Default reconnect_backoff_max_ms should be 30000 ms."""
        config = ModelKafkaEventBusConfig()
        assert config.reconnect_backoff_max_ms == 30000


class TestReconnectBackoffCustomValues:
    """Test custom values can be set via constructor."""

    @pytest.mark.unit
    def test_custom_reconnect_backoff_ms(self) -> None:
        """Should accept a custom reconnect_backoff_ms value."""
        config = ModelKafkaEventBusConfig(
            reconnect_backoff_ms=1000,
            reconnect_backoff_max_ms=20000,
        )
        assert config.reconnect_backoff_ms == 1000

    @pytest.mark.unit
    def test_custom_reconnect_backoff_max_ms(self) -> None:
        """Should accept a custom reconnect_backoff_max_ms value."""
        config = ModelKafkaEventBusConfig(
            reconnect_backoff_ms=500,
            reconnect_backoff_max_ms=60000,
        )
        assert config.reconnect_backoff_max_ms == 60000


class TestReconnectBackoffValidator:
    """Test the cross-field validator for reconnect backoff values."""

    @pytest.mark.unit
    def test_validator_raises_when_max_less_than_base(self) -> None:
        """Cross-field validator must raise when max < base."""
        with pytest.raises(ProtocolConfigurationError):
            ModelKafkaEventBusConfig(
                reconnect_backoff_ms=5000,
                reconnect_backoff_max_ms=1000,
            )

    @pytest.mark.unit
    def test_validator_passes_when_max_equals_base(self) -> None:
        """Cross-field validator must pass when max == base."""
        config = ModelKafkaEventBusConfig(
            reconnect_backoff_ms=5000,
            reconnect_backoff_max_ms=5000,
        )
        assert config.reconnect_backoff_ms == 5000
        assert config.reconnect_backoff_max_ms == 5000


class TestReconnectBackoffEnvOverrides:
    """Test environment variable overrides for reconnect backoff fields."""

    @pytest.mark.unit
    def test_kafka_reconnect_backoff_ms_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAFKA_RECONNECT_BACKOFF_MS env var should override reconnect_backoff_ms."""
        monkeypatch.setenv("KAFKA_RECONNECT_BACKOFF_MS", "1500")
        config = ModelKafkaEventBusConfig.default()
        assert config.reconnect_backoff_ms == 1500

    @pytest.mark.unit
    def test_kafka_reconnect_backoff_max_ms_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAFKA_RECONNECT_BACKOFF_MAX_MS env var should override reconnect_backoff_max_ms."""
        monkeypatch.setenv("KAFKA_RECONNECT_BACKOFF_MAX_MS", "45000")
        config = ModelKafkaEventBusConfig.default()
        assert config.reconnect_backoff_max_ms == 45000


class TestReconnectBackoffDefaultFactory:
    """Test default() factory includes reconnect backoff fields."""

    @pytest.mark.unit
    def test_default_factory_includes_reconnect_backoff_ms(self) -> None:
        """default() factory should return expected reconnect_backoff_ms default."""
        config = ModelKafkaEventBusConfig.default()
        assert config.reconnect_backoff_ms == 2000

    @pytest.mark.unit
    def test_default_factory_includes_reconnect_backoff_max_ms(self) -> None:
        """default() factory should return expected reconnect_backoff_max_ms default."""
        config = ModelKafkaEventBusConfig.default()
        assert config.reconnect_backoff_max_ms == 30000


class TestReconnectBackoffJsonSchema:
    """Test reconnect backoff fields appear in model JSON schema."""

    @pytest.mark.unit
    def test_fields_appear_in_json_schema(self) -> None:
        """reconnect_backoff_ms and reconnect_backoff_max_ms must appear in JSON schema."""
        schema = ModelKafkaEventBusConfig.model_json_schema()
        properties = schema.get("properties", {})
        assert "reconnect_backoff_ms" in properties
        assert "reconnect_backoff_max_ms" in properties
