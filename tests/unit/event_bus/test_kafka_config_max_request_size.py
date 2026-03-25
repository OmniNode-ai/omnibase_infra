# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for max_request_size field on ModelKafkaEventBusConfig."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)


@pytest.mark.unit
class TestMaxRequestSizeConfig:
    """Tests for max_request_size config field."""

    def test_max_request_size_default(self) -> None:
        """max_request_size defaults to 4MB."""
        config = ModelKafkaEventBusConfig()
        assert config.max_request_size == 4 * 1024 * 1024  # 4,194,304

    def test_max_request_size_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """KAFKA_MAX_REQUEST_SIZE env var overrides default."""
        monkeypatch.setenv("KAFKA_MAX_REQUEST_SIZE", "8388608")
        config = ModelKafkaEventBusConfig().apply_environment_overrides()
        assert config.max_request_size == 8388608

    def test_max_request_size_validation_minimum(self) -> None:
        """max_request_size rejects values below 1KB."""
        with pytest.raises(ValidationError):
            ModelKafkaEventBusConfig(max_request_size=500)

    def test_max_request_size_validation_maximum(self) -> None:
        """max_request_size rejects values above 50MB."""
        with pytest.raises(ValidationError):
            ModelKafkaEventBusConfig(max_request_size=60_000_000)

    def test_max_request_size_custom_value(self) -> None:
        """max_request_size accepts valid custom values."""
        config = ModelKafkaEventBusConfig(max_request_size=2_000_000)
        assert config.max_request_size == 2_000_000
