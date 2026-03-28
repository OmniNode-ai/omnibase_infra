# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for max_request_size on ModelKafkaProducerConfig (OMN-6320)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)


@pytest.mark.unit
class TestModelKafkaProducerConfigMaxRequestSize:
    """Verify max_request_size field on ModelKafkaProducerConfig."""

    def test_default_is_4mb(self) -> None:
        """Default max_request_size is 4 MB."""
        config = ModelKafkaProducerConfig()
        assert config.max_request_size == 4 * 1024 * 1024

    def test_custom_value_accepted(self) -> None:
        """Explicit max_request_size within valid range is accepted."""
        config = ModelKafkaProducerConfig(max_request_size=8_000_000)
        assert config.max_request_size == 8_000_000

    def test_rejects_below_minimum(self) -> None:
        """Values below 1 KB are rejected."""
        with pytest.raises(ValidationError):
            ModelKafkaProducerConfig(max_request_size=500)

    def test_rejects_above_maximum(self) -> None:
        """Values above 50 MB are rejected."""
        with pytest.raises(ValidationError):
            ModelKafkaProducerConfig(max_request_size=60_000_000)

    def test_from_env_reads_kafka_max_request_size(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() picks up KAFKA_MAX_REQUEST_SIZE env var."""
        monkeypatch.setenv("KAFKA_MAX_REQUEST_SIZE", "8388608")
        config = ModelKafkaProducerConfig.from_env()
        assert config.max_request_size == 8_388_608

    def test_from_env_default_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """from_env() uses 4 MB default when env var is not set."""
        monkeypatch.delenv("KAFKA_MAX_REQUEST_SIZE", raising=False)
        config = ModelKafkaProducerConfig.from_env()
        assert config.max_request_size == 4 * 1024 * 1024
