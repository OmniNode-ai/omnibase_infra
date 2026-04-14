# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""TDD tests for OMN-8783: Kafka bootstrap overlay split.

Asserts that no code path silently falls back to localhost:19092 when
KAFKA_BOOTSTRAP_SERVERS is unset. Images must hard-fail if the overlay
does not provide the broker address.
"""

from __future__ import annotations

import pytest

from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)


@pytest.mark.unit
class TestKafkaBootstrapNoLocalhostFallback:
    """Verify that from_env() raises when KAFKA_BOOTSTRAP_SERVERS is absent."""

    def test_from_env_raises_when_bootstrap_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() must raise KeyError, not silently use localhost:19092."""
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        with pytest.raises(KeyError, match="KAFKA_BOOTSTRAP_SERVERS"):
            ModelKafkaProducerConfig.from_env()

    def test_from_env_uses_env_var_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() uses KAFKA_BOOTSTRAP_SERVERS when set."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
        config = ModelKafkaProducerConfig.from_env()
        assert config.bootstrap_servers == "redpanda:9092"

    def test_from_env_does_not_return_localhost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """from_env() must never silently return localhost:19092."""
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        try:
            config = ModelKafkaProducerConfig.from_env()
            assert "localhost" not in config.bootstrap_servers, (
                "from_env() must not fall back to localhost when KAFKA_BOOTSTRAP_SERVERS is unset"
            )
        except (KeyError, ValueError):
            pass  # Expected — hard-fail is correct behavior
