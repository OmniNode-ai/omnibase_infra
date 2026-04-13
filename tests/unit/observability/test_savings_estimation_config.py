# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ConfigSavingsEstimation env-var fallback behavior. [OMN-7837]"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestSavingsEstimationConfig:
    """Verify bootstrap-server env-var resolution and fail-loud behavior."""

    def test_loads_from_fallback_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(
            "OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", raising=False
        )
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker-fallback:9092")

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )

        cfg = ConfigSavingsEstimation()
        assert cfg.kafka_bootstrap_servers == "broker-fallback:9092"

    def test_prefix_wins_over_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", "broker-specific:9092"
        )
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker-fallback:9092")

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )

        cfg = ConfigSavingsEstimation()
        assert cfg.kafka_bootstrap_servers == "broker-specific:9092"

    def test_loads_from_prefix_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", "broker-specific:9092"
        )
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )

        cfg = ConfigSavingsEstimation()
        assert cfg.kafka_bootstrap_servers == "broker-specific:9092"

    def test_raises_when_neither_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(
            "OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", raising=False
        )
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )

        with pytest.raises(ValidationError):
            ConfigSavingsEstimation()

    def test_raises_when_both_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", "")
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "")

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )

        with pytest.raises(ValidationError):
            ConfigSavingsEstimation()
