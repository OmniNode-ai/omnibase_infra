# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ConfigSavingsEstimation env-var fallback behavior. [OMN-7837]"""

from __future__ import annotations

from pathlib import Path

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

    def test_default_topics_include_dispatch_outcome_evaluated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", "broker-specific:9092"
        )

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )
        from omnibase_infra.topics import topic_keys
        from omnibase_infra.topics.service_topic_registry import ServiceTopicRegistry

        registry = ServiceTopicRegistry.from_defaults()
        cfg = ConfigSavingsEstimation()

        assert registry.resolve(topic_keys.DISPATCH_OUTCOME_EVALUATED) in (
            cfg.consumed_topics
        )

    def test_every_consumed_topic_is_declared_in_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gate the OMN-13533 regression class.

        The savings-estimation runtime consumer subscribes to the topics in
        ``ConfigSavingsEstimation.consumed_topics``. The contract-driven topic
        provisioner (``scripts/create_kafka_topics.py``) only creates topics it
        finds in a contract's ``event_bus.subscribe_topics``. If a consumed topic
        is NOT declared in the contract, the broker never provisions it and the
        consumer's subscribe loop error-storms ``topic not found in cluster`` on
        every fresh redeploy (validator-catch.v1 + hook-context-injected.v1 did
        exactly this: 174 + 220 errors / 15m).

        This test fails if any runtime-consumed topic is missing from the
        node_savings_estimation_compute contract's declared subscribe_topics.
        """
        import yaml

        from omnibase_infra.services.observability.savings_estimation.config import (
            ConfigSavingsEstimation,
        )

        monkeypatch.setenv(
            "OMNIBASE_INFRA_SAVINGS_KAFKA_BOOTSTRAP_SERVERS", "broker-specific:9092"
        )

        contract_path = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_savings_estimation_compute"
            / "contract.yaml"
        )
        contract = yaml.safe_load(contract_path.read_text())
        declared = set(contract["event_bus"]["subscribe_topics"])

        cfg = ConfigSavingsEstimation()
        consumed = set(cfg.consumed_topics)

        undeclared = consumed - declared
        assert not undeclared, (
            "Runtime-consumed topics are not declared in "
            "node_savings_estimation_compute/contract.yaml subscribe_topics, so "
            "the contract-driven provisioner will not create them on the broker "
            f"(consumer-start error storm risk, OMN-13533): {sorted(undeclared)}"
        )
