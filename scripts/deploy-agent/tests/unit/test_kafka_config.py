# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deploy-agent Kafka config must use one explicit control bus."""

from __future__ import annotations

import pytest
from deploy_agent.kafka_config import (
    ModelDeployAgentKafkaConfig,
    load_deploy_agent_kafka_config_from_env,
)
from pydantic import ValidationError

_TEST_PASSWORD = "test-sasl-secret"


def test_missing_bootstrap_servers_fails_without_localhost_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    with pytest.raises(RuntimeError, match="no localhost fallback"):
        load_deploy_agent_kafka_config_from_env()


def test_plaintext_config_uses_explicit_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda.example:9092")
    monkeypatch.delenv("KAFKA_SASL_USERNAME", raising=False)
    monkeypatch.delenv("KAFKA_SASL_PASSWORD", raising=False)

    config = load_deploy_agent_kafka_config_from_env()

    assert config.bootstrap_servers == "redpanda.example:9092"
    assert config.security_protocol == "PLAINTEXT"
    assert config.consumer_kwargs() == {
        "bootstrap_servers": "redpanda.example:9092",
        "security_protocol": "PLAINTEXT",
    }


def test_sasl_config_uses_kafka_python_cloud_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "pkc.example.confluent.cloud:9092")
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "key")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _TEST_PASSWORD)

    config = load_deploy_agent_kafka_config_from_env()

    assert config.security_protocol == "SASL_SSL"
    assert config.consumer_kwargs() == {
        "bootstrap_servers": "pkc.example.confluent.cloud:9092",
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "PLAIN",
        "sasl_plain_username": "key",
        "sasl_plain_password": _TEST_PASSWORD,
    }
    assert config.producer_kwargs() == config.consumer_kwargs()


def test_sasl_credentials_require_sasl_ssl() -> None:
    with pytest.raises(ValidationError, match="KAFKA_SECURITY_PROTOCOL"):
        ModelDeployAgentKafkaConfig(
            bootstrap_servers="pkc.example.confluent.cloud:9092",
            security_protocol="PLAINTEXT",
            sasl_username="key",
            sasl_password=_TEST_PASSWORD,
        )
