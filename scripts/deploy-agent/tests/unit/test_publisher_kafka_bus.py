# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Completion publishing must use the configured deploy-agent Kafka bus."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from deploy_agent.events import TOPIC_REBUILD_COMPLETED
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from deploy_agent.publisher import publish_result

_TEST_PASSWORD = "test-sasl-secret"


def test_publish_result_uses_configured_kafka_bus() -> None:
    config = ModelDeployAgentKafkaConfig(
        bootstrap_servers="pkc.example.confluent.cloud:9092",
        security_protocol="SASL_SSL",
        sasl_username="key",
        sasl_password=_TEST_PASSWORD,
    )
    producer = MagicMock()

    with patch("deploy_agent.publisher.KafkaProducer", return_value=producer) as ctor:
        ok = publish_result({"correlation_id": "abc-123"}, config)

    assert ok is True
    ctor.assert_called_once()
    kwargs = ctor.call_args.kwargs
    assert kwargs["bootstrap_servers"] == "pkc.example.confluent.cloud:9092"
    assert kwargs["security_protocol"] == "SASL_SSL"
    assert kwargs["sasl_plain_username"] == "key"
    assert kwargs["sasl_plain_password"] == _TEST_PASSWORD
    producer.send.assert_called_once_with(
        TOPIC_REBUILD_COMPLETED,
        key="deploy-result/abc-123",
        value={"correlation_id": "abc-123"},
    )
    producer.flush.assert_called_once_with(timeout=30)
    producer.close.assert_called_once()
