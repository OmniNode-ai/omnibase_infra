# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The agent creates the topics only it publishes before the first send (OMN-17427).

Read on the lab 2026-10-17: a lab deploy agent logged "Topic
onex.evt.deploy.dev-tag-heartbeat.v1 not found in cluster metadata" and every heartbeat
publish timed out. No contract declares that topic or the dev-lane-drift topic, and a broker
the services/topics.yaml provisioner has not run against lacks them: the agent is their only
producer, so the agent creates each once per process before its first publish to it. Topics
another node declares are left alone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from deploy_agent import publisher
from deploy_agent.events import TOPIC_DEV_LANE_DRIFT, TOPIC_DEV_TAG_HEARTBEAT
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from kafka.errors import TopicAlreadyExistsError

from omnibase_infra.topics.topic_namespace import apply_topic_namespace

_CONFIG = ModelDeployAgentKafkaConfig(
    bootstrap_servers="broker.example:9092", security_protocol="PLAINTEXT"
)


@pytest.fixture(autouse=True)
def _fresh_process() -> Iterator[None]:
    publisher._ENSURED_TOPICS.clear()
    yield
    publisher._ENSURED_TOPICS.clear()


def _created(admin: MagicMock) -> list[str]:
    return [t.name for call in admin.create_topics.call_args_list for t in call.args[0]]


@pytest.mark.parametrize(
    ("publish", "topic"),
    [
        (publisher.publish_dev_tag_heartbeat, TOPIC_DEV_TAG_HEARTBEAT),
        (publisher.publish_dev_lane_drift, TOPIC_DEV_LANE_DRIFT),
    ],
)
def test_an_agent_owned_topic_is_created_before_the_first_publish(
    publish: Callable[[dict[str, Any], ModelDeployAgentKafkaConfig], bool], topic: str
) -> None:
    admin, producer = MagicMock(), MagicMock()
    with (
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin) as ctor,
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        assert publish({"instance": "lab-a", "repository": "omnimarket"}, _CONFIG)
    assert ctor.call_args.kwargs["bootstrap_servers"] == "broker.example:9092"
    assert _created(admin) == [apply_topic_namespace(topic)]
    admin.close.assert_called_once()
    assert producer.send.call_args.args[0] == apply_topic_namespace(topic)


def test_a_topic_that_exists_still_publishes_and_is_created_once() -> None:
    admin, producer = MagicMock(), MagicMock()
    admin.create_topics.side_effect = TopicAlreadyExistsError()
    with (
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin),
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        assert publisher.publish_dev_tag_heartbeat({"instance": "a"}, _CONFIG)
        assert publisher.publish_dev_tag_heartbeat({"instance": "a"}, _CONFIG)
    assert admin.create_topics.call_count == 1
    assert producer.send.call_count == 2


def test_an_admin_failure_is_retried_on_the_next_publish() -> None:
    admin, producer = MagicMock(), MagicMock()
    admin.create_topics.side_effect = [OSError("broker down"), None]
    with (
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin),
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        publisher.publish_dev_tag_heartbeat({"instance": "a"}, _CONFIG)
        publisher.publish_dev_tag_heartbeat({"instance": "a"}, _CONFIG)
    assert admin.create_topics.call_count == 2


def test_a_topic_another_node_declares_is_not_created() -> None:
    """Negative control: the completion topic belongs to a contract, not to the agent."""
    admin, producer = MagicMock(), MagicMock()
    with (
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin) as ctor,
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        assert publisher.publish_result({"correlation_id": "c"}, _CONFIG)
    ctor.assert_not_called()


@pytest.mark.parametrize(("brokers", "expected_rf"), [([1, 2, 3], 2), ([1], 1)])
def test_the_replication_factor_is_resolved_by_the_policy_for_the_measured_brokers(
    brokers: list[int], expected_rf: int
) -> None:
    """OMN-15395: no literal replication factor; the measured broker count drives it."""
    admin, producer = MagicMock(), MagicMock()
    admin.describe_cluster.return_value = {"brokers": brokers}
    with (
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin),
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        assert publisher.publish_dev_tag_heartbeat({"instance": "a"}, _CONFIG)
    (created,) = admin.create_topics.call_args.args[0]
    assert created.replication_factor == expected_rf
