# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue event's topic is created before its first publish (OMN-26316).

Read on h201: the deploy agent logged "Topic onex.evt.deploy.dev-merge-build-overdue.v1 not found
in cluster metadata" about ten times a second when the overdue publish fired. The agent is the only
producer of that contract-declared topic, so the sender ensures it first, once per process.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from deploy_agent import publisher
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from pydantic import BaseModel

from omnibase_infra.nodes.node_deploy_job_reading_effect import (
    runtime_deploy_job_reading as caller,
)
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

_CONTRACT = (
    Path(caller.__file__).parent / "contract.yaml"
)  # the node declares both topics
_SETTINGS = yaml.safe_load(_CONTRACT.read_text())["settings"]
_TOPIC = _SETTINGS["dev_merge_build_overdue_topic"]
_COMMAND_TOPIC = _SETTINGS["dev_merge_build_command_topic"]
_CONFIG = ModelDeployAgentKafkaConfig(
    bootstrap_servers="broker.example:9092", security_protocol="PLAINTEXT"
)


@pytest.fixture(autouse=True)
def _fresh_process() -> Iterator[None]:
    publisher._ENSURED_TOPICS.clear()
    yield
    publisher._ENSURED_TOPICS.clear()


class _Event(BaseModel):
    request_id: str


def _sender() -> caller.AgentDevMergeSender:
    with patch(
        "deploy_agent.kafka_config.load_deploy_agent_kafka_config_from_env",
        return_value=_CONFIG,
    ):
        return caller.AgentDevMergeSender(_TOPIC, _COMMAND_TOPIC)


def test_overdue_topic_ensured_before_first_publish_on_a_broker_without_it() -> None:
    admin, producer = MagicMock(), MagicMock()
    event = _Event(request_id="r1")
    sender = _sender()
    with (
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin),
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        sender.publish_event(event)
        sender.publish_event(event)
    created = [t.name for c in admin.create_topics.call_args_list for t in c.args[0]]
    assert created == [apply_topic_namespace(_TOPIC)]
    assert producer.send.call_count == 2
    assert producer.send.call_args.args[0] == apply_topic_namespace(_TOPIC)
