# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The overdue event's topic exists on the broker before the sender's first record (OMN-26316).

A recording broker stands in for Kafka and logs every admin and producer call in one ordered
list. The sender is the production ``AgentDevMergeSender`` over the real ``deploy_agent``
publisher, so the order asserted here is the order a broker without the topic would see.
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

pytestmark = [pytest.mark.integration]

_SETTINGS = yaml.safe_load(
    (Path(caller.__file__).parent / "contract.yaml").read_text()
)["settings"]
_TOPIC = _SETTINGS["dev_merge_build_overdue_topic"]
_COMMAND_TOPIC = _SETTINGS["dev_merge_build_command_topic"]
_CONFIG = ModelDeployAgentKafkaConfig(
    bootstrap_servers="broker.example:9092", security_protocol="PLAINTEXT"
)


class _Event(BaseModel):
    request_id: str


@pytest.fixture(autouse=True)
def _fresh_process() -> Iterator[None]:
    publisher._ENSURED_TOPICS.clear()
    yield
    publisher._ENSURED_TOPICS.clear()


def test_topic_is_created_before_the_first_record_is_sent() -> None:
    calls: list[tuple[str, str]] = []
    admin, producer = MagicMock(), MagicMock()
    admin.create_topics.side_effect = lambda specs: calls.extend(
        ("create", spec.name) for spec in specs
    )
    producer.send.side_effect = lambda topic, **_: calls.append(("send", topic))
    with (
        patch(
            "deploy_agent.kafka_config.load_deploy_agent_kafka_config_from_env",
            return_value=_CONFIG,
        ),
        patch("deploy_agent.publisher.KafkaAdminClient", return_value=admin),
        patch("deploy_agent.publisher.KafkaProducer", return_value=producer),
    ):
        sender = caller.AgentDevMergeSender(_TOPIC, _COMMAND_TOPIC)
        sender.publish_event(_Event(request_id="r1"))
        sender.publish_event(_Event(request_id="r2"))
    topic = apply_topic_namespace(_TOPIC)
    assert calls == [("create", topic), ("send", topic), ("send", topic)]
