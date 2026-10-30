# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract discovery through aiokafka's returned responses and readiness.

Only the admin client's network boundaries are replaced. The real client builds
CreateTopics requests and returns decoded Kafka protocol responses without
raising their per-topic errors. No Docker daemon or live lane is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiokafka.admin import AIOKafkaAdminClient
from aiokafka.protocol.admin import (
    CreateTopicsRequest,
    CreateTopicsRequest_v1,
    CreateTopicsResponse_v1,
)
from aiokafka.protocol.metadata import MetadataResponse_v0

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

TOPICS = tuple(f"onex.evt.omnibase-infra.refusal-probe-{i}.v1" for i in range(3))
PHYSICAL = tuple(f"prepr1.{topic}" for topic in TOPICS)
CORE_LIMIT = "Can not increase partition count due to core limit"


def _provisioner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TopicProvisioner:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr1")
    monkeypatch.setenv("ONEX_RUNTIME_NODE_SELECTION", "node_refusal_probe")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092")
    for name in (
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS",
        "ONEX_RUNTIME_NODE_EXCLUSION",
    ):
        monkeypatch.delenv(name, raising=False)
    node = tmp_path / "node_refusal_probe"
    node.mkdir()
    (node / "contract.yaml").write_text(
        "name: node_refusal_probe\nversion: 1.0.0\nnamespace: onex.stamped\n"
        "event_bus:\n  publish_topics:\n"
        + "".join(f"    - {topic}\n" for topic in TOPICS)
        + "published_events:\n"
        + "".join(
            f"  - topic: {topic}\n    event_type: RefusalProbeEvent\n"
            "    topic_config:\n      partitions: 1\n      replication_factor: 1\n"
            for topic in TOPICS
        ),
        encoding="utf-8",
    )
    return TopicProvisioner(
        bootstrap_servers="localhost:19092", contracts_root=tmp_path
    )


def _network(
    monkeypatch: pytest.MonkeyPatch,
    topics: dict[str, int],
    *,
    ceiling: int | None = None,
    answer: tuple[int, str] | None = None,
) -> tuple[list[str], AsyncMock]:
    """Return wire-level broker answers; leave the real admin methods intact."""
    requested: list[str] = []

    async def metadata(_admin: AIOKafkaAdminClient, _request: object) -> object:
        response = MetadataResponse_v0(
            brokers=[(1, "localhost", 19092)],
            topics=[
                (
                    0,
                    topic,
                    [(0, i, 1, [1], [1]) for i in range(partitions)],
                )
                for topic, partitions in topics.items()
            ],
        )
        return MetadataResponse_v0.decode(response.encode())

    async def create(
        _admin: AIOKafkaAdminClient, request: CreateTopicsRequest
    ) -> object:
        wire = request.build(CreateTopicsRequest_v1)
        name, partitions, replicas, _assignments, _configs = wire.create_topic_requests[
            0
        ]
        requested.append(name)
        assert partitions == replicas == 1
        if answer is not None:
            errno, message = answer
        elif ceiling is not None and sum(topics.values()) + partitions > ceiling:
            errno, message = 37, CORE_LIMIT
        else:
            errno, message = 0, ""
        if errno in (0, 36):
            topics[name] = partitions
        response = CreateTopicsResponse_v1(topic_errors=[(name, errno, message)])
        return CreateTopicsResponse_v1.decode(response.encode())

    # Close the real client as well as observing cleanup on an aborted pass.
    original_close = AIOKafkaAdminClient.close

    async def close(admin: AIOKafkaAdminClient) -> None:
        await closed(admin)

    closed = AsyncMock(side_effect=original_close)
    monkeypatch.setattr(AIOKafkaAdminClient, "start", AsyncMock())
    monkeypatch.setattr(AIOKafkaAdminClient, "close", close)
    monkeypatch.setattr(AIOKafkaAdminClient, "_send_request", metadata)
    monkeypatch.setattr(AIOKafkaAdminClient, "_send_to_controller", create)
    return requested, closed


async def test_ceiling_aborts_without_caching_refused_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provisioner = _provisioner(tmp_path, monkeypatch)
    topics = {"onex.evt.omnibase-infra.held.v1": 4, "prepr2.held": 3}
    requested, closed = _network(monkeypatch, topics, ceiling=8)

    with pytest.raises(TopicReplicationPolicyError) as caught:
        await provisioner.ensure_provisioned_topics_exist()

    assert requested == list(PHYSICAL[:2]), "stop at the first capacity refusal"
    assert PHYSICAL[0] in topics
    assert PHYSICAL[1] not in topics
    assert set(provisioner._created_specs) == {PHYSICAL[0]}
    assert provisioner._existing_topics is not None
    assert PHYSICAL[1] not in provisioner._existing_topics
    assert "1 created this pass" in str(caught.value)
    assert "onex=4, prepr2=3" in str(caught.value)
    assert CORE_LIMIT in str(caught.value)
    closed.assert_awaited_once()
    readiness = await provisioner.confirm_topics_ready(
        TOPICS, config=ModelTopicReadinessConfig(max_attempts=1)
    )
    assert not readiness.is_ready
    assert readiness.ready_topics == PHYSICAL[:1]


async def test_capacity_control_creates_confirms_and_reuses_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provisioner = _provisioner(tmp_path, monkeypatch)
    topics: dict[str, int] = {}
    requested, _closed = _network(monkeypatch, topics, ceiling=3)
    result = await provisioner.ensure_provisioned_topics_exist()
    assert result["status"] == "success"
    assert result["created"] == list(PHYSICAL)
    readiness = await provisioner.confirm_topics_ready(
        TOPICS, config=ModelTopicReadinessConfig(max_attempts=1)
    )
    assert readiness.is_ready
    assert readiness.ready_topics == PHYSICAL
    second = await provisioner.ensure_provisioned_topics_exist()
    assert second["created"] == []
    assert second["existing"] == list(PHYSICAL)
    assert requested == list(PHYSICAL), "the second pass sends no CreateTopics"


@pytest.mark.parametrize(
    ("errno", "message"),
    [(36, "Topic already exists"), (38, "Invalid replication factor"), (7, "Timeout")],
)
async def test_returned_wire_errors_reach_the_existing_classifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, errno: int, message: str
) -> None:
    provisioner = _provisioner(tmp_path, monkeypatch)
    requested, closed = _network(monkeypatch, {}, answer=(errno, message))
    if errno == 38:
        with pytest.raises(TopicReplicationPolicyError):
            await provisioner.ensure_provisioned_topics_exist()
        assert requested == list(PHYSICAL[:1])
    else:
        result = await provisioner.ensure_provisioned_topics_exist()
        assert result["created"] == []
        assert result["existing"] == (list(PHYSICAL) if errno == 36 else [])
        assert result["failed"] == (list(PHYSICAL) if errno == 7 else [])
        assert result["status"] == ("success" if errno == 36 else "unavailable")
    assert not provisioner._created_specs
    closed.assert_awaited_once()
