# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The provisioner creates each workload's hot-loading state topic compacted (OMN-21226).

Only the network boundary is faked. The spec is the one the template model declares,
the provisioner is the real ``TopicProvisioner``, and the assertion reads the
``cleanup.policy`` that reached ``CreateTopics``. A spec that declared another policy
makes the compacted test fail, and the applied topic is the control that it is not
compacted.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.models.contract_hotload.model_contract_hotload_workload_topic import (
    ModelContractHotloadWorkloadTopic,
)
from omnibase_infra.runtime.runtime_profile import _PROFILES

pytestmark = pytest.mark.unit


class _FakeTopicAlreadyExistsError(Exception):
    pass


class _NewTopic:
    def __init__(
        self,
        *,
        name: str,
        num_partitions: int,
        replication_factor: int,
        topic_configs: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.topic_configs = topic_configs


class _Broker:
    def __init__(self) -> None:
        self.topics: dict[str, int] = {}
        self.created: dict[str, dict[str, str]] = {}


@contextmanager
def _patched_admin(broker: _Broker) -> Iterator[None]:
    class _FakeAdmin:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def describe_cluster(self) -> dict[str, object]:
            return {"brokers": [{"node_id": 1}]}

        async def describe_topics(
            self, topics: Sequence[str] | None = None
        ) -> list[dict[str, object]]:
            names = list(broker.topics) if topics is None else list(topics)
            return [
                {
                    "topic": name,
                    "error_code": 0 if name in broker.topics else 3,
                    "partitions": [
                        {"partition": i, "leader": 1, "replicas": [1]}
                        for i in range(broker.topics.get(name, 0))
                    ],
                }
                for name in names
            ]

        async def create_topics(self, new_topics: Sequence[_NewTopic]) -> None:
            for topic in new_topics:
                if topic.name in broker.topics:
                    raise _FakeTopicAlreadyExistsError(topic.name)
                broker.topics[topic.name] = topic.num_partitions
                broker.created[topic.name] = dict(topic.topic_configs or {})

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(
                AIOKafkaAdminClient=_FakeAdmin, NewTopic=_NewTopic
            ),
            "aiokafka.errors": MagicMock(
                TopicAlreadyExistsError=_FakeTopicAlreadyExistsError
            ),
        },
    ):
        yield


@pytest.fixture
def empty_contracts_root(tmp_path: Path) -> Path:
    root = tmp_path / "contracts"
    root.mkdir()
    return root


@pytest.fixture(autouse=True)
def _no_partition_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", raising=False)
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("workload", sorted(_PROFILES))
async def test_the_state_topic_is_created_compacted_from_its_declaration(
    workload: str, empty_contracts_root: Path
) -> None:
    topic = ModelContractHotloadWorkloadTopic.state(workload)
    broker = _Broker()
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=empty_contracts_root
    )

    with _patched_admin(broker):
        created = await provisioner.ensure_topic_exists(
            topic_name=topic.suffix, spec=topic.spec
        )

    assert created is True
    assert broker.created[topic.suffix]["cleanup.policy"] == "compact"


@pytest.mark.asyncio
async def test_control_the_applied_topic_is_not_created_compacted(
    empty_contracts_root: Path,
) -> None:
    topic = ModelContractHotloadWorkloadTopic.applied("main")
    broker = _Broker()
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=empty_contracts_root
    )

    with _patched_admin(broker):
        assert (
            await provisioner.ensure_topic_exists(
                topic_name=topic.suffix, spec=topic.spec
            )
            is True
        )

    assert broker.created[topic.suffix]["cleanup.policy"] == "delete"
