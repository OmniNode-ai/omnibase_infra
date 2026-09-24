# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The topic provisioner works in the lane's PHYSICAL topic names (OMN-19404).

Measured on the fourth pre-PR slot boot (2026-09-24, omnibase_infra#3944 at
116e920b7, epic OMN-18888). The slot's runtime ran with
``KAFKA_TOPIC_NAMESPACE=prepr1``, and its consumers subscribed to the prefixed
names the transport seam (OMN-18891) produces. The provisioner ignored the
namespace: it created and confirmed the canonical names, which on the shared
broker are the DEV lane's topics and already exist. So readiness passed, no
``prepr1.*`` topic existed, and the slot's writers failed with
``UnknownTopicOrPartitionError``.

Only the network boundary is faked. The contract is real and read by the real
extractor, and the fake broker holds a real topic set, so each test asserts on
the names that actually reached the broker.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

pytestmark = pytest.mark.unit

_TOPICS = (
    "onex.evt.omnibase-infra.namespace-probe-a.v1",
    "onex.evt.omnibase-infra.namespace-probe-b.v1",
)


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
    """A broker holding real topic names, with partition counts."""

    def __init__(self, topics: dict[str, int]) -> None:
        self.topics = dict(topics)
        self.created: list[str] = []
        self.described: list[str] = []


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
            broker.described.extend(names)
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
                broker.created.append(topic.name)

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
def contracts_root(tmp_path: Path) -> Path:
    root = tmp_path / "contracts"
    node_dir = root / "node_namespace_probe"
    node_dir.mkdir(parents=True)
    lines = [
        "name: node_namespace_probe",
        "version: 1.0.0",
        "namespace: onex.stamped",
        "event_bus:",
        "  publish_topics:",
    ]
    lines += [f"    - {name}" for name in _TOPICS]
    lines.append("published_events:")
    for index, name in enumerate(_TOPICS):
        lines += [
            f'  - topic: "{name}"',
            f'    event_type: "NamespaceProbeEvent{index}"',
            "    topic_config:",
            "      partitions: 1",
            "      replication_factor: 1",
        ]
    (node_dir / "contract.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture
def no_partition_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_partition_cap")
async def test_unset_namespace_creates_the_canonical_names(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: every declared lane's behaviour is unchanged."""
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    broker = _Broker({})
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )
    with _patched_admin(broker):
        result = await provisioner.ensure_provisioned_topics_exist()
    for name in _TOPICS:
        assert name in broker.created
        assert name in result["created"]
    assert not any(n.startswith("prepr1.") for n in broker.created)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_partition_cap")
async def test_a_namespaced_lane_creates_its_own_topics_beside_the_dev_lanes(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dev lane's canonical topics exist; the slot's must still be created."""
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr1")
    broker = _Broker(dict.fromkeys(_TOPICS, 1))
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )
    with _patched_admin(broker):
        result = await provisioner.ensure_provisioned_topics_exist()
    for name in _TOPICS:
        assert f"prepr1.{name}" in broker.created
        assert f"prepr1.{name}" in result["created"]
        assert name not in result["existing"], (
            "the dev lane's canonical topic was counted as this lane's own"
        )
    assert not any(not n.startswith("prepr1.") for n in broker.created)


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_partition_cap")
async def test_ensure_topic_exists_takes_a_canonical_name_to_the_physical_one(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr1")
    broker = _Broker(dict.fromkeys(_TOPICS, 1))
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )
    with _patched_admin(broker):
        assert await provisioner.ensure_topic_exists(topic_name=_TOPICS[0]) is True
        # Idempotent: an already-physical name is not prefixed twice.
        assert (
            await provisioner.ensure_topic_exists(topic_name=f"prepr1.{_TOPICS[0]}")
            is True
        )
    assert broker.created == [f"prepr1.{_TOPICS[0]}"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_partition_cap")
async def test_readiness_is_confirmed_on_the_physical_topic_not_the_dev_lanes(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_TOPIC_NAMESPACE", "prepr1")
    broker = _Broker(dict.fromkeys(_TOPICS, 1))
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )
    with _patched_admin(broker):
        one_shot = ModelTopicReadinessConfig(max_attempts=1)
        before = await provisioner.confirm_topics_ready([_TOPICS[0]], config=one_shot)
        assert not before.is_ready, "the dev lane's topic confirmed a slot's readiness"
        broker.topics[f"prepr1.{_TOPICS[0]}"] = 1
        after = await provisioner.confirm_topics_ready([_TOPICS[0]], config=one_shot)
    assert after.is_ready
    assert f"prepr1.{_TOPICS[0]}" in broker.described
    assert _TOPICS[0] not in broker.described
