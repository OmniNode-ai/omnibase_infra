# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The provisioner and the bus agree on the physical topic name [OMN-19404].

The OMN-18891 isolation test reads the bus's publish and subscribe seams
together. This one adds the seam it did not read, the topic provisioner, and
checks it against the other two: the name the provisioner CREATES and CONFIRMS
must be the name the consumer SUBSCRIBES to and the transport PUBLISHES on.

That agreement is what failed on the fourth pre-PR slot boot (2026-09-24,
omnibase_infra#3944 at 116e920b7). The bus subscribed to ``prepr1.<topic>``
while the provisioner created and confirmed the canonical ``<topic>``, which on
the shared broker is the dev lane's own topic and already existed. Readiness
passed, no ``prepr1.*`` topic existed, and the slot's consumers failed with
``UnknownTopicOrPartitionError``.

Only the admin client's network boundary is replaced. The contract is read by
the real extractor, and the consumer and transport are the shipped classes.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = [pytest.mark.integration]

BOOTSTRAP = "localhost:9092"
SLOT = "prepr1"
TOPIC = "onex.evt.omnibase-infra.namespace-seam-probe.v1"


class _TopicExistsError(Exception):
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


@contextmanager
def _broker(topics: dict[str, int]) -> Iterator[list[str]]:
    """A broker that holds ``topics``; yields the list of names created on it."""
    created: list[str] = []

    class _Admin:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def describe_cluster(self) -> dict[str, object]:
            return {"brokers": [{"node_id": 1}]}

        async def describe_topics(
            self, names: Sequence[str] | None = None
        ) -> list[dict[str, object]]:
            wanted = list(topics) if names is None else list(names)
            return [
                {
                    "topic": name,
                    "error_code": 0 if name in topics else 3,
                    "partitions": [
                        {"partition": i, "leader": 1, "replicas": [1]}
                        for i in range(topics.get(name, 0))
                    ],
                }
                for name in wanted
            ]

        async def create_topics(self, new_topics: Sequence[_NewTopic]) -> None:
            for topic in new_topics:
                if topic.name in topics:
                    raise _TopicExistsError(topic.name)
                topics[topic.name] = topic.num_partitions
                created.append(topic.name)

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(AIOKafkaAdminClient=_Admin, NewTopic=_NewTopic),
            "aiokafka.errors": MagicMock(TopicAlreadyExistsError=_TopicExistsError),
        },
    ):
        yield created


@pytest.fixture
def contracts_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", raising=False)
    node_dir = tmp_path / "contracts" / "node_namespace_seam_probe"
    node_dir.mkdir(parents=True)
    (node_dir / "contract.yaml").write_text(
        "name: node_namespace_seam_probe\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        f"    - {TOPIC}\n"
        "published_events:\n"
        f'  - topic: "{TOPIC}"\n'
        '    event_type: "NamespaceSeamProbeEvent"\n'
        "    topic_config:\n"
        "      partitions: 1\n"
        "      replication_factor: 1\n",
        encoding="utf-8",
    )
    return tmp_path / "contracts"


def _bus_names(topic: str) -> tuple[str, tuple[str, ...]]:
    """The subscribed and published names the shipped bus classes use."""
    bus = EventBusKafka(config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP))
    with patch(
        "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaConsumer"
    ) as consumer_cls:
        bus._build_consumer(topic, "group", "instance", "earliest")
    transport = KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers=BOOTSTRAP),
        topics=(topic,),
    )
    return consumer_cls.call_args.args[0], transport._physical_topics


@pytest.mark.asyncio
async def test_a_namespaced_runtime_provisions_the_topic_its_bus_uses(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the dev lane's canonical topic already on the broker."""
    monkeypatch.setenv(TOPIC_NAMESPACE_ENV_VAR, SLOT)
    subscribed, published = _bus_names(TOPIC)
    assert subscribed == f"{SLOT}.{TOPIC}"

    broker_topics = {TOPIC: 1}
    provisioner = TopicProvisioner(
        bootstrap_servers=BOOTSTRAP, contracts_root=contracts_root
    )
    with _broker(broker_topics) as created:
        await provisioner.ensure_provisioned_topics_exist()
        readiness = await provisioner.confirm_topics_ready(
            [TOPIC], config=ModelTopicReadinessConfig(max_attempts=1)
        )

    assert subscribed in created, "the provisioner did not create the bus's topic"
    assert published == (subscribed,)
    assert readiness.is_ready
    assert readiness.topics == (subscribed,), (
        "readiness was confirmed on a topic the bus does not use"
    )


@pytest.mark.asyncio
async def test_an_unset_namespace_provisions_the_canonical_topic_its_bus_uses(
    contracts_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Positive control: every declared lane runs with the namespace unset."""
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)
    subscribed, published = _bus_names(TOPIC)
    assert subscribed == TOPIC

    provisioner = TopicProvisioner(
        bootstrap_servers=BOOTSTRAP, contracts_root=contracts_root
    )
    with _broker({}) as created:
        await provisioner.ensure_provisioned_topics_exist()

    assert TOPIC in created
    assert published == (TOPIC,)
    assert not any(name.startswith(f"{SLOT}.") for name in created)
