# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A topic the broker refuses for its partition ceiling is not a created topic (OMN-17427).

The planted fixture is the lab host on 2026-10-30. The shared dev broker held
4827 partitions (the dev lane 2389, an abandoned pre-PR slot ``prepr2`` 2262,
tenants and internals 176) under a ceiling of 7000. A slot boot asked it for
2267 topics, the broker answered the 93 past the ceiling with
``INVALID_PARTITIONS`` ("Can not increase partition count due to core limit",
broker log ``topic_invalid_partitions_core_limit``), and ``aiokafka`` 0.13 handed
those answers back as ``response.topic_errors`` without raising. The provisioner
read none of them, recorded 2267 topics as created, and the runtime booted with
2174: its consumers reported ``topic_absent`` (``error_code=3``), no required
contract attached, and the boot script waited 2400 s.

The fake admin below behaves the way the real client does: it returns a
``CreateTopicsResponse``-shaped object carrying ``topic_errors`` and raises
nothing, which is the one property the earlier fakes (they raised the driver
error) did not model and the reason their ``except`` branches looked reachable.
Each failure case has its positive control beside it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiokafka.admin import NewTopic

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.models.event_bus.model_create_topic_refusal import (
    ModelCreateTopicRefusal,
)

pytestmark = [pytest.mark.unit]

CORE_LIMIT_MESSAGE = "Can not increase partition count due to core limit"
INVALID_PARTITIONS = 37
TOPIC_ALREADY_EXISTS = 36
INVALID_REPLICATION_FACTOR = 38
REQUEST_TIMED_OUT = 7

PROD_TOPIC = "onex.evt.test-producer.example-event.v1"  # onex-topic-allow: unit fixture


class _TopicAlreadyExistsError(Exception):
    """Stand-in for ``aiokafka.errors.TopicAlreadyExistsError``."""


@dataclass
class _Broker:
    """A broker that refuses past a partition ceiling the way Redpanda does."""

    #: ``namespace -> partitions`` the broker holds before the pass starts.
    held: dict[str, int] = field(default_factory=dict)
    #: Partitions the broker will hold in total. ``None`` is unbounded.
    ceiling: int | None = None
    #: Per-topic error to answer for every topic, overriding the ceiling.
    answer: tuple[int, str | None] | None = None
    created: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    def existing(self) -> list[str]:
        return [
            f"{ns}.onex.evt.fixture.t{i}.v1"
            for ns, count in self.held.items()
            for i in range(count)
        ]

    def used(self) -> int:
        return sum(self.held.values()) + len(self.created)

    def respond(self, topic: str) -> SimpleNamespace:
        errors: list[tuple[str, int, str | None]]
        if self.answer is not None:
            errors = [(topic, *self.answer)]
        elif self.ceiling is not None and self.used() + 1 > self.ceiling:
            self.refused.append(topic)
            errors = [(topic, INVALID_PARTITIONS, CORE_LIMIT_MESSAGE)]
        else:
            self.created.append(topic)
            errors = [(topic, 0, None)]
        return SimpleNamespace(throttle_time_ms=0, topic_errors=errors)


@contextmanager
def _patched_admin(broker: _Broker) -> Iterator[None]:
    class _FakeAdmin:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def describe_cluster(self) -> dict[str, object]:
            return {"brokers": [{"node_id": 1, "host": "b", "port": 9092}]}

        async def describe_topics(
            self, topics: Sequence[str] | None = None
        ) -> list[dict[str, object]]:
            return [
                {
                    "topic": name,
                    "error_code": 0,
                    "partitions": [{"partition": 0, "leader": 1, "replicas": [1]}],
                }
                for name in broker.existing()
            ]

        async def create_topics(
            self, new_topics: Sequence[NewTopic]
        ) -> SimpleNamespace:
            # Returns the response and raises nothing: the real client's shape.
            return broker.respond(new_topics[0].name)

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(
                AIOKafkaAdminClient=_FakeAdmin, NewTopic=NewTopic
            ),
            "aiokafka.errors": MagicMock(
                TopicAlreadyExistsError=_TopicAlreadyExistsError
            ),
        },
    ):
        yield


def _contract(root: Path, topics: int) -> Path:
    node_dir = root / "node_example"
    node_dir.mkdir(exist_ok=True)
    lines = [
        "name: node_example",
        "version: 1.0.0",
        "namespace: onex.stamped",
        "event_bus:",
        "  publish_topics:",
        *[f"    - onex.evt.fixture.event-{i}.v1" for i in range(topics)],
        "published_events:",
    ]
    for i in range(topics):
        lines += [
            f'  - topic: "onex.evt.fixture.event-{i}.v1"',
            '    event_type: "ExampleEvent"',
            "    topic_config:",
            "      partitions: 1",
            "      replication_factor: 1",
        ]
    (node_dir / "contract.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
    monkeypatch.delenv("KAFKA_SECURITY_PROTOCOL", raising=False)
    monkeypatch.delenv("KAFKA_SASL_MECHANISM", raising=False)


def _provisioner(root: Path) -> TopicProvisioner:
    return TopicProvisioner(bootstrap_servers="broker:9092", contracts_root=root)


async def test_a_broker_at_its_ceiling_fails_the_pass_and_names_the_holder(
    tmp_path: Path,
) -> None:
    """The incident: refused topics must not be recorded as created."""
    broker = _Broker(held={"onex": 40, "prepr2": 30}, ceiling=70 + 5)
    with _patched_admin(broker):
        with pytest.raises(TopicReplicationPolicyError) as excinfo:
            await _provisioner(_contract(tmp_path, 8)).ensure_provisioned_topics_exist()

    message = str(excinfo.value)
    assert len(broker.created) == 5  # only what fitted under the ceiling
    assert "partition ceiling" in message
    assert "5 created this pass" in message
    assert "onex=40, prepr2=30" in message  # the abandoned slot is named, largest first
    assert "prepr_teardown_slot.py reap" in message
    # Fail-closed rides the existing path: a TopicReplicationPolicyError is
    # re-raised past every best-effort boundary in the runtime.


async def test_positive_control_a_broker_with_room_creates_every_topic(
    tmp_path: Path,
) -> None:
    broker = _Broker(held={"onex": 40}, ceiling=10_000)
    with _patched_admin(broker):
        result = await _provisioner(
            _contract(tmp_path, 8)
        ).ensure_provisioned_topics_exist()

    assert result["status"] == "success"
    assert broker.refused == []
    assert result["failed"] == []
    assert len(result["created"]) == len(broker.created) >= 8


async def test_a_refused_topic_is_never_reported_as_created(tmp_path: Path) -> None:
    """Whatever the broker refuses with is not in ``created``."""
    broker = _Broker(answer=(REQUEST_TIMED_OUT, "timed out"))
    with _patched_admin(broker):
        result = await _provisioner(
            _contract(tmp_path, 3)
        ).ensure_provisioned_topics_exist()

    assert result["created"] == []
    assert len(result["failed"]) >= 3
    assert result["status"] == "unavailable"


async def test_an_already_exists_answer_is_existing_not_created(
    tmp_path: Path,
) -> None:
    broker = _Broker(answer=(TOPIC_ALREADY_EXISTS, "topic already exists"))
    with _patched_admin(broker):
        result = await _provisioner(
            _contract(tmp_path, 3)
        ).ensure_provisioned_topics_exist()

    assert result["created"] == []
    assert result["failed"] == []
    assert "onex.evt.fixture.event-0.v1" in result["existing"]


async def test_an_unhostable_replication_factor_answer_is_now_fail_closed(
    tmp_path: Path,
) -> None:
    """The OMN-15395 D5 branch was unreachable against the real client."""
    broker = _Broker(answer=(INVALID_REPLICATION_FACTOR, "not enough brokers"))
    with _patched_admin(broker):
        with pytest.raises(TopicReplicationPolicyError):
            await _provisioner(_contract(tmp_path, 3)).ensure_provisioned_topics_exist()


async def test_invalid_partitions_that_is_not_a_limit_degrades_not_aborts(
    tmp_path: Path,
) -> None:
    """The code alone is not a ceiling; the broker's message names one."""
    broker = _Broker(answer=(INVALID_PARTITIONS, "Number of partitions must be > 0"))
    with _patched_admin(broker):
        result = await _provisioner(
            _contract(tmp_path, 3)
        ).ensure_provisioned_topics_exist()

    assert result["created"] == []
    assert len(result["failed"]) >= 3


def test_the_response_reader_handles_the_shapes_it_will_meet() -> None:
    refused = SimpleNamespace(
        topic_errors=[("t", INVALID_PARTITIONS, CORE_LIMIT_MESSAGE)]
    )
    refusal = ModelCreateTopicRefusal.from_response(refused, "t")
    assert refusal is not None
    assert refusal.errno == INVALID_PARTITIONS
    assert refusal.is_partition_ceiling

    ok = SimpleNamespace(topic_errors=[("t", 0, None)])
    assert ModelCreateTopicRefusal.from_response(ok, "t") is None
    # another topic's answer is not this topic's refusal
    assert ModelCreateTopicRefusal.from_response(refused, "another-topic") is None
    assert ModelCreateTopicRefusal.from_response(None, "t") is None  # a fake
    bad = SimpleNamespace(topic_errors="oops")
    assert ModelCreateTopicRefusal.from_response(bad, "t") is None
    short = SimpleNamespace(topic_errors=[("t", 5)])
    assert ModelCreateTopicRefusal.from_response(short, "t") is not None
    # the code alone is not a ceiling: the message names the limit
    not_a_limit = ModelCreateTopicRefusal(topic="t", errno=INVALID_PARTITIONS)
    assert not not_a_limit.is_partition_ceiling
    assert not ModelCreateTopicRefusal(
        topic="t", errno=7, broker_message="limit"
    ).is_partition_ceiling
