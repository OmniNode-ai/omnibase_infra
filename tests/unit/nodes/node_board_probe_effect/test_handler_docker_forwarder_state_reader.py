# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19930: the docker reader turns a lab forwarder's state into an observation.

The docker outputs below are recordings: ``docker inspect`` state and
``docker logs`` lines as the .201 lab's ``omninode-gateway-forwarder`` printed
them on 2026-09-28, where the dev cloud broker refused the webhook-delivery
topic. The last test pins the parser to the real emitter: it drives
``KafkaTransport`` into a refusal, formats the record with the forwarder's own
log format, and parses it back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from aiokafka.errors import TopicAuthorizationFailedError

from omnibase_infra.event_bus import kafka_transport
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_forwarder_state_reader import (
    HandlerDockerForwarderStateReader,
    parse_refused_topics,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    ModelForwarderRefusedTopicRequest,
)
from omnibase_infra.topics.topic_namespace import TOPIC_NAMESPACE_ENV_VAR

pytestmark = [pytest.mark.unit]

SLUG = "beta-gateway-canary-79afa7263852"
WEBHOOK = f"tenant-{SLUG}.onex.cmd.github.webhook-delivery.v1"
DECLARED = "b-1.dev-cloud.example:9098"
NOW = datetime(2026, 9, 28, 16, 10, tzinfo=UTC)

REFUSED_LOG = f"""\
2026-09-28 16:01:03,223 WARNING omnibase_infra.event_bus.kafka_transport kafka_transport_topic_refused topic={WEBHOOK} group=tenant-{SLUG}-gateway-forwarder-inbound error=TopicAuthorizationFailedError retry_in_seconds=300.0
2026-09-28 16:06:14,919 WARNING omnibase_infra.event_bus.kafka_transport kafka_transport_topic_refused topic={WEBHOOK} group=tenant-{SLUG}-gateway-forwarder-inbound error=TopicAuthorizationFailedError retry_in_seconds=300.0
"""
ADMITTED_LOG = (
    REFUSED_LOG
    + f"2026-09-28 16:08:00,000 INFO omnibase_infra.event_bus.kafka_transport kafka_transport_topic_admitted topic={WEBHOOK} group=tenant-{SLUG}-gateway-forwarder-inbound\n"
)
STATE = {"Running": True, "StartedAt": "2026-09-28T13:50:23.846688704Z"}
REF_MAP = f"gateway.cloud.kafka.broker: {DECLARED}\n"


def _request() -> ModelForwarderRefusedTopicRequest:
    return ModelForwarderRefusedTopicRequest(
        subject_lane="dev",
        forwarder_container="omninode-gateway-forwarder",
        declared_cloud_broker=DECLARED,
        cloud_broker_ref="gateway.cloud.kafka.broker",
        retry_interval_seconds=300,
    )


class _Runner:
    """Answers the three docker calls from recordings, and logs every argv."""

    def __init__(
        self,
        *,
        state: object = STATE,
        logs: str = REFUSED_LOG,
        ref_map: str = REF_MAP,
        inspect_rc: int = 0,
    ) -> None:
        self.state = state
        self.logs = logs
        self.ref_map = ref_map
        self.inspect_rc = inspect_rc
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: Sequence[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        if args[1] == "inspect":
            if self.inspect_rc:
                return subprocess.CompletedProcess(
                    args, self.inspect_rc, "", "Error: No such object"
                )
            return subprocess.CompletedProcess(args, 0, json.dumps(self.state), "")
        if args[1] == "logs":
            # The forwarder logs to stderr; docker logs replays it on stderr.
            return subprocess.CompletedProcess(args, 0, "", self.logs)
        if args[1] == "exec":
            return subprocess.CompletedProcess(args, 0, self.ref_map, "")
        raise AssertionError(f"unexpected docker call {args}")


def _observe(runner: _Runner):  # type: ignore[no-untyped-def]
    reader = HandlerDockerForwarderStateReader(runner=runner, now=lambda: NOW)
    return asyncio.run(reader.observe(_request()))


def test_a_refused_topic_in_the_window_is_observed() -> None:
    runner = _Runner()
    observation = _observe(runner)
    assert observation.read_ok
    assert observation.running
    assert observation.refused_topics == (WEBHOOK,)
    assert observation.observed_cloud_broker == DECLARED
    assert observation.started_at == datetime(2026, 9, 28, 13, 50, 23, tzinfo=UTC)
    # The log window covers one retry interval plus the margin, and nothing
    # older: a refusal that was admitted long ago must not count.
    logs_call = next(call for call in runner.calls if call[1] == "logs")
    assert logs_call[logs_call.index("--since") + 1] == f"{observation.window_seconds}s"
    assert observation.window_seconds >= 300


def test_a_topic_admitted_after_its_refusal_is_not_refused() -> None:
    assert _observe(_Runner(logs=ADMITTED_LOG)).refused_topics == ()


def test_a_stopped_container_is_observed_as_not_running() -> None:
    observation = _observe(
        _Runner(state={"Running": False, "StartedAt": STATE["StartedAt"]})
    )
    assert observation.read_ok
    assert not observation.running


def test_a_zone_less_start_time_is_read_as_utc() -> None:
    state = {"Running": True, "StartedAt": "2026-09-28T13:50:23.846688704"}
    observation = _observe(_Runner(state=state))
    assert observation.started_at == datetime(2026, 9, 28, 13, 50, 23, tzinfo=UTC)


def test_a_missing_container_is_an_unreadable_observation() -> None:
    observation = _observe(_Runner(inspect_rc=1))
    assert not observation.read_ok
    assert "No such object" in observation.read_error


def test_a_ref_map_without_the_ref_leaves_the_broker_unknown() -> None:
    assert _observe(_Runner(ref_map="other.ref: x:1\n")).observed_cloud_broker == ""


def test_the_parser_reads_the_real_emitters_refusal_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Two-sided pin: KafkaTransport's own WARNING, parsed by the reader."""
    monkeypatch.delenv(TOPIC_NAMESPACE_ENV_VAR, raising=False)

    class _Cluster:
        unauthorized_topics = {WEBHOOK}

        @staticmethod
        def partitions_for_topic(topic: str) -> set[int] | None:
            return None if topic == WEBHOOK else {0}

    class _Consumer:
        def __init__(self, *topics: str, **_kwargs: object) -> None:
            self.topics = topics
            self._client = type("C", (), {"cluster": _Cluster()})()

        async def start(self) -> None:
            if WEBHOOK in self.topics:
                raise TopicAuthorizationFailedError(WEBHOOK)

        async def stop(self) -> None:
            return None

        async def getmany(self, **_kwargs: object) -> dict[object, list[object]]:
            return {}

        def assignment(self) -> set[object]:
            return set()

    class _Producer:
        def __init__(self, **_kwargs: object) -> None:
            return None

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(kafka_transport, "AIOKafkaConsumer", _Consumer)
    monkeypatch.setattr(kafka_transport, "AIOKafkaProducer", _Producer)
    monkeypatch.setattr(KafkaTransport, "_prime", lambda self, consumer: _noop())

    transport = KafkaTransport(
        config=ModelKafkaEventBusConfig(bootstrap_servers="localhost:9092"),
        group="g",
        topics=(
            WEBHOOK,
            f"tenant-{SLUG}.onex.cmd.omnibase-infra.delegation-request.v1",
        ),
        refused_topic_retry_seconds=300.0,
    )
    with caplog.at_level(logging.WARNING, logger=kafka_transport.logger.name):
        asyncio.run(transport.start())
    # The forwarder's entrypoint logs with this exact format (gateway_forwarder.main).
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    text = "\n".join(formatter.format(record) for record in caplog.records)
    assert parse_refused_topics(text) == (WEBHOOK,)
    assert transport.refused_topics == frozenset({WEBHOOK})


async def _noop() -> None:
    return None
