# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex lease`` talks to the orchestrator over the bus and never to etcd (OMN-21424).

Each of the three commands is produced by the verb on the command topic the
orchestrator subscribes to, parses as the orchestrator's input model for that
topic, and is answered from the orchestrator's event topics. The responder here
is the orchestrator's own handler classes behind a stub store; the kernel's
wiring of the real contracts over a real etcd is covered by the integration
suites.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
import yaml
from click.testing import CliRunner

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.generated.enum_omnibase_infra_topic import (
    EnumOmnibaseInfraTopic as Topic,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.handlers.lease import lease_lane_bus
from omnibase_infra.handlers.lease.handler_lease_orchestrator import (
    HandlerLeaseAcquire,
    HandlerLeaseRelease,
    HandlerLeaseRenew,
)
from omnibase_infra.handlers.lease.lease_bus_client import (
    REPLY_TOPICS,
    LeaseReplyTimeoutError,
    request_lease,
)
from omnibase_infra.handlers.lease.lease_cli import lease_group
from omnibase_infra.handlers.lease.lease_store import LeaseStore
from omnibase_infra.models.lease import (
    ModelLeaseAcquireRequest,
    ModelLeaseReleaseRequest,
    ModelLeaseRenewRequest,
    ModelLeaseReply,
)

pytestmark = pytest.mark.unit

COMMAND_TOPICS = {
    "acquire": Topic.CMD_LEASE_ACQUIRE_REQUESTED_V1.value,
    "renew": Topic.CMD_LEASE_RENEW_REQUESTED_V1.value,
    "release": Topic.CMD_LEASE_RELEASE_REQUESTED_V1.value,
}
_REPO = Path(__file__).resolve().parents[4]
_LEASE_ID = UUID(int=7)
_OUTCOME_TOPIC = dict(
    zip(
        (
            "lease-granted",
            "lease-refused",
            "lease-renewed",
            "lease-released",
            "lease-expired",
        ),
        REPLY_TOPICS,
        strict=True,
    )
)


class _StubStore:
    """Answers like the etcd store for one held lease ``_LEASE_ID``."""

    async def acquire(self, request: ModelLeaseAcquireRequest) -> ModelLeaseReply:
        if request.kind != "exclusive":
            raise ValueError(f"lease kind {request.kind!r} is not declared")
        return ModelLeaseReply(
            outcome="lease-granted",
            kind=request.kind,
            resource=request.resource,
            holder=request.holder,
            lease_id=_LEASE_ID,
            fence=41,
            request_id=request.request_id,
            ttl_s=30,
        )

    async def renew(self, kind: str, resource: str, lease_id: UUID) -> ModelLeaseReply:
        if kind != "exclusive":
            raise ValueError(f"lease kind {kind!r} is not declared")
        return ModelLeaseReply(
            outcome="lease-renewed",
            kind=kind,
            resource=resource,
            lease_id=lease_id,
            fence=41,
            ttl_s=30,
        )

    async def release(
        self, kind: str, resource: str, lease_id: UUID
    ) -> ModelLeaseReply:
        return ModelLeaseReply(
            outcome="lease-released",
            kind=kind,
            resource=resource,
            lease_id=lease_id,
            fence=41,
        )


class _Orchestrator:
    """The orchestrator's handlers, subscribed to the command topics of a bus.

    Records every command message it consumes, parsed as the orchestrator's input
    model for the topic, and publishes the handler's reply on the reply topic of
    its outcome.
    """

    def __init__(self, bus: EventBusInmemory) -> None:
        store = cast("LeaseStore", _StubStore())
        self._bus = bus
        self._handlers: dict[str, Callable[..., Awaitable[ModelLeaseReply]]] = {
            COMMAND_TOPICS["acquire"]: HandlerLeaseAcquire(store).handle,
            COMMAND_TOPICS["renew"]: HandlerLeaseRenew(store).handle,
            COMMAND_TOPICS["release"]: HandlerLeaseRelease(store).handle,
        }
        self.consumed: list[tuple[str, object]] = []

    async def start(self) -> None:
        for topic in self._handlers:
            await self._bus.subscribe(
                topic,
                group_id="stub-lease-orchestrator",
                on_message=self._consume(topic),
            )

    def _consume(self, topic: str) -> Callable[[ModelEventMessage], object]:
        models = {
            COMMAND_TOPICS["acquire"]: ModelLeaseAcquireRequest,
            COMMAND_TOPICS["renew"]: ModelLeaseRenewRequest,
            COMMAND_TOPICS["release"]: ModelLeaseReleaseRequest,
        }

        async def consume(message: ModelEventMessage) -> None:
            envelope = ModelEventEnvelope[object].model_validate_json(message.value)
            request = models[topic].model_validate(envelope.payload)
            self.consumed.append((topic, request))
            reply = await self._handlers[topic](request)
            out = ModelEventEnvelope[object](
                payload=reply.model_dump(mode="json"),
                correlation_id=envelope.correlation_id,
                event_type=_OUTCOME_TOPIC[reply.outcome],
            )
            await self._bus.publish(
                _OUTCOME_TOPIC[reply.outcome], None, out.model_dump_json().encode()
            )

        return consume


class _Lane:
    """What the patched ``open_lane_bus`` hands the verb: one bus with an orchestrator."""

    def __init__(self) -> None:
        self.orchestrators: list[_Orchestrator] = []
        self.lanes: list[str] = []

    def opener(self) -> Callable[[str, Path | None], object]:
        @asynccontextmanager
        async def open_lane_bus(
            lane: str, omnibase_path: Path | None
        ) -> AsyncIterator[EventBusInmemory]:
            self.lanes.append(lane)
            bus = EventBusInmemory(environment="test", group="omn-21424-verb")
            await bus.start()
            orchestrator = _Orchestrator(bus)
            await orchestrator.start()
            self.orchestrators.append(orchestrator)
            try:
                yield bus
            finally:
                await bus.close()

        return open_lane_bus


@pytest.fixture
def lane(monkeypatch: pytest.MonkeyPatch) -> _Lane:
    state = _Lane()
    monkeypatch.setattr(lease_lane_bus, "open_lane_bus", state.opener())
    return state


def _verb(*args: str) -> tuple[int, dict[str, object]]:
    result = CliRunner().invoke(
        lease_group, ["--lane", "dev", *args], catch_exceptions=False
    )
    lines = result.stdout.strip().splitlines()
    return result.exit_code, json.loads(lines[-1]) if lines else {}


def test_acquire_is_produced_on_its_command_topic_and_answered(lane: _Lane) -> None:
    request_id = uuid4()
    code, reply = _verb(
        "acquire",
        "--kind",
        "exclusive",
        "--resource",
        "pr:verb",
        "--holder",
        "lane-a",
        "--request-id",
        str(request_id),
    )
    assert code == 0
    assert (reply["outcome"], reply["fence"], reply["lease_id"]) == (
        "lease-granted",
        41,
        str(_LEASE_ID),
    )
    assert lane.lanes == ["dev"]
    (consumed,) = lane.orchestrators[0].consumed
    assert consumed[0] == "onex.cmd.omnibase-infra.lease-acquire-requested.v1"
    assert consumed[1] == ModelLeaseAcquireRequest(
        kind="exclusive", resource="pr:verb", holder="lane-a", request_id=request_id
    )


def test_renew_is_produced_on_its_command_topic_and_answered(lane: _Lane) -> None:
    code, reply = _verb(
        "renew",
        "--kind",
        "exclusive",
        "--resource",
        "pr:verb",
        "--lease-id",
        str(_LEASE_ID),
    )
    assert code == 0
    assert (reply["outcome"], reply["fence"]) == ("lease-renewed", 41)
    (consumed,) = lane.orchestrators[0].consumed
    assert consumed[0] == "onex.cmd.omnibase-infra.lease-renew-requested.v1"
    assert consumed[1] == ModelLeaseRenewRequest(
        kind="exclusive", resource="pr:verb", lease_id=_LEASE_ID
    )


def test_release_is_produced_on_its_command_topic_and_answered(lane: _Lane) -> None:
    code, reply = _verb(
        "release",
        "--kind",
        "exclusive",
        "--resource",
        "pr:verb",
        "--lease-id",
        str(_LEASE_ID),
    )
    assert code == 0
    assert reply["outcome"] == "lease-released"
    (consumed,) = lane.orchestrators[0].consumed
    assert consumed[0] == "onex.cmd.omnibase-infra.lease-release-requested.v1"
    assert consumed[1] == ModelLeaseReleaseRequest(
        kind="exclusive", resource="pr:verb", lease_id=_LEASE_ID
    )


def test_a_renew_of_an_undeclared_kind_is_answered_with_a_refusal_not_silence(
    lane: _Lane,
) -> None:
    code, reply = _verb(
        "renew", "--kind", "nope", "--resource", "r", "--lease-id", str(_LEASE_ID)
    )
    assert code == 3
    assert reply["outcome"] == "lease-refused"
    assert str(reply["lease_id"]) == str(_LEASE_ID)
    assert "nope" in str(reply["reason"])


def test_a_refused_acquire_exits_3_with_the_orchestrators_reason(lane: _Lane) -> None:
    code, reply = _verb("acquire", "--kind", "nope", "--resource", "r", "--holder", "h")
    assert code == 3
    assert reply["outcome"] == "lease-refused"
    assert "nope" in str(reply["reason"])


def test_a_command_without_a_lane_is_refused_before_any_bus_is_opened(
    lane: _Lane,
) -> None:
    result = CliRunner().invoke(
        lease_group,
        ["acquire", "--kind", "exclusive", "--resource", "r", "--holder", "h"],
        env={"ONEX_LEASE_LANE": ""},
    )
    assert result.exit_code == 2, result.output
    assert "ONEX_LEASE_LANE" in result.output
    assert lane.orchestrators == []


async def test_an_unrelated_reply_is_not_taken_for_the_answer() -> None:
    bus = EventBusInmemory(environment="test", group="omn-21424-unrelated")
    await bus.start()
    try:
        orchestrator = _Orchestrator(bus)
        await orchestrator.start()
        mine = ModelLeaseAcquireRequest(
            kind="exclusive", resource="r", holder="me", request_id=uuid4()
        )
        stray = ModelEventEnvelope[object](
            payload=ModelLeaseReply(
                outcome="lease-granted",
                kind="exclusive",
                resource="r",
                holder="someone-else",
                lease_id=uuid4(),
                fence=1,
                request_id=uuid4(),
            ).model_dump(mode="json"),
            correlation_id=uuid4(),
            event_type=REPLY_TOPICS[0],
        )
        await bus.publish(REPLY_TOPICS[0], None, stray.model_dump_json().encode())
        reply = await request_lease(bus, mine, timeout_s=5)
        assert reply.request_id == mine.request_id
        assert reply.holder == "me"
    finally:
        await bus.close()


async def test_no_orchestrator_is_a_named_timeout() -> None:
    bus = EventBusInmemory(environment="test", group="omn-21424-silent")
    await bus.start()
    try:
        with pytest.raises(LeaseReplyTimeoutError, match="node_lease_orchestrator"):
            await request_lease(
                bus,
                ModelLeaseRenewRequest(kind="k", resource="r", lease_id=uuid4()),
                timeout_s=0.2,
            )
    finally:
        await bus.close()


def test_the_verbs_topics_are_the_orchestrators_declared_topics() -> None:
    contract = yaml.safe_load(
        (
            _REPO / "src/omnibase_infra/nodes/node_lease_orchestrator/contract.yaml"
        ).read_text()
    )
    bus = contract["event_bus"]
    assert set(COMMAND_TOPICS.values()) == set(bus["subscribe_topics"])
    assert set(REPLY_TOPICS) == set(bus["publish_topics"])


def test_the_verb_is_declared_the_publisher_of_each_command_topic() -> None:
    declared = yaml.safe_load(
        (_REPO / "src/omnibase_infra/cli/topics.yaml").read_text()
    )["topics"]
    assert set(COMMAND_TOPICS.values()) <= set(declared)


def test_the_verb_imports_no_part_of_the_etcd_write_path() -> None:
    """Only ``show`` reads the store; no command verb opens an etcd client."""
    source = (_REPO / "src/omnibase_infra/handlers/lease/lease_cli.py").read_text()
    commands = source.split("def _command(")[1].split("@lease_group.command()")[0]
    assert "LeaseEtcdGateway" not in commands
    assert "LeaseStore" not in commands


def test_a_lane_the_workspace_does_not_declare_is_refused_by_name(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        lease_group,
        [
            "--lane",
            "no-such-lane",
            "--omnibase-path",
            str(tmp_path),
            "acquire",
            "--kind",
            "exclusive",
            "--resource",
            "r",
            "--holder",
            "h",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "no-such-lane" in result.output
