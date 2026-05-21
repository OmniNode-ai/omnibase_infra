# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for Pattern B one-shot terminal waits."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import ClassVar

import pytest

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.runtime_local_ingress import RuntimeLocalIngressRoute
from omnibase_infra.runtime.service_pattern_b_broker import RuntimePatternBBroker

pytestmark = [pytest.mark.integration]


def _route() -> RuntimeLocalIngressRoute:
    return RuntimeLocalIngressRoute(
        node_name="node_delegate_skill_orchestrator",
        contract_name="delegate_skill.orchestrate",
        command_topic="onex.cmd.omnimarket.delegate-skill.v1",
        event_type="omnimarket.delegate-skill",
        terminal_event="onex.evt.omnimarket.delegate-skill-completed.v1",
        terminal_events=("onex.evt.omnimarket.delegate-skill-completed.v1",),
        contract_path="/tmp/node_delegate_skill_orchestrator/contract.yaml",  # noqa: S108
        package_name="omnimarket",
    )


class _TerminalConsumer:
    created: ClassVar[list[_TerminalConsumer]] = []

    def __init__(self, *topics: object, **kwargs: object) -> None:
        self.topics = topics
        self.kwargs = kwargs
        self.messages: asyncio.Queue[SimpleNamespace] = asyncio.Queue()
        self.assigned_partitions: set[object] = set()
        self.seeked_to_end = False
        self.started = False
        self.stopped = False
        type(self).created.append(self)

    async def start(self) -> None:
        self.started = True

    def partitions_for_topic(self, topic: str) -> set[int]:
        return {0, 1} if self.started and topic else set()

    def assign(self, partitions: set[object]) -> None:
        self.assigned_partitions = partitions

    def assignment(self) -> set[object]:
        return self.assigned_partitions

    async def seek_to_end(self, *_assignment: object) -> None:
        self.seeked_to_end = True

    async def getone(self) -> SimpleNamespace:
        return await self.messages.get()

    async def stop(self) -> None:
        self.stopped = True


class _KafkaLikeTransport:
    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "redpanda:9092"

    def __init__(self, route: RuntimeLocalIngressRoute) -> None:
        self._route = route

    def _build_auth_kwargs(self) -> dict[str, object]:
        return {}

    async def publish(
        self,
        _topic: str,
        _key: bytes | None,
        value: bytes,
        _headers: object | None = None,
    ) -> None:
        consumer = _TerminalConsumer.created[-1]
        assert consumer.kwargs["group_id"] is None
        assert consumer.topics == ()
        assert consumer.assigned_partitions
        assert consumer.seeked_to_end is True

        command_envelope = ModelEventEnvelope[object].model_validate_json(value)
        terminal_envelope = ModelEventEnvelope[object](
            payload={"status": "completed", "response": "stability-smoke-ok"},
            correlation_id=command_envelope.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=self._route.terminal_event,
            source_tool="node_delegate_skill_orchestrator",
        )
        await consumer.messages.put(
            SimpleNamespace(
                topic=self._route.terminal_event,
                value=terminal_envelope.model_dump_json().encode(),
            )
        )


@pytest.mark.asyncio
async def test_pattern_b_terminal_waiter_uses_ungrouped_assigned_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _TerminalConsumer.created = []
    monkeypatch.setattr(
        "omnibase_infra.runtime.service_pattern_b_broker.AIOKafkaConsumer",
        _TerminalConsumer,
    )
    route = _route()
    broker = RuntimePatternBBroker(
        _KafkaLikeTransport(route),
        command_topic="onex.cmd.omnibase-infra.pattern-b-dispatch.v1",
        routes={"delegate_skill.orchestrate": route},
    )
    command = ModelDispatchBusCommand(
        command_name="delegate_skill.orchestrate",
        requester="codex",
        payload={"prompt": "stability smoke"},
        response_topic="onex.evt.pattern-b.dispatch-completed.v1",
        timeout_seconds=1,
    )

    resolved_route, result = await broker.dispatch_request(command)

    assert resolved_route == route
    assert result.status == "completed"
    assert result.payload == {"status": "completed", "response": "stability-smoke-ok"}
    assert _TerminalConsumer.created[0].stopped is True
