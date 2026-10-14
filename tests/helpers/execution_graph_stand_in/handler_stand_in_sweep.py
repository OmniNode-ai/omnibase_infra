# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The stand-in sweep's orchestrator handler: every stage is a bus hop (OMN-21071).

It reads its stages from its own contract's ``execution_graph``, and each
stage's command and terminal topics from that stage node's contract, through
the same registry ``onex node`` resolves names with. It subscribes to every
stage terminal before it publishes a command, keys each answer by the sweep's
correlation id and the graph node id, ignores any other answer, and returns a
typed failure naming the stage whose terminal did not arrive by the request's
deadline. It holds no consumer of its own stages: whatever answers them is
whatever the runtime hosts.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import ClassVar, Protocol, cast

from omnibase_infra.cli import contract_registry
from tests.helpers.execution_graph_stand_in.model_stand_in_sweep import (
    ModelStandInStageRequest,
    ModelStandInSweepRequest,
    ModelStandInSweepResult,
)


class _Message(Protocol):
    value: bytes


class _Bus(Protocol):
    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None: ...

    async def subscribe(
        self,
        topic: str,
        node_identity: None = None,
        on_message: Callable[[_Message], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
    ) -> Callable[[], Awaitable[None]]: ...


def _mapping(value: object) -> dict[str, object]:
    return cast("dict[str, object]", value) if isinstance(value, dict) else {}


def _stages(orchestrator: str) -> list[tuple[str, str, str]]:
    """``(node_id, command_topic, terminal_topic)`` of each graph node, in order."""
    graph = _mapping(
        _mapping(
            _mapping(
                contract_registry.node_contract(orchestrator).get(
                    "workflow_coordination"
                )
            ).get("workflow_definition")
        ).get("execution_graph")
    )
    stages: list[tuple[str, str, str]] = []
    for raw in cast("list[object]", graph.get("nodes") or []):
        node = _mapping(raw)
        stage_node = str(_mapping(node.get("step_config")).get("node"))
        stage_contract = contract_registry.node_contract(stage_node)
        command = str(_mapping(stage_contract.get("runtime_dispatch"))["command_topic"])
        stages.append(
            (str(node["node_id"]), command, str(stage_contract["terminal_event"]))
        )
    return stages


class HandlerStandInSweep:
    """Run the stand-in sweep named by ``ORCHESTRATOR`` over the bus."""

    ORCHESTRATOR: ClassVar[str] = "node_stand_in_sweep_orchestrator"

    def __init__(self, event_bus: object) -> None:
        self._bus = cast("_Bus", event_bus)

    async def handle(
        self, request: ModelStandInSweepRequest
    ) -> ModelStandInSweepResult:
        loop = asyncio.get_running_loop()
        stages = _stages(self.ORCHESTRATOR)
        waiting = {node_id: loop.create_future() for node_id, _, _ in stages}
        ignored = 0
        wanted = str(request.correlation_id)

        def _listener(node_id: str) -> Callable[[_Message], Awaitable[None]]:
            async def _on_terminal(message: _Message) -> None:
                nonlocal ignored
                body = _mapping(json.loads(message.value))
                answer = waiting[node_id]
                if (
                    body.get("correlation_id") != wanted
                    or body.get("stage_key") != node_id
                    or answer.done()
                ):
                    ignored += 1
                    return
                answer.set_result(str(body.get("value")))

            return _on_terminal

        unsubscribe = [
            await self._bus.subscribe(
                terminal,
                on_message=_listener(node_id),
                group_id=f"{self.ORCHESTRATOR}.{node_id}.{wanted}",
            )
            for node_id, _, terminal in stages
        ]
        try:
            for node_id, command, _ in stages:
                stage_request = ModelStandInStageRequest(
                    correlation_id=request.correlation_id,
                    stage_key=node_id,
                    operand=request.operand,
                )
                await self._bus.publish(
                    command, None, stage_request.model_dump_json().encode("utf-8")
                )
            await asyncio.wait(waiting.values(), timeout=request.stage_deadline_s)
        finally:
            for unsub in unsubscribe:
                await unsub()

        missing = [node_id for node_id, answer in waiting.items() if not answer.done()]
        if missing:
            return ModelStandInSweepResult(
                correlation_id=request.correlation_id,
                status="failed",
                ignored_terminals=ignored,
                why=f"stage-terminal-missing:{','.join(missing)}",
            )
        return ModelStandInSweepResult(
            correlation_id=request.correlation_id,
            status="completed",
            stage_values={
                node_id: answer.result() for node_id, answer in waiting.items()
            },
            ignored_terminals=ignored,
        )


class HandlerStandInStalledSweep(HandlerStandInSweep):
    """The same sweep under the orchestrator whose graph names the silent stage."""

    ORCHESTRATOR: ClassVar[str] = "node_stand_in_stalled_sweep_orchestrator"
