# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerA2ATask."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import aiohttp
import pytest
from pytest_httpserver import HTTPServer
from werkzeug.wrappers import Response

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_core.models.delegation.model_remote_task_state import ModelRemoteTaskState
from omnibase_core.models.delegation.model_target_agent import ModelTargetAgent
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_remote_agent_invoke_effect.handlers.handler_a2a_task import (
    HandlerA2ATask,
    map_remote_status,
)


class RecordingRepository:
    def __init__(self) -> None:
        self.rows: list[ModelRemoteTaskState] = []

    async def upsert(
        self,
        state: ModelRemoteTaskState,
        *,
        correlation_id=None,
    ) -> None:
        del correlation_id
        self.rows.append(state)

    async def get_by_remote_task_handle(
        self,
        remote_task_handle: str,
        *,
        correlation_id=None,
    ) -> ModelRemoteTaskState | None:
        del correlation_id
        for row in reversed(self.rows):
            if row.remote_task_handle == remote_task_handle:
                return row
        return None


class RecordingEventBus:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bytes | None, bytes]] = []

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        self.messages.append((topic, key, value))


def _command() -> ModelInvocationCommand:
    return ModelInvocationCommand(
        task_id=uuid4(),
        correlation_id=uuid4(),
        invocation_kind=EnumInvocationKind.AGENT,
        agent_protocol=EnumAgentProtocol.A2A,
        target_ref="agent:local-a2a-smoke",
        payload={"prompt": ModelSchemaValue.from_value("triage these findings")},
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("submitted", EnumAgentTaskLifecycleType.SUBMITTED),
        ("accepted", EnumAgentTaskLifecycleType.ACCEPTED),
        ("working", EnumAgentTaskLifecycleType.PROGRESS),
        ("in_progress", EnumAgentTaskLifecycleType.PROGRESS),
        ("completed", EnumAgentTaskLifecycleType.COMPLETED),
        ("failed", EnumAgentTaskLifecycleType.FAILED),
        ("timed_out", EnumAgentTaskLifecycleType.TIMED_OUT),
        ("canceled", EnumAgentTaskLifecycleType.CANCELED),
    ],
)
def test_status_mapping_table(
    raw_status: str,
    expected: EnumAgentTaskLifecycleType,
) -> None:
    assert map_remote_status(raw_status) is expected


@pytest.mark.unit
def test_status_mapping_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown remote agent status"):
        map_remote_status("mystery")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_emits_submitted_event_and_persists(
    httpserver: HTTPServer,
) -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 15, tzinfo=UTC)

    httpserver.expect_request("/tasks.send", method="POST").respond_with_json(
        {
            "remote_task_handle": "remote-123",
            "status": "submitted",
            "artifacts": [],
        }
    )

    async with aiohttp.ClientSession() as session:
        handler = HandlerA2ATask(
            repository=repo,  # type: ignore[arg-type]
            event_bus=event_bus,
            target_registry={
                "agent:local-a2a-smoke": ModelTargetAgent(
                    agent_ref="agent:local-a2a-smoke",
                    protocol=EnumAgentProtocol.A2A,
                    base_url=httpserver.url_for(""),
                    protocol_version="0.3",
                )
            },
            lifecycle_topic="onex.evt.omnibase-infra.agent-task-lifecycle.v1",
            http_session=session,
            clock=lambda: now,
        )

        response = await handler.submit(command)

    assert response.remote_task_handle == "remote-123"
    assert len(repo.rows) == 2
    assert repo.rows[0].remote_task_handle is None
    assert repo.rows[1].remote_task_handle == "remote-123"
    assert len(event_bus.messages) == 1

    topic, key, value = event_bus.messages[0]
    assert topic == "onex.evt.omnibase-infra.agent-task-lifecycle.v1"
    assert key == str(command.task_id).encode("utf-8")

    envelope = ModelEventEnvelope[object].model_validate(json.loads(value))
    payload = envelope.payload
    assert isinstance(payload, dict)
    assert payload["lifecycle_type"] == "SUBMITTED"
    assert payload["remote_task_handle"] is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watch_emits_transitions_until_completed(httpserver: HTTPServer) -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 20, tzinfo=UTC)
    poll_counter = {"count": 0}

    def _watch_handler(request):
        del request
        poll_counter["count"] += 1
        if poll_counter["count"] == 1:
            return Response(
                json.dumps(
                    {
                        "remote_task_handle": "remote-123",
                        "status": "in_progress",
                        "artifacts": [],
                        "error": None,
                    }
                ),
                status=200,
                content_type="application/json",
            )
        return Response(
            json.dumps(
                {
                    "remote_task_handle": "remote-123",
                    "status": "completed",
                    "artifacts": [{"result": {"summary": "done"}}],
                    "error": None,
                }
            ),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/tasks.send", method="POST").respond_with_json(
        {
            "remote_task_handle": "remote-123",
            "status": "submitted",
            "artifacts": [],
        }
    )
    httpserver.expect_request("/tasks.get", method="GET").respond_with_handler(
        _watch_handler
    )

    async with aiohttp.ClientSession() as session:
        handler = HandlerA2ATask(
            repository=repo,  # type: ignore[arg-type]
            event_bus=event_bus,
            target_registry={
                "agent:local-a2a-smoke": ModelTargetAgent(
                    agent_ref="agent:local-a2a-smoke",
                    protocol=EnumAgentProtocol.A2A,
                    base_url=httpserver.url_for(""),
                    protocol_version="0.3",
                )
            },
            lifecycle_topic="onex.evt.omnibase-infra.agent-task-lifecycle.v1",
            http_session=session,
            clock=lambda: now,
            poll_interval_seconds=0.0,
        )
        await handler.submit(command)
        events = await handler.watch(
            remote_task_handle="remote-123",
            correlation_id=command.correlation_id,
        )

    assert [event.lifecycle_type for event in events] == [
        EnumAgentTaskLifecycleType.PROGRESS,
        EnumAgentTaskLifecycleType.ARTIFACT,
        EnumAgentTaskLifecycleType.COMPLETED,
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watch_dedups_same_status_repeats(httpserver: HTTPServer) -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 25, tzinfo=UTC)
    poll_counter = {"count": 0}

    def _watch_handler(request):
        del request
        poll_counter["count"] += 1
        if poll_counter["count"] < 3:
            return Response(
                json.dumps(
                    {
                        "remote_task_handle": "remote-456",
                        "status": "in_progress",
                        "artifacts": [],
                        "error": None,
                    }
                ),
                status=200,
                content_type="application/json",
            )
        return Response(
            json.dumps(
                {
                    "remote_task_handle": "remote-456",
                    "status": "completed",
                    "artifacts": [],
                    "error": None,
                }
            ),
            status=200,
            content_type="application/json",
        )

    httpserver.expect_request("/tasks.send", method="POST").respond_with_json(
        {
            "remote_task_handle": "remote-456",
            "status": "submitted",
            "artifacts": [],
        }
    )
    httpserver.expect_request("/tasks.get", method="GET").respond_with_handler(
        _watch_handler
    )

    async with aiohttp.ClientSession() as session:
        handler = HandlerA2ATask(
            repository=repo,  # type: ignore[arg-type]
            event_bus=event_bus,
            target_registry={
                "agent:local-a2a-smoke": ModelTargetAgent(
                    agent_ref="agent:local-a2a-smoke",
                    protocol=EnumAgentProtocol.A2A,
                    base_url=httpserver.url_for(""),
                    protocol_version="0.3",
                )
            },
            lifecycle_topic="onex.evt.omnibase-infra.agent-task-lifecycle.v1",
            http_session=session,
            clock=lambda: now,
            poll_interval_seconds=0.0,
        )
        updated_command = command.model_copy(
            update={"task_id": uuid4(), "correlation_id": uuid4()}
        )
        await handler.submit(updated_command)
        events = await handler.watch(
            remote_task_handle="remote-456",
            correlation_id=updated_command.correlation_id,
        )

    assert [event.lifecycle_type for event in events] == [
        EnumAgentTaskLifecycleType.PROGRESS,
        EnumAgentTaskLifecycleType.COMPLETED,
    ]
