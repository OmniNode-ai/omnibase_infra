# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerA2ATask."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation.model_a2a_task_request import ModelA2ATaskRequest
from omnibase_core.models.delegation.model_a2a_task_response import ModelA2ATaskResponse
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_core.models.delegation.model_remote_task_state import ModelRemoteTaskState
from omnibase_core.models.delegation.model_target_agent import ModelTargetAgent
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.nodes.node_remote_agent_invoke_effect.handlers.handler_a2a_task import (
    HandlerA2ATask,
    ProtocolA2ATransport,
    map_remote_status,
)
from omnibase_infra.nodes.node_remote_agent_invoke_effect.services import (
    EventBusLifecycleEventSink,
)


class RecordingRepository:
    def __init__(self) -> None:
        self.rows: list[ModelRemoteTaskState] = []
        self.artifacts: dict[object, list[dict[str, ModelSchemaValue]]] = {}

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

    async def get_last_emitted_artifact_payload(
        self,
        task_id,
        *,
        correlation_id=None,
    ) -> "list[dict[str, ModelSchemaValue]] | None":  # noqa: UP037
        del correlation_id
        return self.artifacts.get(task_id)

    async def set_last_emitted_artifact_payload(
        self,
        task_id,
        payload: list[dict[str, ModelSchemaValue]],
        *,
        correlation_id=None,
    ) -> None:
        del correlation_id
        self.artifacts[task_id] = payload


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


class FakeTransport(ProtocolA2ATransport):
    def __init__(
        self,
        *,
        submit_response: ModelA2ATaskResponse,
        get_responses: list[ModelA2ATaskResponse] | None = None,
        submit_error: "Exception | None" = None,  # noqa: UP037
        get_error: "Exception | None" = None,  # noqa: UP037
    ) -> None:
        self.submit_response = submit_response
        self.get_responses = list(get_responses or [])
        self.submit_error = submit_error
        self.get_error = get_error
        self.submit_requests: list[tuple[ModelTargetAgent, ModelA2ATaskRequest]] = []
        self.get_requests: list[tuple[ModelTargetAgent, str]] = []

    async def submit(
        self,
        *,
        target: ModelTargetAgent,
        request_model: ModelA2ATaskRequest,
    ) -> ModelA2ATaskResponse:
        self.submit_requests.append((target, request_model))
        if self.submit_error is not None:
            raise self.submit_error
        return self.submit_response

    async def get_task(
        self,
        *,
        target: ModelTargetAgent,
        remote_task_handle: str,
    ) -> ModelA2ATaskResponse:
        self.get_requests.append((target, remote_task_handle))
        if self.get_error is not None:
            raise self.get_error
        if not self.get_responses:
            msg = "No queued get_task responses"
            raise AssertionError(msg)
        return self.get_responses.pop(0)


def _command() -> ModelInvocationCommand:
    return ModelInvocationCommand(
        task_id=uuid4(),
        correlation_id=uuid4(),
        invocation_kind=EnumInvocationKind.AGENT,
        agent_protocol=EnumAgentProtocol.A2A,
        target_ref="agent:local-a2a-smoke",
        payload={"prompt": ModelSchemaValue.from_value("triage these findings")},
    )


def _target_registry() -> dict[str, ModelTargetAgent]:
    return {
        "agent:local-a2a-smoke": ModelTargetAgent(
            target_ref="agent:local-a2a-smoke",
            protocol=EnumAgentProtocol.A2A,
            base_url="http://127.0.0.1:8011/a2a/app",
        )
    }


def _lifecycle_sink(event_bus: RecordingEventBus) -> EventBusLifecycleEventSink:
    return EventBusLifecycleEventSink(
        event_bus=event_bus,
        lifecycle_topic="onex.evt.omnibase-infra.agent-task-lifecycle.v1",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("submitted", EnumAgentTaskLifecycleType.SUBMITTED),
        ("accepted", EnumAgentTaskLifecycleType.ACCEPTED),
        ("working", EnumAgentTaskLifecycleType.PROGRESS),
        ("in_progress", EnumAgentTaskLifecycleType.PROGRESS),
        ("input-required", EnumAgentTaskLifecycleType.PROGRESS),
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
async def test_zero_arg_handler_fails_closed_for_unknown_target() -> None:
    handler = HandlerA2ATask()

    with pytest.raises(ValueError, match="Unknown target_ref"):
        await handler.submit(_command())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_emits_submitted_event_and_persists() -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 15, tzinfo=UTC)
    transport = FakeTransport(
        submit_response=ModelA2ATaskResponse(
            remote_task_handle="remote-123",
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            artifacts=[],
            error=None,
        )
    )

    handler = HandlerA2ATask(
        repository=repo,  # type: ignore[arg-type]
        lifecycle_sink=_lifecycle_sink(event_bus),
        target_registry=_target_registry(),
        clock=lambda: now,
        transport=transport,
    )

    response = await handler.submit(command)

    assert response.remote_task_handle == "remote-123"
    assert len(repo.rows) == 2
    assert repo.rows[0].remote_task_handle is None
    assert repo.rows[1].remote_task_handle == "remote-123"
    assert len(event_bus.messages) == 1
    assert len(transport.submit_requests) == 1
    _target, request_model = transport.submit_requests[0]
    assert request_model.input["prompt"].to_value() == "triage these findings"

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
async def test_watch_emits_transitions_until_completed() -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 20, tzinfo=UTC)
    transport = FakeTransport(
        submit_response=ModelA2ATaskResponse(
            remote_task_handle="remote-123",
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            artifacts=[],
            error=None,
        ),
        get_responses=[
            ModelA2ATaskResponse(
                remote_task_handle="remote-123",
                status=EnumAgentTaskLifecycleType.PROGRESS,
                artifacts=[],
                error=None,
            ),
            ModelA2ATaskResponse(
                remote_task_handle="remote-123",
                status=EnumAgentTaskLifecycleType.COMPLETED,
                artifacts=[
                    {
                        "report": ModelSchemaValue.from_value(
                            {"summary": "done", "priority": "high"}
                        )
                    }
                ],
                error=None,
            ),
        ],
    )

    handler = HandlerA2ATask(
        repository=repo,  # type: ignore[arg-type]
        lifecycle_sink=_lifecycle_sink(event_bus),
        target_registry=_target_registry(),
        clock=lambda: now,
        poll_interval_seconds=0.0,
        transport=transport,
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
    assert events[1].artifact is not None
    assert events[1].artifact["report"].to_value() == {
        "summary": "done",
        "priority": "high",
    }
    assert repo.artifacts[command.task_id] == [
        {"report": ModelSchemaValue.from_value({"summary": "done", "priority": "high"})}
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watch_dedups_same_status_repeats() -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 25, tzinfo=UTC)
    transport = FakeTransport(
        submit_response=ModelA2ATaskResponse(
            remote_task_handle="remote-456",
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            artifacts=[],
            error=None,
        ),
        get_responses=[
            ModelA2ATaskResponse(
                remote_task_handle="remote-456",
                status=EnumAgentTaskLifecycleType.PROGRESS,
                artifacts=[],
                error=None,
            ),
            ModelA2ATaskResponse(
                remote_task_handle="remote-456",
                status=EnumAgentTaskLifecycleType.PROGRESS,
                artifacts=[],
                error=None,
            ),
            ModelA2ATaskResponse(
                remote_task_handle="remote-456",
                status=EnumAgentTaskLifecycleType.COMPLETED,
                artifacts=[],
                error=None,
            ),
        ],
    )

    handler = HandlerA2ATask(
        repository=repo,  # type: ignore[arg-type]
        lifecycle_sink=_lifecycle_sink(event_bus),
        target_registry=_target_registry(),
        clock=lambda: now,
        poll_interval_seconds=0.0,
        transport=transport,
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


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watch_uses_persisted_artifact_dedupe_state() -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 28, tzinfo=UTC)
    artifact = {"report": ModelSchemaValue.from_value({"summary": "done"})}
    remote_task_handle = "remote-restart"
    row = ModelRemoteTaskState(
        task_id=command.task_id,
        invocation_kind=command.invocation_kind,
        protocol=command.agent_protocol,
        target_ref=command.target_ref,
        remote_task_handle=remote_task_handle,
        correlation_id=command.correlation_id,
        status=EnumAgentTaskLifecycleType.PROGRESS,
        last_remote_status=EnumAgentTaskLifecycleType.PROGRESS.value,
        last_emitted_event_type=EnumAgentTaskLifecycleType.PROGRESS,
        submitted_at=now,
        updated_at=now,
    )
    await repo.upsert(row)
    await repo.set_last_emitted_artifact_payload(
        command.task_id,
        [artifact],
        correlation_id=command.correlation_id,
    )
    transport = FakeTransport(
        submit_response=ModelA2ATaskResponse(
            remote_task_handle="unused",
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            artifacts=[],
            error=None,
        ),
        get_responses=[
            ModelA2ATaskResponse(
                remote_task_handle=remote_task_handle,
                status=EnumAgentTaskLifecycleType.COMPLETED,
                artifacts=[artifact],
                error=None,
            ),
        ],
    )

    handler = HandlerA2ATask(
        repository=repo,  # type: ignore[arg-type]
        lifecycle_sink=_lifecycle_sink(event_bus),
        target_registry=_target_registry(),
        clock=lambda: now,
        poll_interval_seconds=0.0,
        transport=transport,
    )
    events = await handler.watch(
        remote_task_handle=remote_task_handle,
        correlation_id=command.correlation_id,
    )

    assert [event.lifecycle_type for event in events] == [
        EnumAgentTaskLifecycleType.COMPLETED
    ]
    assert len(event_bus.messages) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_submit_failure_persists_failed_lifecycle_before_raising() -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    now = datetime(2026, 4, 25, 17, 30, tzinfo=UTC)
    transport = FakeTransport(
        submit_response=ModelA2ATaskResponse(
            remote_task_handle="unused",
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            artifacts=[],
            error=None,
        ),
        submit_error=RuntimeError("peer unavailable"),
    )

    handler = HandlerA2ATask(
        repository=repo,  # type: ignore[arg-type]
        lifecycle_sink=_lifecycle_sink(event_bus),
        target_registry=_target_registry(),
        clock=lambda: now,
        transport=transport,
    )

    with pytest.raises(InfraUnavailableError, match="a2a_submit failed"):
        await handler.submit(command)

    assert repo.rows[-1].status is EnumAgentTaskLifecycleType.FAILED
    assert repo.rows[-1].last_remote_status == "submit_failed"
    assert repo.rows[-1].remote_task_handle is None
    assert len(event_bus.messages) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_watch_times_out_non_terminal_task() -> None:
    repo = RecordingRepository()
    event_bus = RecordingEventBus()
    command = _command()
    times = [
        datetime(2026, 4, 25, 17, 35, 0, tzinfo=UTC),
        datetime(2026, 4, 25, 17, 35, 2, tzinfo=UTC),
    ]
    transport = FakeTransport(
        submit_response=ModelA2ATaskResponse(
            remote_task_handle="remote-timeout",
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            artifacts=[],
            error=None,
        ),
    )

    def _clock() -> datetime:
        return times.pop(0) if times else datetime(2026, 4, 25, 17, 35, 3, tzinfo=UTC)

    handler = HandlerA2ATask(
        repository=repo,  # type: ignore[arg-type]
        lifecycle_sink=_lifecycle_sink(event_bus),
        target_registry=_target_registry(),
        clock=_clock,
        poll_interval_seconds=0.0,
        transport=transport,
    )
    await handler.submit(command)
    times[:] = [
        datetime(2026, 4, 25, 17, 35, 0, tzinfo=UTC),
        datetime(2026, 4, 25, 17, 35, 2, tzinfo=UTC),
    ]

    events = await handler.watch(
        remote_task_handle="remote-timeout",
        correlation_id=command.correlation_id,
        max_watch_seconds=0.5,
    )

    assert [event.lifecycle_type for event in events] == [
        EnumAgentTaskLifecycleType.TIMED_OUT
    ]
    assert repo.rows[-1].status is EnumAgentTaskLifecycleType.TIMED_OUT
