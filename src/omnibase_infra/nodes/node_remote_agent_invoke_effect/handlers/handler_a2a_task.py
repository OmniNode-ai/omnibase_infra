# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A2A submit/watch handler for node_remote_agent_invoke_effect."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

import aiohttp
import httpx
from a2a.client.base_client import BaseClient
from a2a.client.client import ClientConfig
from a2a.client.client_factory import ClientFactory
from a2a.types import (
    Message,
    MessageSendConfiguration,
    Part,
    Role,
    TaskQueryParams,
    TextPart,
)
from a2a.types import Task as A2ATask

from omnibase_core.enums.enum_agent_protocol import EnumAgentProtocol
from omnibase_core.enums.enum_agent_task_lifecycle_type import (
    EnumAgentTaskLifecycleType,
)
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation.model_a2a_task_request import ModelA2ATaskRequest
from omnibase_core.models.delegation.model_a2a_task_response import ModelA2ATaskResponse
from omnibase_core.models.delegation.model_agent_task_lifecycle_event import (
    ModelAgentTaskLifecycleEvent,
)
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_core.models.delegation.model_remote_task_state import ModelRemoteTaskState
from omnibase_core.models.delegation.model_target_agent import ModelTargetAgent
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_remote_agent_invoke_effect.persistence.remote_task_state_repository import (
    RemoteTaskStateRepository,
)
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

_STATUS_MAP: dict[str, EnumAgentTaskLifecycleType] = {
    "submitted": EnumAgentTaskLifecycleType.SUBMITTED,
    "accepted": EnumAgentTaskLifecycleType.ACCEPTED,
    "artifact": EnumAgentTaskLifecycleType.ARTIFACT,
    "working": EnumAgentTaskLifecycleType.PROGRESS,
    "in_progress": EnumAgentTaskLifecycleType.PROGRESS,
    "input_required": EnumAgentTaskLifecycleType.PROGRESS,
    "auth_required": EnumAgentTaskLifecycleType.PROGRESS,
    "completed": EnumAgentTaskLifecycleType.COMPLETED,
    "failed": EnumAgentTaskLifecycleType.FAILED,
    "timed_out": EnumAgentTaskLifecycleType.TIMED_OUT,
    "canceled": EnumAgentTaskLifecycleType.CANCELED,
}
_TERMINAL_LIFECYCLES = {
    EnumAgentTaskLifecycleType.COMPLETED,
    EnumAgentTaskLifecycleType.FAILED,
    EnumAgentTaskLifecycleType.TIMED_OUT,
    EnumAgentTaskLifecycleType.CANCELED,
}


class ProtocolA2ATransport(Protocol):
    """Small transport boundary so unit tests can avoid protocol plumbing."""

    async def submit(
        self,
        *,
        target: ModelTargetAgent,
        request_model: ModelA2ATaskRequest,
    ) -> ModelA2ATaskResponse: ...

    async def get_task(
        self,
        *,
        target: ModelTargetAgent,
        remote_task_handle: str,
    ) -> ModelA2ATaskResponse: ...


def map_remote_status(raw_status: str) -> EnumAgentTaskLifecycleType:
    """Map A2A peer status strings into typed lifecycle events."""
    normalized = raw_status.strip().lower().replace("-", "_")
    try:
        return _STATUS_MAP[normalized]
    except KeyError as exc:
        msg = f"Unknown remote agent status: {raw_status}"
        raise ValueError(msg) from exc


class A2ASdkTransport:
    """Real A2A transport backed by the upstream A2A SDK."""

    def __init__(self, *, request_timeout_seconds: float = 180.0) -> None:
        self._request_timeout_seconds = request_timeout_seconds

    async def submit(
        self,
        *,
        target: ModelTargetAgent,
        request_model: ModelA2ATaskRequest,
    ) -> ModelA2ATaskResponse:
        async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
            a2a_client = await ClientFactory.connect(
                target.base_url,
                client_config=ClientConfig(
                    streaming=False,
                    polling=True,
                    httpx_client=client,
                ),
            )
            typed_client = _require_base_client(a2a_client)
            message = Message(
                message_id=str(request_model.correlation_id),
                role=Role.user,
                parts=[Part(root=TextPart(text=_build_prompt(request_model)))],
            )
            async for event in typed_client.send_message(
                message,
                configuration=MessageSendConfiguration(blocking=False),
            ):
                if isinstance(event, Message):
                    msg = "Unexpected direct message response from A2A peer"
                    raise ValueError(msg)
                task, _update = event
                return _task_to_response(task)

        msg = "A2A peer returned no task from send_message"
        raise ValueError(msg)

    async def get_task(
        self,
        *,
        target: ModelTargetAgent,
        remote_task_handle: str,
    ) -> ModelA2ATaskResponse:
        async with httpx.AsyncClient(timeout=self._request_timeout_seconds) as client:
            a2a_client = await ClientFactory.connect(
                target.base_url,
                client_config=ClientConfig(
                    streaming=False,
                    polling=True,
                    httpx_client=client,
                ),
            )
            typed_client = _require_base_client(a2a_client)
            task = await typed_client.get_task(
                TaskQueryParams(id=remote_task_handle, history_length=20)
            )
        return _task_to_response(task)


def _require_base_client(client: object) -> BaseClient:
    if isinstance(client, BaseClient):
        return client
    msg = f"Unsupported A2A client type: {type(client).__name__}"
    raise TypeError(msg)


def _build_prompt(request_model: ModelA2ATaskRequest) -> str:
    payload = {key: value.to_value() for key, value in request_model.input.items()}
    prompt = payload.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    return json.dumps(
        {
            "skill_id": request_model.skill_ref,
            "correlation_id": str(request_model.correlation_id),
            "input": payload,
        },
        indent=2,
        sort_keys=True,
    )


def _task_to_response(task: A2ATask) -> ModelA2ATaskResponse:
    status = getattr(task.status.state, "value", str(task.status.state))
    return ModelA2ATaskResponse(
        remote_task_handle=task.id,
        status=status,
        artifacts=_extract_artifacts(task),
        error=_extract_error(task),
    )


def _extract_error(task: A2ATask) -> str | None:
    state = getattr(task.status.state, "value", str(task.status.state)).lower()
    if state not in {"failed", "canceled", "timed_out"}:
        return None
    status_message = getattr(task.status, "message", None)
    if status_message is None:
        return None
    texts = _extract_texts(getattr(status_message, "parts", None) or [])
    if texts:
        return "\n".join(texts)
    return None


def _extract_artifacts(task: A2ATask) -> list[dict[str, ModelSchemaValue]]:
    task_artifacts = getattr(task, "artifacts", None) or []
    extracted: list[dict[str, ModelSchemaValue]] = []
    for artifact in task_artifacts:
        payload = _parts_to_payload(getattr(artifact, "parts", None) or [])
        if payload is None:
            continue
        if isinstance(payload, dict):
            extracted.append(
                {
                    key: ModelSchemaValue.from_value(value)
                    for key, value in payload.items()
                }
            )
            continue
        extracted.append({"text": ModelSchemaValue.from_value(payload)})
    return extracted


def _parts_to_payload(parts: Sequence[object]) -> object | None:
    texts = _extract_texts(parts)
    if not texts:
        return None
    combined = "\n".join(texts).strip()
    if not combined:
        return None
    try:
        parsed = json.loads(combined)
    except json.JSONDecodeError:
        return combined
    return parsed


def _extract_texts(parts: Sequence[object]) -> list[str]:
    texts: list[str] = []
    for part in parts:
        candidate = getattr(part, "root", part)
        kind = getattr(candidate, "kind", None)
        if kind == "text":
            text = getattr(candidate, "text", None)
            if isinstance(text, str) and text:
                texts.append(text)
    return texts


class HandlerA2ATask:
    """Submit-path handler for A2A task invocation."""

    def __init__(
        self,
        *,
        repository: RemoteTaskStateRepository,
        event_bus: ProtocolEventBusLike,
        target_registry: Mapping[str, ModelTargetAgent],
        lifecycle_topic: str,
        http_session: aiohttp.ClientSession | None = None,
        clock: Callable[[], datetime] | None = None,
        poll_interval_seconds: float = 1.0,
        transport: ProtocolA2ATransport | None = None,
    ) -> None:
        self._repository = repository
        self._event_bus = event_bus
        self._target_registry = target_registry
        self._lifecycle_topic = lifecycle_topic
        self._http_session = http_session
        self._clock = clock or (lambda: datetime.now(UTC))
        self._poll_interval_seconds = poll_interval_seconds
        self._transport = transport or A2ASdkTransport()

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def submit(self, command: ModelInvocationCommand) -> ModelA2ATaskResponse:
        """Submit a command to a remote A2A peer and persist the returned handle."""
        target = self._resolve_target(command.target_ref)
        now = self._clock()

        submitted_state = ModelRemoteTaskState(
            task_id=command.task_id,
            invocation_kind=command.invocation_kind,
            protocol=command.agent_protocol,
            target_ref=command.target_ref,
            remote_task_handle=None,
            correlation_id=command.correlation_id,
            status=EnumAgentTaskLifecycleType.SUBMITTED,
            last_remote_status="submitted",
            last_emitted_event_type=EnumAgentTaskLifecycleType.SUBMITTED,
            submitted_at=now,
            updated_at=now,
        )
        await self._repository.upsert(submitted_state)
        await self._emit_submitted(
            command=command, occurred_at=now, remote_task_handle=None
        )

        request_model = ModelA2ATaskRequest(
            skill_ref=command.target_ref,
            input=command.payload,
            correlation_id=command.correlation_id,
        )
        response = await self._transport.submit(
            target=target, request_model=request_model
        )

        await self._repository.upsert(
            submitted_state.model_copy(
                update={
                    "remote_task_handle": response.remote_task_handle,
                    "last_remote_status": response.status,
                    "updated_at": self._clock(),
                }
            )
        )
        return response

    async def watch(
        self,
        remote_task_handle: str,
        correlation_id: UUID,
    ) -> list[ModelAgentTaskLifecycleEvent]:
        """Poll tasks.get until the remote task reaches a terminal lifecycle."""
        row = await self._repository.get_by_remote_task_handle(
            remote_task_handle,
            correlation_id=correlation_id,
        )
        if row is None:
            msg = f"Unknown remote_task_handle: {remote_task_handle}"
            raise ValueError(msg)

        target = self._resolve_target(row.target_ref)
        last_seen = row.last_emitted_event_type
        events: list[ModelAgentTaskLifecycleEvent] = []
        last_artifact_payload: list[dict[str, ModelSchemaValue]] | None = None

        while True:
            response = await self._transport.get_task(
                target=target,
                remote_task_handle=remote_task_handle,
            )
            mapped = map_remote_status(response.status)
            now = self._clock()

            if response.artifacts and response.artifacts != last_artifact_payload:
                artifact_event = await self._emit_lifecycle(
                    task_id=row.task_id,
                    correlation_id=correlation_id,
                    lifecycle_type=EnumAgentTaskLifecycleType.ARTIFACT,
                    remote_task_handle=remote_task_handle,
                    artifact=response.artifacts[0],
                    occurred_at=now,
                    remote_status=response.status,
                    error=response.error,
                )
                await self._repository.upsert(
                    row.model_copy(
                        update={
                            "status": mapped,
                            "last_remote_status": response.status,
                            "last_emitted_event_type": EnumAgentTaskLifecycleType.ARTIFACT,
                            "updated_at": now,
                            "completed_at": now
                            if mapped in _TERMINAL_LIFECYCLES
                            else None,
                            "error": response.error,
                        }
                    ),
                    correlation_id=correlation_id,
                )
                row = row.model_copy(
                    update={
                        "status": mapped,
                        "last_remote_status": response.status,
                        "last_emitted_event_type": EnumAgentTaskLifecycleType.ARTIFACT,
                        "updated_at": now,
                        "completed_at": now if mapped in _TERMINAL_LIFECYCLES else None,
                        "error": response.error,
                    }
                )
                events.append(artifact_event)
                last_artifact_payload = response.artifacts

            if mapped is last_seen:
                if mapped in _TERMINAL_LIFECYCLES:
                    return events
                await asyncio.sleep(self._poll_interval_seconds)
                continue

            lifecycle_event = await self._emit_lifecycle(
                task_id=row.task_id,
                correlation_id=correlation_id,
                lifecycle_type=mapped,
                remote_task_handle=remote_task_handle,
                artifact=response.artifacts[0] if response.artifacts else None,
                occurred_at=now,
                remote_status=response.status,
                error=response.error,
            )
            await self._repository.upsert(
                row.model_copy(
                    update={
                        "status": mapped,
                        "last_remote_status": response.status,
                        "last_emitted_event_type": mapped,
                        "updated_at": now,
                        "completed_at": now if mapped in _TERMINAL_LIFECYCLES else None,
                        "error": response.error,
                    }
                ),
                correlation_id=correlation_id,
            )
            row = row.model_copy(
                update={
                    "status": mapped,
                    "last_remote_status": response.status,
                    "last_emitted_event_type": mapped,
                    "updated_at": now,
                    "completed_at": now if mapped in _TERMINAL_LIFECYCLES else None,
                    "error": response.error,
                }
            )
            last_seen = mapped
            events.append(lifecycle_event)
            if mapped in _TERMINAL_LIFECYCLES:
                return events
            await asyncio.sleep(self._poll_interval_seconds)

    def _resolve_target(self, target_ref: str) -> ModelTargetAgent:
        try:
            target = self._target_registry[target_ref]
        except KeyError as exc:
            msg = f"Unknown target_ref: {target_ref}"
            raise ValueError(msg) from exc
        if target.protocol is not EnumAgentProtocol.A2A:
            msg = f"Target {target_ref} is not configured for A2A"
            raise ValueError(msg)
        return target

    async def _emit_submitted(
        self,
        *,
        command: ModelInvocationCommand,
        occurred_at: datetime,
        remote_task_handle: str | None,
    ) -> None:
        await self._emit_lifecycle(
            task_id=command.task_id,
            correlation_id=command.correlation_id,
            lifecycle_type=EnumAgentTaskLifecycleType.SUBMITTED,
            remote_task_handle=remote_task_handle,
            artifact={
                "target_ref": ModelSchemaValue.from_value(command.target_ref),
            },
            occurred_at=occurred_at,
            remote_status="submitted",
        )

    async def _emit_lifecycle(
        self,
        *,
        task_id: UUID,
        correlation_id: UUID,
        lifecycle_type: EnumAgentTaskLifecycleType,
        remote_task_handle: str | None,
        occurred_at: datetime,
        artifact: dict[str, ModelSchemaValue] | None = None,
        remote_status: str | None = None,
        error: str | None = None,
    ) -> ModelAgentTaskLifecycleEvent:
        lifecycle_event = ModelAgentTaskLifecycleEvent(
            task_id=task_id,
            correlation_id=correlation_id,
            lifecycle_type=lifecycle_type,
            remote_task_handle=remote_task_handle,
            artifact=artifact,
            occurred_at=occurred_at,
            remote_status=remote_status,
            error=error,
        )
        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=correlation_id,
            payload=lifecycle_event.model_dump(mode="json"),
        )
        await self._event_bus.publish(
            self._lifecycle_topic,
            key=str(task_id).encode("utf-8"),
            value=envelope.model_dump_json().encode("utf-8"),
        )
        return lifecycle_event


__all__ = [
    "A2ASdkTransport",
    "HandlerA2ATask",
    "ProtocolA2ATransport",
    "_STATUS_MAP",
    "_TERMINAL_LIFECYCLES",
    "map_remote_status",
]
