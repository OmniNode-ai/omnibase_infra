# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A2A submit-path handler for node_remote_agent_invoke_effect."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from uuid import UUID

import aiohttp

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
    "working": EnumAgentTaskLifecycleType.PROGRESS,
    "in_progress": EnumAgentTaskLifecycleType.PROGRESS,
    "completed": EnumAgentTaskLifecycleType.COMPLETED,
    "failed": EnumAgentTaskLifecycleType.FAILED,
    "timed_out": EnumAgentTaskLifecycleType.TIMED_OUT,
    "canceled": EnumAgentTaskLifecycleType.CANCELED,
}


def map_remote_status(raw_status: str) -> EnumAgentTaskLifecycleType:
    """Map A2A peer status strings into typed lifecycle events."""
    normalized = raw_status.strip().lower()
    try:
        return _STATUS_MAP[normalized]
    except KeyError as exc:
        msg = f"Unknown remote agent status: {raw_status}"
        raise ValueError(msg) from exc


class HandlerA2ATask:
    """Submit-path handler for A2A task invocation."""

    def __init__(
        self,
        *,
        repository: RemoteTaskStateRepository,
        event_bus: ProtocolEventBusLike,
        target_registry: Mapping[str, ModelTargetAgent],
        lifecycle_topic: str,
        http_session: aiohttp.ClientSession,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._repository = repository
        self._event_bus = event_bus
        self._target_registry = target_registry
        self._lifecycle_topic = lifecycle_topic
        self._http_session = http_session
        self._clock = clock or (lambda: datetime.now(UTC))

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
        response = await self._post_tasks_send(
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

    async def _post_tasks_send(
        self,
        *,
        target: ModelTargetAgent,
        request_model: ModelA2ATaskRequest,
    ) -> ModelA2ATaskResponse:
        async with self._http_session.post(
            f"{target.base_url.rstrip('/')}/tasks.send",
            json=request_model.model_dump(mode="json", by_alias=True),
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        return ModelA2ATaskResponse.model_validate(payload)

    async def _emit_submitted(
        self,
        *,
        command: ModelInvocationCommand,
        occurred_at: datetime,
        remote_task_handle: str | None,
    ) -> None:
        lifecycle_event = ModelAgentTaskLifecycleEvent(
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
        envelope = ModelEventEnvelope[dict[str, object]](
            correlation_id=command.correlation_id,
            payload=lifecycle_event.model_dump(mode="json"),
        )
        await self._event_bus.publish(
            self._lifecycle_topic,
            key=str(command.task_id).encode("utf-8"),
            value=envelope.model_dump_json().encode("utf-8"),
        )


__all__ = ["HandlerA2ATask", "_STATUS_MAP", "map_remote_status"]
