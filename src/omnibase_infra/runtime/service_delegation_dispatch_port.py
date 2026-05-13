# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-owned delegation dispatch port for consumer-facing delegation handlers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from importlib import import_module
from typing import Literal, Protocol, cast
from uuid import UUID, uuid4

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)

_INTERNAL_COMMAND_NAME = "node_delegation_orchestrator"
_REQUESTER = "delegate-skill-runtime-port"
_DEFAULT_TIMEOUT_SECONDS = 300


class ProtocolDelegationPortEventBus(Protocol):
    """Event bus surface required by RuntimeDelegationDispatchPort."""

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: object = None,
    ) -> None: ...

    async def subscribe(
        self,
        topic: str,
        node_identity: object | None = None,
        on_message: Callable[[object], Awaitable[None]] | None = None,
        **kwargs: object,
    ) -> Callable[[], Awaitable[None]]: ...


class RuntimeDelegationDispatchPort:
    """Dispatch delegate-skill requests into the runtime Pattern B broker."""

    def __init__(
        self,
        *,
        event_bus: ProtocolDelegationPortEventBus,
        command_topic: str | None = None,
        response_topic: str | None = None,
    ) -> None:
        default_command_topic, default_response_topic = _load_topic_defaults()
        self._event_bus = event_bus
        self._command_topic = command_topic or default_command_topic
        self._response_topic = response_topic or default_response_topic

    async def dispatch(
        self,
        *,
        prompt: str,
        task_type: str,
        correlation_id: UUID,
        max_tokens: int,
        source_file_path: str | None,
        source_session_id: str | None,
        wait: bool,
    ) -> dict[str, object]:
        dispatch_correlation_id = uuid4()
        request = ModelDelegationRequest(
            prompt=prompt,
            task_type=cast("Literal['test', 'document', 'research']", task_type),
            source_session_id=source_session_id,
            source_file_path=source_file_path,
            correlation_id=dispatch_correlation_id,
            max_tokens=max_tokens,
            emitted_at=datetime.now(UTC),
        )
        command = ModelDispatchBusCommand(
            command_name=_INTERNAL_COMMAND_NAME,
            requester=_REQUESTER,
            payload=cast("dict[str, object]", request.model_dump(mode="json")),
            correlation_id=dispatch_correlation_id,
            response_topic=self._response_topic,
            timeout_seconds=_DEFAULT_TIMEOUT_SECONDS,
        )

        if not wait:
            await self._publish_command(command)
            return {
                "status": "completed",
                "correlation_id": str(correlation_id),
                "content": "",
                "delegated_to": "runtime",
                "model_name": "",
                "quality_gate_passed": False,
            }

        unsubscribe, queue = await self._subscribe_for_result(dispatch_correlation_id)
        try:
            await self._publish_command(command)
            terminal = await asyncio.wait_for(
                queue.get(), timeout=command.timeout_seconds
            )
        except TimeoutError:
            return {
                "status": "timeout",
                "correlation_id": str(correlation_id),
                "error_message": (
                    f"timed out after {command.timeout_seconds:g}s waiting for "
                    "delegation result"
                ),
            }
        finally:
            await unsubscribe()

        result: dict[str, object] = {
            "status": terminal.status,
            "correlation_id": str(correlation_id),
        }
        if terminal.error_message:
            result["error_message"] = terminal.error_message
        if isinstance(terminal.payload, dict):
            result.update(_flatten_terminal_payload(terminal.payload))
        return result

    async def _publish_command(self, command: ModelDispatchBusCommand) -> None:
        envelope = ModelEventEnvelope[ModelDispatchBusCommand](
            payload=command,
            correlation_id=command.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=self._command_topic,
            source_tool=_REQUESTER,
        )
        await self._event_bus.publish(
            self._command_topic,
            None,
            envelope.model_dump_json(exclude_none=True).encode("utf-8"),
            None,
        )

    async def _subscribe_for_result(
        self, dispatch_correlation_id: UUID
    ) -> tuple[
        Callable[[], Awaitable[None]], asyncio.Queue[ModelDispatchBusTerminalResult]
    ]:
        queue: asyncio.Queue[ModelDispatchBusTerminalResult] = asyncio.Queue()

        async def on_message(message: object) -> None:
            value = _message_value(message)
            if not isinstance(value, (bytes, str)):
                return
            try:
                envelope = ModelEventEnvelope[
                    ModelDispatchBusTerminalResult
                ].model_validate_json(value)
            except ValueError:
                return
            if envelope.payload.correlation_id != dispatch_correlation_id:
                return
            await queue.put(envelope.payload)

        unsubscribe = await self._event_bus.subscribe(
            self._response_topic,
            None,
            on_message,
            group_id=f"delegate-skill-runtime-port-{dispatch_correlation_id.hex}",
        )
        return unsubscribe, queue


def _message_value(message: object) -> object:
    raw = getattr(message, "value", None)
    if isinstance(raw, bytearray):
        return bytes(raw)
    return raw


def _flatten_terminal_payload(payload: Mapping[str, object]) -> dict[str, object]:
    nested_payload = payload.get("payload")
    if isinstance(nested_payload, dict):
        flattened = dict(nested_payload)
        topic = payload.get("topic")
        if isinstance(topic, str) and topic:
            flattened["terminal_topic"] = topic
        return flattened
    return dict(payload)


def _load_topic_defaults() -> tuple[str, str]:
    runtime_client = import_module("omnimarket.adapters.codex.runtime_client")
    default_command_topic = cast(
        "Callable[[], str]", runtime_client.__dict__["default_command_topic"]
    )
    default_response_topic = cast(
        "Callable[[], str]", runtime_client.__dict__["default_response_topic"]
    )
    return default_command_topic(), default_response_topic()


__all__ = ["RuntimeDelegationDispatchPort"]
