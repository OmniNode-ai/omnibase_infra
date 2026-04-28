# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pattern B broker process for contract-driven command to terminal-result flow."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from uuid import UUID

from omnibase_core.models.dispatch.model_dispatch_bus_command import (
    ModelDispatchBusCommand,
)
from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.runtime.runtime_local_ingress import RuntimeLocalIngressRoute
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message


def _broker_group_id(command_topic: str) -> str:
    normalized = command_topic.replace(".", "-")
    return f"pattern-b-broker-{normalized}"


def _terminal_group_id(correlation_id: UUID) -> str:
    return f"pattern-b-broker-terminal-{correlation_id}"


def _error_result(
    correlation_id: UUID,
    *,
    status: str,
    error_message: str,
) -> ModelDispatchBusTerminalResult:
    return ModelDispatchBusTerminalResult(
        correlation_id=correlation_id,
        status=status,
        error_message=error_message,
        completed_at=datetime.now(UTC),
    )


class RuntimePatternBBroker:
    """Long-lived broker that normalizes worker terminal events for external CLIs."""

    def __init__(
        self,
        event_bus: ProtocolPatternBBrokerTransport,
        *,
        command_topic: str,
        routes: Mapping[str, RuntimeLocalIngressRoute],
    ) -> None:
        self._event_bus = event_bus
        self._command_topic = command_topic
        self._routes = dict(routes)
        self._unsubscribe: Callable[[], Awaitable[None]] | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def is_running(self) -> bool:
        return self._unsubscribe is not None

    async def start(self) -> None:
        if self._unsubscribe is not None:
            return

        async def on_command(message: ModelEventMessage) -> None:
            task = asyncio.create_task(self._handle_command_message(message))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._unsubscribe = await self._event_bus.subscribe(
            self._command_topic,
            group_id=_broker_group_id(self._command_topic),
            on_message=on_command,
        )

    async def stop(self) -> None:
        if self._unsubscribe is not None:
            await self._unsubscribe()
            self._unsubscribe = None
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()

    async def _handle_command_message(self, message: ModelEventMessage) -> None:
        command_envelope = ModelEventEnvelope[
            ModelDispatchBusCommand
        ].model_validate_json(message.value)
        command = command_envelope.payload
        _route, result = await self.dispatch_request(command)
        await self._publish_terminal_result(command.response_topic, result)

    async def dispatch_request(
        self,
        command: ModelDispatchBusCommand,
    ) -> tuple[RuntimeLocalIngressRoute | None, ModelDispatchBusTerminalResult]:
        correlation_id = command.correlation_id
        route = self._routes.get(command.command_name)
        if route is None:
            return None, _error_result(
                correlation_id,
                status="failed",
                error_message=f"Unknown Pattern B route '{command.command_name}'",
            )

        if route.terminal_event is None:
            return route, _error_result(
                correlation_id,
                status="failed",
                error_message=(
                    f"Route '{command.command_name}' does not declare a terminal_event"
                ),
            )

        terminal_queue: asyncio.Queue[object] = asyncio.Queue(maxsize=1)

        async def on_terminal(message: ModelEventMessage) -> None:
            terminal_envelope = ModelEventEnvelope[object].model_validate_json(
                message.value
            )
            if terminal_envelope.correlation_id != correlation_id:
                return
            if terminal_queue.empty():
                await terminal_queue.put(terminal_envelope.payload)

        unsubscribe_terminal = await self._event_bus.subscribe(
            route.terminal_event,
            group_id=_terminal_group_id(correlation_id),
            on_message=on_terminal,
        )
        try:
            worker_envelope = ModelEventEnvelope[object](
                payload=command.payload,
                correlation_id=correlation_id,
                envelope_timestamp=datetime.now(UTC),
                event_type=route.event_type,
                source_tool="pattern-b-broker",
                target_tool=route.contract_name,
            )
            await self._event_bus.publish(
                route.command_topic,
                None,
                worker_envelope.model_dump_json().encode("utf-8"),
                None,
            )
            try:
                terminal_payload = await asyncio.wait_for(
                    terminal_queue.get(),
                    timeout=command.timeout_seconds,
                )
            except TimeoutError:
                return route, _error_result(
                    correlation_id,
                    status="timeout",
                    error_message=(
                        "Timed out waiting for Pattern B broker terminal event."
                    ),
                )

            return route, ModelDispatchBusTerminalResult(
                correlation_id=correlation_id,
                status="completed",
                payload=terminal_payload,
                completed_at=datetime.now(UTC),
            )
        except Exception as exc:  # noqa: BLE001
            return route, _error_result(
                correlation_id,
                status="failed",
                error_message=sanitize_error_message(exc),
            )
        finally:
            try:
                await unsubscribe_terminal()
            except Exception:  # noqa: BLE001
                pass

    async def _publish_terminal_result(
        self,
        response_topic: str,
        result: ModelDispatchBusTerminalResult,
    ) -> None:
        envelope = ModelEventEnvelope[ModelDispatchBusTerminalResult](
            payload=result,
            correlation_id=result.correlation_id,
            envelope_timestamp=datetime.now(UTC),
            event_type=response_topic,
            source_tool="pattern-b-broker",
            target_tool="pattern-b-client",
            payload_type=ModelDispatchBusTerminalResult.__name__,
        )
        await self._event_bus.publish(
            response_topic,
            None,
            envelope.model_dump_json().encode("utf-8"),
            None,
        )


__all__ = ["RuntimePatternBBroker"]
