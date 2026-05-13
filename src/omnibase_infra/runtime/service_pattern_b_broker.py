# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pattern B broker process for contract-driven command to terminal-result flow."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from uuid import UUID

from aiokafka import AIOKafkaConsumer

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

_LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class TerminalPayload:
    payload: object
    topic: str


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
        command_envelope = ModelEventEnvelope[object].model_validate_json(message.value)
        command = self._decode_dispatch_command_payload(command_envelope.payload)
        _route, result = await self.dispatch_request(command)
        await self._publish_terminal_result(command.response_topic, result)

    @staticmethod
    def _decode_dispatch_command_payload(payload: object) -> ModelDispatchBusCommand:
        if isinstance(payload, dict):
            command_payload = dict(payload)
            if (
                "target_runtime_address" in command_payload
                and "target_runtime_address" not in ModelDispatchBusCommand.model_fields
            ):
                command_payload.pop("target_runtime_address")
            return ModelDispatchBusCommand.model_validate(command_payload)

        return ModelDispatchBusCommand.model_validate(payload)

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

        terminal_topics = _terminal_topics(route)
        if not terminal_topics:
            return route, _error_result(
                correlation_id,
                status="failed",
                error_message=f"Route '{command.command_name}' does not declare terminal events",
            )

        try:
            terminal = (
                await self._dispatch_and_wait_with_direct_kafka_consumer(command, route)
                if self._supports_direct_kafka_terminal_consumer()
                else await self._dispatch_and_wait_with_event_bus_subscription(
                    command,
                    route,
                )
            )
        except TimeoutError:
            return route, _error_result(
                correlation_id,
                status="timeout",
                error_message=(
                    "Timed out waiting for Pattern B broker terminal event."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return route, _error_result(
                correlation_id,
                status="failed",
                error_message=sanitize_error_message(exc),
            )

        status = _status_for_terminal_topic(route, terminal.topic)
        return route, ModelDispatchBusTerminalResult(
            correlation_id=correlation_id,
            status=status,
            payload=terminal.payload,
            error_message=(
                _terminal_error_message(terminal.payload)
                if status == "failed"
                else None
            ),
            completed_at=datetime.now(UTC),
        )

    async def _dispatch_and_wait_with_event_bus_subscription(
        self,
        command: ModelDispatchBusCommand,
        route: RuntimeLocalIngressRoute,
    ) -> TerminalPayload:
        correlation_id = command.correlation_id
        terminal_queue: asyncio.Queue[TerminalPayload] = asyncio.Queue(maxsize=1)
        terminal_topics = _terminal_topics(route)
        if not terminal_topics:
            raise RuntimeError(f"Route '{command.command_name}' has no terminal event")

        async def on_terminal(message: ModelEventMessage, topic: str) -> None:
            terminal_envelope = ModelEventEnvelope[object].model_validate_json(
                message.value
            )
            if terminal_envelope.correlation_id != correlation_id:
                return
            if terminal_queue.empty():
                await terminal_queue.put(
                    TerminalPayload(payload=terminal_envelope.payload, topic=topic)
                )

        def terminal_callback(
            topic: str,
        ) -> Callable[[ModelEventMessage], Awaitable[None]]:
            async def callback(message: ModelEventMessage) -> None:
                await on_terminal(message, topic)

            return callback

        unsubscribe_terminals: list[Callable[[], Awaitable[None]]] = []
        for topic in terminal_topics:
            unsubscribe = await self._event_bus.subscribe(
                topic,
                group_id=_terminal_group_id(correlation_id),
                on_message=terminal_callback(topic),
            )
            unsubscribe_terminals.append(
                cast("Callable[[], Awaitable[None]]", unsubscribe)
            )
        try:
            await self._publish_worker_command(command, route)
            return await asyncio.wait_for(
                terminal_queue.get(),
                timeout=command.timeout_seconds,
            )
        finally:
            for unsubscribe_terminal in unsubscribe_terminals:
                try:
                    await unsubscribe_terminal()
                except Exception:  # noqa: BLE001
                    pass

    async def _dispatch_and_wait_with_direct_kafka_consumer(
        self,
        command: ModelDispatchBusCommand,
        route: RuntimeLocalIngressRoute,
    ) -> TerminalPayload:
        terminal_topics = _terminal_topics(route)
        if not terminal_topics:
            raise RuntimeError(f"Route '{command.command_name}' has no terminal event")

        kafka_event_bus = self._kafka_event_bus()
        config = getattr(kafka_event_bus, "config", SimpleNamespace())
        consumer = AIOKafkaConsumer(
            *terminal_topics,
            bootstrap_servers=self._kafka_bootstrap_servers(),
            group_id=_terminal_group_id(command.correlation_id),
            auto_offset_reset="latest",
            enable_auto_commit=False,
            session_timeout_ms=getattr(config, "session_timeout_ms", 45000),
            heartbeat_interval_ms=getattr(config, "heartbeat_interval_ms", 15000),
            max_poll_interval_ms=getattr(config, "max_poll_interval_ms", 300000),
            retry_backoff_ms=getattr(config, "reconnect_backoff_ms", 2000),
            **self._direct_kafka_auth_kwargs(),
        )

        try:
            await asyncio.wait_for(
                consumer.start(), timeout=min(30, command.timeout_seconds)
            )
            await self._wait_for_kafka_assignment(consumer, command.timeout_seconds)
            await consumer.seek_to_end(*consumer.assignment())
            await self._publish_worker_command(command, route)

            deadline = asyncio.get_running_loop().time() + command.timeout_seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                message = await asyncio.wait_for(consumer.getone(), timeout=remaining)
                terminal_envelope = ModelEventEnvelope[object].model_validate_json(
                    message.value
                )
                if terminal_envelope.correlation_id == command.correlation_id:
                    topic = getattr(message, "topic", None)
                    if not isinstance(topic, str) or not topic:
                        topic = str(
                            terminal_envelope.event_type or route.terminal_event
                        )
                    return TerminalPayload(
                        payload=terminal_envelope.payload,
                        topic=topic,
                    )
        finally:
            try:
                await consumer.stop()
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "Failed to stop Pattern B terminal Kafka consumer: %s",
                    sanitize_error_message(exc),
                )

    async def _wait_for_kafka_assignment(
        self,
        consumer: AIOKafkaConsumer,
        timeout_seconds: int,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + min(30, timeout_seconds)
        while not consumer.assignment():
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError
            await asyncio.sleep(0.05)

    def _supports_direct_kafka_terminal_consumer(self) -> bool:
        return (
            hasattr(self._event_bus, "_bootstrap_servers")
            and hasattr(self._event_bus, "_build_auth_kwargs")
            and hasattr(self._event_bus, "config")
        )

    def _direct_kafka_auth_kwargs(self) -> dict[str, object]:
        build_auth_kwargs = getattr(  # noqa: B009
            self._kafka_event_bus(),
            "_build_auth_kwargs",
        )
        auth_kwargs = cast(
            "Callable[[], Mapping[str, object] | None]",
            build_auth_kwargs,
        )()
        return dict(auth_kwargs or {})

    def _kafka_bootstrap_servers(self) -> str:
        bootstrap_servers = getattr(  # noqa: B009
            self._kafka_event_bus(),
            "_bootstrap_servers",
        )
        if not isinstance(bootstrap_servers, str):
            raise RuntimeError("Kafka event bus bootstrap servers are not configured")
        return bootstrap_servers

    def _kafka_event_bus(self) -> object:
        return self._event_bus

    async def _publish_worker_command(
        self,
        command: ModelDispatchBusCommand,
        route: RuntimeLocalIngressRoute,
    ) -> None:
        worker_envelope = ModelEventEnvelope[object](
            payload=command.payload,
            correlation_id=command.correlation_id,
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


def _terminal_topics(route: RuntimeLocalIngressRoute) -> tuple[str, ...]:
    if route.terminal_events:
        return route.terminal_events
    if route.terminal_event is not None:
        return (route.terminal_event,)
    return ()


def _status_for_terminal_topic(route: RuntimeLocalIngressRoute, topic: str) -> str:
    success_topic = route.terminal_event
    if success_topic is None:
        terminal_topics = _terminal_topics(route)
        success_topic = terminal_topics[0] if terminal_topics else ""
    return "completed" if topic == success_topic else "failed"


def _terminal_error_message(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    nested_payload = payload.get("payload")
    if isinstance(nested_payload, dict):
        for key in ("failure_reason", "error_message", "error"):
            value = nested_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    for key in ("failure_reason", "error_message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


__all__ = ["RuntimePatternBBroker"]
