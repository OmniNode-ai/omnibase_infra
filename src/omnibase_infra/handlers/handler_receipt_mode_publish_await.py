# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21341: synchronous receipt-mode bus client for the Claude Code hook.

Exports HandlerReceiptModePublishAwait; its event model is in models/. The hook adapter loads it by the module:attribute spec
omnibase_infra.handlers.handler_receipt_mode_publish_await:HandlerReceiptModePublishAwait.

The hook loads this client without arguments. Lane resolution and credentials
stay in infra; the caller receives only correlated topic/payload pairs without
an import dependency on omnimarket.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from functools import partial
from typing import cast
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter, ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.cli.delegate_lane import DelegateLaneSelectionError
from omnibase_infra.cli.delegate_lane_credentials import (
    DelegateLaneCredentialError,
    resolve_lane_binding,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bind_lane_client_transport,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.model_receipt_mode_event import ModelReceiptModeEvent

logger = logging.getLogger(__name__)
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class HandlerReceiptModePublishAwait:
    """Publish one command and collect its replies until an action or deadline."""

    def __init__(
        self,
        bus_factory: Callable[[], EventBusKafka] | None = None,
        *,
        lane: str = "dev",
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._bus_factory = bus_factory
        self._lane = lane
        # None lets resolve_lane_binding read the process environment, the
        # source a zero-argument hook loader has.
        self._environ = environ

    @property
    def handler_type(self) -> EnumHandlerType:
        """Return the architectural role of this handler.

        Returns:
            EnumHandlerType.INFRA_HANDLER - Infrastructure transport handler
            exchanging one command and its replies over the event bus.
        """
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Return the behavioral classification of this handler.

        Returns:
            EnumHandlerTypeCategory.EFFECT - Side-effecting I/O operations
            (a bus publish and subscriptions).
        """
        return EnumHandlerTypeCategory.EFFECT

    @asynccontextmanager
    async def _bus(self) -> AsyncIterator[EventBusKafka]:
        if self._bus_factory is not None:
            yield self._bus_factory()
            return

        try:
            selection, transport = resolve_lane_binding(
                self._lane, environ=self._environ
            )
        except (DelegateLaneSelectionError, DelegateLaneCredentialError) as exc:
            raise RuntimeError(str(exc)) from exc

        # Kafka resolves the transport during its lifecycle, not just construction.
        with (
            bind_lane_client_transport(transport)
            if transport is not None
            else nullcontext()
        ):
            yield EventBusKafka.from_bootstrap(selection.bootstrap_servers)

    def publish_and_await(
        self,
        *,
        topic: str,
        command_type: str,
        payload: dict[str, JsonValue],
        awaited_events: tuple[str, ...],
        deadline_ms: int,
    ) -> tuple[ModelReceiptModeEvent, ...]:
        """Run one async exchange; the deadline includes bus startup."""
        expires_at = time.monotonic() + deadline_ms / 1000
        return asyncio.run(
            self._publish_and_await(
                topic=topic,
                command_type=command_type,
                payload=payload,
                awaited_events=awaited_events,
                expires_at=expires_at,
            )
        )

    async def _publish_and_await(
        self,
        *,
        topic: str,
        command_type: str,
        payload: dict[str, JsonValue],
        awaited_events: tuple[str, ...],
        expires_at: float,
    ) -> tuple[ModelReceiptModeEvent, ...]:
        correlation_id = uuid4()
        invocation_id = payload.get("invocation_id")
        envelope = ModelEventEnvelope[dict[str, object]](
            payload=dict(payload),
            correlation_id=correlation_id,
            event_type=command_type,
            # The hook command belongs to the operator's own session; no tenant.
            tenant_id=None,
        )
        events: list[ModelReceiptModeEvent] = []
        finished = asyncio.Event()

        async def receive(subscribed_topic: str, message: ModelEventMessage) -> None:
            if finished.is_set() or time.monotonic() >= expires_at:
                return
            try:
                incoming = _JSON_OBJECT.validate_json(message.value)
            except ValidationError:
                logger.debug("Ignoring malformed receipt-mode event")
                return
            reply = incoming.get("payload")
            if not isinstance(reply, dict):
                return
            correlated = str(incoming.get("correlation_id")) == str(correlation_id)
            echoed = (
                "invocation_id" in payload
                and "invocation_id" in reply
                and reply.get("invocation_id") == invocation_id
            )
            if not (correlated or echoed):
                return
            events.append(ModelReceiptModeEvent(topic=subscribed_topic, payload=reply))
            if "action" in reply:
                finished.set()

        async with self._bus() as bus, AsyncExitStack() as cleanup:
            # The stack runs every unsubscribe and closes even if cleanup raises.
            cleanup.push_async_callback(bus.close)
            timeout = asyncio.timeout(max(0, expires_at - time.monotonic()))
            try:
                async with timeout:
                    await bus.start()
                    group_id = f"receipt-mode-client.{uuid4().hex[:12]}"
                    supports_latest = (
                        "auto_offset_reset"
                        in inspect.signature(bus.subscribe).parameters
                    )
                    for awaited_topic in awaited_events:
                        callback = partial(receive, awaited_topic)
                        if supports_latest:
                            # Signature inspection establishes this transport extension.
                            subscribe = cast(
                                "Callable[..., Awaitable[Callable[[], Awaitable[None]]]]",
                                bus.subscribe,
                            )
                            unsubscribe = await subscribe(
                                awaited_topic,
                                on_message=callback,
                                group_id=group_id,
                                auto_offset_reset="latest",
                            )
                        else:
                            unsubscribe = await bus.subscribe(
                                awaited_topic, on_message=callback, group_id=group_id
                            )
                        cleanup.push_async_callback(unsubscribe)
                    await bus.publish(
                        topic,
                        key=str(correlation_id).encode("utf-8"),
                        value=json.dumps(envelope.model_dump(mode="json")).encode(),
                    )
                    await finished.wait()
            except TimeoutError:
                if not timeout.expired():
                    raise
        return tuple(events)
