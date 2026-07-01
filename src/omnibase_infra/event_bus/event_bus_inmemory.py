# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Thin infra adapter over the canonical core in-memory event bus.

The single canonical ``EventBusInmemory`` transport lives in
``omnibase_core.event_bus.event_bus_inmemory`` (OMN-7062/OMN-7077). The
861-line infra duplicate that previously lived here has been deleted
(OMN-13419) so there is exactly one in-memory transport implementation.

This module is a *thin* adapter, NOT a second implementation. It subclasses
the core bus, inheriting all of its state management (deque history, async
lock, topic offsets, subscriber registry, per-subscriber circuit-breaker
accounting). It re-expresses only the surface that the infra runtime and
``ProtocolEventBusLike`` genuinely require and that the pinned core release
does not yet provide:

1. Infra-typed messages — the canonical core bus builds
   ``omnibase_core`` ``ModelEventMessage``/``ModelEventHeaders`` whose
   ``schema_version`` is a ``ModelSemVer``; the infra runtime, Kafka bus, node
   handlers, and 50+ tests are typed against the infra
   ``ModelEventMessage`` whose ``schema_version`` is a ``str``. The adapter
   overrides ``publish`` to build and deliver the infra message types so every
   downstream consumer receives exactly the model it always has (one boundary
   instead of coercion scattered across every handler/projection).
2. Infra error taxonomy — the infra kernel, dispatchers, and error-handling
   tests are typed against ``InfraUnavailableError`` /
   ``ProtocolConfigurationError`` (the core bus raises ``ModelOnexError``).
3. Dict-shaped ``health_check()`` — ``RuntimeHostProcess.health_check()`` and
   ``ServiceRuntimeHealthMonitor`` read ``health["started"]`` /
   ``health["subscriber_count"]`` etc.; the core bus returns a
   ``TypedDictEventBusHealth`` with only ``healthy``/``connected``/``status``.
4. ``get_consumer_groups()`` — part of ``ProtocolEventBusLike`` (alongside
   ``EventBusKafka``) and consumed by ``ServiceRuntimeHealthMonitor``; absent
   on the pinned core bus.

Everything else — ``subscribe``/``unsubscribe``, history, offsets, lifecycle,
the circuit-breaker manager methods — is inherited unchanged from the core
bus. When the core ``ModelEventHeaders``/``ModelEventMessage`` model shape,
``health_check`` shape, and ``get_consumer_groups`` land in a released core
version, this adapter collapses to a bare re-export and can be deleted.

Protocol Compatibility:
    ProtocolEventBus / ProtocolEventBusLike via duck typing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from omnibase_core.enums.enum_consumer_group_purpose import (
    EnumConsumerGroupPurpose as _CoreEnumConsumerGroupPurpose,
)
from omnibase_core.event_bus.event_bus_inmemory import (
    EventBusInmemory as _CoreEventBusInmemory,
)
from omnibase_core.models.event_bus.model_event_message import (
    ModelEventMessage as _CoreModelEventMessage,
)
from omnibase_infra.enums import EnumConsumerGroupPurpose, EnumInfraTransportType
from omnibase_infra.errors import (
    InfraUnavailableError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
)
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus.protocol_node_identity import (
        ProtocolNodeIdentity as _CoreProtocolNodeIdentity,
    )

logger = logging.getLogger(__name__)


class EventBusInmemory(_CoreEventBusInmemory):
    """Infra-facing adapter over the canonical core in-memory event bus.

    Inherits all transport behavior from
    ``omnibase_core.event_bus.event_bus_inmemory.EventBusInmemory`` and only
    overrides the infra-contract boundary (error taxonomy, dict-shaped
    ``health_check``, consumer-group introspection).
    """

    def __init__(
        self,
        environment: str = "local",
        group: str = "default",
        max_history: int = 1000,
        circuit_breaker_threshold: int = 5,
    ) -> None:
        """Initialize the bus, raising the infra error type on bad config."""
        if circuit_breaker_threshold < 1:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INMEMORY,
                operation="init",
                target_name="inmemory_event_bus",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"circuit_breaker_threshold must be a positive integer, "
                f"got {circuit_breaker_threshold}",
                context=context,
                parameter="circuit_breaker_threshold",
                value=circuit_breaker_threshold,
            )
        super().__init__(
            environment=environment,
            group=group,
            max_history=max_history,
            circuit_breaker_threshold=circuit_breaker_threshold,
        )

    @property
    def adapter(self) -> EventBusInmemory:
        """No separate adapter for in-memory -- returns self."""
        return self

    async def subscribe(  # type: ignore[override]
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
    ) -> Callable[[], Awaitable[None]]:
        """Subscribe with infra-typed identity/enum/callback.

        Re-declares the inherited core ``subscribe`` with the infra
        ``ModelNodeIdentity`` / ``EnumConsumerGroupPurpose`` / infra-message
        callback so infra call sites type-check. The core implementation only
        reads ``identity`` attributes and ``purpose.value`` and stores the
        callback opaquely, so delegation is behavior-preserving (OMN-13419).
        """
        return await super().subscribe(
            topic,
            cast("_CoreProtocolNodeIdentity | None", node_identity),
            cast(
                "Callable[[_CoreModelEventMessage], Awaitable[None]] | None",
                on_message,
            ),
            group_id=group_id,
            purpose=cast("_CoreEnumConsumerGroupPurpose", purpose),
            required_for_readiness=required_for_readiness,
        )

    async def publish(  # type: ignore[override]
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> None:
        """Publish a message, delivering infra-typed ``ModelEventMessage``.

        Overrides the core publish so the bus emits the infra
        ``ModelEventMessage``/``ModelEventHeaders`` (``schema_version: str``)
        that the infra runtime, node handlers, and tests expect, instead of the
        core models (``schema_version: ModelSemVer``). State (offsets, history,
        per-subscriber circuit breaker) is the inherited core state; only the
        message construction and the not-started error type differ.
        """
        if not self._started:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INMEMORY,
                operation="publish",
                target_name=f"event_bus.{self._environment}",
                correlation_id=(
                    headers.correlation_id if headers is not None else uuid4()
                ),
            )
            raise InfraUnavailableError(
                "Event bus not started. Call start() first.",
                context=context,
                topic=topic,
            )

        if headers is None:
            headers = ModelEventHeaders(
                source=f"{self._environment}.{self._group}",
                event_type=topic,
                timestamp=datetime.now(UTC),
            )

        async with self._lock:
            offset = self._topic_offsets[topic]
            self._topic_offsets[topic] = offset + 1

            message = ModelEventMessage(
                topic=topic,
                key=key,
                value=value,
                headers=headers,
                offset=str(offset),
                partition=0,
            )
            # OMN-13419: the inherited core deque is typed for the core
            # ModelEventMessage; the adapter intentionally stores the infra-typed
            # message (same structural shape, str schema_version) so consumers
            # receive the infra model. Drops to a re-export once core aligns.
            self._event_history.append(message)  # type: ignore[arg-type]
            subscribers = list(self._subscribers.get(topic, []))

        # Deliver outside the lock; mirror the inherited circuit-breaker policy.
        for group_id, callback in subscribers:
            failure_key = (topic, group_id)

            async with self._lock:
                failure_count = self._subscriber_failures.get(failure_key, 0)

            if failure_count >= self._max_consecutive_failures:
                logger.warning(
                    "Subscriber circuit breaker open - skipping callback",
                    extra={
                        "topic": topic,
                        "group_id": group_id,
                        "consecutive_failures": failure_count,
                        "correlation_id": str(headers.correlation_id),
                    },
                )
                continue

            try:
                # Callback typed for core ModelEventMessage; infra-typed
                # message is structurally identical (OMN-13419).
                await callback(message)  # type: ignore[arg-type]
                async with self._lock:
                    if failure_key in self._subscriber_failures:
                        del self._subscriber_failures[failure_key]
            except Exception as e:
                async with self._lock:
                    self._subscriber_failures[failure_key] = (
                        self._subscriber_failures.get(failure_key, 0) + 1
                    )
                    current_failure_count = self._subscriber_failures[failure_key]
                logger.exception(
                    "Subscriber callback failed",
                    extra={
                        "topic": topic,
                        "group_id": group_id,
                        "error": str(e),
                        "consecutive_failures": current_failure_count,
                        "correlation_id": str(headers.correlation_id),
                    },
                )

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        """Serialize an envelope to JSON bytes and publish via infra ``publish``.

        Mirrors the core serialization contract but builds infra-typed headers
        and routes through this class's ``publish`` so an infra
        ``ModelEventMessage`` is delivered, and re-expresses serialization
        failures as the infra ``ProtocolConfigurationError``.
        """
        envelope_dict: object
        if hasattr(envelope, "model_dump"):
            envelope_dict = envelope.model_dump(mode="json")
        elif hasattr(envelope, "dict"):
            envelope_dict = envelope.dict()
        elif isinstance(envelope, dict):
            envelope_dict = envelope
        else:
            envelope_dict = envelope

        try:
            value = json.dumps(envelope_dict).encode("utf-8")
        except TypeError as e:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.INMEMORY,
                operation="publish_envelope",
                target_name=f"event_bus.{self._environment}",
                correlation_id=uuid4(),
            )
            raise ProtocolConfigurationError(
                f"Envelope is not JSON-serializable: {e}. "
                f"Ensure envelope is a Pydantic model (with model_dump), dict, "
                f"or JSON-compatible primitive. Got type: "
                f"{type(envelope).__name__}",
                context=context,
                parameter="envelope",
                value=str(type(envelope)),
            ) from e

        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type=topic,
            content_type="application/json",
            timestamp=datetime.now(UTC),
        )
        await self.publish(topic, key, value, headers)

    async def broadcast_to_environment(
        self,
        command: str,
        payload: dict[str, object],
        target_environment: str | None = None,
    ) -> None:
        """Broadcast a command to an environment using infra-typed headers."""
        env = target_environment or self._environment
        topic = f"{env}.broadcast"
        value = json.dumps({"command": command, "payload": payload}).encode("utf-8")
        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type="broadcast",
            content_type="application/json",
            timestamp=datetime.now(UTC),
        )
        await self.publish(topic, None, value, headers)

    async def send_to_group(
        self,
        command: str,
        payload: dict[str, object],
        target_group: str,
    ) -> None:
        """Send a command to a specific group using infra-typed headers."""
        topic = f"{self._environment}.{target_group}"
        value = json.dumps({"command": command, "payload": payload}).encode("utf-8")
        headers = ModelEventHeaders(
            source=f"{self._environment}.{self._group}",
            event_type="group_command",
            content_type="application/json",
            timestamp=datetime.now(UTC),
        )
        await self.publish(topic, None, value, headers)

    async def health_check(self) -> dict[str, object]:  # type: ignore[override]
        """Return infra dict-shaped health (consumed by the runtime kernel).

        ``RuntimeHostProcess.health_check()`` and
        ``ServiceRuntimeHealthMonitor`` read ``started`` / ``subscriber_count``
        etc., so the adapter preserves the historical infra dict shape rather
        than the core ``TypedDictEventBusHealth``.
        """
        async with self._lock:
            subscriber_count = sum(len(subs) for subs in self._subscribers.values())
            topic_count = len(self._subscribers)
            history_size = len(self._event_history)

        return {
            "healthy": self._started,
            "started": self._started,
            "environment": self._environment,
            "group": self._group,
            "subscriber_count": subscriber_count,
            "topic_count": topic_count,
            "history_size": history_size,
        }

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        """Return active topic/group keys mapped to effective consumer group IDs.

        Part of ``ProtocolEventBusLike`` (shared with ``EventBusKafka``) and
        consumed by ``ServiceRuntimeHealthMonitor``; absent on the pinned core
        bus so it is re-expressed here over the inherited subscriber state.
        """
        return {
            (topic, group_id): group_id
            for topic, subscribers in self._subscribers.items()
            for group_id, _callback in subscribers
        }


__all__: list[str] = ["EventBusInmemory", "ModelEventHeaders", "ModelEventMessage"]
