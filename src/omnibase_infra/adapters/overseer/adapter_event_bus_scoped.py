# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope-enforcing wrapper adapter for ProtocolEventBusLike.

Wraps any ProtocolEventBusLike implementation with an allowed-action set.
Any call to a guarded action not in the set raises InvariantViolation before
the underlying bus is reached.

Related Tickets:
    - OMN-8065: Task 7 — Scoped wrapper adapters for TicketService, EventBus, LLMProvider
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from omnibase_infra.errors.error_invariant_violation import InvariantViolation
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike

_PROTOCOL_DOMAIN = "event_bus"

_ACTION_PUBLISH = "publish"
_ACTION_PUBLISH_ENVELOPE = "publish_envelope"
_ACTION_SUBSCRIBE = "subscribe"
_ACTION_CLOSE = "close"


class AdapterEventBusScoped:
    """Scope-enforcing wrapper around a ProtocolEventBusLike implementation.

    Delegates publish and subscribe calls to the inner bus after checking
    the allowed-action set.  Raises InvariantViolation on any disallowed
    action before any I/O occurs.

    Args:
        inner: Any object satisfying ProtocolEventBusLike.
        allowed_actions: Frozenset of action name strings that callers may
            invoke.  Pass an empty frozenset to deny all actions.
    """

    def __init__(
        self,
        inner: ProtocolEventBusLike,
        allowed_actions: frozenset[str],
    ) -> None:
        self._inner = inner
        self._allowed_actions = allowed_actions

    def _check(self, action_name: str) -> None:
        if action_name not in self._allowed_actions:
            raise InvariantViolation(
                action_name=action_name,
                protocol_domain=_PROTOCOL_DOMAIN,
                allowed_actions=tuple(sorted(self._allowed_actions)),
            )

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        self._check(_ACTION_PUBLISH)
        await self._inner.publish(topic=topic, key=key, value=value)

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        self._check(_ACTION_PUBLISH_ENVELOPE)
        await self._inner.publish_envelope(envelope, topic, key=key)

    async def subscribe(
        self,
        topic: str,
        identity: object,
        handler: Callable[[object], Awaitable[None]],
    ) -> Callable[[], Awaitable[None]]:
        self._check(_ACTION_SUBSCRIBE)
        inner = self._inner
        # Why: Optional dependency or runtime adapter exposes this attribute dynamically.
        return await inner.subscribe(topic, identity, handler)  # type: ignore[attr-defined, no-any-return]

    async def close(self) -> None:
        self._check(_ACTION_CLOSE)
        # Why: Optional dependency or runtime adapter exposes this attribute dynamically.
        await self._inner.close()  # type: ignore[attr-defined]


__all__: list[str] = ["AdapterEventBusScoped"]
