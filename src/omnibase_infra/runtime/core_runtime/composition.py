# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Composition-root orchestration for the S6 core runtime (epic OMN-14717, §c).

Ties the S6 pieces together behind the ``ONEX_CORE_RUNTIME_TOPICS`` allowlist:

* :func:`parse_core_runtime_topics` — read the comma-separated allowlist (DEFAULT EMPTY
  ⇒ ``build_core_runtime`` is never called and there is ZERO behavior change; the legacy
  kernel owns everything — the single-lever rollback state).
* :func:`build_core_runtime` — build the routing map (§a), the real DLQ resolver (§b),
  assert the single-owner split (§c.3), wrap the transport with the phantom alarm (§d),
  and construct ``RuntimeDispatch``. Returns a :class:`CoreRuntimeHandle` that owns the
  supervised loop lifecycle (§c.5).

The transport (S3 ``KafkaTransport`` for prod, S2 ``InMemoryTransport`` for local/CI) is
constructed by the CALLER — the composition root reuses the already-resolved event-bus
config and does not re-probe. ONE transport instance is BOTH ``consumer`` and
``producer``; the phantom monitor's counting decorator wraps only the consumer face.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from omnibase_core.protocols.runtime.protocol_transport_message import (
    ProtocolTransportMessage,
)
from omnibase_core.runtime.runtime_dispatch import RuntimeDispatch
from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.core_runtime.dlq_resolver import (
    build_delegation_dlq_resolver,
    load_contract_dlq_topics,
)
from omnibase_infra.runtime.core_runtime.phantom_alarm import PhantomAlarmMonitor
from omnibase_infra.runtime.core_runtime.routing_map_builder import (
    HandlerResolver,
    ModelResolver,
    build_routing_map,
    import_model_cls,
)
from omnibase_infra.runtime.core_runtime.single_owner import assert_single_owner_split
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)

logger = logging.getLogger(__name__)

CORE_RUNTIME_TOPICS_ENV = "ONEX_CORE_RUNTIME_TOPICS"

__all__ = [
    "CORE_RUNTIME_TOPICS_ENV",
    "CoreRuntimeHandle",
    "CoreTransport",
    "build_core_runtime",
    "parse_core_runtime_topics",
]


class CoreTransport(Protocol):
    """Structural transport that is BOTH consumer and producer (KafkaTransport /
    InMemoryTransport). One instance serves ``RuntimeDispatch(consumer=, producer=)``."""

    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def poll(
        self, *, max_messages: int, timeout_ms: int
    ) -> Sequence[ProtocolTransportMessage]: ...
    async def commit(self, message: object) -> None: ...
    async def nack(self, message: object) -> None: ...
    async def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Mapping[str, bytes],
    ) -> None: ...


def parse_core_runtime_topics(env: Mapping[str, str]) -> frozenset[str]:
    """Parse ``ONEX_CORE_RUNTIME_TOPICS`` into a topic allowlist (default EMPTY).

    Comma-separated, whitespace-stripped, empties dropped. An unset or blank value
    yields ``frozenset()`` — the strict no-op / rollback state.
    """
    raw = env.get(CORE_RUNTIME_TOPICS_ENV, "")
    topics = {piece.strip() for piece in raw.split(",")}
    return frozenset(t for t in topics if t)


@dataclass
class CoreRuntimeHandle:
    """Owns the constructed ``RuntimeDispatch`` and its supervised loop lifecycle (§c.5)."""

    dispatch: RuntimeDispatch
    transport: object
    monitor: PhantomAlarmMonitor
    dlq_provision_topics: frozenset[str]
    core_runtime_topics: frozenset[str]
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    def start(self) -> asyncio.Task[None]:
        """Start the supervised dispatch loop as a background task."""
        if self._task is not None:
            return self._task
        self._task = asyncio.create_task(
            self.dispatch.run(), name="onex-core-runtime-dispatch"
        )
        logger.info(
            "S6 core runtime: dispatch loop started for topics=%s",
            sorted(self.core_runtime_topics),
        )
        return self._task

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def is_loop_healthy(self) -> bool:
        """Readiness tri-state input: a crashed loop task is a health failure (§c.5).

        A task that finished with an exception (or finished at all while we did not ask
        it to stop) is unhealthy. A live, running task is healthy.
        """
        if self._task is None:
            return False
        if not self._task.done():
            return True
        # Done unexpectedly: surface the exception if any.
        exc = self._task.exception() if not self._task.cancelled() else None
        if exc is not None:
            logger.error(
                "S6 core runtime: dispatch loop crashed: %r — runtime readiness FAIL.",
                exc,
            )
        return False

    async def stop(self) -> None:
        """Stop the loop cleanly and close the transport (§c.5 shutdown)."""
        self.dispatch.stop()
        if self._task is not None:
            try:
                await self._task
            except asyncio.CancelledError:  # boundary-ok: cleanup absorbs cancellation
                pass
            except Exception:  # noqa: BLE001 — cleanup-resilience-ok: never fail shutdown
                logger.warning(
                    "S6 core runtime: dispatch loop raised during stop (non-fatal).",
                    exc_info=True,
                )
            finally:
                self._task = None
        # run() closes the consumer face already; close defensively (idempotent).
        close = getattr(self.transport, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001 — cleanup-resilience-ok
                logger.warning(
                    "S6 core runtime: transport close raised during stop (non-fatal).",
                    exc_info=True,
                )


def build_core_runtime(
    *,
    core_runtime_topics: frozenset[str],
    contracts: Sequence[ModelDiscoveredContract],
    transport: CoreTransport,
    handler_resolver: HandlerResolver,
    legacy_subscribed_topics: frozenset[str],
    model_resolver: ModelResolver = import_model_cls,
    published_events_loader: Callable[
        [Path], Mapping[str, str]
    ] = load_published_events_map,
    contract_dlq_loader: Callable[[Path], Sequence[str]] = load_contract_dlq_topics,
) -> CoreRuntimeHandle:
    """Build the S6 ``RuntimeDispatch`` for the allowlisted topics (§c.1-c.3, §d).

    Precondition: ``core_runtime_topics`` is non-empty (the caller must short-circuit on
    the empty allowlist so the empty path constructs NOTHING). ``transport`` is the
    constructed transport instance that is BOTH consumer and producer.
    """
    if not core_runtime_topics:
        raise ValueError(
            "build_core_runtime called with an empty allowlist; the caller must "
            "short-circuit on the empty allowlist (zero behavior change)."
        )

    routing_map = build_routing_map(
        contracts,
        core_runtime_topics,
        handler_resolver=handler_resolver,
        model_resolver=model_resolver,
        published_events_loader=published_events_loader,
    )
    dlq_resolver, dlq_provision_topics = build_delegation_dlq_resolver(
        contracts,
        core_runtime_topics,
        contract_dlq_loader=contract_dlq_loader,
    )
    assert_single_owner_split(
        core_runtime_topics=core_runtime_topics,
        routing_map=routing_map,
        legacy_subscribed_topics=legacy_subscribed_topics,
        contracts=contracts,
    )

    monitor = PhantomAlarmMonitor(
        transport,
        core_runtime_topics=core_runtime_topics,
    )
    # The counting consumer is a valid ProtocolTransportConsumer; RuntimeDispatch's private
    # structural mirror narrows the message type, so present it via the CoreTransport face
    # (consumer methods) which is structurally what RuntimeDispatch consumes.
    dispatch = RuntimeDispatch(
        consumer=cast("CoreTransport", monitor.consumer),
        producer=transport,
        routing_map=routing_map,
        dlq_topic_resolver=dlq_resolver,
    )
    logger.info(
        "S6 core runtime built: topics=%s dlq_provision=%s",
        sorted(core_runtime_topics),
        sorted(dlq_provision_topics),
    )
    return CoreRuntimeHandle(
        dispatch=dispatch,
        transport=transport,
        monitor=monitor,
        dlq_provision_topics=dlq_provision_topics,
        core_runtime_topics=core_runtime_topics,
    )
