# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998 -- the runtime error triage node reaches its database pool through
the runtime's own wiring path.

``node_runtime_error_triage_effect`` declares ``db_pool`` required and its
handler defaults the parameter to ``None``. ``_build_runtime_handler_dependencies``
gave the pool to two other handlers and never to this one, so the auto-wired
handler was built with no pool and every runtime error it received logged
``db_pool not configured, suppressing event`` and was dropped (1,291 in 20
minutes on the dev runtime).

This drives the same path the kernel takes: discovery reads the real contract,
``wire_from_manifest`` builds the handler from the dependency map the kernel
builds, and an error event published on the contract's subscribe topic is
dispatched through the engine. The pool is a recording stand-in whose
``acquire`` raises, so a handler that reaches its pool is visible as a recorded
acquire and a handler built without one is visible as none.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import omnibase_infra.nodes.node_runtime_error_triage_effect as _triage_pkg
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_kernel import _build_runtime_handler_dependencies

pytestmark = pytest.mark.integration

_CONTRACT = Path(str(_triage_pkg.__file__)).parent / "contract.yaml"
_EVENT_TYPE = "omnibase-infra.runtime-error"


class _RecordingPool:
    """Stands in for an asyncpg pool: records each acquire, then refuses it."""

    def __init__(self) -> None:
        self.acquires = 0

    def acquire(self) -> object:
        self.acquires += 1
        raise RuntimeError("recording pool refuses the connection")


async def _dispatch_one_error(pool: _RecordingPool | None) -> None:
    manifest = discover_contracts_from_paths([_CONTRACT])
    assert not manifest.errors, manifest.errors
    (contract,) = manifest.contracts
    assert contract.event_bus is not None
    topic = contract.event_bus.subscribe_topics[0]

    bus = EventBusInmemory(environment="triage-pool", group="triage-pool")
    await bus.start()
    try:
        engine = MessageDispatchEngine()
        report = await wire_from_manifest(
            ModelAutoWiringManifest(contracts=tuple(manifest.contracts)),
            engine,
            event_bus=bus,
            environment="local",
            materialized_explicit_dependencies=_build_runtime_handler_dependencies(
                pool
            ),
        )
        assert report.total_failed == 0
        engine.freeze()
        event = ModelRuntimeErrorEvent(
            logger_family="aiokafka",
            log_level="ERROR",
            message_template="consumer lost its coordinator",
            raw_message="consumer lost its coordinator",
            error_category="kafka_consumer",
            severity="error",
            fingerprint="0123456789abcdef",
        )
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=event.model_dump(mode="json"),
            correlation_id=uuid4(),
            event_type=_EVENT_TYPE,
        )
        await engine.dispatch(topic, envelope)
    finally:
        await bus.close()


async def test_the_wired_triage_handler_reaches_the_runtime_pool() -> None:
    pool = _RecordingPool()
    await _dispatch_one_error(pool)
    assert pool.acquires > 0, (
        "the auto-wired triage handler never reached the runtime pool: it was "
        "built with db_pool=None and suppressed the event"
    )


async def test_a_runtime_with_no_pool_still_wires_and_suppresses() -> None:
    """Negative control: with no pool the handler wires and touches no pool."""
    await _dispatch_one_error(None)
