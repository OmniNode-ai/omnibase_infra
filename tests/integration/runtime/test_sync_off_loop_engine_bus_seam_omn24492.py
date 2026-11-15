# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24492 seam: wiring -> real dispatch engine -> real in-memory event bus.

The callback-level tests (test_handler_wiring_sync_off_loop.py) call the dispatch
callback directly. This file drives the same change through the seam the runtime
uses: ``wire_from_manifest`` registers the callback with a real
``MessageDispatchEngine`` and subscribes it on a real ``EventBusInmemory``; one
command is published on the bus; a plain-function handler blocks for 0.6 s.

Reads:
  1. golden: the handler runs on a worker thread and the loop keeps beating;
  2. control: with the placement caller replaced by the old direct call, the same
     scenario stalls the loop for the block time (so the golden reading can fail);
  3. error: a plain handler that raises and an ``async def handle_async`` that
     raises reach ``MessageDispatchEngine.dispatch`` as the same failure.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_entry_method_placement.handler import (
    HandlerEntryMethodPlacement,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

BLOCK_SECONDS = 0.6
HEARTBEAT_INTERVAL_SECONDS = 0.01
GOLDEN_MAX_GAP_SECONDS = 0.25
CONTROL_MIN_GAP_SECONDS = 0.48
DELIVERY_TIMEOUT_SECONDS = 10.0

_THIS_MODULE = "tests.integration.runtime.test_sync_off_loop_engine_bus_seam_omn24492"
_PATCH_IMPORT_HANDLER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_TOPIC = "onex.cmd.omnibase-infra.entry-placement-probe.v1"
_EVENT_TYPE = "omnibase-infra.entry-placement-probe"

# Filled by the stub handlers: the thread each ran on, and a flag the test waits on.
_RAN_ON: list[int] = []
_RAN = threading.Event()


class ProbeBoomError(Exception):
    """Raised by the stub handlers."""


class ModelProbeCommand(BaseModel):
    """The command the stub handlers take."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: UUID


class ModelProbeResult(BaseModel):
    """What the stub handlers answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    probe_id: UUID


class HandlerBlockingSync:
    """A plain-function entry method that blocks its thread for BLOCK_SECONDS."""

    def handle(self, command: ModelProbeCommand) -> ModelProbeResult:
        _RAN_ON.append(threading.get_ident())
        time.sleep(BLOCK_SECONDS)
        _RAN.set()
        return ModelProbeResult(probe_id=command.probe_id)


class HandlerRaisingSync:
    """A plain-function entry method that raises."""

    def handle(self, command: ModelProbeCommand) -> ModelProbeResult:
        raise ProbeBoomError(f"probe {command.probe_id} failed")


class HandlerRaisingAsync:
    """An ``async def handle_async`` entry method that raises the same error."""

    def handle(self, command: ModelProbeCommand) -> ModelProbeResult:
        raise AssertionError("handle_async is the entry method")

    async def handle_async(self, command: ModelProbeCommand) -> ModelProbeResult:
        raise ProbeBoomError(f"probe {command.probe_id} failed")


def _contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_entry_placement_probe_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name="node_entry_placement_probe_effect",
        package_name="omnibase_infra",
        event_bus=ModelEventBusWiring(subscribe_topics=(_TOPIC,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="Probe", module=_THIS_MODULE),
                    event_model=ModelHandlerRef(
                        name="ModelProbeCommand", module=_THIS_MODULE
                    ),
                    operation="probe.run",
                ),
            ),
        ),
    )


def _envelope(probe_id: UUID) -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        payload={"probe_id": str(probe_id)},
        correlation_id=uuid4(),
        event_type=_EVENT_TYPE,
    )


async def _wired(handler_cls: type, bus: EventBusInmemory) -> MessageDispatchEngine:
    engine = MessageDispatchEngine()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_PATCH_IMPORT_HANDLER, lambda *_a, **_k: handler_cls)
        await wire_from_manifest(
            ModelAutoWiringManifest(contracts=(_contract(),)),
            engine,
            event_bus=bus,
            environment="local",
        )
    engine.freeze()
    return engine


async def _heartbeat_gap_while(
    publish_and_wait: Callable[[], Awaitable[None]],
) -> float:
    """Largest gap between 10 ms heartbeat wakeups while ``publish_and_wait`` runs."""
    beats: list[float] = []

    async def heartbeat() -> None:
        while True:
            beats.append(time.monotonic())
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS * 2)
    try:
        await publish_and_wait()
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS * 3)
    finally:
        beat.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat
    return max(later - earlier for earlier, later in itertools.pairwise(beats))


async def _one_command_over_the_bus() -> float:
    """Publish one command on a real in-memory bus; return the largest loop gap."""
    _RAN_ON.clear()
    _RAN.clear()
    bus = EventBusInmemory(environment="test", group="omn-24492-seam")
    await bus.start()
    try:
        await _wired(HandlerBlockingSync, bus)

        async def publish_and_wait() -> None:
            envelope = _envelope(uuid4())
            await bus.publish(
                _TOPIC, None, envelope.model_dump_json().encode("utf-8"), None
            )
            deadline = time.monotonic() + DELIVERY_TIMEOUT_SECONDS
            while not _RAN.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)

        return await _heartbeat_gap_while(publish_and_wait)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_golden_a_blocking_plain_handler_runs_off_the_loop_beside_the_bus() -> (
    None
):
    gap = await _one_command_over_the_bus()

    assert _RAN.is_set(), "the command never reached the handler"
    assert _RAN_ON and _RAN_ON[0] != threading.get_ident()
    assert gap < GOLDEN_MAX_GAP_SECONDS


@pytest.mark.asyncio
async def test_control_the_old_direct_call_stalls_the_bus_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def direct_caller(
        self: HandlerEntryMethodPlacement, entry: Callable[[object], object]
    ) -> Callable[[object], Awaitable[object]]:
        async def call(argument: object) -> object:
            result = entry(argument)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        return call

    monkeypatch.setattr(HandlerEntryMethodPlacement, "caller", direct_caller)

    gap = await _one_command_over_the_bus()

    assert _RAN.is_set(), "the command never reached the handler"
    assert _RAN_ON and _RAN_ON[0] == threading.get_ident()
    assert gap >= CONTROL_MIN_GAP_SECONDS


async def _dispatch_failure(handler_cls: type) -> tuple[str, str]:
    bus = EventBusInmemory(environment="test", group="omn-24492-seam-error")
    await bus.start()
    try:
        engine = await _wired(handler_cls, bus)
        result = await engine.dispatch(_TOPIC, _envelope(uuid4()))
    finally:
        await bus.close()
    return str(result.status), str(result.error_message)


@pytest.mark.asyncio
async def test_error_a_plain_handler_failure_reaches_dispatch_like_an_async_one() -> (
    None
):
    plain_status, plain_error = await _dispatch_failure(HandlerRaisingSync)
    async_status, async_error = await _dispatch_failure(HandlerRaisingAsync)

    assert plain_status == async_status == "handler_error"
    assert "ProbeBoomError: probe " in plain_error
    assert "ProbeBoomError: probe " in async_error
