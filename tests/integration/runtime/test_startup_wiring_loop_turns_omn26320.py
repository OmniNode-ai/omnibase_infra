# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Startup wiring must hand the event loop back between items (OMN-26320).

``wire_from_manifest`` prepares every discovered contract and imports every
handler class in loops that never awaited anything that suspends, so the loop
thread that serves /health and the bus consumers was held for the whole pass.
A lab run of the real kernel with the dev contract set measured one contiguous
12.4 s block there (807 contracts prepared, 1834 handler classes imported).

The fixture ``tests/fixtures/omn26320/devboot_1000_wiring_block_profile.json``
holds the recorded per-call durations of that run. The tests replay the recorded
calls through the real ``wire_from_manifest`` with a real asyncio server
answering /health on the same loop, and count loop turns and wiring calls, never
elapsed time: an earlier version bounded the wall-clock wait of a /health probe
and failed once on a shared CI runner (1.47 s against a 0.8 s limit, 86 probes
answered) with no change to the code under test. A recorded duration is only
summed, never slept, so the verdict is the same on a loaded host.

Needs no services: the blocking calls are replaced by recorders carrying the
recorded length, the dispatch engine is a mock and the server is a loopback
socket.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    PreparedContractWiring,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.auto_wiring.report import (
    EnumWiringOutcome,
    ModelContractWiringResult,
)
from omnibase_infra.runtime.state_io.state_store_adapter import (
    StateIoUnconfiguredError,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "omn26320"
    / "devboot_1000_wiring_block_profile.json"
)

# A /health probe needs a handful of loop turns (accept, read, write, close) and
# every turn lets at most one wiring item run. One uninterrupted pass runs all
# ~2600 recorded calls inside a single probe, so the bound is far from both.
_MAX_WIRING_CALLS_PER_PROBE = 500


def _load_profile() -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return the recorded (name, milliseconds) pairs: prepares, then imports."""
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    prepare = [(name, float(ms)) for name, ms in raw["prepare_contract_wiring_ms"]]
    imports = [(name, float(ms)) for name, ms in raw["import_handler_class_ms"]]
    return prepare, imports


def _manifest(
    prepare: list[tuple[str, float]], imports: list[tuple[str, float]]
) -> tuple[ModelAutoWiringManifest, dict[str, float]]:
    """One contract per recorded prepare call, the recorded handler imports spread over them.

    Returns the manifest and the recorded import milliseconds keyed by the unique
    handler name given to each routing entry.
    """
    contracts = []
    import_ms: dict[str, float] = {}
    count = len(prepare)
    for index, (contract_name, _) in enumerate(prepare):
        entries = []
        for slot, (_, ms) in enumerate(imports[index::count]):
            handler_name = f"Handler{index}x{slot}"
            import_ms[handler_name] = ms
            entries.append(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_name, module=f"omn26320_fixture.m{index}"
                    )
                )
            )
        if not entries:
            handler_name = f"Handler{index}x0"
            import_ms[handler_name] = 0.0
            entries.append(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_name, module=f"omn26320_fixture.m{index}"
                    )
                )
            )
        contracts.append(
            ModelDiscoveredContract(
                name=contract_name,
                node_type="EFFECT_GENERIC",
                contract_version=ModelContractVersion(major=1, minor=0, patch=0),
                contract_path=Path(f"{contract_name}/contract.yaml"),
                entry_point_name=contract_name,
                package_name="omn26320-fixture",
                event_bus=ModelEventBusWiring(
                    subscribe_topics=(f"onex.cmd.omn26320.{index}.v1",),
                    publish_topics=(),
                ),
                handler_routing=ModelHandlerRouting(
                    routing_strategy="operation_match", handlers=entries
                ),
            )
        )
    return ModelAutoWiringManifest(contracts=tuple(contracts)), import_ms


async def _no_instance(container: object, handler_cls: type) -> None:
    return None


async def _no_resolver(container: object) -> None:
    return None


def _skipped(
    contract: ModelDiscoveredContract, environment: str
) -> PreparedContractWiring:
    return PreparedContractWiring(
        contract=contract,
        prepared_wirings=[],
        subscription_topics=[],
        environment=environment,
        skip_result=ModelContractWiringResult(
            contract_name=contract.name,
            package_name=contract.package_name,
            outcome=EnumWiringOutcome.SKIPPED,
            reason="omn26320 replay",
        ),
    )


@dataclass
class _Trace:
    """What the replayed blocking calls saw of the event loop.

    ``ticks`` counts loop turns taken by a sibling task. ``blocks_ms`` holds, for
    each run of consecutive wiring calls with no loop turn between them, the sum
    of their recorded milliseconds: the time the loop thread was held.
    """

    ticks: int = 0
    calls: int = 0
    last_tick: int = -1
    blocks_ms: list[float] = field(default_factory=list)
    prepare_order: list[str] = field(default_factory=list)

    def record_call(self, recorded_ms: float) -> None:
        self.calls += 1
        if not self.blocks_ms or self.ticks != self.last_tick:
            self.blocks_ms.append(0.0)
            self.last_tick = self.ticks
        self.blocks_ms[-1] += recorded_ms


@contextmanager
def _replayed_blocking_calls(
    trace: _Trace, prepare: list[tuple[str, float]], import_ms: dict[str, float]
) -> Iterator[None]:
    """Replace the two blocking wiring calls with recorders of their recorded length."""
    prepare_ms = dict(prepare)

    def blocking_prepare(**kwargs: object) -> PreparedContractWiring:
        contract = kwargs["contract"]
        assert isinstance(contract, ModelDiscoveredContract)
        trace.prepare_order.append(contract.name)
        trace.record_call(prepare_ms[contract.name])
        return _skipped(contract, str(kwargs["environment"]))

    def blocking_import(module_path: str, class_name: str) -> type:
        trace.record_call(import_ms[class_name])
        return object

    with (
        patch.object(handler_wiring, "_prepare_contract_wiring", blocking_prepare),
        patch.object(handler_wiring, "_import_handler_class", blocking_import),
        patch.object(handler_wiring, "_async_resolve_from_container", _no_instance),
        patch.object(handler_wiring, "build_topology_secret_resolver", _no_resolver),
    ):
        yield


async def _serve_health() -> tuple[asyncio.Server, int]:
    async def answer(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        body = b'{"status":"healthy"}'
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(answer, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    return server, port


async def _tick(trace: _Trace, stop: asyncio.Event) -> None:
    """Take one loop turn at a time, so ``trace.ticks`` counts the turns the loop got."""
    while not stop.is_set():
        trace.ticks += 1
        await asyncio.sleep(0)


async def _probe_health(
    port: int,
    trace: _Trace,
    stop: asyncio.Event,
    started: asyncio.Event,
    held: list[int],
) -> None:
    """Probe /health on the wiring loop, recording how many wiring calls ran meanwhile."""
    while not stop.is_set():
        before = trace.calls
        started.set()
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /health HTTP/1.1\r\nHost: probe\r\n\r\n")
        await writer.drain()
        await reader.read()
        writer.close()
        await writer.wait_closed()
        held.append(trace.calls - before)


@dataclass(frozen=True)
class _Measured:
    manifest: ModelAutoWiringManifest
    trace: _Trace
    held: list[int]
    slowest_item_ms: float


def _slowest_item_ms(
    manifest: ModelAutoWiringManifest,
    prepare: list[tuple[str, float]],
    import_ms: dict[str, float],
) -> float:
    """The longest the loop must be held when it is handed back after every contract.

    One item is either one contract's preparation or the handler imports of one
    contract, so the floor of a hold is the slowest of those, not the pass.
    """
    prepare_worst = max(ms for _, ms in prepare)
    imports_worst = max(
        sum(
            import_ms[entry.handler.name] for entry in contract.handler_routing.handlers
        )
        for contract in manifest.contracts
        if contract.handler_routing is not None
    )
    return max(prepare_worst, imports_worst)


async def _no_yield[T](items: Iterable[T]) -> AsyncIterator[T]:
    """The wiring loops as they were before OMN-26320: items run back to back."""
    for item in items:
        yield item


async def _wire_while_probing(*, hand_back: bool) -> _Measured:
    prepare, imports = _load_profile()
    assert len(prepare) == 807
    assert len(imports) == 1834
    manifest, import_ms = _manifest(prepare, imports)
    trace = _Trace()
    held: list[int] = []
    stop = asyncio.Event()
    started = asyncio.Event()
    server, port = await _serve_health()
    ticker = asyncio.create_task(_tick(trace, stop))
    prober = asyncio.create_task(_probe_health(port, trace, stop, started, held))
    # A probe is in flight before wiring begins, so a pass that never hands the
    # loop back leaves it unanswered for the whole pass.
    await started.wait()
    try:
        with _replayed_blocking_calls(trace, prepare, import_ms):
            if hand_back:
                report = await wire_from_manifest(
                    manifest=manifest,
                    dispatch_engine=MagicMock(),
                    event_bus=None,
                    container=MagicMock(),
                    subscribe_immediately=False,
                )
            else:
                with patch.object(handler_wiring, "yield_each", _no_yield):
                    report = await wire_from_manifest(
                        manifest=manifest,
                        dispatch_engine=MagicMock(),
                        event_bus=None,
                        container=MagicMock(),
                        subscribe_immediately=False,
                    )
        assert report.total_failed == 0
        assert len(report.results) == len(manifest.contracts)
    finally:
        stop.set()
        await asyncio.gather(ticker, prober)
        server.close()
        await server.wait_closed()
    return _Measured(
        manifest=manifest,
        trace=trace,
        held=held,
        slowest_item_ms=_slowest_item_ms(manifest, prepare, import_ms),
    )


@pytest.mark.asyncio
async def test_startup_wiring_off_loop() -> None:
    """/health is answered between wiring items while the recorded dev-sized pass runs."""
    measured = await _wire_while_probing(hand_back=True)

    assert measured.held, "no /health probe completed during startup wiring"
    assert max(measured.held) <= _MAX_WIRING_CALLS_PER_PROBE, (
        f"a /health probe waited through {max(measured.held)} wiring calls "
        f"(limit {_MAX_WIRING_CALLS_PER_PROBE}, {len(measured.held)} probes): "
        "the loop was not handed back between items"
    )
    # The loop thread was never held longer than the slowest single item.
    assert max(measured.trace.blocks_ms) <= measured.slowest_item_ms, (
        f"the loop was held {max(measured.trace.blocks_ms):.0f} ms recorded "
        f"without a turn (slowest single item {measured.slowest_item_ms:.0f} ms)"
    )
    # Same wiring order: the loop is handed back, the contracts are not reordered.
    assert measured.trace.prepare_order == [c.name for c in measured.manifest.contracts]


@pytest.mark.asyncio
async def test_startup_wiring_off_loop_control_detects_held_loop() -> None:
    """The measurement flags the pass as it was before the loops handed the loop back.

    Positive control for ``test_startup_wiring_off_loop``: with the wiring loops
    running items back to back, the same replay holds the loop for the whole
    recorded pass and a probe waits through every wiring call.
    """
    measured = await _wire_while_probing(hand_back=False)

    assert max(measured.held) > _MAX_WIRING_CALLS_PER_PROBE
    assert max(measured.trace.blocks_ms) > measured.slowest_item_ms
    # Prepared contracts alone were recorded at more than ten seconds in one block.
    assert max(measured.trace.blocks_ms) > 10_000


async def _wire_expecting_error(
    manifest: ModelAutoWiringManifest,
    error: type[Exception],
    match: str | None = None,
) -> int:
    """Run wiring, expect ``error`` from it, and return how many loop turns a
    sibling task got before the error surfaced."""
    turns = 0
    finished = False

    async def sibling() -> None:
        nonlocal turns
        while not finished:
            turns += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(sibling())
    try:
        with pytest.raises(error, match=match):
            await wire_from_manifest(
                manifest=manifest,
                dispatch_engine=MagicMock(),
                container=MagicMock(),
                subscribe_immediately=False,
            )
    finally:
        finished = True
        await task
    return turns


@pytest.mark.asyncio
async def test_startup_wiring_error_fails_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wiring error still aborts startup with its own typed error.

    The error is raised partway through a pass that hands the loop back between
    contracts, so the sibling task must have run before the error surfaced and
    the error must still reach the awaiting caller unchanged.
    """
    prepare, imports = _load_profile()
    manifest, _ = _manifest(prepare[:20], imports[:60])
    failing = prepare[7][0]

    def prepare_raising(**kwargs: object) -> PreparedContractWiring:
        contract = kwargs["contract"]
        assert isinstance(contract, ModelDiscoveredContract)
        if contract.name == failing:
            raise StateIoUnconfiguredError("state_io seam without its DSN")
        return _skipped(contract, str(kwargs["environment"]))

    def prepare_collected(**kwargs: object) -> PreparedContractWiring:
        contract = kwargs["contract"]
        assert isinstance(contract, ModelDiscoveredContract)
        if contract.name == failing:
            raise ModelOnexError("handler failed to prepare")
        return _skipped(contract, str(kwargs["environment"]))

    with (
        patch.object(handler_wiring, "_import_handler_class", lambda m, c: object),
        patch.object(handler_wiring, "_async_resolve_from_container", _no_instance),
        patch.object(handler_wiring, "build_topology_secret_resolver", _no_resolver),
    ):
        with patch.object(handler_wiring, "_prepare_contract_wiring", prepare_raising):
            turns = await _wire_expecting_error(manifest, StateIoUnconfiguredError)
        assert turns >= 5, f"loop was not handed back before the error ({turns} turns)"

        # Strict mode: a collected per-contract failure is still raised as the
        # typed ModelOnexError, after the whole scan.
        monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
        with patch.object(
            handler_wiring, "_prepare_contract_wiring", prepare_collected
        ):
            turns = await _wire_expecting_error(manifest, ModelOnexError, failing)
        assert turns >= 5, f"loop was not handed back before the error ({turns} turns)"
