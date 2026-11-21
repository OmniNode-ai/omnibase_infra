# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Startup wiring must not hold the event loop that serves /health (OMN-26320).

``wire_from_manifest`` prepares every discovered contract and imports every
handler class in loops that never awaited anything that suspends, so the loop
thread that serves /health and the bus consumers was held for the whole pass.
A lab run of the real kernel with the dev contract set measured one contiguous
12.4 s block there (807 contracts prepared, 1834 handler classes imported).

The fixture ``tests/fixtures/omn26320/devboot_1000_wiring_block_profile.json``
holds the recorded per-call durations of that run. The tests replay them,
scaled by ``_SCALE`` to keep the run short, through the real
``wire_from_manifest`` with a real asyncio server answering /health on the same
loop, and read how long each probe waited.

Needs no services: the blocking calls are replaced by sleeps of the recorded
length, the dispatch engine is a mock and the server is a loopback socket.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
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

# Recorded durations are replayed at one quarter of their length: the total
# block is still ~2.9 s at origin/dev and the worst single call ~0.27 s.
_SCALE = 0.25
# A /health probe must be answered within this long. The slowest single
# replayed call is ~0.27 s, so cooperative yielding stays under it; one
# uninterrupted block of the whole pass (~2.9 s) does not.
_MAX_HEALTH_WAIT_SECONDS = 0.8


def _load_profile() -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    prepare = [
        (name, ms / 1000.0 * _SCALE) for name, ms in raw["prepare_contract_wiring_ms"]
    ]
    imports = [
        (name, ms / 1000.0 * _SCALE) for name, ms in raw["import_handler_class_ms"]
    ]
    return prepare, imports


def _manifest(
    prepare: list[tuple[str, float]], imports: list[tuple[str, float]]
) -> tuple[ModelAutoWiringManifest, dict[str, float]]:
    """One contract per recorded prepare call, the recorded handler imports spread over them.

    Returns the manifest and the recorded import seconds keyed by the unique
    handler name given to each routing entry.
    """
    contracts = []
    import_seconds: dict[str, float] = {}
    count = len(prepare)
    for index, (contract_name, _) in enumerate(prepare):
        entries = []
        for slot, (_, seconds) in enumerate(imports[index::count]):
            handler_name = f"Handler{index}x{slot}"
            import_seconds[handler_name] = seconds
            entries.append(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name=handler_name, module=f"omn26320_fixture.m{index}"
                    )
                )
            )
        if not entries:
            handler_name = f"Handler{index}x0"
            import_seconds[handler_name] = 0.0
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
    return ModelAutoWiringManifest(contracts=tuple(contracts)), import_seconds


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


@contextmanager
def _replayed_blocking_calls(
    prepare: list[tuple[str, float]], import_seconds: dict[str, float]
) -> Iterator[None]:
    """Replace the two blocking wiring calls with sleeps of their recorded length."""
    prepare_s = dict(prepare)

    def blocking_prepare(**kwargs: object) -> PreparedContractWiring:
        contract = kwargs["contract"]
        assert isinstance(contract, ModelDiscoveredContract)
        time.sleep(prepare_s[contract.name])
        return _skipped(contract, str(kwargs["environment"]))

    def blocking_import(module_path: str, class_name: str) -> type:
        time.sleep(import_seconds[class_name])
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


def _probe_health(port: int, stop: threading.Event, waits: list[float]) -> None:
    """Probe /health from another thread, recording how long each answer took."""
    while not stop.is_set():
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=30
            ) as response:
                response.read()
        except OSError:
            pass
        waits.append(time.perf_counter() - started)
        time.sleep(0.02)


async def _wire_while_probing(manifest: ModelAutoWiringManifest) -> list[float]:
    server, port = await _serve_health()
    stop = threading.Event()
    waits: list[float] = []
    prober = threading.Thread(
        target=_probe_health, args=(port, stop, waits), daemon=True
    )
    prober.start()
    # Let the first probe start before wiring begins.
    await asyncio.sleep(0.1)
    try:
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
        await asyncio.to_thread(prober.join, 30)
        server.close()
        await server.wait_closed()
    return waits


@pytest.mark.asyncio
async def test_startup_wiring_off_loop() -> None:
    """/health is answered while the recorded dev-sized wiring pass runs."""
    prepare, imports = _load_profile()
    assert len(prepare) == 807
    assert len(imports) == 1834
    manifest, import_seconds = _manifest(prepare, imports)
    with _replayed_blocking_calls(prepare, import_seconds):
        waits = await _wire_while_probing(manifest)

    assert waits, "no /health probe completed during startup wiring"
    worst = max(waits)
    assert worst < _MAX_HEALTH_WAIT_SECONDS, (
        f"/health waited {worst:.2f}s for the event loop during startup wiring "
        f"(limit {_MAX_HEALTH_WAIT_SECONDS}s, {len(waits)} probes)"
    )


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
