# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel's boot-wiring deferral seams on the real bootstrap() (OMN-21318).

Only a Kafka bus can defer a contract at boot (``plan_runtime_deferral`` asks
the broker which topics are in use), and these tests run on the in-memory bus.
So the kernel's planner is the only replaced deferral seam: the fake returns a
REAL ``BootWiringDeferral`` built from the real manifest the kernel passes,
deferring the bridge contract itself with a preparer that always raises. What
stays real is everything around it — ``bootstrap()``, the health server, the
auto-wiring seam patches of the OMN-21288 harness — so the tests pin the
kernel's call-sites:

* strict boot: ``verify_strict`` runs the deferred contract's preparation and
  the refusal it raises is a boot failure (exit code 1), never a hidden row;
* non-strict boot: the preparation is never attempted, the boot reaches the
  demand watch armed with ``on_attempt`` bound to the attach-readiness gate's
  ``record``, and a clean shutdown still exits 0.
"""

from __future__ import annotations

import asyncio
import signal
import socket
from collections.abc import Callable, Generator
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

import omnibase_infra.nodes.node_contract_resolver_bridge as bridge_package
from omnibase_infra.handlers.boot_wiring_deferral import (
    BootWiringDeferral,
    _FirstRecordRelay,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import (
    build_unwired_contract_results,
    handler_wiring,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport
from omnibase_infra.runtime.health.contract_attach_readiness_gate import (
    ContractAttachReadinessGate,
)
from omnibase_infra.runtime.service_kernel import bootstrap
from tests.unit.runtime.conftest import force_inmemory_runtime_config

if TYPE_CHECKING:
    from collections.abc import Mapping

    from omnibase_infra.event_bus.model_topic_readiness_config import (
        ModelTopicReadinessConfig,
    )
    from omnibase_infra.protocols.protocol_dispatch_result_applier import (
        ProtocolDispatchResultApplier,
    )

BRIDGE_CONTRACT = Path(bridge_package.__file__).parent / "contract.yaml"
BRIDGE_CONTRACT_NAME = "node_contract_resolver_bridge"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_wiring(manifest: ModelAutoWiringManifest) -> ModelAutoWiringReport:
    return ModelAutoWiringReport(
        results=build_unwired_contract_results(
            manifest, reason="wiring engine replaced by the deferral test"
        ),
        duplicates=(),
    )


async def _stop_nothing() -> None:
    return None


def _broken_deferral_plan(
    prepare_calls: list[ModelDiscoveredContract],
    planned: list[BootWiringDeferral],
) -> Callable[..., object]:
    """Stand in for ``plan_runtime_deferral`` on the in-memory bus.

    The deferral is real (real manifest, real wiring module, real readiness
    config from the kernel) and defers exactly the manifest's first contract;
    only its preparation is a recorded, always-raising fake, and its wire is
    None so the demand watch arms without a broker behind it.
    """

    async def broken_prepare(contract: ModelDiscoveredContract) -> NoReturn:
        prepare_calls.append(contract)
        raise RuntimeError("deferred handler is broken (test)")

    async def fake_plan(
        manifest: ModelAutoWiringManifest, **kwargs: object
    ) -> BootWiringDeferral:
        deferral = BootWiringDeferral(
            manifest=manifest,
            deferred=(manifest.contracts[0],),
            relay=_FirstRecordRelay(),
            stop=_stop_nothing,
            wire=None,
            prepare=broken_prepare,
            wiring=handler_wiring,
            readiness_config=cast(
                "ModelTopicReadinessConfig", kwargs["readiness_config"]
            ),
            result_appliers_by_contract=cast(
                "Mapping[str, ProtocolDispatchResultApplier] | None",
                kwargs["result_appliers_by_contract"],
            ),
        )
        planned.append(deferral)
        return deferral

    return fake_plan


@pytest.fixture
def kernel_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[int, None, None]:
    """Boot dependencies replaced except ServiceHealth; yields the HTTP port."""
    force_inmemory_runtime_config(monkeypatch, tmp_path)
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.setenv("RUNTIME_PROFILE", "main")
    monkeypatch.delenv("ONEX_RUNTIME_NODE_SELECTION", raising=False)
    monkeypatch.delenv("ONEX_RUNTIME_FOLDED_PROFILES", raising=False)
    port = _free_port()
    monkeypatch.setenv("ONEX_HTTP_PORT", str(port))

    async def noop_wire(container: object) -> dict[str, list[str]]:
        return {"services": []}

    runtime = MagicMock()
    runtime.start = AsyncMock()
    runtime.stop = AsyncMock()
    runtime.input_topic = "requests"
    runtime.output_topic = "responses"
    bus = MagicMock(spec=ProtocolEventBusLike)
    bus.start = AsyncMock()
    bus.close = AsyncMock()
    bus.subscribe = AsyncMock(return_value=AsyncMock())
    container = MagicMock()
    container.service_registry.resolve_service = AsyncMock(return_value=MagicMock())
    container.service_registry.register_instance = AsyncMock(return_value="id")
    with (
        patch(
            "omnibase_infra.runtime.service_kernel.wire_infrastructure_services",
            side_effect=noop_wire,
        ),
        patch(
            "omnibase_infra.runtime.service_kernel.ModelONEXContainer",
            return_value=container,
        ),
        patch(
            "omnibase_infra.runtime.service_kernel.RuntimeHostProcess",
            return_value=runtime,
        ),
        patch(
            "omnibase_infra.backends.auto_configure.select_event_bus",
            return_value=bus,
        ),
    ):
        yield port


async def test_strict_boot_refuses_a_broken_deferred_handler(
    kernel_harness: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_WIRING_STRICT_MODE", "1")
    monkeypatch.delenv("ONEX_STRICT_DISPATCHER_COVERAGE", raising=False)
    discovered = discover_contracts_from_paths([BRIDGE_CONTRACT])
    assert not discovered.errors
    prepare_calls: list[ModelDiscoveredContract] = []
    planned: list[BootWiringDeferral] = []
    wired_manifests: list[ModelAutoWiringManifest] = []

    async def wire(**kwargs: object) -> ModelAutoWiringReport:
        manifest = kwargs["manifest"]
        assert isinstance(manifest, ModelAutoWiringManifest)
        wired_manifests.append(manifest)
        return _no_wiring(manifest)

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.discover_contracts",
            return_value=discovered,
        ),
        patch("omnibase_infra.runtime.auto_wiring.wire_from_manifest", new=wire),
        patch(
            "omnibase_infra.runtime.service_kernel.boot_wiring_deferral"
            ".plan_runtime_deferral",
            new=_broken_deferral_plan(prepare_calls, planned),
        ),
    ):
        exit_code = await asyncio.wait_for(bootstrap(), timeout=60)
    assert exit_code == 1
    # verify_strict ran the deferred contract's preparation exactly once, on
    # the contract the deferral took out of the wired manifest.
    assert [contract.name for contract in prepare_calls] == [BRIDGE_CONTRACT_NAME]
    assert len(planned) == 1
    assert planned[0].deferred_names == frozenset({BRIDGE_CONTRACT_NAME})
    assert wired_manifests == [planned[0].wired_manifest]
    # The refused boot left no health server behind on the port.
    with pytest.raises(aiohttp.ClientConnectorError):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{kernel_harness}/api/nodes/health"
            ):
                pass


async def test_non_strict_boot_survives_a_broken_deferred_handler(
    kernel_harness: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = asyncio.get_running_loop()
    monkeypatch.delenv("ONEX_WIRING_STRICT_MODE", raising=False)
    monkeypatch.delenv("ONEX_STRICT_DISPATCHER_COVERAGE", raising=False)
    shutdown_ready: asyncio.Future[Callable[[], None]] = loop.create_future()
    watch_ran: asyncio.Future[None] = loop.create_future()

    def capture_shutdown(
        sig: signal.Signals, callback: Callable[..., None], *args: object
    ) -> None:
        # Exercise the kernel's real shutdown callback without installing a
        # process-wide handler or sending SIGTERM to the pytest process.
        if sig == signal.SIGTERM:
            shutdown_ready.set_result(partial(callback, *args))

    discovered = discover_contracts_from_paths([BRIDGE_CONTRACT])
    assert not discovered.errors
    prepare_calls: list[ModelDiscoveredContract] = []
    planned: list[BootWiringDeferral] = []
    wired_manifests: list[ModelAutoWiringManifest] = []

    async def wire(**kwargs: object) -> ModelAutoWiringReport:
        manifest = kwargs["manifest"]
        assert isinstance(manifest, ModelAutoWiringManifest)
        wired_manifests.append(manifest)
        return _no_wiring(manifest)

    real_watch = BootWiringDeferral.watch

    async def watch_spy(*args: object, **kwargs: object) -> None:
        # autospec may hand the bound instance through or not; this boot
        # planned exactly one deferral, and it is the instance the kernel holds.
        await real_watch(planned[-1], **kwargs)
        if not watch_ran.done():
            watch_ran.set_result(None)

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.discover_contracts",
            return_value=discovered,
        ),
        patch("omnibase_infra.runtime.auto_wiring.wire_from_manifest", new=wire),
        patch(
            "omnibase_infra.runtime.service_kernel.boot_wiring_deferral"
            ".plan_runtime_deferral",
            new=_broken_deferral_plan(prepare_calls, planned),
        ),
        patch.object(
            BootWiringDeferral, "watch", autospec=True, side_effect=watch_spy
        ) as watch_mock,
        patch.object(loop, "add_signal_handler", side_effect=capture_shutdown),
    ):
        task = asyncio.create_task(bootstrap())
        try:
            await asyncio.wait_for(watch_ran, timeout=60)
            shutdown = await asyncio.wait_for(shutdown_ready, timeout=30)
            shutdown()
            assert await asyncio.wait_for(task, timeout=30) == 0
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    # The deferral never moved the contract out of what non-strict boot checks:
    # verify_strict returned before preparing anything, and the kernel wired
    # the manifest without the deferred contract.
    assert prepare_calls == []
    assert len(planned) == 1
    assert wired_manifests == [planned[0].wired_manifest]
    assert BRIDGE_CONTRACT_NAME not in {
        contract.name for contract in wired_manifests[0].contracts
    }
    # The demand watch ran exactly once, with its attach attempts bound to the
    # readiness gate's record.
    assert watch_mock.await_count == 1
    on_attempt = watch_mock.await_args.kwargs["on_attempt"]
    assert on_attempt.__name__ == "record"
    assert isinstance(on_attempt.__self__, ContractAttachReadinessGate)
