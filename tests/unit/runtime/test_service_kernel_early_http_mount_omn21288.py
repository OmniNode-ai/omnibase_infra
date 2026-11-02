# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract-declared HTTP mount is served before consumer wiring (OMN-21288, T16).

The contract resolver's compose service must not wait on the main runtime's
health (docker/catalog/services/contract-resolver.yaml, OMN-18843), so the
runtime has to answer the mount while ``wire_from_manifest`` is still running.
These tests drive the real ``bootstrap()`` with a real ``ServiceHealth`` on a
real port; only the wiring engine is replaced, by one that blocks until the
test lets it go.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

import omnibase_infra.nodes.node_contract_resolver_bridge as bridge_package
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring import build_unwired_contract_results
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport
from omnibase_infra.runtime.service_kernel import bootstrap
from tests.unit.runtime.conftest import force_inmemory_runtime_config

BRIDGE_CONTRACT = Path(bridge_package.__file__).parent / "contract.yaml"
FIXTURES = "tests.unit.handlers.http_mount.asgi_fixtures"
RESOLVE_BODY = {
    "base_profile_ref": {"profile": "effect_idempotent", "version": "1.0.0"},
    "patches": [],
    "options": {"include_diff": False},
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_wiring(manifest: ModelAutoWiringManifest) -> ModelAutoWiringReport:
    return ModelAutoWiringReport(
        results=build_unwired_contract_results(
            manifest, reason="wiring engine replaced by the early-mount test"
        ),
        duplicates=(),
    )


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


async def _stop_when_up(task: asyncio.Task[int]) -> int:
    """End a booted kernel the way an operator does: SIGTERM."""
    await asyncio.sleep(0.2)
    os.kill(os.getpid(), signal.SIGTERM)
    return await asyncio.wait_for(task, timeout=30)


async def test_mount_answers_while_wiring_is_still_running(kernel_harness: int) -> None:
    loop = asyncio.get_running_loop()
    wiring_reached: asyncio.Future[None] = loop.create_future()
    let_wiring_finish: asyncio.Future[None] = loop.create_future()
    wiring_finished = False

    async def slow_wire(**kwargs: object) -> ModelAutoWiringReport:
        nonlocal wiring_finished
        wiring_reached.set_result(None)
        await let_wiring_finish
        wiring_finished = True
        manifest = kwargs["manifest"]
        assert isinstance(manifest, ModelAutoWiringManifest)
        return _no_wiring(manifest)

    discovered = discover_contracts_from_paths([BRIDGE_CONTRACT])
    assert not discovered.errors
    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.discover_contracts",
            return_value=discovered,
        ),
        patch("omnibase_infra.runtime.auto_wiring.wire_from_manifest", new=slow_wire),
    ):
        task = asyncio.create_task(bootstrap())
        try:
            await asyncio.wait_for(wiring_reached, timeout=60)
            assert not wiring_finished
            base = f"http://127.0.0.1:{kernel_harness}"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base}/api/nodes/health") as response:
                    assert response.status == 200
                    assert (await response.json())["status"] == "ok"
                async with session.post(
                    f"{base}/api/nodes/contract.resolve", json=RESOLVE_BODY
                ) as response:
                    assert response.status == 200
                # The runtime's own liveness still refuses until the runtime attaches.
                async with session.get(f"{base}/health") as response:
                    assert response.status == 503
            assert not wiring_finished
        finally:
            let_wiring_finish.set_result(None)
        assert await _stop_when_up(task) == 0


async def test_failing_factory_fails_boot_before_wiring_starts(
    kernel_harness: int, tmp_path: Path
) -> None:
    contract = tmp_path / "planted" / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text(
        "name: node_http_mount_planted\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "http_mount:\n"
        f'  factory: "{FIXTURES}:create_app_planted_error"\n'
        '  path_prefix: "/api/nodes"\n',
        encoding="utf-8",
    )
    discovered = discover_contracts_from_paths([contract])
    wiring_calls: list[object] = []

    async def wire(**kwargs: object) -> ModelAutoWiringReport:
        wiring_calls.append(kwargs["manifest"])
        raise RuntimeError("wiring was reached")

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.discover_contracts",
            return_value=discovered,
        ),
        patch("omnibase_infra.runtime.auto_wiring.wire_from_manifest", new=wire),
    ):
        exit_code = await asyncio.wait_for(bootstrap(), timeout=60)
    assert exit_code == 1
    assert wiring_calls == []
    with pytest.raises(aiohttp.ClientConnectorError):
        async with aiohttp.ClientSession() as session:
            await session.get(f"http://127.0.0.1:{kernel_harness}/api/nodes/health")
