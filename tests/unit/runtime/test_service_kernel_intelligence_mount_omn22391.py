# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime serves the intelligence API from its contract-declared mount (OMN-22391, T15).

omniintelligence declares ``http_mount`` (``/api/v1``) in the contract of
``node_pattern_storage_effect``; the separate ``intelligence-api`` container is gone, so
the runtime's own port is the only place the API answers. These tests drive the real
``bootstrap()`` with a real ``ServiceHealth`` on a real port and a contract that
declares the same prefix. The ASGI factory is a stand-in with the intelligence API's
absolute ``/api/v1`` routes, because the omniintelligence package is not a dependency of
this repository.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable
from functools import partial
from pathlib import Path
from unittest.mock import patch

import aiohttp
import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport
from omnibase_infra.runtime.service_kernel import bootstrap
from tests.unit.runtime.test_service_kernel_early_http_mount_omn21288 import (
    _no_wiring,
    kernel_harness,
)

FIXTURES = "tests.unit.handlers.http_mount.asgi_fixtures"
INTELLIGENCE_PREFIX = "/api/v1"


def _declare_intelligence_mount(tmp_path: Path, factory: str) -> Path:
    contract = tmp_path / "intelligence" / "contract.yaml"
    contract.parent.mkdir()
    contract.write_text(
        "name: node_pattern_storage_effect\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "http_mount:\n"
        f'  factory: "{factory}"\n'
        f'  path_prefix: "{INTELLIGENCE_PREFIX}"\n',
        encoding="utf-8",
    )
    return contract


async def test_failing_intelligence_mount_refuses_to_become_ready(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """AC2: a planted factory error stops boot; nothing listens, wiring never starts."""
    port: int = request.getfixturevalue("kernel_harness")
    discovered = discover_contracts_from_paths(
        [_declare_intelligence_mount(tmp_path, f"{FIXTURES}:create_app_planted_error")]
    )
    assert not discovered.errors
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
    async with aiohttp.ClientSession() as session:
        for path in ("/health", f"{INTELLIGENCE_PREFIX}/patterns"):
            with pytest.raises(aiohttp.ClientConnectorError):
                async with session.get(f"http://127.0.0.1:{port}{path}"):
                    pass


async def test_intelligence_api_answers_on_the_runtime_port(
    request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    """AC1 (in-repo form): the mounted API answers GET /api/v1/patterns on main's port."""
    port: int = request.getfixturevalue("kernel_harness")
    loop = asyncio.get_running_loop()
    shutdown_ready: asyncio.Future[Callable[[], None]] = loop.create_future()
    wiring_reached: asyncio.Future[None] = loop.create_future()
    let_wiring_finish: asyncio.Future[None] = loop.create_future()

    def capture_shutdown(
        sig: signal.Signals, callback: Callable[..., None], *args: object
    ) -> None:
        if sig == signal.SIGTERM:
            shutdown_ready.set_result(partial(callback, *args))

    async def slow_wire(**kwargs: object) -> ModelAutoWiringReport:
        wiring_reached.set_result(None)
        await let_wiring_finish
        manifest = kwargs["manifest"]
        assert isinstance(manifest, ModelAutoWiringManifest)
        return _no_wiring(manifest)

    discovered = discover_contracts_from_paths(
        [_declare_intelligence_mount(tmp_path, f"{FIXTURES}:create_app")]
    )
    assert not discovered.errors
    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.discover_contracts",
            return_value=discovered,
        ),
        patch("omnibase_infra.runtime.auto_wiring.wire_from_manifest", new=slow_wire),
        patch.object(loop, "add_signal_handler", side_effect=capture_shutdown),
    ):
        task = asyncio.create_task(bootstrap())
        try:
            await asyncio.wait_for(wiring_reached, timeout=60)
            base = f"http://127.0.0.1:{port}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base}{INTELLIGENCE_PREFIX}/patterns", params={"domain": "code"}
                ) as response:
                    assert response.status == 200
                    assert await response.json() == {
                        "domain": "code",
                        "source": "mounted-app",
                    }
                # The mount never takes the runtime's own /health.
                async with session.get(f"{base}/health") as response:
                    assert response.status == 503
        finally:
            let_wiring_finish.set_result(None)
            try:
                shutdown = await asyncio.wait_for(shutdown_ready, timeout=30)
                shutdown()
                assert await asyncio.wait_for(task, timeout=30) == 0
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
