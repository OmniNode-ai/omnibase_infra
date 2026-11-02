# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract-declared ASGI app answers on the runtime's real HTTP port (OMN-21288).

Contracts are loaded from disk through ``discover_contracts_from_paths`` (the
production discovery entry point), the server listens on a real TCP port, and
the requests are real HTTP over that port. Nothing is mocked except the runtime
object the health routes delegate to.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration

FIXTURES = "tests.unit.handlers.http_mount.asgi_fixtures"


def _contract(tmp_path: Path, factory: str) -> Path:
    path = tmp_path / factory / "contract.yaml"
    path.parent.mkdir()
    path.write_text(
        "name: node_http_mount_live\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "http_mount:\n"
        f'  factory: "{FIXTURES}:{factory}"\n'
        '  path_prefix: "/api/v1"\n',
        encoding="utf-8",
    )
    return path


async def test_mounted_app_answers_on_the_runtime_port(tmp_path: Path) -> None:
    manifest = discover_contracts_from_paths([_contract(tmp_path, "create_app")])
    assert not manifest.errors
    server = ServiceHealth(runtime=MagicMock(), port=0, host="127.0.0.1")
    await server.start()
    try:
        server.attach_manifest(manifest)
        assert server._mount_task is not None
        await server._mount_task
        assert server._runner is not None
        port = int(server._runner.addresses[0][1])
        async with aiohttp.ClientSession() as session:
            url = f"http://127.0.0.1:{port}/api/v1/patterns?domain=code"
            async with session.get(url) as response:
                assert response.status == 200
                assert await response.json() == {
                    "domain": "code",
                    "source": "mounted-app",
                }
            async with session.get(f"http://127.0.0.1:{port}/health") as response:
                assert "status" in await response.json()
    finally:
        await server.stop()


async def test_planted_factory_error_fails_boot(tmp_path: Path) -> None:
    manifest = discover_contracts_from_paths(
        [_contract(tmp_path, "create_app_planted_error")]
    )
    server = ServiceHealth(runtime=MagicMock(), port=0, host="127.0.0.1")
    await server.start()
    try:
        with pytest.raises(RuntimeHostError, match="planted factory error"):
            server.attach_manifest(manifest)
    finally:
        await server.stop()
