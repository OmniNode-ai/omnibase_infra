# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Check the contract body cap on the runtime's HTTP listener."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.services.health_checker import ServiceHealth

pytestmark = pytest.mark.integration


async def test_contract_body_cap_on_runtime_listener(tmp_path: Path) -> None:
    exact = b'{"body":"1234"}'
    over = b'{"body":"12345"}'
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: node_http_mount_body_cap\n"
        "node_type: EFFECT_GENERIC\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "http_mount:\n"
        '  factory: "tests.unit.handlers.http_mount.asgi_fixtures:create_app"\n'
        '  path_prefix: "/api/v1"\n'
        f"  max_body_bytes: {len(exact)}\n",
        encoding="utf-8",
    )
    manifest = discover_contracts_from_paths([contract])
    assert manifest.errors == ()

    server = ServiceHealth(runtime=MagicMock(), port=0, host="127.0.0.1")
    await server.start()
    try:
        server.attach_manifest(manifest)
        assert server._mount_task is not None
        await server._mount_task
        assert server._runner is not None
        port = int(server._site.port)
        url = f"http://127.0.0.1:{port}/api/v1/echo"
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.post(url, data=exact, headers=headers) as response:
                assert response.status == 200
                assert await response.json() == {"body": "1234"}
            async with session.post(url, data=over, headers=headers) as response:
                assert response.status == 413
            async with session.post(url, data=exact, headers=headers) as response:
                assert response.status == 200
                assert await response.json() == {"body": "1234"}
    finally:
        await server.stop()
