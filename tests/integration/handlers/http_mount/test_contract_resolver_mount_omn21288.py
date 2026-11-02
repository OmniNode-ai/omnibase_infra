# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract resolver is served from a contract-declared mount (OMN-21288, T16).

``node_contract_resolver_bridge`` declares the resolver's ASGI factory and the
``/api/nodes`` prefix in its own ``contract.yaml``. The contract is loaded from
disk through the production discovery entry point, the runtime's aiohttp server
listens on a real port, and every request below is real HTTP over that port.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiohttp

import omnibase_infra.nodes.node_contract_resolver_bridge as bridge_package
from omnibase_infra.handlers.http_mount import AsgiMounts
from omnibase_infra.models.model_http_mount import ModelHttpMount
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.services.health_checker import ServiceHealth

BRIDGE_CONTRACT = Path(bridge_package.__file__).parent / "contract.yaml"
RESOLVE_BODY = {
    "base_profile_ref": {"profile": "effect_idempotent", "version": "1.0.0"},
    "patches": [],
    "options": {
        "include_diff": False,
        "include_overlay_refs": True,
        "normalize_for_hash": True,
    },
}


def test_bridge_contract_declares_the_resolver_mount() -> None:
    manifest = discover_contracts_from_paths([BRIDGE_CONTRACT])
    assert not manifest.errors
    (mount,) = AsgiMounts.declared(manifest.contracts)
    assert mount == ModelHttpMount(
        factory="omnibase_infra.services.contract_resolver.main:create_mounted_app",
        path_prefix="/api/nodes",
    )


async def test_declared_resolver_serves_health_and_a_resolution_in_process() -> None:
    manifest = discover_contracts_from_paths([BRIDGE_CONTRACT])
    server = ServiceHealth(runtime=MagicMock(), port=0, host="127.0.0.1")
    await server.start()
    try:
        server.attach_manifest(manifest)
        assert server._mount_task is not None
        await server._mount_task
        assert server._runner is not None
        base = f"http://127.0.0.1:{int(server._runner.addresses[0][1])}"
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/api/nodes/health") as response:
                assert response.status == 200
                body = await response.json()
                assert body["status"] == "ok"
                assert body["service"] == "node_contract_resolver_bridge"
            async with session.post(
                f"{base}/api/nodes/contract.resolve", json=RESOLVE_BODY
            ) as response:
                assert response.status == 200
                resolved = await response.json()
                assert len(resolved["resolved_hash"]) == 64
                assert resolved["overlay_refs"] == []
            async with session.post(
                f"{base}/api/nodes/contract.resolve", json={"invalid": "field"}
            ) as response:
                assert response.status == 422
    finally:
        await server.stop()
