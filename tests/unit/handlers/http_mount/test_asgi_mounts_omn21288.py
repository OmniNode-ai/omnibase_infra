# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract-declared ASGI factory is served on the runtime's port (OMN-21288).

Real aiohttp server, real FastAPI application, real HTTP requests. The
factories live in ``asgi_fixtures``; the prefix and factory reach the server
only through a ``contract.yaml`` parsed by the production discovery code.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.handlers.http_mount import AsgiMounts
from omnibase_infra.models.model_http_mount import ModelHttpMount
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.models.model_auto_wiring_manifest import (
    ModelAutoWiringManifest,
)
from omnibase_infra.services.health_checker import ServiceHealth
from tests.unit.handlers.http_mount import asgi_fixtures

pytestmark = pytest.mark.unit

FIXTURES = "tests.unit.handlers.http_mount.asgi_fixtures"


def _manifest(tmp_path: Path, *blocks: str) -> ModelAutoWiringManifest:
    contracts = []
    for index, block in enumerate(blocks):
        path = tmp_path / f"contract_{index}.yaml"
        path.write_text(
            f"name: node_mount_{index}\n"
            "node_type: EFFECT_GENERIC\n"
            "contract_version: {major: 1, minor: 0, patch: 0}\n"
            f"{block}\n",
            encoding="utf-8",
        )
        contracts.append(
            _parse_contract(
                contract_path=path,
                entry_point_name=f"node_mount_{index}",
                package_name="fixture",
                package_version="0.0.0",
            )
        )
    return ModelAutoWiringManifest(contracts=tuple(contracts))


def _mount_block(prefix: str, factory: str) -> str:
    return f'http_mount:\n  factory: "{FIXTURES}:{factory}"\n  path_prefix: "{prefix}"'


async def _server() -> ServiceHealth:
    server = ServiceHealth(runtime=MagicMock(), port=0, host="127.0.0.1")
    await server.start()
    return server


async def _mounted(server: ServiceHealth, manifest: ModelAutoWiringManifest) -> None:
    """Attach the manifest, as the kernel does, and wait for the mounts to start."""
    server.attach_manifest(manifest)
    if server._mount_task is not None:
        await server._mount_task


def _bound_port(server: ServiceHealth) -> int:
    assert server._runner is not None
    return int(server._runner.addresses[0][1])


async def test_declared_factory_is_served_on_the_runtime_port(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, _mount_block("/api/v1", "create_app"))
    assert AsgiMounts.declared(manifest.contracts) == (
        ModelHttpMount(factory=f"{FIXTURES}:create_app", path_prefix="/api/v1"),
    )
    server = await _server()
    try:
        await _mounted(server, manifest)
        base = f"http://127.0.0.1:{_bound_port(server)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/api/v1/patterns?domain=code") as response:
                assert response.status == 200
                assert await response.json() == {
                    "domain": "code",
                    "source": "mounted-app",
                }
            async with session.post(f"{base}/api/v1/echo", json={"a": "b"}) as response:
                assert response.status == 200
                assert await response.json() == {"a": "b"}
            async with session.get(f"{base}/api/v1/missing") as response:
                assert response.status == 404
            # The runtime's own routes keep answering on the same port.
            async with session.get(f"{base}/health") as response:
                assert response.status in (200, 503)
                assert "status" in await response.json()
            async with session.get(f"{base}/not-mounted") as response:
                assert response.status == 404
        assert asgi_fixtures.LIFESPAN_LOG[-1] == "startup"
    finally:
        await server.stop()
    assert asgi_fixtures.LIFESPAN_LOG[-1] == "shutdown"


@pytest.mark.parametrize("factory", ["create_app_planted_error", "create_not_an_app"])
async def test_a_failing_factory_fails_boot_and_mounts_nothing(
    tmp_path: Path, factory: str
) -> None:
    manifest = _manifest(
        tmp_path,
        _mount_block("/api/v1", "create_app"),
        _mount_block("/broken", factory),
    )
    server = await _server()
    try:
        with pytest.raises(RuntimeHostError, match="/broken"):
            await _mounted(server, manifest)
        # All or nothing: the healthy mount declared first is not left serving.
        base = f"http://127.0.0.1:{_bound_port(server)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/api/v1/patterns") as response:
                assert response.status == 404
    finally:
        await server.stop()


async def test_a_failing_lifespan_takes_the_whole_server_out_of_service(
    tmp_path: Path,
) -> None:
    manifest = _manifest(
        tmp_path,
        _mount_block("/api/v1", "create_app"),
        _mount_block("/broken", "create_app_failing_lifespan"),
    )
    server = await _server()
    try:
        with pytest.raises(RuntimeHostError, match="database unreachable"):
            await _mounted(server, manifest)
        base = f"http://127.0.0.1:{_bound_port(server)}"
        async with aiohttp.ClientSession() as session:
            for path in ("/api/v1/patterns", "/health", "/ready"):
                async with session.get(f"{base}{path}") as response:
                    assert response.status == 503, path
                    assert (await response.json())["status"] == "unavailable"
    finally:
        await server.stop()


async def test_a_malformed_declaration_fails_boot(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path, 'http_mount:\n  factory: "pkg.mod:create_app"\n  path_prefix: "/"'
    )
    server = await _server()
    try:
        with pytest.raises(RuntimeHostError, match="malformed http_mount"):
            await _mounted(server, manifest)
    finally:
        await server.stop()


async def test_an_unimportable_factory_fails_boot(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        'http_mount:\n  factory: "no.such.module:create_app"\n  path_prefix: "/x"',
    )
    server = await _server()
    try:
        with pytest.raises(RuntimeHostError, match="/x"):
            await _mounted(server, manifest)
    finally:
        await server.stop()


@pytest.mark.parametrize("prefix", ["/health", "/v1", "/v1/introspection"])
async def test_a_mount_may_not_shadow_a_runtime_route(
    tmp_path: Path, prefix: str
) -> None:
    manifest = _manifest(tmp_path, _mount_block(prefix, "create_app"))
    server = await _server()
    try:
        with pytest.raises(RuntimeHostError, match="shadow"):
            await _mounted(server, manifest)
    finally:
        await server.stop()


async def test_overlapping_mounts_fail_boot(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        _mount_block("/api", "create_app"),
        _mount_block("/api/v1", "create_app"),
    )
    server = await _server()
    try:
        with pytest.raises(RuntimeHostError, match="overlaps"):
            await _mounted(server, manifest)
    finally:
        await server.stop()


async def test_a_contract_without_a_mount_changes_nothing(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "description: plain")
    server = await _server()
    try:
        await _mounted(server, manifest)
        base = f"http://127.0.0.1:{_bound_port(server)}"
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base}/api/v1/patterns") as response:
                assert response.status == 404
    finally:
        await server.stop()


async def test_a_second_attach_mounts_nothing_and_a_restart_mounts_again(
    tmp_path: Path,
) -> None:
    """The kernel attaches before wiring and again for introspection (OMN-21288).

    The second call keeps the first start task and registers no route twice.
    After stop() the server builds a fresh router, so attaching again serves.
    """
    manifest = _manifest(tmp_path, _mount_block("/api/v1", "create_app"))
    server = await _server()
    try:
        await _mounted(server, manifest)
        first_task = server._mount_task
        assert first_task is not None
        await _mounted(server, manifest)
        assert server._mount_task is first_task
        base = f"http://127.0.0.1:{_bound_port(server)}"
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base}/api/v1/echo", json={"a": "b"}) as response:
                assert response.status == 200
    finally:
        await server.stop()
    await server.start()
    try:
        await _mounted(server, manifest)
        assert server._mount_task is not first_task
        base = f"http://127.0.0.1:{_bound_port(server)}"
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base}/api/v1/echo", json={"a": "b"}) as response:
                assert response.status == 200
    finally:
        await server.stop()


@pytest.mark.parametrize(
    "block",
    [
        {"factory": "pkg.mod:create_app", "path_prefix": "/"},
        {"factory": "pkg.mod:create_app", "path_prefix": "/api/"},
        {"factory": "pkg.mod:create_app", "path_prefix": "api"},
        {"factory": "pkg.mod.create_app", "path_prefix": "/api"},
        {"factory": "pkg.mod:create_app"},
        {"factory": "pkg.mod:create_app", "path_prefix": "/api", "port": 9},
    ],
)
def test_malformed_declarations_are_refused(block: dict[str, str | int]) -> None:
    with pytest.raises(ValueError):
        ModelHttpMount.from_contract_block(block)
