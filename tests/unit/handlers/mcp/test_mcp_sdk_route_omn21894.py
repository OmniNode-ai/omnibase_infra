# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The MCP SDK protocol route answers a real ``initialize`` (OMN-21894).

Real uvicorn listener, real SDK client. No mock stands in for the protocol path.
"""

from __future__ import annotations

import asyncio
import re
import socket
import tomllib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from omnibase_infra.handlers.mcp.transport_streamable_http import (
    TransportMCPStreamableHttp,
)
from omnibase_infra.handlers.models.mcp import ModelMcpHandlerConfig

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_KEY = "mount-probe-token"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@asynccontextmanager
async def _serving(
    config: ModelMcpHandlerConfig,
) -> AsyncIterator[int]:
    transport = TransportMCPStreamableHttp(config)
    app = transport.create_app([], lambda _name, _args: None)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=config.port, log_level="warning")
    )
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.02)
        yield config.port
    finally:
        server.should_exit = True
        await task


@pytest.mark.parametrize("stateless", [True, False])
async def test_sdk_initialize_completes_on_the_declared_path(stateless: bool) -> None:
    config = ModelMcpHandlerConfig(
        host="127.0.0.1",
        port=_free_port(),
        path="/mcp",
        api_keys=(_KEY,),
        stateless=stateless,
    )
    async with _serving(config) as port:
        async with httpx.AsyncClient(headers={"X-MCP-API-Key": _KEY}) as http:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp", http_client=http
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    result = await asyncio.wait_for(session.initialize(), 10)
                    assert result.server_info.name
                    assert (await session.list_tools()).tools == []


async def test_route_refuses_a_request_without_a_key() -> None:
    config = ModelMcpHandlerConfig(
        host="127.0.0.1", port=_free_port(), path="/mcp", api_keys=(_KEY,)
    )
    async with _serving(config) as port:
        async with httpx.AsyncClient() as http:
            response = await http.post(f"http://127.0.0.1:{port}/mcp", json={})
    assert response.status_code == 401


def _specifier(text: str, name: str) -> str:
    match = re.search(rf'"{name}([^"]*)"', text)
    assert match, f"{name} not pinned"
    return match.group(1).replace(" ", "")


def test_one_mcp_sdk_pin_across_pyproject_and_runtime_image() -> None:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    declared = [d for d in pyproject["project"]["dependencies"] if d.startswith("mcp")]
    assert len(declared) == 1
    pinned = declared[0].removeprefix("mcp").replace(" ", "")
    dockerfile = (_REPO_ROOT / "docker" / "Dockerfile.runtime").read_text("utf-8")
    assert _specifier(dockerfile, "mcp") == pinned


async def test_sdk_initialize_completes_through_the_runtime_mount(
    tmp_path: Path,
) -> None:
    """The declared mount serves the SDK route on the runtime listener."""
    from tests.unit.handlers.http_mount.test_asgi_mounts_omn21288 import (
        _bound_port,
        _manifest,
        _mounted,
        _server,
    )

    config = ModelMcpHandlerConfig(
        host="127.0.0.1", port=1, path="/mcp", api_keys=(_KEY,)
    )
    block = (
        'http_mount:\n  factory: "omnibase_infra.handlers.mcp.mount_factory:'
        'create_mcp_app"\n  path_prefix: "/mcp"\n'
        "  dependencies: [config, tools, tool_executor]"
    )
    server = await _server()
    try:
        server._http_mounts.provide_dependency("config", config)
        server._http_mounts.provide_dependency("tools", ())
        server._http_mounts.provide_dependency("tool_executor", lambda _n, _a: None)
        await _mounted(server, _manifest(tmp_path, block))
        port = _bound_port(server)
        async with httpx.AsyncClient(headers={"X-MCP-API-Key": _KEY}) as http:
            async with streamable_http_client(
                f"http://127.0.0.1:{port}/mcp", http_client=http
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    result = await asyncio.wait_for(session.initialize(), 10)
                    assert result.server_info.name
    finally:
        await server.stop()
