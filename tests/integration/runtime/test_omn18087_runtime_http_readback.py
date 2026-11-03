# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime HTTP consumers use the provider and retain their ownership rules."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.adapters.adapter_onex_tool_execution import AdapterONEXToolExecution
from omnibase_infra.handlers.handler_http import HandlerHttpRest
from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition
from omnibase_infra.runtime.models.model_http_client_config import ModelHttpClientConfig
from omnibase_infra.runtime.providers.provider_http_client import ProviderHttpClient

pytestmark = pytest.mark.integration


class _ReadbackServer:
    def __init__(self) -> None:
        self.hits: list[str] = []
        hits = self.hits

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                hits.append(self.command)
                body = b'{"ready": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.do_GET()

            def log_message(self, fmt: str, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/readback"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _ReadbackServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


async def test_runtime_http_handler_uses_provider_and_closes_its_client() -> None:
    provider = ProviderHttpClient(ModelHttpClientConfig(timeout_seconds=17))
    client = await provider.create()
    handler = HandlerHttpRest(MagicMock())
    with (
        patch.object(
            ProviderHttpClient, "create", new=AsyncMock(return_value=client)
        ) as create_client,
        _ReadbackServer() as server,
    ):
        await handler.initialize({})
        try:
            response = await handler.execute(
                {"operation": "http.get", "payload": {"url": server.url}}
            )
            assert response.result["payload"]["status_code"] == 200
            assert server.hits == ["GET"]
            assert client.timeout.connect == 17
            create_client.assert_awaited_once()
        finally:
            await handler.shutdown()
    assert client.is_closed


async def test_tool_adapter_uses_temporary_provider_client_and_closes_it() -> None:
    provider = ProviderHttpClient(ModelHttpClientConfig(follow_redirects=False))
    client = await provider.create()
    supplied_provider = MagicMock(spec=ProviderHttpClient)
    supplied_provider.create = AsyncMock(return_value=client)
    adapter = AdapterONEXToolExecution(
        MagicMock(), http_client_provider=supplied_provider
    )
    with _ReadbackServer() as server:
        result = await adapter.execute(
            ModelMCPToolDefinition(
                name="readback", description="readback", endpoint=server.url
            ),
            {},
            uuid4(),
        )
        assert result == {"success": True, "result": {"ready": True}}
        assert server.hits == ["POST"]
        supplied_provider.create.assert_awaited_once()
    assert client.is_closed


async def test_tool_adapter_preserves_caller_owned_client() -> None:
    provider = MagicMock(spec=ProviderHttpClient)
    provider.create = AsyncMock()
    async with httpx.AsyncClient() as client:
        adapter = AdapterONEXToolExecution(
            MagicMock(), http_client=client, http_client_provider=provider
        )
        with _ReadbackServer() as server:
            result = await adapter.execute(
                ModelMCPToolDefinition(
                    name="readback", description="readback", endpoint=server.url
                ),
                {},
                uuid4(),
            )
            assert result["success"] is True
            assert server.hits == ["POST"]
        assert not client.is_closed
        provider.create.assert_not_awaited()
