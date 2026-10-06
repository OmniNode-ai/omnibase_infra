# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A 402 over a real socket reaches the MCP caller as a fixed refusal (OMN-20610).

The unit tests answer through ``httpx.MockTransport``. These tests run both
adapters, ``ONEXToMCPAdapter.invoke_tool`` over ``AdapterONEXToolExecution``,
against a loopback HTTP server that answers every POST with a 402 whose body
carries a ``payTo`` value, so the wire path is the one production takes.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.adapters.adapter_onex_tool_execution import (
    PAYMENT_REFUSED_MESSAGE,
    AdapterONEXToolExecution,
)
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import ONEXToMCPAdapter

PAY_TO = "0xPAYTOSECRET1234567890abcdef"
BODY = json.dumps({"accepts": [{"payTo": PAY_TO}]}).encode()


class _Server:
    def __init__(self) -> None:
        self.hits = 0
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", "0")))
                outer.hits += 1
                self.send_response(402)
                self.send_header("content-type", "application/json")
                self.send_header("payment-required", PAY_TO)
                self.send_header("content-length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)

            def log_message(self, fmt: str, *args: object) -> None:
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/execute"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> _Server:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.mark.integration
class TestMcpToolPaymentRefusalOverRealSocket:
    async def test_402_is_one_request_fixed_sentence_no_breaker_failure(self) -> None:
        with _Server() as server:
            async with httpx.AsyncClient() as client:
                execution = AdapterONEXToolExecution(
                    container=MagicMock(), http_client=client
                )
                adapter = ONEXToMCPAdapter(node_executor=execution)
                await adapter.register_node_as_tool(
                    node_name="paid_tool",
                    description="d",
                    parameters=[],
                    version="1.0.0",
                    timeout_seconds=5,
                )
                adapter._tool_cache["paid_tool"].execution_endpoint = server.url
                result = await adapter.invoke_tool("paid_tool", {}, uuid4())
        assert server.hits == 1
        assert execution._circuit_breaker_failures == 0
        assert result == {
            "content": [{"type": "text", "text": PAYMENT_REFUSED_MESSAGE}],
            "isError": True,
        }
        assert PAY_TO not in json.dumps(result)
