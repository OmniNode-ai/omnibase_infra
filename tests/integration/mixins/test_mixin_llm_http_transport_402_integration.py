# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A 402 over a real socket is one attempt, no retry, no breaker failure (OMN-20608).

The unit tests answer through ``httpx.MockTransport``. These tests send the
request through a real ``httpx.AsyncClient`` to a loopback HTTP server, so the
wire path (connection, status line, headers, body) is the one production takes.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.errors import InfraPaymentRequiredError, InfraUnavailableError
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport

PAYLOAD = {"messages": [{"role": "user", "content": "hello"}]}
HEADER_VALUE = "H" * 4096
BODY = b'{"pay_to":"' + b"B" * 2000 + b'"}'


class _Harness(MixinLlmHttpTransport):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._init_llm_http_transport(
            target_name="loopback-llm",
            max_timeout_seconds=30.0,
            max_retry_after_seconds=5.0,
            http_client=client,
        )


class _Server:
    """Loopback HTTP server that answers every POST with a fixed status."""

    def __init__(self, status: int) -> None:
        self.hits = 0
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("content-length", "0")))
                outer.hits += 1
                self.send_response(status)
                self.send_header("content-type", "application/json")
                if status == 402:
                    self.send_header("payment-required", HEADER_VALUE)
                self.send_header("content-length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)

            def log_message(self, fmt: str, *args: object) -> None:
                return

        self._httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/v1/chat/completions"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> _Server:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "t" * 32)
    monkeypatch.setenv("LLM_ENDPOINT_CIDR_ALLOWLIST", "127.0.0.0/8")
    monkeypatch.setenv("LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST", "api.z.ai")
    for cls in (MixinLlmHttpTransport, _Harness):
        cls._LOCAL_LLM_CIDRS = None
        cls._CLOUD_LLM_HOSTS = None
    yield
    for cls in (MixinLlmHttpTransport, _Harness):
        cls._LOCAL_LLM_CIDRS = None
        cls._CLOUD_LLM_HOSTS = None


@pytest.mark.integration
class TestPaymentRequiredOverRealSocket:
    async def test_402_is_one_request_on_the_wire_and_no_breaker_failure(self) -> None:
        with _Server(402) as server:
            async with httpx.AsyncClient() as client:
                harness = _Harness(client)
                before = harness._circuit_breaker_failures
                with pytest.raises(InfraPaymentRequiredError) as exc_info:
                    await harness._execute_llm_http_call(
                        url=server.url,
                        payload=PAYLOAD,
                        correlation_id=uuid4(),
                        max_retries=3,
                    )
        err = exc_info.value
        assert server.hits == 1
        assert harness._circuit_breaker_failures == before
        assert err.status_code == 402
        assert err.header_sha256 == hashlib.sha256(HEADER_VALUE.encode()).hexdigest()
        assert err.body_sha256 == hashlib.sha256(BODY).hexdigest()
        assert err.body_byte_length == len(BODY)

    async def test_500_over_the_same_socket_still_retries(self) -> None:
        with _Server(500) as server:
            async with httpx.AsyncClient() as client:
                harness = _Harness(client)
                with pytest.raises(InfraUnavailableError):
                    await harness._execute_llm_http_call(
                        url=server.url,
                        payload=PAYLOAD,
                        correlation_id=uuid4(),
                        max_retries=1,
                    )
        assert server.hits == 2
        assert harness._circuit_breaker_failures >= 1
