# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""configure_tracing exports a dispatch span to ``<endpoint>/v1/traces`` over real HTTP (OMN-22097).

The receiver stands in for Phoenix: it answers 405 to anything but ``POST /v1/traces``, as
Phoenix answered the old exporter's POST to ``/``. The span arrives only when the exporter
URL carries the signal path.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.tracing import configure_tracing

pytestmark = pytest.mark.integration


class _Payload:
    user_id = "u-1"


@pytest.mark.asyncio
async def test_dispatch_span_is_posted_to_v1_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[str, bytes]] = []
    refused: list[str] = []

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.path == "/v1/traces":
                received.append((self.path, body))
                self.send_response(200)
            else:
                refused.append(self.path)
                self.send_response(405)
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    installed: list[TracerProvider] = []

    def _install(provider: TracerProvider) -> None:
        installed.append(provider)

    try:
        with patch.object(trace, "set_tracer_provider", _install):
            assert configure_tracing() is True
        (provider,) = installed
        monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)

        engine = MessageDispatchEngine()

        async def dispatcher(envelope: ModelEventEnvelope[object]) -> str:
            return "output.topic.v1"

        engine.register_dispatcher(
            dispatcher_id="event-handler",
            dispatcher=dispatcher,
            category=EnumMessageCategory.EVENT,
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="event-route",
                topic_pattern="*.user.events.*",
                message_category=EnumMessageCategory.EVENT,
                dispatcher_id="event-handler",
            )
        )
        engine.freeze()
        await engine.dispatch(
            "dev.user.events.v1", ModelEventEnvelope(payload=_Payload())
        )

        assert provider.force_flush()
    finally:
        server.shutdown()
        server.server_close()

    assert refused == []
    assert len(received) == 1
    assert b"message_dispatch_engine.dispatch" in received[0][1]
