# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch spans reach a real OTLP HTTP receiver through configure_tracing."""

from __future__ import annotations

import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Thread

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

pytestmark = pytest.mark.integration

# A fresh interpreter uses the real once-only global provider and BatchSpanProcessor.
# No OpenTelemetry symbol is patched: this exercises the production import sites.
_DISPATCH = """
import asyncio
from opentelemetry import trace
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.tracing import configure_tracing

async def main():
    assert configure_tracing() is True
    engine = MessageDispatchEngine()
    async def dispatch(envelope):
        return "output.topic.v1"
    engine.register_dispatcher(
        dispatcher_id="http-test", dispatcher=dispatch, category=EnumMessageCategory.EVENT,
    )
    engine.register_route(ModelDispatchRoute(
        route_id="http-route", topic_pattern="*.user.events.*",
        message_category=EnumMessageCategory.EVENT, dispatcher_id="http-test",
    ))
    engine.freeze()
    result = await engine.dispatch(
        "dev.user.events.v1", ModelEventEnvelope(payload={"user_id": "u-1"}),
    )
    assert result.is_successful(), result
    assert trace.get_tracer_provider().force_flush(timeout_millis=5000)
    trace.get_tracer_provider().shutdown()

asyncio.run(main())
"""


@pytest.mark.parametrize("suffix", ["", "/", "/v1/traces", "/v1/traces/"])
def test_dispatch_span_reaches_otlp_http_receiver(suffix: str) -> None:
    """A real export must POST one dispatch span and its attributes to /v1/traces."""
    requests: Queue[tuple[str, str, bytes]] = Queue()

    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.put((self.path, self.headers["Content-Type"], body))
            self.send_response(200 if self.path == "/v1/traces" else 405)
            self.end_headers()

        def log_message(self, fmt: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Receiver)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        environment = dict(os.environ)
        environment.update(
            OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{server.server_port}{suffix}",
            OTEL_SERVICE_NAME="landing-dispatch-http",
            OTEL_TRACES_EXPORTER="otlp",
            OTEL_EXPORTER_OTLP_COMPRESSION="none",
            OTEL_EXPORTER_OTLP_TRACES_COMPRESSION="none",
            OTEL_EXPORTER_OTLP_TRACES_TIMEOUT="2",
            PYTHONDONTWRITEBYTECODE="1",
        )
        completed = subprocess.run(
            [sys.executable, "-c", _DISPATCH],
            env=environment,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        path, content_type, body = requests.get(timeout=5)
        assert path == "/v1/traces"
        assert content_type == "application/x-protobuf"
        assert requests.empty(), "One dispatch must produce exactly one export"
        exported = ExportTraceServiceRequest.FromString(body)
        (resource_spans,) = exported.resource_spans
        resource = {item.key: item.value for item in resource_spans.resource.attributes}
        assert resource["service.name"].string_value == "landing-dispatch-http"
        (scope_spans,) = resource_spans.scope_spans
        (span,) = scope_spans.spans
        assert span.name == "message_dispatch_engine.dispatch"
        attributes = {item.key: item.value for item in span.attributes}
        assert (
            attributes["omninode.dispatch.topic"].string_value == "dev.user.events.v1"
        )
        assert attributes["omninode.dispatch.category"].string_value == "event"
        assert attributes["omninode.dispatch.success"].bool_value is True
        assert attributes["omninode.dispatch.duration_ms"].double_value >= 0
        assert span.end_time_unix_nano >= span.start_time_unix_nano
        assert any(span.trace_id)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
