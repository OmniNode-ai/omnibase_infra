# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A model server that answers 503 is a COULD_NOT_RUN input error, not a traceback (OMN-26534).

Scheduled run 45237853027 (2026-11-23T18:23Z) ended in step "Drive one customer
delegation and grade it" with ``urllib.error.HTTPError: HTTP Error 503: Service
Unavailable`` raised from ``tokens_predicted`` (``GET /metrics`` on the model
host) inside ``observe_live``: the read after the configured run sat outside the
``try`` that guards the first read, so the probe died with no record and the
next step, ``test -s c13-customer-local.json``, failed too. Every model-server
read goes through ``_http_get``, so the guard lives there; ``main`` already turns
a ``ProbeInputError`` into a COULD_NOT_RUN record and exit 2 (see
``test_a_public_model_host_exits_could_not_run``). The server here is a real
HTTP listener answering the recorded status, so the exception is the one urllib
raises, not a stand-in.
"""

from __future__ import annotations

import http.server
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from scripts.ci import c13_customer_local_probe as probe


@contextmanager
def _server(status: int, reason: str, body: bytes = b"") -> Iterator[str]:
    """A llama-server that answers every path with one status."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(status, reason)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.unit
@pytest.mark.live_contact(
    "tests/fixtures/omn26534/c13-metrics-503-run-45237853027.json"
)
def test_a_503_from_the_model_server_is_a_probe_input_error_naming_it(
    recorded_response: dict[str, Any],
) -> None:
    response = recorded_response["response"]
    with _server(response["status"], response["reason"]) as base_url:
        with pytest.raises(probe.ProbeInputError, match="is not answering") as caught:
            probe._http_get(f"{base_url}/metrics")
    assert f"{base_url}/metrics" in str(caught.value)
    assert "503" in str(caught.value)


@pytest.mark.unit
def test_a_refused_connection_is_a_probe_input_error_too() -> None:
    with _server(200, "OK") as base_url:
        pass
    with pytest.raises(probe.ProbeInputError, match="is not answering"):
        probe._http_get(f"{base_url}/metrics")


@pytest.mark.unit
def test_a_healthy_server_still_returns_its_body() -> None:
    """Positive control: the guard does not eat a 200."""
    with _server(200, "OK", b"llamacpp:tokens_predicted_total 7\n") as base_url:
        assert probe._http_get(f"{base_url}/metrics") == (
            "llamacpp:tokens_predicted_total 7\n"
        )
