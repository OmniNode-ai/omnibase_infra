# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The local-model doctor check judges every local chat backend on the wire (OMN-20489).

Two real ``http.server`` model servers listen on the loopback interface. The
shipped ``GatewayTransportHttpx`` transport probes each one's ``/v1/models``, so
a later local rung whose declared model its server does not serve is reported
even when the first rung is healthy.
"""

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks.check_delegation_local_model import (
    CheckDelegationLocalModel,
)
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)

pytestmark = pytest.mark.integration

_SERVED = "local/served-model"
_STALE = "local/stale-model"


def _models_handler(seen_paths: list[str]) -> type[BaseHTTPRequestHandler]:
    class _ModelsHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            seen_paths.append(self.path)
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps({"data": [{"id": _SERVED}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            del args

    return _ModelsHandler


@pytest.fixture
def model_server() -> Iterator[Callable[[], tuple[str, list[str]]]]:
    servers: list[tuple[HTTPServer, threading.Thread]] = []

    def start() -> tuple[str, list[str]]:
        seen_paths: list[str] = []
        server = HTTPServer(("127.0.0.1", 0), _models_handler(seen_paths))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return (f"http://127.0.0.1:{server.server_address[1]}", seen_paths)

    try:
        yield start
    finally:
        for server, thread in servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def _write_overlay(path: Path, backends: list[tuple[str, str]]) -> None:
    lines = ["backends:"]
    for index, (origin, model) in enumerate(backends):
        lines += [
            f"  - backend_id: local-{index}",
            "    tier: local",
            f"    endpoint_url: {origin}/v1/chat/completions",
            f"    model_name: {model}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_mismatch_on_a_later_local_backend_is_reported_over_http(
    tmp_path: Path, model_server: Callable[[], tuple[str, list[str]]]
) -> None:
    first, first_seen = model_server()
    second, second_seen = model_server()
    overlay = tmp_path / "overlay.yaml"
    _write_overlay(overlay, [(first, _SERVED), (second, _STALE)])

    check = CheckDelegationLocalModel(overlay_path=overlay, environ={})
    diagnosis = check.diagnose()
    result = check.run()

    assert diagnosis.fault is EnumDelegationDoctorFault.LOCAL_MODEL_ID_MISMATCH
    assert diagnosis.detail == (
        f"The overlay declares '{_STALE}', but the server reports {_SERVED}."
    )
    assert result.status is EnumHealthStatusValue.UNHEALTHY
    assert result.message.startswith("[local_model_id_mismatch] ")
    assert first_seen == ["/v1/models", "/v1/models"]
    assert second_seen == ["/v1/models", "/v1/models"]


def test_every_healthy_local_backend_is_probed_once_over_http(
    tmp_path: Path, model_server: Callable[[], tuple[str, list[str]]]
) -> None:
    first, first_seen = model_server()
    second, second_seen = model_server()
    overlay = tmp_path / "overlay.yaml"
    _write_overlay(overlay, [(first, _SERVED), (first, _SERVED), (second, _SERVED)])

    diagnosis = CheckDelegationLocalModel(overlay_path=overlay, environ={}).diagnose()

    assert diagnosis.fault is None
    assert diagnosis.detail == (
        f"Local model '{_SERVED}' is serving at {first}. "
        f"Local model '{_SERVED}' is serving at {second}."
    )
    assert first_seen == ["/v1/models"]
    assert second_seen == ["/v1/models"]
