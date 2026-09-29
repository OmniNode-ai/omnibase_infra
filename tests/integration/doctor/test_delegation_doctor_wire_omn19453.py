# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Delegation doctor checks over a real local socket (OMN-19453).

The unit suite injects fake transports.  This suite runs the shipped
``GatewayTransportHttpx`` transport against a real ``http.server`` listening on
the loopback interface, so the whoami request the doctor sends (path, header,
status handling) and the connection-refused mapping are exercised on the wire.
"""

import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from omnibase_core.enums.enum_health_status_value import EnumHealthStatusValue
from omnibase_infra.doctor.checks.check_delegation_gateway import (
    CheckDelegationGateway,
)
from omnibase_infra.doctor.checks.check_delegation_key import CheckDelegationKey
from omnibase_infra.doctor.enum_delegation_doctor_fault import (
    EnumDelegationDoctorFault,
)
from omnibase_infra.gateway.client.gateway_identity_verifier import (
    GATEWAY_WHOAMI_PATH,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)

pytestmark = pytest.mark.integration

_GOOD_KEY = "onxk_omn19453-good-key"  # pragma: allowlist secret
_BAD_KEY = "onxk_omn19453-bad-key"  # pragma: allowlist secret


class _WhoamiHandler(BaseHTTPRequestHandler):
    """Serve the gateway whoami path: 200 for the good key, 401 otherwise."""

    seen_paths: list[str] = []

    def do_GET(self) -> None:
        type(self).seen_paths.append(self.path)
        if self.path != GATEWAY_WHOAMI_PATH:
            self.send_response(404)
            self.end_headers()
            return
        authorised = self.headers.get("x-api-key") == _GOOD_KEY
        body = (
            b'{"tenant_id": "11111111-1111-4111-8111-111111111111",'
            b' "tenant_slug": "acme"}'
            if authorised
            else b"{}"
        )
        self.send_response(200 if authorised else 401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        del format, args


@pytest.fixture
def gateway_url() -> Iterator[str]:
    _WhoamiHandler.seen_paths = []
    server = HTTPServer(("127.0.0.1", 0), _WhoamiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def _save(home: Path, *, base_url: str, api_key: str) -> None:
    StoreGatewayCredential(onex_home=home).save_api_key(
        tenant_slug="acme",
        api_key=api_key,
        base_url=base_url,
    )


def test_reachable_gateway_and_good_key_are_healthy(
    tmp_path: Path, gateway_url: str
) -> None:
    _save(tmp_path, base_url=gateway_url, api_key=_GOOD_KEY)

    gateway = CheckDelegationGateway(onex_home=tmp_path).run()
    key = CheckDelegationKey(onex_home=tmp_path).run()

    assert gateway.status == EnumHealthStatusValue.HEALTHY
    assert key.status == EnumHealthStatusValue.HEALTHY
    assert set(_WhoamiHandler.seen_paths) == {GATEWAY_WHOAMI_PATH}


def test_wrong_key_is_named_from_a_real_401(tmp_path: Path, gateway_url: str) -> None:
    _save(tmp_path, base_url=gateway_url, api_key=_BAD_KEY)

    diagnosis = CheckDelegationKey(onex_home=tmp_path).diagnose()
    result = CheckDelegationKey(onex_home=tmp_path).run()

    assert diagnosis.fault == EnumDelegationDoctorFault.WRONG_KEY
    assert result.status == EnumHealthStatusValue.UNHEALTHY
    assert _BAD_KEY not in result.message


def test_refused_connection_is_named_gateway_down(tmp_path: Path) -> None:
    _save(tmp_path, base_url=_closed_port_url(), api_key=_GOOD_KEY)

    diagnosis = CheckDelegationGateway(onex_home=tmp_path).diagnose()

    assert diagnosis.fault == EnumDelegationDoctorFault.GATEWAY_DOWN
    assert str(tmp_path / "config.yaml") in diagnosis.fix
