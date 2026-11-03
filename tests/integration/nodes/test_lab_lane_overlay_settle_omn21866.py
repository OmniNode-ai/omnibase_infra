# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exercise the resolver's HTTP reader, contract policy and busy handler together."""

from __future__ import annotations

import http.server
import json
import threading
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from tests.ci.test_resolve_lab_lane_deploy_wait_omn19811 import LANE_A, resolver
from tests.ci.test_resolve_lab_lane_overlay_settle_omn21866 import NEXT_JOB, SETTLING

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("snapshot", "expected_code", "expected_reason"),
    [(SETTLING, 0, ""), (NEXT_JOB, 1, "state=deploying")],
    ids=["overlay-settle-proceeds", "compose-deploy-waits"],
)
def test_agent_http_readback_drives_resolver_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    snapshot: dict[str, Any],
    expected_code: int,
    expected_reason: str,
) -> None:
    """A 503 agent body remains readable and the handler grades that real body."""
    reads: list[str] = []

    class Agent(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            reads.append(self.path)
            surface = self.path.removeprefix("/")
            self.send_response(503 if surface == "health" else 200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(snapshot[surface]).encode())

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    real_open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open

    def open_agent(url: str, timeout: float) -> Any:
        # Keep overlay validation on the declared lane; route only its HTTP
        # transport to this test's server, as a lab agent would answer it.
        assert url.startswith(LANE_A["deploy_agent_url"] + "/")
        return real_open(
            f"http://127.0.0.1:{server.server_port}{urlsplit(url).path}",
            timeout=timeout,
        )

    clock = [0.0]
    monkeypatch.setattr(resolver, "_open", open_agent)
    monkeypatch.setattr(resolver, "http_health", lambda url, timeout: None)
    monkeypatch.setattr(resolver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        resolver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    env_file = tmp_path / "env"
    try:
        code = resolver.main(
            {
                "LAB_LANES_JSON": json.dumps([LANE_A]),
                "LANE_REQUIRE": "gateway_url",
                "LANE_DEPLOY_WAIT_SECONDS": "1",
                "LANE_DEPLOY_POLL_SECONDS": "1",
                "LANE_DEPLOY_WAIT_WHILE_BUSY": "TrUe",
                "GITHUB_ENV": str(env_file),
            }
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert code == expected_code
    assert reads.count("/health") >= 2
    assert reads.count("/queue") >= 2
    if expected_code == 0:
        assert f"LANE_NAME={LANE_A['name']}\n" in env_file.read_text()
        assert clock[0] == 0
    else:
        assert expected_reason in capsys.readouterr().out
        assert not env_file.exists()
        assert clock[0] == 1
